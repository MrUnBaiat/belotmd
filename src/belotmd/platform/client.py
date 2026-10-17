"""
client.py — Python end of the Node bridge. Message plumbing only.

Owns the `node bridge.js` subprocess, the local WebSocket to it, and the
outgoing message vocabulary. It holds no game state: raw frames go straight
to the callbacks in arrival order.
"""

import asyncio
import json
import os
import secrets
import socket
import subprocess
import time
import websockets
# Imported by name: `websockets.exceptions` is a lazy attribute in current
# versions, so touching it inside an `except` clause raises AttributeError
# WHILE handling the original error, and the real failure is lost.
from websockets.exceptions import WebSocketException

from . import tournaments as tour
from .protocol import (CHAR_TO_CARD, LEAVE_NAMES,  # noqa: F401  (re-exported)
                       LEAVE_KICKED, LEAVE_OTHER_SESSION,
                       LEAVE_POSITION_CHANGED, NOT_STARTED,
                       RECOVER_LATER, RECOVER_NOW, RECOVER_SOON, RECOVER_STOP,
                       SUIT_TO_INT, between_matches, leave_recovery,
                       match_underway)
from .tables import CREATE_TABLE_BODY, find_table, is_joinable, table_id_of


class TableCreationRefused(RuntimeError):
    """belot.md refused to create a table several times in a row.

    It does this, silently, once an account has left too many matches before
    they finished: the create request lands back on the home page and no table
    exists. It lifts on its own, but not within minutes, so retrying is
    pointless -- and a run that keeps retrying looks, from the outside, exactly
    like a run playing nothing for hours.
    """


class TableJoinRefused(RuntimeError):
    """The table we were told to join will not have us, and asking again is
    the wrong answer: a password table that refused the seat several times in
    a row (each further try is one more wrong password), or a table whose
    owner removed us."""


class TournamentError(RuntimeError):
    """The tournament run cannot go on: belot.md refused the enrolment, offers
    no tournament for our status, never answered at all, or the account was
    opened somewhere else while it played."""


class BelotClient:
    def __init__(self, cookies: str, bridge_port: int = 0,
                 reconnect: bool = True, retry_delay_s: float = 300.0,
                 rejoin_delay_s: float = 5.0, table_mode: str = "lobby",
                 table_id: str = None, table_creator: str = None,
                 join_poll_s: float = 2.0, join_max_polls: int = 20,
                 pair_restart_pause_s: float = 10.0,
                 max_create_failures: int = 3,
                 table_password: str = None,
                 tournament_min_status: int = 5,
                 tournament_enroll: bool = True,
                 tournament_poll_s: float = tour.POLL_S):
        """`reconnect` keeps the bot looking for tables instead of exiting.

        Two delays, because two very different situations end a session:
          rejoin_delay_s  a match ended or the table dissolved -- go straight
                          back out and find another one;
          retry_delay_s   there is nothing to join (no open tables, or we were
                          kicked). Waiting is the only useful move, and
                          hammering the lobby every few seconds is rude.

        `bridge_port` 0 means "pick a free one", which is what makes two
        accounts on one machine safe. With the old fixed 8765 the second
        client's daemon could not bind, and the client then connected to the
        FIRST client's bridge -- and was handed another account's frames, hand
        included. The token below closes the remaining gap: a stale or foreign
        daemon on the port we picked is detected before we send any cookie.
        """
        self.cookies = cookies
        self.bridge_port = bridge_port
        self.bridge_token = secrets.token_hex(16)
        # Which table to sit at. "lobby" is the historic behaviour (the first
        # open public table). A pair uses the next two: the host "create"s a
        # table and the guest "join"s the one that host made. "tournament"
        # enrols in one tournament and plays the tables it seats us at.
        self.table_mode = table_mode
        self.table_id = table_id
        self.table_creator = table_creator
        # A private table's password, sent with every seat request in "join"
        # mode, and the refused seats in a row at such a table.
        self.table_password = table_password or None
        self._refused_joins = 0
        # Watching the lobby for our partner's table: how often, and how many
        # fruitless looks before standing down for a while.
        self.join_poll_s = float(join_poll_s)
        self.join_max_polls = int(join_max_polls)
        self.pair_restart_pause_s = float(pair_restart_pause_s)
        # Refused creates in a row, and the error that ends the session once
        # there are too many. Raised only after the loop has closed cleanly.
        self.max_create_failures = int(max_create_failures)
        self._create_failures = 0
        self.fatal = None
        self._join_polls = 0
        # The table we last sat at, and a note to go straight back to it
        # rather than asking the lobby (see the LEAVE handler).
        self._last_table_id = None
        self._rejoin_hint = None
        self.reconnect = reconnect
        self.retry_delay_s = float(retry_delay_s)
        self.rejoin_delay_s = float(rejoin_delay_s)
        self.sessions_played = 0
        self.ws = None
        self.node_process = None
        self.my_player_id = None
        self.room_id = None
        self._queue = None
        self._consumer = None
        self._phase = NOT_STARTED
        self._in_room = False
        # Set while a LEAVE_TABLE of our own is in flight. "We asked to leave"
        # normally ends the session; when we left in order to make a better
        # table, it must not.
        self._leaving_to_restart = False
        # Losing the connection to a table (see `_after_a_drop`): when it last
        # happened, the room it took with it, and until when a failed attempt
        # to get back is retried at once rather than after the long wait.
        self._last_drop_at = None
        self._lost_room = None
        self._rescue_until = 0.0
        # True when the room we are in is the match we lost the connection
        # to, rather than a new table -- the bot must not treat it as fresh.
        self.resumed_match = False

        # --- "tournament" mode ----------------------------------------------
        # Which tournaments we may enter: exactly the account's own status.
        self.tournament_min_status = int(tournament_min_status)
        # False when this run may only go back to a tournament it is already
        # in, never enrol in a new one (a supervisor resuming after a crash).
        self.tournament_enroll = bool(tournament_enroll)
        self.tournament_poll_s = float(tournament_poll_s)
        # Set by whoever is stopping the run. In the waiting room it means
        # withdraw now; in the bracket it means stop once we are out of it.
        self.stop_requested = False
        # Why our tournament ended, once it has: the session's normal end.
        self.tournament_over = None
        # Answers tournaments.php actually gave: proof of life while there are
        # no frames to be had, in a waiting room or between two rounds.
        self.tournament_answers = 0
        # What happened, for the session log. Our own results only.
        self.tournament = {}
        self._t_failures = 0          # failed calls in a row
        self._t_enrolls = 0           # enrolments sent
        self._t_withdraws = 0         # withdrawals sent
        self._t_refusal = None        # the last enrolment refusal, unconfirmed
        self._t_stale_seat = 0        # polls offering a seat at a decided table
        self._t_in = False            # the page has shown us in a tournament
        self._t_said = None           # last status line, so polls stay quiet

    # The local socket to the bridge: no keepalive, and no flow control on
    # reading. It is a loopback link to our own child process, whose death
    # closes it at once, so a keepalive detects nothing there -- but it can
    # kill a healthy link. A ping is answered only while the event loop reads,
    # and the loop stops reading when the unread frames reach `max_queue` (as
    # during a recovery wait) or while an agent's decision blocks it. Live
    # (2026-09-29) that closed the socket, 1011 "keepalive ping timeout", just
    # after the bridge had rejoined a match; the bridge, its client gone, left
    # the table, and the match was abandoned. A bridge that hangs without
    # dying shows up instead as a run that receives no frames.
    BRIDGE_SOCKET = {"ping_interval": None, "max_queue": None}

    # Failed tournaments.php calls in a row, before it has answered even once,
    # that end the run: a logged-out cookie gets a page instead of JSON, every
    # time, and polling it for an hour proves nothing more.
    MAX_TOURNAMENT_FAILURES = 5
    # Enrolments, and withdrawals, sent before giving up on the site taking
    # one. Each is re-sent only when the next poll shows it did not land.
    MAX_ENROLLS = 3
    MAX_WITHDRAWS = 3
    # Polls (3 s each) that a seat at a table whose match is already decided
    # is declined, before it is taken anyway (see `_on_tournament_state`).
    STALE_SEAT_POLLS = 10

    # Refused seats in a row at a password table before giving up on it. A
    # table that vanished or filled as we sat down explains one or two; after
    # that the likeliest cause is the password, and every further try is
    # another wrong one sent to somebody's private table.
    MAX_REFUSED_JOINS = 3

    # How long to wait for `node bridge.js` to bind its port.
    DAEMON_TIMEOUT_S = 15.0
    DAEMON_POLL_S = 0.25
    # How long to wait for that daemon to identify itself.
    HELLO_TIMEOUT_S = 5.0
    # A lost connection is rejoined at once -- but one that keeps dropping is
    # rejoined on the ordinary short cadence, not in a tight loop.
    DROP_REPEAT_S = 60.0
    # After losing the connection in a match, a failed attempt to get back is
    # retried on the short cadence for this long rather than after the long
    # wait: the match is being played without us meanwhile.
    MATCH_RESCUE_S = 120.0

    @staticmethod
    def _free_port():
        """Ask the OS for a free port. Racy in principle -- another process
        could take it between here and the daemon's bind -- but the daemon then
        exits with a clear message and the token check makes the silent
        wrong-bridge outcome impossible."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    # Tells bridge.js that its stdin is the pipe below, and to live exactly as
    # long as it stays open.
    LIFELINE_ARG = "--lifeline"

    @staticmethod
    def _own_process_group():
        """Popen arguments that keep console control signals away from the
        bridge.

        A supervisor stops a bot at the end of the MATCH by sending it a break
        (Windows: CTRL_BREAK_EVENT to the bot's process group) or an interrupt
        (a terminal's Ctrl-C reaches the whole foreground group). The bot traps
        it and finishes the match -- but a bridge in the same group died on the
        spot (exit 0xC000013A, reproduced), the socket dropped, and the table
        was abandoned mid-match: the one thing the graceful stop exists to
        prevent, and something belot.md penalises.
        """
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _start_node_daemon(self):
        if not self.bridge_port:
            self.bridge_port = self._free_port()
        print(f"[SDK] Launching background Node.js bridge on port "
              f"{self.bridge_port}...")
        bridge_path = os.path.join(os.path.dirname(__file__), "bridge.js")
        if not os.path.exists(bridge_path):
            raise FileNotFoundError(f"bridge.js not found at {bridge_path}")
        try:
            # Out of our process group, the bridge no longer dies with our
            # console, so its lifetime is tied to ours by a pipe instead: the
            # OS closes our end however this process ends -- close(), an
            # exception, or a hard kill that never runs close() -- and the
            # bridge exits on EOF. Without that, a killed bot would leave a
            # daemon behind that is still seated at the table.
            self.node_process = subprocess.Popen(
                ["node", bridge_path, str(self.bridge_port), self.bridge_token,
                 self.LIFELINE_ARG],
                stdin=subprocess.PIPE, **self._own_process_group())
        except FileNotFoundError:
            raise RuntimeError(
                "`node` is not on PATH -- the bridge daemon cannot start."
            ) from None

    async def _await_daemon(self, uri):
        """Poll until the daemon accepts a connection.

        The old code slept a flat 1.5s and connected blind. On a cold start,
        a loaded machine or a slower Node boot that races the bind and raises
        an unhandled ConnectionRefusedError -- which looks exactly like a
        belot.md outage in the logs. Poll instead, and surface a real error if
        the daemon died (bad bridge.js, port 8765 already in use, missing
        dependency) rather than retrying into a timeout.
        """
        deadline = time.monotonic() + self.DAEMON_TIMEOUT_S
        attempt = 0
        while True:
            attempt += 1
            rc = self.node_process.poll() if self.node_process else None
            if rc is not None:
                raise RuntimeError(
                    f"bridge daemon exited with code {rc} before accepting a "
                    f"connection -- check its stderr (port {self.bridge_port} "
                    f"already in use? `npm install` not run?)"
                )
            try:
                ws = await websockets.connect(uri, **self.BRIDGE_SOCKET)
                await self._verify_bridge(ws)
                if attempt > 1:
                    print(f"[SDK] Bridge daemon ready after {attempt} attempts.")
                return ws
            except (OSError, WebSocketException) as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"bridge daemon did not accept a connection on "
                        f"{uri} within {self.DAEMON_TIMEOUT_S}s ({exc})"
                    ) from exc
                await asyncio.sleep(self.DAEMON_POLL_S)

    async def _verify_bridge(self, ws):
        """The daemon's first frame must be OUR token.

        Anything else means we are talking to someone else's bridge -- another
        account's, or a stale one left on this port -- and continuing would
        send this account's cookies down it and accept that account's frames as
        ours. Refuse before a single cookie leaves the process.
        """
        try:
            raw = await asyncio.wait_for(ws.recv(), self.HELLO_TIMEOUT_S)
            hello = json.loads(raw)
        except (asyncio.TimeoutError, ValueError, WebSocketException) as exc:
            await ws.close()
            raise RuntimeError(
                f"the daemon on port {self.bridge_port} did not identify "
                f"itself ({type(exc).__name__}). Another program, or an old "
                f"bridge.js, is listening there."
            ) from exc

        if hello.get("event") != "HELLO" or hello.get("token") != self.bridge_token:
            await ws.close()
            raise RuntimeError(
                f"the daemon on port {self.bridge_port} is not ours (first "
                f"frame was {hello!r}). Refusing to hand it this account's "
                f"cookies."
            )

    async def connect(self, on_state_callback, on_message_callback=None):
        self._start_node_daemon()
        uri = f"ws://127.0.0.1:{self.bridge_port}"

        async with await self._await_daemon(uri) as ws:
            self.ws = ws
            print("[SDK] Connected to bridge daemon. Releasing join request...")
            await self._request_table(ws)

            self._queue = asyncio.Queue()
            consumer = asyncio.create_task(
                self._consume(on_state_callback, on_message_callback))
            self._consumer = consumer          # keep a strong reference

            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                event = msg.get("event")

                if event == "CONNECTED":
                    self.room_id = msg.get("roomId")
                    self.my_player_id = msg.get("playerId")
                    self._in_room = True
                    self._phase = NOT_STARTED
                    self._join_polls = 0
                    self._create_failures = 0
                    self._refused_joins = 0
                    # Back in the match we lost, or somewhere new? The bridge
                    # returns to an active game first, so it is the same room
                    # whenever the platform still had us in it.
                    self.resumed_match = (
                        time.monotonic() < self._rescue_until
                        and self.room_id is not None
                        and self.room_id == self._lost_room)
                    self._rescue_until = 0.0
                    self._lost_room = None
                    self.sessions_played += 1
                    print(f"[SDK] Joined Game Room: {self.room_id} | Player ID: {self.my_player_id}")
                    if self.resumed_match:
                        print("[SDK] Back in the match we lost the connection "
                              "to.")
                    # Also what takes our seat back from the platform's bot,
                    # if it stood in while we were away.
                    await self.deactivate_bot()

                elif event == "STATE":
                    # Serialized, in-order delivery. The previous
                    # asyncio.create_task() here ran handlers CONCURRENTLY over
                    # a shared StateSynchronizer: a handler that awaited (e.g.
                    # sending a declaration) resumed against an env another
                    # frame had already overwritten, producing an all-zero
                    # legal mask and silently dropping our turn.
                    # It also leaked the task -- asyncio only keeps a weak
                    # reference, so an un-stored task can be GC'd mid-flight.
                    #
                    # Remember the phase rather than latching "a match has
                    # started": one room hosts many consecutive matches, so a
                    # latch never returns and every legal reseat between them
                    # reads as impossible.
                    self._phase = (msg.get("data") or {}).get(
                        "currentPhase", self._phase)
                    await self._queue.put(("STATE", msg["data"]))

                elif event == "MESSAGE" and on_message_callback:
                    # Same queue, so STATE/MESSAGE ordering is preserved.
                    await self._queue.put(("MESSAGE", (msg.get("type"),
                                                       msg.get("data"))))

                elif event == "LOBBY":
                    # "join" mode, step two: the lobby came back, so pick our
                    # partner's table out of it.
                    wanted = find_table(msg.get("mese"),
                                        table_id=self.table_id,
                                        creator=self.table_creator)
                    who = self.table_creator or self.table_id

                    if wanted is not None and is_joinable(wanted):
                        found = table_id_of(wanted)
                        print(f"[SDK] Found {who}'s table {found}; taking a seat.")
                        self._join_polls = 0
                        await self._request_table(ws, table_id=found)
                        continue

                    # Either it is not listed yet, or strangers have taken
                    # every seat. A full table can never become ours, so there
                    # is nothing to watch for: stand down, let the host delete
                    # it, and start the next attempt together.
                    self._join_polls += 1
                    full = wanted is not None
                    spent = self._join_polls >= self.join_max_polls
                    if full or spent:
                        why = ("it filled up before we got a seat" if full else
                               f"{self._join_polls} looks and it never appeared")
                        self._join_polls = 0
                        if not await self._recover(
                                ws, RECOVER_SOON,
                                f"{who}'s table: {why} -- standing down",
                                delay=self.pair_restart_pause_s):
                            break
                    else:
                        if not await self._recover(
                                ws, RECOVER_SOON,
                                f"{who}'s table is not in the lobby yet "
                                f"({self._join_polls}/{self.join_max_polls})",
                                delay=self.join_poll_s):
                            break

                elif event == "TOURNAMENT":
                    # "tournament" mode: an answer from tournaments.php. Each
                    # one leads to exactly one next request, so there is never
                    # more than one poll in flight.
                    if not await self._on_tournament(ws, msg):
                        break

                elif event == "ERROR":
                    code = msg.get("code")
                    detail = msg.get("message") or code
                    print(f"[SDK Error]: {detail}")
                    if code == "NO_ROOM":
                        # The bridge's answer to a move sent while no room was
                        # open: our agent acting on a frame from before the
                        # room closed. It says nothing about any join, so it
                        # must not be read as one failing. Live (2026-09-29) it
                        # was, and started a second five-minute wait while the
                        # bridge was already back in the match.
                        continue
                    if self.table_mode == "tournament":
                        # Taking the seat the tournament gave us failed --
                        # gameplay_new.php had no room on it yet, or the
                        # network. The tournament knows whether the seat is
                        # still ours, so ask it rather than any lobby.
                        if not self._in_room:
                            print(f"[SDK][Tournament] could not take our seat "
                                  f"({detail}); asking the tournament again.")
                            await self._poll_tournament(ws, self.tournament_poll_s)
                        continue
                    rescuing = time.monotonic() < self._rescue_until
                    # An error while we are NOT in a room means the join
                    # itself failed -- almost always "No public open tables
                    # available." Without this the bot sat idle forever
                    # waiting for frames from a room it never entered.
                    #
                    # Only a create the platform turned down counts towards
                    # giving up. A network failure is not a refusal, and
                    # stopping on three of them said the account was barred
                    # when it was not.
                    if (not self._in_room and self.table_mode == "create"
                            and code == "CREATE_REFUSED"):
                        self._create_failures += 1
                        if self._create_failures >= self.max_create_failures:
                            self.fatal = TableCreationRefused(
                                f"belot.md refused to create a table "
                                f"{self._create_failures} times in a row "
                                f"(last: {detail}). It does this after an "
                                f"account has left too many matches before "
                                f"they finished. It lifts on its own, later.")
                            print("=" * 72)
                            print(f"[SDK][STOP] {self.fatal}")
                            print("            Not retrying: nothing will play "
                                  "until the platform allows creating again.")
                            print("=" * 72)
                            break
                    if not self._in_room:
                        # The long wait exists for "the lobby is empty", and
                        # neither case here is that. Our partner's table not
                        # being listed yet is a matter of seconds -- and in
                        # join mode EVERY failure is, because the table we want
                        # is being created, filled and deleted on that
                        # timescale. Waiting 300s there once left the guest
                        # asleep while the host churned through five tables.
                        soon = (msg.get("code") == "TABLE_NOT_FOUND"
                                or self.table_mode == "join")
                        # A failed join is a spent look like any other, and it
                        # keeps the guest on its 2s cadence rather than the
                        # 5s one meant for "a match just ended".
                        joining = self.table_mode == "join"
                        delay = self.join_poll_s if joining else None
                        if rescuing and code != "CREATE_REFUSED":
                            # Getting back to a match we lost the connection
                            # to failed -- most likely the same blip. The match
                            # goes on without us, so try again on the short
                            # cadence, and straight to that table: the lobby
                            # lists it as full, one of the seats being ours.
                            soon = True
                            delay = self.rejoin_delay_s
                            if self._last_table_id:
                                self._rejoin_hint = self._last_table_id
                        elif joining:
                            self._join_polls += 1
                            if (self.table_password
                                    and code == "TABLE_NOT_FOUND"):
                                self._refused_joins += 1
                                if self._refused_joins >= self.MAX_REFUSED_JOINS:
                                    self._give_up_on_the_table(
                                        f"{self._table_name()} refused the "
                                        f"seat {self._refused_joins} times in "
                                        f"a row (last: {detail}). Most likely "
                                        f"the password is wrong")
                                    break
                        if not await self._recover(
                                ws, RECOVER_SOON if soon else RECOVER_LATER,
                                f"could not join a table ({detail})",
                                delay=delay):
                            break

                elif event == "LEAVE":
                    code = msg.get("code")
                    label = LEAVE_NAMES.get(code, "UNKNOWN")
                    # Read before a rejoin resets the phase.
                    mid_match = not between_matches(self._phase)
                    self._in_room = False
                    restart_pause = None
                    if self.table_mode == "tournament":
                        if not await self._left_tournament_table(
                                ws, code, label, mid_match):
                            break
                        continue
                    if code == LEAVE_KICKED and self.table_mode == "join":
                        # We came to this table by name, and its owner took us
                        # off it. Looking it up again would sit us straight
                        # back down, every few seconds, for as long as they
                        # kept removing us.
                        self._give_up_on_the_table(
                            f"removed from {self._table_name()} by its owner "
                            f"({label}, {code})")
                        break
                    if code == LEAVE_POSITION_CHANGED and self._last_table_id:
                        # The host rotated the seats. Nobody was removed: the
                        # platform still has us at this table, so go back to it
                        # directly. Asking the lobby reads our own seat as
                        # taken, and a guest that stands down from its own
                        # table leaves the host holding READY until it deletes
                        # a table that was perfectly good.
                        self._rejoin_hint = self._last_table_id
                    if self._leaving_to_restart:
                        # We asked for this. The policy for a leave we
                        # requested is to STOP -- correct when a person means
                        # "I am done", fatal when the bot means "this table is
                        # no good, make another". As the creator, leaving also
                        # deletes the table, so the close can arrive as I_LEFT
                        # or as TABLE_REMOVED; neither should end the run.
                        self._leaving_to_restart = False
                        action = RECOVER_SOON
                        restart_pause = self.pair_restart_pause_s
                        print(f"[SDK] left that table on purpose ({label}); "
                              f"settling for {restart_pause:.0f}s, then making "
                              f"another.")
                    else:
                        action = leave_recovery(code, mid_match=mid_match)
                        if (action == RECOVER_NOW
                                and code != LEAVE_POSITION_CHANGED):
                            restart_pause = self._after_a_drop(mid_match)

                    if label == "UNKNOWN":
                        print(f"[SDK][WARN] unrecognised close code {code}; "
                              + ("treating it as a dropped connection, since a "
                                 "match is under way." if mid_match else
                                 "treating it as recoverable."))
                    if (code == LEAVE_POSITION_CHANGED
                            and match_underway(self._phase)):
                        # A cheap assertion on a platform invariant: the host
                        # can only rotate seats while the table is IDLE -- before
                        # a match, or between them. Mid-hand it cannot happen,
                        # so if it ever does, something about the platform is
                        # not what we think it is. We still resync rather than
                        # end an unattended run, and the synchronizer will flag
                        # the hand as degraded.
                        print("=" * 72)
                        print(f"[SDK][VIOLATION] {label} ({code}) AFTER the "
                              f"match began (phase {self._phase}). This "
                              f"is not supposed to be possible.")
                        print("                 Seat indices have moved, so "
                              "every seat-indexed belief is now wrong.")
                        print("                 Resyncing, but keep the "
                              "recording -- this is worth understanding.")
                        print("=" * 72)

                    if not await self._recover(
                            ws, action, f"{label} ({code})",
                            avoid_last=(code == LEAVE_KICKED),
                            delay=restart_pause):
                        break

            await self._queue.put((None, None))     # drain sentinel
            await consumer

        # Outside the connection block, so the bridge socket is closed and the
        # consumer drained before the caller hears about it.
        if self.fatal is not None:
            raise self.fatal

    def _table_name(self):
        """The table "join" mode is after, for a message."""
        return (f"{self.table_creator}'s table" if self.table_creator
                else f"table {self.table_id}")

    def _give_up_on_the_table(self, why):
        """End the session for good; `connect` raises this once it has closed."""
        self.fatal = TableJoinRefused(f"{why}. Not going back.")
        print("=" * 72)
        print(f"[SDK][STOP] {self.fatal}")
        print("=" * 72)

    # ------------------------------------------------------------ tournaments
    #
    # One tournament per session. Its page is the only authority on where we
    # stand, so every step starts by asking it (`getState`), and each answer
    # leads to exactly one next request:
    #
    #   none       not enrolled yet  -> enrol in one for our status
    #              enrolled before   -> it is over for us: end the session
    #   enrolled   the waiting room  -> poll; withdraw at once if stopping
    #   playing    our table is set  -> take the seat (the room gameplay_new.php
    #                                   names; never a lobby table)
    #              out of it         -> withdraw from the bracket, as the site's
    #                                   own "continue" button does, and end
    #              anything else     -> poll: the other tables are still playing
    #
    # A stop requested while we are in the bracket waits until we are out of
    # it: walking away mid-tournament abandons a partner the draw gave us, and
    # belot.md penalises an account that leaves matches.

    async def _ask_tournament(self, ws, op, **extra):
        """One call to tournaments.php, through the bridge. The answer comes
        back as a TOURNAMENT event."""
        request = {"action": "TOURNAMENT", "op": op, "cookies": self.cookies}
        request.update(extra)
        await ws.send(json.dumps(request))

    async def _poll_tournament(self, ws, delay):
        """Ask where we stand, after `delay` seconds."""
        if delay:
            await asyncio.sleep(delay)
        await self._ask_tournament(ws, "state")

    async def _take_tournament_seat(self, ws):
        """The tournament has seated us: enter the room gameplay_new.php names.

        The bridge reads that page before anything else, and in this mode it
        falls back to nothing -- no lobby, no table of its own choosing.
        """
        self._in_room = False
        self._phase = NOT_STARTED
        await ws.send(json.dumps({"action": "CONNECT", "cookies": self.cookies,
                                  "avoidLast": False,
                                  "table": {"mode": "tournament"}}))

    def _say_tournament(self, line):
        """A status line, printed only when it changes: polls come every 3 s."""
        if line != self._t_said:
            self._t_said = line
            print(f"[SDK][Tournament] {line}")

    def _end_tournament_run(self, reason=None, fatal=None):
        """End the session: normally, for `reason`, or with `fatal`, which
        `connect` raises once it has closed. -> False, to leave the loop."""
        if fatal is not None:
            self.fatal = fatal
            print("=" * 72)
            print(f"[SDK][STOP] {fatal}")
            print("=" * 72)
        else:
            self.tournament_over = reason
            self.tournament.setdefault("result", reason)
            print(f"[SDK][Tournament] done: {reason}.")
        return False

    async def _on_tournament(self, ws, msg):
        """An answer from tournaments.php. -> False to end the session."""
        op = msg.get("op")
        data = msg.get("data")
        if msg.get("error") is not None or not isinstance(data, dict):
            self._t_failures += 1
            why = msg.get("error") or "an answer that is not an object"
            print(f"[SDK][Tournament] {op} failed ({self._t_failures} in a "
                  f"row): {why}")
            if (not self.tournament_answers
                    and self._t_failures >= self.MAX_TOURNAMENT_FAILURES):
                return self._end_tournament_run(fatal=TournamentError(
                    f"tournaments.php failed {self._t_failures} times in a row "
                    f"without answering once (last: {why}). Is the login "
                    f"cookie still valid?"))
            # Whatever the call was, the next poll shows whether it landed.
            await self._poll_tournament(ws, self.tournament_poll_s)
            return True
        self._t_failures = 0

        if op == "enroll":
            # Only the next poll says whether we are in: a refusal such as
            # "already enrolled" is no refusal at all, and a success the
            # page does not reflect is no enrolment. The refusal is kept, and
            # ends the run only if the poll finds us in no tournament.
            self._t_refusal = data.get("error")
            print("[SDK][Tournament] " + (
                f"belot.md refused the enrolment: {self._t_refusal}"
                if self._t_refusal else "enrolment accepted."))
            await self._poll_tournament(ws, 0)
            return True

        if op == "withdraw":
            refusal = data.get("error")
            if refusal:
                print(f"[SDK][Tournament] belot.md refused the withdrawal: "
                      f"{refusal}")
            else:
                print("[SDK][Tournament] withdrawn.")
                self.tournament["withdrew"] = True
            await self._poll_tournament(
                ws, self.tournament_poll_s if refusal else 0)
            return True

        return await self._on_tournament_state(ws, data)

    async def _on_tournament_state(self, ws, state):
        """Where we stand -> the one next step. -> False to end the session."""
        self.tournament_answers += 1
        player_state = state.get("playerState")

        if player_state == tour.NONE:
            if self._t_in:
                # We withdrew, our tournament finished, or the waiting room
                # closed under us. Either way it is over, and a session
                # plays one.
                if (not self.tournament.get("result")
                        and self.tournament.get("withdrew")):
                    self.tournament["result"] = ("withdrew from the waiting "
                                                 "room before the start")
                return self._end_tournament_run(
                    reason=self.tournament.get("result")
                    or "the tournament no longer lists us")
            if self._t_refusal:
                hint = (" -- this account's status is below the tournament's "
                        "minimum" if self._t_refusal == "no_status" else "")
                return self._end_tournament_run(fatal=TournamentError(
                    f"belot.md refused the enrolment: {self._t_refusal}{hint}"))
            if self.stop_requested:
                return self._end_tournament_run(
                    reason="stopped before enrolling")
            if not self.tournament_enroll:
                return self._end_tournament_run(
                    reason="not in a tournament, and this session may not "
                           "enrol in another")
            return await self._enrol(ws, state.get("tournaments"))

        if player_state in (tour.ENROLLED, tour.PLAYING):
            self._t_in = True
            self._t_refusal = None

        if player_state == tour.ENROLLED:
            if self.stop_requested and self._t_withdraws < self.MAX_WITHDRAWS:
                self._t_withdraws += 1
                print("[SDK][Tournament] stopping in the waiting room: "
                      "withdrawing now.")
                await self._ask_tournament(ws, "withdraw")
                return True
            info = state.get("tournamentInfo") or {}
            self._say_tournament(
                f"waiting room: {info.get('numPlayers', '?')}/"
                f"{info.get('maxPlayers', '?')} players")
            await self._poll_tournament(ws, self.tournament_poll_s)
            return True

        if player_state == tour.PLAYING:
            self._note_tournament(state)
            if tour.is_out(state):
                if self._t_withdraws < self.MAX_WITHDRAWS:
                    self._t_withdraws += 1
                    print(f"[SDK][Tournament] {self.tournament['result']}; "
                          f"leaving the bracket.")
                    await self._ask_tournament(ws, "withdraw")
                    return True
                return self._end_tournament_run(
                    reason=self.tournament["result"])
            if state.get("isAtTable"):
                ours = tour.our_table(state, self.my_player_id)
                # Our latest table already having a winner means the site
                # still counts us at the table that just finished: taking that
                # seat would only walk back into a finished match. But only
                # for so long -- a site that seats us before it lists the next
                # round must not leave us waiting while that round starts.
                self._t_stale_seat += 1
                if (ours is None or ours["won"] is None
                        or self._t_stale_seat > self.STALE_SEAT_POLLS):
                    print(f"[SDK][Tournament] our table for the "
                          f"{state.get('phase')} is ready; taking the seat.")
                    self._t_said = None
                    self._t_stale_seat = 0
                    await self._take_tournament_seat(ws)
                    return True
            else:
                self._t_stale_seat = 0
            self._say_tournament(
                f"in the {state.get('phase')}; waiting for our table"
                + (" (a stop is pending: it takes effect once we are out)"
                   if self.stop_requested else ""))
            await self._poll_tournament(ws, self.tournament_poll_s)
            return True

        print(f"[SDK][Tournament][WARN] unrecognised playerState "
              f"{player_state!r}; asking again.")
        await self._poll_tournament(ws, self.tournament_poll_s)
        return True

    async def _enrol(self, ws, tournaments):
        """Enter one tournament for our status: the fullest, which starts
        soonest. -> False to end the session."""
        status = self.tournament_min_status
        pick = tour.pick_tournament(tournaments, status)
        if pick is None:
            if not tour.offers_status(tournaments, status):
                return self._end_tournament_run(fatal=TournamentError(
                    f"belot.md lists no tournament with minimum status "
                    f"{status}"))
            self._say_tournament(f"every tournament with minimum status "
                                 f"{status} is full; waiting for a place")
            await self._poll_tournament(ws, self.tournament_poll_s)
            return True
        if self._t_enrolls >= self.MAX_ENROLLS:
            return self._end_tournament_run(fatal=TournamentError(
                f"enrolled {self._t_enrolls} times, and the site still lists "
                f"us in no tournament"))
        self._t_enrolls += 1
        slot = str(pick.get("id"))
        stake = tour.as_int(pick.get("stake"))
        self.tournament.update(slot=slot, stake=stake, min_status=status)
        print(f"[SDK][Tournament] enrolling in tournament {slot} (minimum "
              f"status {status}, stake {stake}, "
              f"{pick.get('numPlayers')}/{pick.get('maxPlayers')} enrolled).")
        await self._ask_tournament(ws, "enroll", id=slot)
        return True

    def _note_tournament(self, state):
        """Keep the session log's account current: the tournament's id, our
        score at each stage, and once we are out, how it ended for us."""
        if state.get("tournamentId") is not None:
            self.tournament["id"] = str(state["tournamentId"])
        tables = tour.our_tables(state, self.my_player_id)
        if tables:
            self.tournament.setdefault("stages", {}).update(tables)
        if tour.is_out(state) and "result" not in self.tournament:
            self.tournament["result"] = self._tournament_result(state)
            self.tournament["cashback"] = bool(state.get("gotCashback"))

    def _tournament_result(self, state):
        """How the tournament ended for us, in words."""
        place = tour.placing(state, self.my_player_id)
        if place == "won":
            return "won the tournament"
        if place == "second":
            return "second in the tournament"
        ours = tour.our_table(state, self.my_player_id)
        if ours is not None:
            score = (f" ({ours['us']}-{ours['them']})"
                     if ours["us"] is not None and ours["them"] is not None
                     else "")
            return f"eliminated in the {ours['stage']}{score}"
        return ("eliminated" if state.get("isEliminated")
                else "the tournament finished")

    async def _left_tournament_table(self, ws, code, label, mid_match):
        """A tournament table closed. -> False to end the session.

        Every close but one goes back to the tournament, which says what comes
        next: the end of a match played out (we leave at phase 14 ourselves,
        I_LEFT, or the site closes the table, TABLE_REMOVED), or a dropped
        connection, after which the tournament still has the seat as ours.
        """
        self._leaving_to_restart = False
        if code == LEAVE_OTHER_SESSION:
            # This account was opened somewhere else -- most likely a browser
            # on the tournament page, which goes to the table by itself.
            # Taking the seat back would throw them out, and they us, for ever.
            return self._end_tournament_run(fatal=TournamentError(
                f"this account was opened somewhere else ({label}, {code}), "
                f"and whoever opened it has the seat now. Close every other "
                f"belot.md tab of this account before a tournament run"))
        delay = 0.0
        if (leave_recovery(code, mid_match=mid_match) == RECOVER_NOW
                and code != LEAVE_POSITION_CHANGED):
            delay = self._after_a_drop(mid_match) or 0.0
        print(f"[SDK][Tournament] left the table ({label}, {code}); asking "
              f"the tournament where we stand.")
        self._t_said = None
        await self._poll_tournament(ws, delay)
        return True

    def _after_a_drop(self, mid_match):
        """We lost the room without being removed from it. -> the pause before
        going back, or None for at once.

        Going back is the whole recovery: the bridge's CONNECT returns to an
        active game before it does anything else. At once the first time; on
        the short cadence if it keeps happening, so a connection that keeps
        failing is not retried in a tight loop.
        """
        now = time.monotonic()
        again = (self._last_drop_at is not None
                 and now - self._last_drop_at < self.DROP_REPEAT_S)
        self._last_drop_at = now
        if mid_match:
            self._rescue_until = now + self.MATCH_RESCUE_S
            self._lost_room = self.room_id
            print("[SDK] Lost the connection in the middle of a match; going "
                  "back to it (the platform keeps our seat).")
        if self._last_table_id:
            # Joining: straight to that table, not via the lobby.
            self._rejoin_hint = self._last_table_id
        return self.rejoin_delay_s if again else None

    async def _request_table(self, ws, avoid_last=False, table_id=None):
        """Ask the bridge to find a table and join it.

        `avoid_last` tells it to skip the table it most recently reserved --
        the one that just removed us. Without that the lobby pick, which takes
        the first open table, would very likely hand us straight back to it.

        In "join" mode this takes two steps: ask for the lobby, pick our
        partner's table HERE (in Python, where the rule is tested), and come
        back with its id. The bridge never decides which table is ours.
        """
        self._in_room = False
        self._phase = NOT_STARTED

        if self.table_mode == "tournament":
            # Tables come from the tournament, never from a lobby: ask it
            # where we stand, and its answer says what to do next.
            await self._ask_tournament(ws, "state")
            return

        if table_id is None and self._rejoin_hint:
            # We were moved off our seat, not off the table. The lobby would
            # report that table as full -- our own seat is one of the taken
            # ones -- and we would stand down from a table that is ours. Ask
            # for it directly instead: the bridge rejoins whatever room the
            # platform still has us in before it looks at this id at all.
            table_id = self._rejoin_hint
            self._rejoin_hint = None

        if self.table_mode == "join" and table_id is None:
            await ws.send(json.dumps({"action": "LOBBY",
                                      "cookies": self.cookies}))
            return

        request = {"action": "CONNECT", "cookies": self.cookies,
                   "avoidLast": bool(avoid_last)}
        if self.table_mode == "create":
            request["table"] = {"mode": "create", "body": CREATE_TABLE_BODY}
        elif self.table_mode == "join":
            request["table"] = {"mode": "join", "tableId": table_id}
            if self.table_password:
                request["table"]["password"] = self.table_password
            self._last_table_id = table_id
        await ws.send(json.dumps(request))

    async def _recover(self, ws, action, reason, avoid_last=False, delay=None):
        """Act on a recovery decision. -> True to keep the session alive.

        `delay` overrides the wait the action would imply, for the cases that
        have their own cadence: watching the lobby for a partner's table, and
        standing down after a pairing failed.
        """
        if action == RECOVER_STOP:
            print(f"[SDK] Session ended: {reason}. Not rejoining.")
            return False
        if not self.reconnect:
            print(f"[SDK] {reason}; --once was requested, so stopping.")
            return False

        if delay is None:
            delay = {RECOVER_NOW: 0.0,
                     RECOVER_SOON: self.rejoin_delay_s,
                     RECOVER_LATER: self.retry_delay_s}[action]
        if delay:
            print(f"[SDK] {reason}; looking for another table in "
                  f"{delay:.0f}s. ({self.sessions_played} played so far)")
            await asyncio.sleep(delay)
        else:
            print(f"[SDK] {reason}; rejoining now.")
        await self._request_table(ws, avoid_last=avoid_last)
        return True

    async def _consume(self, on_state_callback, on_message_callback):
        """Single consumer: exactly one handler touches the synchronizer at a
        time, and frames are applied in arrival order."""
        while True:
            kind, payload = await self._queue.get()
            try:
                if kind is None:
                    return
                if kind == "STATE":
                    await on_state_callback(payload, self.my_player_id)
                elif kind == "MESSAGE" and on_message_callback:
                    await on_message_callback(*payload)
            except Exception:
                import traceback
                print("[SDK][ERROR] handler raised:")
                traceback.print_exc()
            finally:
                self._queue.task_done()

    async def deactivate_bot(self):
        await self._send("BOT_ACTIVATION", "deactivate")

    async def send_ready(self, is_ready: bool = True):
        await self._send("READY", {"value": is_ready})

    async def pass_turn(self):
        await self._send("PASS", {})

    async def bid_trump(self, suit_int: int):
        await self._send("TRUMP_CHOOSE", suit_int)

    async def play_card_char(self, card_char: str):
        await self._send("PLAY_CARD", card_char)

    async def swap_seven(self):
        """Phase 8 (SWAP_SEVEN): exchange our 7 of trump for the face-up card.

        Payload mirrors PASS -- an empty object, msgpack 0x80. A captured
        outgoing frame reads

            0x0d ROOM_DATA | 0xAA fixstr(10) "SWAP_SEVEN" | <one byte>

        and exactly one trailing byte rules out a card char (a1 45), an
        object (81 ab ...) and an omitted payload (no byte at all). The card
        is not sent: the server already knows who holds the 7.
        """
        await self._send("SWAP_SEVEN", {})

    async def show_combination(self, value: str):
        """Announce a declaration. Payload shape mirrors PLAY_CARD /
        TRUMP_CHOOSE, which both send a bare scalar; the server broadcasts
        SHOW_COMBINATION {"who": seat, "value": "<code>"} in response."""
        await self._send("SHOW_COMBINATION", value)

    async def leave_table(self, restart: bool = True):
        """Leave the table. **As its creator this DELETES it**, and everyone
        sitting there is kicked.

        `restart=True` marks the leave as deliberate and temporary, so the
        close that follows is recovered from rather than treated as the end of
        the session. Payload `{}`, matching PASS and SWAP_SEVEN.
        """
        self._leaving_to_restart = bool(restart)
        await self._send("LEAVE_TABLE", {})

    async def change_players_position(self):
        """Ask the table to move the other players around.

        [UNVERIFIED] Captured from the web client, where it carries no payload
        we have been able to decode, and the reply is Colyseus-encoded state.
        Observed effect: the other three seats rotate while the sender stays
        put, so at most two sends put a chosen partner opposite. Read the
        result from the next STATE frame, never from the reply.

        Only the table's creator appears to be able to do this. The other
        players are dropped with close code 4005 (POSITION_CHANGED) and rejoin
        immediately, which the recovery policy already handles.
        """
        await self._send("CHANGE_PLAYERS_POSITION", {})

    async def cut_deck(self, cut_index: int = 15):
        await self._send("PUSH_CARD", {"cardPushed": cut_index})

    async def _send(self, action_type: str, payload):
        if self.ws:
            await self.ws.send(json.dumps({"action": "SEND", "type": action_type, "payload": payload}))

    def close(self):
        proc = self.node_process
        if proc:
            proc.terminate()
            # Our end of the lifeline, so the bridge sees EOF even if the
            # terminate did not land.
            lifeline = getattr(proc, "stdin", None)
            if lifeline is not None:
                try:
                    lifeline.close()
                except OSError:
                    pass