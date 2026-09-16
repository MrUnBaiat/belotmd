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

from .protocol import (CHAR_TO_CARD, LEAVE_NAMES,  # noqa: F401  (re-exported)
                       LEAVE_KICKED, LEAVE_POSITION_CHANGED, NOT_STARTED,
                       RECOVER_LATER, RECOVER_NOW, RECOVER_SOON, RECOVER_STOP,
                       SUIT_TO_INT, leave_recovery, match_underway)
from .tables import CREATE_TABLE_BODY, find_table, is_joinable, table_id_of


class BelotClient:
    def __init__(self, cookies: str, bridge_port: int = 0,
                 reconnect: bool = True, retry_delay_s: float = 300.0,
                 rejoin_delay_s: float = 5.0, table_mode: str = "lobby",
                 table_id: str = None, table_creator: str = None,
                 join_poll_s: float = 2.0, join_max_polls: int = 20,
                 pair_restart_pause_s: float = 10.0):
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
        # open public table). A pair uses the other two: the host "create"s a
        # table and the guest "join"s the one that host made.
        self.table_mode = table_mode
        self.table_id = table_id
        self.table_creator = table_creator
        # Watching the lobby for our partner's table: how often, and how many
        # fruitless looks before standing down for a while.
        self.join_poll_s = float(join_poll_s)
        self.join_max_polls = int(join_max_polls)
        self.pair_restart_pause_s = float(pair_restart_pause_s)
        self._join_polls = 0
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

    # How long to wait for `node bridge.js` to bind its port.
    DAEMON_TIMEOUT_S = 15.0
    DAEMON_POLL_S = 0.25
    # How long to wait for that daemon to identify itself.
    HELLO_TIMEOUT_S = 5.0

    @staticmethod
    def _free_port():
        """Ask the OS for a free port. Racy in principle -- another process
        could take it between here and the daemon's bind -- but the daemon then
        exits with a clear message and the token check makes the silent
        wrong-bridge outcome impossible."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def _start_node_daemon(self):
        if not self.bridge_port:
            self.bridge_port = self._free_port()
        print(f"[SDK] Launching background Node.js bridge on port "
              f"{self.bridge_port}...")
        bridge_path = os.path.join(os.path.dirname(__file__), "bridge.js")
        if not os.path.exists(bridge_path):
            raise FileNotFoundError(f"bridge.js not found at {bridge_path}")
        try:
            self.node_process = subprocess.Popen(
                ["node", bridge_path, str(self.bridge_port), self.bridge_token])
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
                ws = await websockets.connect(uri)
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
                    self.sessions_played += 1
                    print(f"[SDK] Joined Game Room: {self.room_id} | Player ID: {self.my_player_id}")
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

                elif event == "ERROR":
                    detail = msg.get("message") or msg.get("code")
                    print(f"[SDK Error]: {detail}")
                    # An error while we are NOT in a room means the join
                    # itself failed -- almost always "No public open tables
                    # available." Without this the bot sat idle forever
                    # waiting for frames from a room it never entered.
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
                        if joining:
                            self._join_polls += 1
                        if not await self._recover(
                                ws, RECOVER_SOON if soon else RECOVER_LATER,
                                f"could not join a table ({detail})",
                                delay=self.join_poll_s if joining else None):
                            break

                elif event == "LEAVE":
                    code = msg.get("code")
                    label = LEAVE_NAMES.get(code, "UNKNOWN")
                    self._in_room = False
                    restart_pause = None
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
                        action = leave_recovery(code)

                    if label == "UNKNOWN":
                        print(f"[SDK][WARN] unrecognised close code {code}; "
                              f"treating it as recoverable.")
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
        if self.node_process:
            self.node_process.terminate()