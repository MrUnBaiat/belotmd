"""
client.py — Python end of the Node bridge. Message plumbing only.

Owns the `node bridge.js` subprocess, the local WebSocket to it, and the
outgoing message vocabulary. It holds no game state: raw frames go straight
to the callbacks in arrival order.
"""

import asyncio
import json
import os
import subprocess
import time
import websockets

from .protocol import (CHAR_TO_CARD, LEAVE_NAMES,  # noqa: F401  (re-exported)
                       LEAVE_KICKED, LEAVE_POSITION_CHANGED, PUSH_CARDS,
                       RECOVER_LATER,
                       RECOVER_NOW, RECOVER_SOON, RECOVER_STOP,
                       SUIT_TO_INT, leave_recovery)


class BelotClient:
    def __init__(self, cookies: str, bridge_port: int = 8765,
                 reconnect: bool = True, retry_delay_s: float = 300.0,
                 rejoin_delay_s: float = 5.0):
        """`reconnect` keeps the bot looking for tables instead of exiting.

        Two delays, because two very different situations end a session:
          rejoin_delay_s  a match ended or the table dissolved -- go straight
                          back out and find another one;
          retry_delay_s   there is nothing to join (no open tables, or we were
                          kicked). Waiting is the only useful move, and
                          hammering the lobby every few seconds is rude.
        """
        self.cookies = cookies
        self.bridge_port = bridge_port
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
        self._match_started = False
        self._in_room = False

    # How long to wait for `node bridge.js` to bind BRIDGE_PORT.
    DAEMON_TIMEOUT_S = 15.0
    DAEMON_POLL_S = 0.25

    def _start_node_daemon(self):
        print("[SDK] Launching background Node.js bridge...")
        bridge_path = os.path.join(os.path.dirname(__file__), "bridge.js")
        if not os.path.exists(bridge_path):
            raise FileNotFoundError(f"bridge.js not found at {bridge_path}")
        try:
            self.node_process = subprocess.Popen(["node", bridge_path])
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
                if attempt > 1:
                    print(f"[SDK] Bridge daemon ready after {attempt} attempts.")
                return ws
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"bridge daemon did not accept a connection on "
                        f"{uri} within {self.DAEMON_TIMEOUT_S}s ({exc})"
                    ) from exc
                await asyncio.sleep(self.DAEMON_POLL_S)

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
                    self._match_started = False
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
                    # The match is underway from the deck cut (phase 2),
                    # not from the first bid. Sticky for the room session:
                    # a cancelled deal drops back through phases 1-2 without
                    # meaning we returned to the lobby.
                    if (msg.get("data") or {}).get("currentPhase", 0) >= PUSH_CARDS:
                        self._match_started = True
                    await self._queue.put(("STATE", msg["data"]))

                elif event == "MESSAGE" and on_message_callback:
                    # Same queue, so STATE/MESSAGE ordering is preserved.
                    await self._queue.put(("MESSAGE", (msg.get("type"),
                                                       msg.get("data"))))

                elif event == "ERROR":
                    detail = msg.get("message") or msg.get("code")
                    print(f"[SDK Error]: {detail}")
                    # An error while we are NOT in a room means the join
                    # itself failed -- almost always "No public open tables
                    # available." Without this the bot sat idle forever
                    # waiting for frames from a room it never entered.
                    if not self._in_room:
                        if not await self._recover(ws, RECOVER_LATER,
                                                   f"could not join a table ({detail})"):
                            break

                elif event == "LEAVE":
                    code = msg.get("code")
                    label = LEAVE_NAMES.get(code, "UNKNOWN")
                    self._in_room = False
                    action = leave_recovery(code)

                    if label == "UNKNOWN":
                        print(f"[SDK][WARN] unrecognised close code {code}; "
                              f"treating it as recoverable.")
                    if code == LEAVE_POSITION_CHANGED and self._match_started:
                        # A cheap assertion on a platform invariant: the host
                        # can only rotate seats BETWEEN joining a table and the
                        # match starting. Once it is under way this cannot
                        # happen, so if it ever does, something about the
                        # platform is not what we think it is -- say so
                        # unmistakably. We still resync rather than end an
                        # unattended run, and the synchronizer will flag the
                        # hand as degraded.
                        print("=" * 72)
                        print(f"[SDK][VIOLATION] {label} ({code}) AFTER the "
                              f"match began. This is not supposed to be "
                              f"possible.")
                        print("                 Seat indices have moved, so "
                              "every seat-indexed belief is now wrong.")
                        print("                 Resyncing, but keep the "
                              "recording -- this is worth understanding.")
                        print("=" * 72)

                    if not await self._recover(
                            ws, action, f"{label} ({code})",
                            avoid_last=(code == LEAVE_KICKED)):
                        break

            await self._queue.put((None, None))     # drain sentinel
            await consumer

    async def _request_table(self, ws, avoid_last=False):
        """Ask the bridge to find a table and join it.

        `avoid_last` tells it to skip the table it most recently reserved --
        the one that just removed us. Without that the lobby pick, which takes
        the first open table, would very likely hand us straight back to it.
        """
        self._in_room = False
        self._match_started = False
        await ws.send(json.dumps({"action": "CONNECT",
                                  "cookies": self.cookies,
                                  "avoidLast": bool(avoid_last)}))

    async def _recover(self, ws, action, reason, avoid_last=False):
        """Act on a recovery decision. -> True to keep the session alive."""
        if action == RECOVER_STOP:
            print(f"[SDK] Session ended: {reason}. Not rejoining.")
            return False
        if not self.reconnect:
            print(f"[SDK] {reason}; --once was requested, so stopping.")
            return False

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

    async def cut_deck(self, cut_index: int = 15):
        await self._send("PUSH_CARD", {"cardPushed": cut_index})

    async def _send(self, action_type: str, payload):
        if self.ws:
            await self.ws.send(json.dumps({"action": "SEND", "type": action_type, "payload": payload}))

    def close(self):
        if self.node_process:
            self.node_process.terminate()