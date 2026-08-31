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
                       LEAVE_POSITION_CHANGED, SUIT_TO_INT, TRUMP_CHOOSE_1)


class BelotClient:
    def __init__(self, cookies: str, bridge_port: int = 8765):
        self.cookies = cookies
        self.bridge_port = bridge_port
        self.ws = None
        self.node_process = None
        self.my_player_id = None
        self.room_id = None
        self._queue = None
        self._consumer = None
        self._match_started = False

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
            await ws.send(json.dumps({"action": "CONNECT", "cookies": self.cookies}))

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
                    if (msg.get("data") or {}).get("currentPhase", 0) >= TRUMP_CHOOSE_1:
                        self._match_started = True
                    await self._queue.put(("STATE", msg["data"]))

                elif event == "MESSAGE" and on_message_callback:
                    # Same queue, so STATE/MESSAGE ordering is preserved.
                    await self._queue.put(("MESSAGE", (msg.get("type"),
                                                       msg.get("data"))))

                elif event == "ERROR":
                    print(f"[SDK Error]: {msg.get('message') or msg.get('code')}")

                elif event == "LEAVE":
                    code = msg.get("code")
                    label = LEAVE_NAMES.get(code, "UNKNOWN")
                    if code == LEAVE_POSITION_CHANGED and not self._match_started:
                        print(f"[SDK] {label} ({code}) in the lobby: "
                              f"rejoining...")
                        await ws.send(json.dumps({"action": "CONNECT",
                                                  "cookies": self.cookies}))
                        continue
                    if code == LEAVE_POSITION_CHANGED:
                        # Cheap assertion on the platform invariant. If this
                        # ever fires, seats moved mid-match and every
                        # seat-indexed belief (known_cards, impossible_cards,
                        # hands, team parity) now describes the wrong player --
                        # stopping is correct, reconnecting would not be.
                        print(f"[SDK][WARN] {label} ({code}) AFTER play began "
                              f"-- not supposed to happen; stopping rather "
                              f"than resuming on stale seat state.")
                    print(f"[SDK] Left room session: {label} ({code}).")
                    break

            await self._queue.put((None, None))     # drain sentinel
            await consumer

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