import asyncio
import json
import os
import subprocess
import time
import websockets

CHAR_TO_CARD = {
    "y": "7_diamonds", "z": "8_diamonds", "a": "9_diamonds", "b": "10_diamonds",
    "c": "J_diamonds", "d": "Q_diamonds", "e": "K_diamonds", "f": "A_diamonds",
    "A": "7_hearts", "B": "8_hearts", "g": "9_hearts", "h": "10_hearts",
    "i": "J_hearts", "j": "Q_hearts", "k": "K_hearts", "l": "A_hearts",
    "C": "7_clubs", "D": "8_clubs", "m": "9_clubs", "n": "10_clubs",
    "o": "J_clubs", "p": "Q_clubs", "q": "K_clubs", "r": "A_clubs",
    "E": "7_spades", "F": "8_spades", "s": "9_spades", "t": "10_spades",
    "u": "J_spades", "v": "Q_spades", "w": "K_spades", "x": "A_spades",
}

CARD_TO_CHAR = {v: k for k, v in CHAR_TO_CARD.items()}
SUIT_TO_INT = {"diamonds": 1, "hearts": 2, "clubs": 3, "spades": 4}

class BelotClient:
    def __init__(self, cookies: str, bridge_port: int = 8765):
        self.cookies = cookies
        self.bridge_port = bridge_port
        self.ws = None
        self.node_process = None
        self.my_player_id = None
        self.room_id = None

    def _start_node_daemon(self):
        print("[SDK] Launching background Node.js bridge...")
        bridge_path = os.path.join(os.path.dirname(__file__), "bridge.js")
        self.node_process = subprocess.Popen(["node", bridge_path])
        time.sleep(1.5)

    async def connect(self, on_state_callback, on_message_callback=None):
        self._start_node_daemon()
        uri = f"ws://127.0.0.1:{self.bridge_port}"

        async with websockets.connect(uri) as ws:
            self.ws = ws
            print("[SDK] Connected to bridge daemon. Releasing join request...")
            await ws.send(json.dumps({"action": "CONNECT", "cookies": self.cookies}))

            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                event = msg.get("event")

                if event == "CONNECTED":
                    self.room_id = msg.get("roomId")
                    self.my_player_id = msg.get("playerId")
                    print(f"[SDK] Joined Game Room: {self.room_id} | Player ID: {self.my_player_id}")
                    await self.deactivate_bot()

                elif event == "STATE":
                    asyncio.create_task(on_state_callback(msg["data"], self.my_player_id))

                elif event == "MESSAGE" and on_message_callback:
                    await on_message_callback(msg.get("type"), msg.get("data"))

                elif event == "ERROR":
                    print(f"[SDK Error]: {msg.get('message') or msg.get('code')}")

                elif event == "LEAVE":
                    print(f"[SDK] Left room session ({msg.get('code')}).")
                    break

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