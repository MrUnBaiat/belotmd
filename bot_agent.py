import asyncio
import json
import os
import time
import torch

from client import BelotClient, CHAR_TO_CARD
from model import RecurrentMAPPOModel
from observation import build_observation
from belot_sync import StateSynchronizer, ID_TO_ASCII
from env import BelotEnv
from sync_auditor import FrameRecorder, Auditor

# Insert your active browser session cookie here
BELOT_COOKIES = ""
# train.py writes to CHECKPOINT_DIR/latest_model.pt (default "checkpoints/"),
# so a bare filename here silently falls back to RANDOM WEIGHTS.
MODEL_PATH = "latest_model.pt"
MODEL_SEARCH = [MODEL_PATH,
                os.path.join("checkpoints", "latest_model.pt"),
                os.path.join("checkpoints", "best_model.pt")]

# Verification layer: leave ON until the auditor reports clean matches and
# all PROBE questions are answered, then flip via BELOT_AUDIT=0.
AUDIT = os.environ.get("BELOT_AUDIT", "1") == "1"
FRAMES_LOG = "frames.jsonl"

# If the state hasn't advanced this long after we dispatched an action, we
# assume the server rejected it and re-dispatch (a successful action always
# changes the turn signature, so this can never double-fire a good move).
REDISPATCH_AFTER_S = 3.0

# Declarations. The model has no combo action, so these points are forfeited
# by default. Combination points land in roundTotals.c and NEVER in .p (10
# hands verified: p always totals exactly 162), so declaring cannot perturb
# any observation feature — it only adds match points. Opt in with
# BELOT_AUTO_DECLARE=1 once you have confirmed the outgoing payload shape by
# watching for a SHOW_COMBINATION broadcast carrying your own seat.
AUTO_DECLARE = os.environ.get("BELOT_AUTO_DECLARE", "1") == "1"


class LiveBelotBot:
    def __init__(self, cookies: str, model_path: str):
        self.client = BelotClient(cookies=cookies)
        self.sync_engine = StateSynchronizer()

        # Neural Network Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RecurrentMAPPOModel().to(self.device)

        found = next((p for p in [model_path] + MODEL_SEARCH
                      if p and os.path.exists(p)), None)
        if found:
            checkpoint = torch.load(found, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Bot] Loaded trained weights from {found} "
                  f"(epoch {checkpoint.get('epoch', '?')})")
        else:
            print("=" * 68)
            print("[Bot][CRITICAL] No checkpoint found — PLAYING WITH RANDOM "
                  "WEIGHTS.\n                Searched: "
                  + ", ".join(MODEL_SEARCH) + "\n"
                  "                Any conclusion about play quality is "
                  "meaningless until this is fixed.")
            print("=" * 68)

        self.model.eval()
        self.hidden_state = self._zero_lstm_state()

        # Turn debounce (replaces the old hard dedup, which deadlocked the
        # bot forever if the server ever rejected an action)
        self.last_action_turn_id = None
        self.last_dispatch_ts = 0.0
        self._ready_sent_phase = None
        self._declared = set()          # (hand_id, value) already announced

        if AUDIT:
            self.recorder = FrameRecorder(FRAMES_LOG)
            self.auditor = Auditor(verbose=True)
            print(f"[Bot] AUDIT mode on: recording frames to {FRAMES_LOG}")

    def _zero_lstm_state(self):
        return (torch.zeros(1, 1, 512, device=self.device),
                torch.zeros(1, 1, 512, device=self.device))

    async def on_state_update(self, raw_state: dict, my_player_id: str):
        try:
            await self._on_state_update(raw_state, my_player_id)
        except Exception:
            # client.py fires this via asyncio.create_task; an unhandled
            # exception there is swallowed as "Task exception was never
            # retrieved" and the frame is lost without a trace.
            import traceback
            print("[Bot][ERROR] state handler raised:")
            traceback.print_exc()

    async def _on_state_update(self, raw_state: dict, my_player_id: str):
        my_pos = self.sync_engine.sync(raw_state, my_player_id)
        if my_pos is None:
            return

        # Hand-boundary LSTM reset, driven by the sync engine's own new-hand
        # detection (robust to dropped phase frames), matching training where
        # the hidden state is zeroed at the start of every episode/hand.
        if self.sync_engine.consume_new_hand():
            self.hidden_state = self._zero_lstm_state()
            self.last_action_turn_id = None

        if AUDIT:
            # The verification layer must NEVER take the bot down: a crash
            # here previously killed the whole task, silently dropping the
            # frame (and any turn it contained).
            try:
                self.recorder.record(raw_state, my_player_id)
                self.auditor.check(self.sync_engine, raw_state)
            except Exception as e:
                print(f"[Audit][ERROR] suppressed: {type(e).__name__}: {e}")

        env = self.sync_engine.env
        phase = raw_state.get("currentPhase", 0)
        active_player = raw_state.get("activePlayer", -1)

        if AUTO_DECLARE:
            await self._maybe_declare(raw_state, my_pos)

        # 1. Auto-Ready on game transitions (deduped by phase transition)
        if phase in [0, 13, 14]:
            self.hidden_state = self._zero_lstm_state()
            if self._ready_sent_phase != phase:
                self._ready_sent_phase = phase
                await self.client.send_ready(True)
            return
        self._ready_sent_phase = None
        self._declared = set()          # (hand_id, value) already announced

        # 2. Cut deck phase
        if phase == 2 and active_player == my_pos:
            await self.client.cut_deck(15)
            return

        # 3. Network Action Decision (Bidding or Playing)
        if phase in [6, 7, 10] and active_player == my_pos:
            turn_id = (f"PHASE_{phase}_P{my_pos}"
                       f"_HAND_{len(env.hands[my_pos])}"
                       f"_TRICK_{len(env.current_trick)}")
            now = time.monotonic()
            if (self.last_action_turn_id == turn_id
                    and (now - self.last_dispatch_ts) < REDISPATCH_AFTER_S):
                return
            if self.last_action_turn_id == turn_id:
                print(f"[Bot][WARN] State unchanged {REDISPATCH_AFTER_S}s after "
                      f"dispatch — re-dispatching (possible server rejection): "
                      f"{turn_id}")
            self.last_action_turn_id = turn_id
            self.last_dispatch_ts = now

            await self._select_and_execute_action(my_pos)

    async def _maybe_declare(self, raw_state, my_pos):
        """Announce every combination the server offers us. `value` uses the
        same encoding the server broadcasts back (SHOW_COMBINATION carries
        {"who": seat, "value": "1c"}); multiple offers arrive pipe-separated.
        Deduplicated per hand so repeated state frames cannot spam."""
        players = raw_state.get("players", [])
        if my_pos >= len(players):
            return
        offer = players[my_pos].get("combinationsCanShow") or ""
        for value in [v for v in offer.split("|") if v]:
            key = (self.sync_engine.hand_id, value)
            if key in self._declared:
                continue
            self._declared.add(key)
            print(f"[Bot Action] -> SHOW COMBINATION: {value}")
            await self.client.show_combination(value)

    async def _select_and_execute_action(self, my_pos: int):
        env = self.sync_engine.env

        local_obs, global_obs, legal_mask = build_observation(
            env,
            my_pos,
            self.sync_engine.match_scores
        )

        if not legal_mask.any():
            return

        local_t = torch.from_numpy(local_obs).unsqueeze(0).to(self.device)
        glob_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _, self.hidden_state = self.model(
                local_t, glob_t, self.hidden_state, mask_t, is_sequence=False
            )
            action = int(torch.argmax(dist.probs, dim=-1).item())

        await self._dispatch_action(action, env)

    async def _dispatch_action(self, action: int, env: BelotEnv):
        if env.phase == "BIDDING":
            if action == 32:
                print("[Bot Action] -> PASS")
                await self.client.pass_turn()
            elif action == 33:
                if env.face_up_suit is None:
                    # Transient race: topCard not yet observed this hand.
                    # Action 33 is only legal in round 1, where passing is
                    # always legal, so this fallback can never be illegal.
                    print("[Bot][WARN] face_up_suit unknown; passing defensively")
                    await self.client.pass_turn()
                else:
                    colyseus_suit = env.face_up_suit + 1
                    print(f"[Bot Action] -> ACCEPT FACE-UP (Suit: {colyseus_suit})")
                    await self.client.bid_trump(colyseus_suit)
            elif 34 <= action <= 37:
                colyseus_suit = (action - 34) + 1
                print(f"[Bot Action] -> CHOOSE SUIT ({colyseus_suit})")
                await self.client.bid_trump(colyseus_suit)

        elif env.phase == "PLAYING":
            if 0 <= action <= 31:
                card_char = ID_TO_ASCII[action]
                print(f"[Bot Action] -> PLAY CARD: "
                      f"{CHAR_TO_CARD[card_char]} ({card_char})")
                await self.client.play_card_char(card_char)

    async def on_server_message(self, msg_type, data):
        """belot.md offered our seat a declaration in ~7 hands of the last
        match (combinationsCanShow), which we forfeit because the model has
        no combo action. Logging every server message is the cheapest way to
        learn the protocol's naming so an auto-declare can be added later."""
        try:
            blob = json.dumps(data)[:300]
        except Exception:
            blob = repr(data)[:300]
        print(f"[MSG] {msg_type}: {blob}")

    async def run(self):
        try:
            await self.client.connect(
                on_state_callback=self.on_state_update,
                on_message_callback=self.on_server_message if AUDIT else None,
            )
        finally:
            self.client.close()


if __name__ == "__main__":
    bot = LiveBelotBot(cookies=BELOT_COOKIES, model_path=MODEL_PATH)
    asyncio.run(bot.run())