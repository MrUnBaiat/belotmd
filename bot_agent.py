import asyncio
import json
import os
import time
import torch

from client import BelotClient, CHAR_TO_CARD
from model import RecurrentMAPPOModel
from observation import build_observation
from belot_sync import StateSynchronizer, ID_TO_ASCII, ASCII_TO_ID
from env import BelotEnv
from sync_auditor import (FrameRecorder, Auditor,
                          verify_observation_patch)
import combinations as combo

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
# Announced by default: point combinations (1-5), LESS_THAN_14 and
# BELOT_COMBO. WIN_ALL_HANDS and SURRENDER_BT are never auto-fired.
DECLARE_WIN_ALL = os.environ.get("BELOT_DECLARE_WIN_ALL", "0") == "1"
# Four 8s scores nothing and silences every combination except bella --
# including ours -- so it is declined unless explicitly enabled.
# Four 7s cancels the deal (same effect as LESS_THAN_14) -- declared by
# default. Four 8s silences every combination except bella, ours included --
# declined by default. Both are switchable without touching code.
DECLARE_FOUR_SEVENS = os.environ.get("BELOT_DECLARE_FOUR_SEVENS", "1") == "1"
DECLARE_FOUR_EIGHTS = os.environ.get("BELOT_DECLARE_FOUR_EIGHTS", "0") == "1"

# Seven-swap (phase 8, SWAP_SEVEN): trade the 7 of trump for the face-up
# card. Only reachable after a round-1 accept, where the face-up card is a
# trump strictly better than the 7, so it is always a gain. Forced-trump
# ("BIZON") hands skip phase 8 entirely (5 -> 9), so no guard is needed.
AUTO_SWAP_SEVEN = os.environ.get("BELOT_AUTO_SWAP_SEVEN", "1") == "1"
# Phase 8 opens after EVERY round-1 accept and waits on a timeout rather than
# closing at once, so the payload ladder can finish inside a single window.



class LiveBelotBot:
    def __init__(self, cookies: str, model_path: str):
        self.client = BelotClient(cookies=cookies)
        self.sync_engine = StateSynchronizer()

        # Neural Network Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RecurrentMAPPOModel().to(self.device)

        candidates = [model_path] + [p for p in MODEL_SEARCH if p != model_path]
        found = next((p for p in candidates if p and os.path.exists(p)), None)
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
        # (turn_id, hidden_state_before_this_turn, {actions already refused})
        # Keeps the recurrent state advancing once per DECISION rather than
        # once per DISPATCH, and blocks a refused action on re-dispatch
        # (PLATFORM_NOTES §12).
        self._turn_decision = None
        self._ready_sent_phase = None
        self._declared = set()          # (hand_id, value) already announced
        self._pending_play = None     # (hand_id, trick_no, card) in flight
        self._rejected_plays = 0
        self.seat_bot_controlled = False
        self._takeover_frame = None   # (round, phase) when the seat was lost
        self._swap_hand = None        # hand_id of the open phase-8 window
        self._swap_sent = False       # SWAP_SEVEN sent in that window
        self._swap_skip = False       # decided not to swap this window

        if AUDIT:
            self.recorder = FrameRecorder(FRAMES_LOG)
            self.auditor = Auditor(verbose=True)
            verify_observation_patch()
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

        await self._check_play_accepted(my_pos)
        await self._check_seat_autopilot(raw_state, my_pos)

        # Declarations exist only once all eight cards are dealt (phase 9+).
        # Every other server field -- trump, declarer, topCard, swapSeven --
        # is stale through the deal, so acting on combinationsCanShow before
        # the second deal risks announcing a combination from the PREVIOUS
        # hand.
        if AUTO_DECLARE and phase >= 9 and phase not in (13, 14):
            await self._maybe_declare(raw_state, my_pos)
        if AUTO_SWAP_SEVEN:
            if phase == 8:
                await self._maybe_swap_seven(my_pos, raw_state)
            elif phase >= 9:
                self._resolve_swap_attempt(my_pos)

        # 1. Auto-Ready on game transitions (deduped by phase transition)
        if phase in [0, 13, 14]:
            self.hidden_state = self._zero_lstm_state()
            if self._ready_sent_phase != phase:
                self._ready_sent_phase = phase
                await self.client.send_ready(True)
            return
        self._ready_sent_phase = None

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

            await self._select_and_execute_action(my_pos, turn_id)

    async def _check_play_accepted(self, my_pos):
        """Did the server actually take the card we sent?

        A live session showed the bot dispatching the same card in six
        consecutive tricks: every play was being ignored while the tricks
        resolved around it, so the model was effectively not playing at all.
        Nothing detected that. The card leaving our hand is the server's own
        confirmation; anything else is a rejection worth shouting about.
        """
        if self._pending_play is None:
            return
        hand_id, trick_no, card = self._pending_play
        env = self.sync_engine.env
        if hand_id != self.sync_engine.hand_id:
            self._pending_play = None
            return
        if card not in env.hands[my_pos]:
            self._pending_play = None                 # accepted
            self._rejected_plays = 0
            return

        played_by_us = next((c for s_, c in env.current_trick if s_ == my_pos), None)
        moved_on = env.tricks_played > trick_no
        if played_by_us is None and not moved_on:
            return                                    # still in flight

        self._pending_play = None
        self._rejected_plays += 1
        if played_by_us is not None and played_by_us != card:
            detail = (f"the server recorded {ID_TO_ASCII.get(played_by_us)} "
                      f"for our seat instead")
        else:
            detail = "the trick moved on without it"
        cause = (" (expected: the platform bot holds our seat)"
                 if self.seat_bot_controlled else
                 " -- if this repeats, our turn probably timed out and the "
                 "platform bot has taken the seat")
        print(f"[Bot][ERROR] play {ID_TO_ASCII.get(card)} was NOT accepted: "
              f"{detail}{cause}. (rejected plays this session: "
              f"{self._rejected_plays})")

    async def _check_seat_autopilot(self, raw_state, my_pos):
        """Track the platform bot taking our seat.

        belot.md hands a seat to its own bot once a turn times out, and from
        then on our PLAY_CARD messages are ignored. The hand still shrinks
        because the platform bot is playing it, so our model keeps selecting
        whatever card the platform bot happens not to play -- which is how
        one card ends up "played" in six consecutive tricks.

        By deliberate choice we do NOT try to reclaim the seat: we only log
        the transition loudly, so it is unmistakable in the log and in
        frames.jsonl that everything after this point is the platform bot's
        play and not the model's.
        """
        players = raw_state.get("players", [])
        if my_pos >= len(players):
            return
        flagged = bool(players[my_pos].get("bot"))
        if flagged == self.seat_bot_controlled:
            return
        self.seat_bot_controlled = flagged
        rnd = raw_state.get("round")
        phase = raw_state.get("currentPhase")
        if flagged:
            self._takeover_frame = (rnd, phase)
            print("=" * 72)
            print(f"[Bot][SEAT LOST] The platform bot has taken our seat "
                  f"(round {rnd}, phase {phase}).")
            print("               Our turn timed out -- most likely a move "
                  "the server rejected.")
            print("               From here the server IGNORES our messages: "
                  "every [Bot Action]")
            print("               below is the model talking to itself, and "
                  "the cards actually")
            print("               played are the platform bot's. Not "
                  "reclaiming the seat by design.")
            print("=" * 72)
        else:
            print(f"[Bot][SEAT REGAINED] The server no longer flags our seat "
                  f"as bot-controlled (round {rnd}, phase {phase}); we were "
                  f"out from {self._takeover_frame}.")

    async def _maybe_declare(self, raw_state, my_pos):
        """Announce the point combinations the server offers us. Codes are
        "<type><highest_card>", pipe-separated. Combination points land in
        roundTotals.c and never in .p, so declaring cannot perturb any
        observation feature -- it is free match score."""
        players = raw_state.get("players", [])
        if my_pos >= len(players):
            return
        offer = players[my_pos].get("combinationsCanShow") or ""
        for value in combo.declarable(offer, ASCII_TO_ID,
                                      include_win_all=DECLARE_WIN_ALL,
                                      include_four_sevens=DECLARE_FOUR_SEVENS,
                                      include_four_eights=DECLARE_FOUR_EIGHTS):
            key = (self.sync_engine.hand_id, value)
            if key in self._declared:
                continue
            self._declared.add(key)
            cards, _ = combo.decode_field(value, ASCII_TO_ID)
            print(f"[Bot Action] -> SHOW COMBINATION: {value} "
                  f"({combo.describe(value, ASCII_TO_ID)}"
                  + (f", cards {[ID_TO_ASCII[c] for c in cards]}" if cards else "")
                  + ")")
            await self.client.show_combination(value)

    async def _maybe_swap_seven(self, my_pos, raw_state):
        """Phase 8: trade our 7 of trump for the face-up card.

        Preconditions, all deduced LOCALLY -- the server sends no "you may
        swap" event; it just opens phase 8 after every round-1 accept and
        waits on a timeout:
          1. phase == 8. Round-2 picks go 7 -> 9 and forced-trump ("BIZON")
             hands go 5 -> 9, so both skip the window with no special case.
          2. trump == the face-up suit (implied by 1, asserted anyway).
          3. we hold the 7 of trump -- and it must be among the FIRST FIVE
             cards, since the rest are dealt at phase 9, after the window.
          4. the face-up card is not itself that 7.
          5. the face-up card is not going to us or our partner: swapping
             inside our own team just shuffles a good trump and the 7 around
             for nothing while announcing that we hold the 7.

        Exactly one message per window, payload {} as PASS sends. Whether it
        worked is read back from the state, not assumed.
        """
        env = self.sync_engine.env
        hand_id = self.sync_engine.hand_id
        if env.trump is None or env.face_up_card is None:
            return
        if env.trump != env.face_up_card // 8:      # not a round-1 accept
            return
        seven = env.trump * 8                       # rank 0 == the 7

        if self._swap_hand != hand_id:              # entering a new window
            self._swap_hand = hand_id
            self._swap_sent = False
            self._swap_skip = False

        # Success is visible in the state: swapSeven carries our seat and the
        # 7 leaves our hand. Report as soon as either shows up.
        if self._swap_sent:
            if raw_state.get("swapSeven", -1) == my_pos or seven not in env.hands[my_pos]:
                if not self._swap_skip:
                    self._swap_skip = True          # nothing further to do
                    print("[Bot] SWAP SEVEN confirmed: we now hold "
                          f"{ID_TO_ASCII[env.face_up_card]}.")
            return

        if self._swap_skip or seven not in env.hands[my_pos]:
            return
        if env.face_up_card == seven:
            return

        recipient = self.sync_engine._natural_face_up_recipient()
        if recipient is not None and (recipient - my_pos) % 2 == 0:
            who = "us" if recipient == my_pos else f"our partner (seat {recipient})"
            print(f"[Bot] holding the 7 of trump, but the face-up card goes "
                  f"to {who} -- not swapping.")
            self._swap_skip = True
            return

        self._swap_sent = True
        print(f"[Bot Action] -> SWAP SEVEN ({ID_TO_ASCII[seven]} for "
              f"{ID_TO_ASCII[env.face_up_card]})")
        await self.client.swap_seven()

    def _resolve_swap_attempt(self, my_pos):
        """Window closed: did the swap actually happen?"""
        if not self._swap_sent:
            return
        self._swap_sent = False
        env = self.sync_engine.env
        seven = env.trump * 8 if env.trump is not None else None
        if seven is None:
            return
        if seven in env.hands[my_pos]:
            print(f"[Bot][WARN] SWAP SEVEN had no effect -- still holding "
                  f"{ID_TO_ASCII[seven]}. Either the payload shape is wrong "
                  f"(capture the outgoing frame from the web client) or the "
                  f"server declined it.")
        elif not self._swap_skip:
            print("[Bot] SWAP SEVEN succeeded (detected after the window).")

    async def on_combination(self, seat, value):
        """SHOW_COMBINATION broadcast -> pin those cards in the belief state.
        The state's per-player `combinations` field carries the same data, so
        this is a redundancy against dropped messages, not the only path."""
        self.sync_engine.apply_combination(seat, value)

    async def _select_and_execute_action(self, my_pos: int, turn_id: str):
        env = self.sync_engine.env

        # One decision per turn. A re-dispatch replays from the SAME pre-turn
        # hidden state (so the LSTM advances once per game decision, as in
        # train.py collect_rollout) and blocks every action already refused.
        if self._turn_decision is not None and self._turn_decision[0] == turn_id:
            _, hidden_before, refused = self._turn_decision
        else:
            hidden_before, refused = self.hidden_state, set()
            self._turn_decision = (turn_id, hidden_before, refused)

        local_obs, global_obs, legal_mask = build_observation(
            env,
            my_pos,
            self.sync_engine.match_scores
        )

        if not legal_mask.any():
            print(f"[Bot][WARN] empty legal mask on our turn ({turn_id}); "
                  f"nothing dispatched -- the turn will time out unless a "
                  f"newer frame arrives")
            self.last_action_turn_id = None      # allow an immediate retry
            return

        legal_mask = legal_mask.copy()
        for a in refused:
            legal_mask[a] = 0
        if not legal_mask.any():
            print("[Bot][WARN] every legal action has already been refused "
                  "this turn; giving up rather than looping")
            return

        local_t = torch.from_numpy(local_obs).unsqueeze(0).to(self.device)
        glob_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _, new_hidden = self.model(
                local_t, glob_t, hidden_before, mask_t, is_sequence=False
            )
            action = int(torch.argmax(dist.probs, dim=-1).item())

        # Derived from hidden_before, never chained off a previous retry.
        self.hidden_state = new_hidden
        refused.add(action)

        await self._dispatch_action(action, env)

    def _tag(self):
        """Prefix for action lines so a log read later is never ambiguous."""
        return "[seat bot-controlled] " if self.seat_bot_controlled else ""

    async def _dispatch_action(self, action: int, env: BelotEnv):
        if env.phase == "BIDDING":
            if action == 32:
                print(f"{self._tag()}[Bot Action] -> PASS")
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
                    print(f"{self._tag()}[Bot Action] -> ACCEPT FACE-UP (Suit: {colyseus_suit})")
                    await self.client.bid_trump(colyseus_suit)
            elif 34 <= action <= 37:
                suit = action - 34
                if env.face_up_suit is None:
                    # env.get_legal_actions() compares `suit != face_up_suit`;
                    # with None that is True for ALL FOUR suits, so the mask
                    # can offer the flipped suit -- an illegal round-2 bid.
                    # The server refuses it, the turn times out, and the
                    # platform bot takes the seat (PLATFORM_NOTES §5.1, §12).
                    if env.get_legal_actions()[32]:
                        print("[Bot][CRITICAL] face_up_suit unknown in round 2; "
                              "the suit mask is unreliable -- passing instead")
                        await self.client.pass_turn()
                        return
                    print("[Bot][CRITICAL] face_up_suit unknown and passing is "
                          f"illegal (we are the dealer); bidding suit {suit} "
                          "blind -- this may be refused")
                elif suit == env.face_up_suit:
                    print(f"[Bot][CRITICAL] refusing to bid the face-up suit "
                          f"{suit} in round 2 (illegal); passing instead")
                    if env.get_legal_actions()[32]:
                        await self.client.pass_turn()
                        return
                colyseus_suit = suit + 1
                print(f"{self._tag()}[Bot Action] -> CHOOSE SUIT ({colyseus_suit})")
                await self.client.bid_trump(colyseus_suit)

        elif env.phase == "PLAYING":
            if 0 <= action <= 31:
                card_char = ID_TO_ASCII[action]
                print(f"{self._tag()}[Bot Action] -> PLAY CARD: "
                      f"{CHAR_TO_CARD[card_char]} ({card_char})")
                self._pending_play = (self.sync_engine.hand_id,
                                      env.tricks_played, action)
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
        try:
            if msg_type == "SHOW_COMBINATION" and isinstance(data, dict):
                if AUDIT:
                    self.auditor.note_combination(data.get("who"),
                                                  data.get("value"), "message")
                await self.on_combination(data.get("who"), data.get("value"))
            elif msg_type in ("LESS_THAN_14", "FOUR_OF_SEVEN", "FOUR_OF_EIGHT",
                              "WIN_ALL_HANDS", "SURRENDER_BT", "BIZON"):
                if AUDIT:
                    self.auditor.note_special(msg_type, data)
            elif msg_type == "SWAP_SEVEN" and isinstance(data, dict):
                who = data.get("who")
                seven = ASCII_TO_ID.get(data.get("swappedCard"))
                top = ASCII_TO_ID.get(data.get("topCard"))
                self.sync_engine.apply_seven_swap(who, seven, top)
                if who == self.sync_engine.my_pos:
                    print("[Bot] server broadcast confirms OUR swap")
        except Exception as e:
            print(f"[MSG][ERROR] suppressed: {type(e).__name__}: {e}")

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