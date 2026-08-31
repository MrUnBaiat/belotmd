"""
bot.py — the live belot.md player.

Owns everything that is a *rule of the platform*: when it is our turn, when to
send READY, when the seven-swap window is open, which declarations to announce,
whether the server actually accepted the card we sent, and whether belot.md has
quietly handed our seat to its own bot after a timeout.

It owns no policy. Which card to play and what to bid come from an `Agent`
(see `belotmd.agents.base`), which this module knows only through
`reset()` / `snapshot()` / `restore()` / `act()`. Nothing here imports torch or
knows what an observation vector looks like.
"""

import asyncio
import json
import time

import numpy as np

from .agents import get_agent
from .audit import Auditor, FrameRecorder
from .config import Config
from .game import actions, combinations as combo
from .game.actions import (ACTION_ACCEPT, ACTION_PASS, CARD_ACTIONS,
                           SUIT_ACTIONS, suit_of)
from .platform.client import BelotClient
from .platform.protocol import (ASCII_TO_ID, BID_PHASES, CHAR_TO_CARD,
                                DEAL_PHASES, END_PHASES, ID_TO_ASCII,
                                NOT_STARTED, PLAY, SWAP_SEVEN, DEAL_CARDS_2)
from .platform.sync import StateSynchronizer


class LiveBelotBot:
    """Plays one belot.md session with the supplied agent."""

    def __init__(self, config: Config = None, agent=None):
        self.config = config or Config.from_env()
        self.client = BelotClient(cookies=self.config.require_cookies())
        self.sync_engine = StateSynchronizer()

        # The decision-maker. Anything satisfying belotmd.agents.base.Agent
        # works; nothing below this line knows which one it got.
        self.agent = agent or get_agent(
            self.config.agent, checkpoint=self.config.checkpoint or None
        )
        print(f"[Bot] Agent: {getattr(self.agent, 'name', type(self.agent).__name__)}")

        # Turn debounce (replaces the old hard dedup, which deadlocked the
        # bot forever if the server ever rejected an action)
        self.last_action_turn_id = None
        self.last_dispatch_ts = 0.0
        # (turn_id, agent_snapshot_before_this_turn, {actions already refused})
        # Keeps agent state advancing once per DECISION rather than once per
        # DISPATCH, and blocks a refused action on re-dispatch.
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

        self.audit = self.config.audit
        if self.audit:
            self.recorder = FrameRecorder(self.config.frames_path)
            self.auditor = Auditor(verbose=True)
            print(f"[Bot] AUDIT mode on: recording frames to "
                  f"{self.config.frames_path}")

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

        # Hand boundary, detected by the sync engine itself (robust to dropped
        # phase frames). A recurrent agent zeroes its hidden state here, which
        # is what training did at the start of every episode.
        if self.sync_engine.consume_new_hand():
            self.agent.reset()
            self.last_action_turn_id = None

        if self.audit:
            # The verification layer must NEVER take the bot down: a crash
            # here previously killed the whole task, silently dropping the
            # frame (and any turn it contained).
            try:
                self.recorder.record(raw_state, my_player_id)
                self.auditor.check(self.sync_engine, raw_state)
            except Exception as e:
                print(f"[Audit][ERROR] suppressed: {type(e).__name__}: {e}")

        state = self.sync_engine.state
        phase = raw_state.get("currentPhase", 0)
        active_player = raw_state.get("activePlayer", -1)

        await self._check_play_accepted(my_pos)
        await self._check_seat_autopilot(raw_state, my_pos)

        # Declarations exist only once all eight cards are dealt (phase 9+).
        # Every other server field -- trump, declarer, topCard, swapSeven --
        # is stale through the deal, so acting on combinationsCanShow before
        # the second deal risks announcing a combination from the PREVIOUS
        # hand.
        if (self.config.auto_declare and phase >= DEAL_CARDS_2
                and phase not in END_PHASES):
            await self._maybe_declare(raw_state, my_pos)
        if self.config.auto_swap_seven:
            if phase == SWAP_SEVEN:
                await self._maybe_swap_seven(my_pos, raw_state)
            elif phase >= DEAL_CARDS_2:
                self._resolve_swap_attempt(my_pos)

        # 1. Auto-Ready on game transitions (deduped by phase transition)
        if phase == NOT_STARTED or phase in END_PHASES:
            self.agent.reset()
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
        #
        # A seat plays exactly ONE card per trick, and the live gate has to
        # enforce that structurally. belot.md publishes a
        # transient frame in which our card is already on the table and our
        # hand has shrunk, but `activePlayer` has NOT yet moved on (the tell is
        # timeleft dropping to 0). Captured live, 6 ms wide:
        #
        #   activePlayer=3  cards "yEBdel"  no cardPlayed        <- our turn
        #   activePlayer=3  cards "yEBdl"   cardPlayed 'e'  t=0  <- THIS ONE
        #   activePlayer=0  cards "yEBdl"   cardPlayed 'e'       <- moved on
        #
        # `active_player == my_pos` is satisfied on the middle frame, and the
        # debounce key changes in the very same instant
        # (HAND_6_TRICK_0 -> HAND_5_TRICK_1), so neither guard blocks it: the
        # bot dispatches a second card into a trick it has already played
        # into. The server refuses it and _check_play_accepted reports
        # "the server recorded e for our seat instead".
        already_played = any(s_ == my_pos for s_, _ in state.current_trick)
        if phase == PLAY and already_played:
            return                       # our card is already on the table

        if phase in (*BID_PHASES, PLAY) and active_player == my_pos:
            turn_id = (f"PHASE_{phase}_P{my_pos}"
                       f"_HAND_{len(state.hands[my_pos])}"
                       f"_TRICK_{len(state.current_trick)}")
            now = time.monotonic()
            redispatch_after = self.config.redispatch_after_s
            if (self.last_action_turn_id == turn_id
                    and (now - self.last_dispatch_ts) < redispatch_after):
                return
            if self.last_action_turn_id == turn_id:
                print(f"[Bot][WARN] State unchanged {redispatch_after}s after "
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
        state = self.sync_engine.state
        if hand_id != self.sync_engine.hand_id:
            self._pending_play = None
            return
        if card not in state.hands[my_pos]:
            self._pending_play = None                 # accepted
            self._rejected_plays = 0
            return

        played_by_us = next((c for s_, c in state.current_trick if s_ == my_pos), None)
        moved_on = state.tricks_played > trick_no
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
                                      include_win_all=self.config.declare_win_all,
                                      include_four_sevens=self.config.declare_four_sevens,
                                      include_four_eights=self.config.declare_four_eights):
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
        state = self.sync_engine.state
        hand_id = self.sync_engine.hand_id
        if state.trump is None or state.face_up_card is None:
            return
        if state.trump != state.face_up_card // 8:      # not a round-1 accept
            return
        seven = state.trump * 8                       # rank 0 == the 7

        if self._swap_hand != hand_id:              # entering a new window
            self._swap_hand = hand_id
            self._swap_sent = False
            self._swap_skip = False

        # Success is visible in the state: swapSeven carries our seat and the
        # 7 leaves our hand. Report as soon as either shows up.
        if self._swap_sent:
            if raw_state.get("swapSeven", -1) == my_pos or seven not in state.hands[my_pos]:
                if not self._swap_skip:
                    self._swap_skip = True          # nothing further to do
                    print("[Bot] SWAP SEVEN confirmed: we now hold "
                          f"{ID_TO_ASCII[state.face_up_card]}.")
            return

        if self._swap_skip or seven not in state.hands[my_pos]:
            return
        if state.face_up_card == seven:
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
              f"{ID_TO_ASCII[state.face_up_card]})")
        await self.client.swap_seven()

    def _resolve_swap_attempt(self, my_pos):
        """Window closed: did the swap actually happen?"""
        if not self._swap_sent:
            return
        self._swap_sent = False
        state = self.sync_engine.state
        seven = state.trump * 8 if state.trump is not None else None
        if seven is None:
            return
        if seven in state.hands[my_pos]:
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
        """Ask the agent for one action and put it on the wire.

        One decision per turn. A re-dispatch replays from the SAME pre-turn
        agent state (so a recurrent policy advances once per game decision, as
        it did in training) and blocks every action the server already refused.
        """
        state = self.sync_engine.state

        if self._turn_decision is not None and self._turn_decision[0] == turn_id:
            _, snapshot, refused = self._turn_decision
            self.agent.restore(snapshot)          # replay, don't chain
        else:
            snapshot, refused = self.agent.snapshot(), set()
            self._turn_decision = (turn_id, snapshot, refused)

        legal_mask = state.get_legal_actions().astype(np.int8)
        if not legal_mask.any():
            print(f"[Bot][WARN] empty legal mask on our turn ({turn_id}); "
                  f"nothing dispatched -- the turn will time out unless a "
                  f"newer frame arrives")
            self.last_action_turn_id = None      # allow an immediate retry
            return

        for a in refused:
            legal_mask[a] = 0
        if not legal_mask.any():
            print("[Bot][WARN] every legal action has already been refused "
                  "this turn; giving up rather than looping")
            return

        action = self.agent.act(state, my_pos, self.sync_engine.match_scores,
                                legal_mask)

        if not legal_mask[action]:
            # An agent that ignores the mask would get the move refused and
            # eventually lose the seat to the platform bot. Say so loudly
            # rather than letting it look like a server problem.
            print(f"[Bot][CRITICAL] agent returned illegal action {action} "
                  f"({actions.describe(action, ID_TO_ASCII)}); not dispatching")
            return

        refused.add(action)
        await self._dispatch_action(action, state)

    def _tag(self):
        """Prefix for action lines so a log read later is never ambiguous."""
        return "[seat bot-controlled] " if self.seat_bot_controlled else ""

    async def _dispatch_action(self, action: int, state):
        """Translate an action index into the server message that performs it."""
        if state.phase == "BIDDING":
            if action == ACTION_PASS:
                print(f"{self._tag()}[Bot Action] -> PASS")
                await self.client.pass_turn()

            elif action == ACTION_ACCEPT:
                if state.face_up_suit is None:
                    # Transient race: topCard not yet observed this hand.
                    # ACTION_ACCEPT is only legal in round 1, where passing is
                    # always legal, so this fallback can never be illegal.
                    print("[Bot][WARN] face_up_suit unknown; passing defensively")
                    await self.client.pass_turn()
                else:
                    colyseus_suit = state.face_up_suit + 1
                    print(f"{self._tag()}[Bot Action] -> ACCEPT FACE-UP "
                          f"(Suit: {colyseus_suit})")
                    await self.client.bid_trump(colyseus_suit)

            elif action in SUIT_ACTIONS:
                suit = suit_of(action)
                if state.face_up_suit is None:
                    # get_legal_actions() compares `suit != face_up_suit`; with
                    # None that is True for ALL FOUR suits, so the mask can
                    # offer the flipped suit -- an illegal round-2 bid. The
                    # server refuses it, the turn times out, and the platform
                    # bot takes the seat.
                    if state.get_legal_actions()[ACTION_PASS]:
                        print("[Bot][CRITICAL] face_up_suit unknown in round 2; "
                              "the suit mask is unreliable -- passing instead")
                        await self.client.pass_turn()
                        return
                    print("[Bot][CRITICAL] face_up_suit unknown and passing is "
                          f"illegal (we are the dealer); bidding suit {suit} "
                          "blind -- this may be refused")
                elif suit == state.face_up_suit:
                    print(f"[Bot][CRITICAL] refusing to bid the face-up suit "
                          f"{suit} in round 2 (illegal); passing instead")
                    if state.get_legal_actions()[ACTION_PASS]:
                        await self.client.pass_turn()
                        return
                colyseus_suit = suit + 1
                print(f"{self._tag()}[Bot Action] -> CHOOSE SUIT ({colyseus_suit})")
                await self.client.bid_trump(colyseus_suit)

        elif state.phase == "PLAYING":
            if action in CARD_ACTIONS:
                card_char = ID_TO_ASCII[action]
                print(f"{self._tag()}[Bot Action] -> PLAY CARD: "
                      f"{CHAR_TO_CARD[card_char]} ({card_char})")
                self._pending_play = (self.sync_engine.hand_id,
                                      state.tricks_played, action)
                await self.client.play_card_char(card_char)

    async def on_server_message(self, msg_type, data):
        """Broadcasts that carry belief information the state frames do not.

        SHOW_COMBINATION and SWAP_SEVEN each pin cards into the belief state.
        The per-player `combinations` / `swapSeven` state fields carry the same
        facts, so these are redundancy against a dropped message rather than
        the only path — but they arrive first, and they are unambiguous.

        This is deliberately NOT gated on audit mode: the auditor's bookkeeping
        is, the belief updates are not.
        """
        try:
            blob = json.dumps(data)[:300]
        except Exception:
            blob = repr(data)[:300]
        print(f"[MSG] {msg_type}: {blob}")
        try:
            if msg_type == "SHOW_COMBINATION" and isinstance(data, dict):
                if self.audit:
                    self.auditor.note_combination(data.get("who"),
                                                  data.get("value"), "message")
                await self.on_combination(data.get("who"), data.get("value"))
            elif msg_type in ("LESS_THAN_14", "FOUR_OF_SEVEN", "FOUR_OF_EIGHT",
                              "WIN_ALL_HANDS", "SURRENDER_BT", "BIZON"):
                if self.audit:
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
                on_message_callback=self.on_server_message,
            )
        finally:
            self.client.close()