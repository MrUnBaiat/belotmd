"""
sync.py — raw belot.md state frames -> a reconstructed `BelotState`.

The hard part of the project. The server publishes a partial, occasionally
stale view of the table; this module turns a stream of those frames into a
consistent local game state, including the belief information (voids, declared
combinations, seven-swaps) that no single frame contains.

Field-by-field trust rules are documented in docs/PLATFORM_NOTES.md.
"""

import json
import re
import numpy as np

from ..game import combinations as combo
from ..game.state import BelotState
from .protocol import (ASCII_TO_ID, BID_PHASES, DEAL_PHASES,  # noqa: F401
                       END_PHASES, ID_TO_ASCII, SWAP_PHASE)


_BOLT_RE = re.compile(r"^\s*BT[-_ ]?(\d+)\s*$", re.IGNORECASE)


def _bolt_marker(cell):
    """LIVE-DECODED: a scoreTable/roundTotals cell of the form 'BT-N' means
    the team BOLTED that round -- they gained 0 match points, and it was
    their N-th bolt. The cumulative score carries forward unchanged.
    Verified across a full match: 35 -> 'BT-1' -> 51 with b=16 that round
    (35 + 16 = 51), so the bolt row itself contributed nothing.
    Returns N, or None if the cell is not a bolt marker.
    NOTE: naive int-coercion of 'BT-1' yields -1, which silently corrupted
    match_scores (fed to the model as obs[185]/obs[186])."""
    if isinstance(cell, str):
        m = _BOLT_RE.match(cell)
        if m:
            return int(m.group(1))
    return None


def _to_int(v, default=0):
    """belot.md sometimes puts non-integers in scoreTable cells (observed
    live: a str where an int was expected, which crashed observation.py via
    match_scores). Coerce defensively and never let it reach the model."""
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+", v)
        return int(m.group()) if m else default
    return default


def _last_cards_str(raw_state) -> str:
    """Server sends lastCards as a plain string ("xFtu"); tolerate a list too."""
    lc = raw_state.get("lastCards") or ""
    return "".join(lc) if isinstance(lc, list) else lc


class StateSynchronizer:
    def __init__(self):
        self.state = BelotState()
        # BelotState() deals a fictional random hand, and ~1/8 of the time
        # (face-up Jack) self-finalizes bidding, leaving a belief pin behind.
        # A frame arriving before the first _start_new_hand() would then see a
        # pin for a card nobody holds -- a warning that appeared in roughly one
        # replay in eight and in no other. Wipe it up front so the synchronizer
        # starts from nothing whatever the deck happened to be.
        self._neutralize_fictional_deal()
        self.my_pos = None
        self.match_scores = [0, 0]

        self._known_last_cards_str = ""
        self._prev_phase = -1
        # seat -> (cardOrder, card_id). Survives table wipes and bundled
        # frames; partitioned (not clobbered) on trick completion.
        self._trick_plays = {}
        # Force a clean shadow-reset on the first actionable frame
        # (cold start AND the bridge.js mid-hand rejoin path).
        self._pending_reset = True
        self._new_hand_event = False
        # Leader chain: first trick led by the declarer, every later trick by
        # the previous trick's winner. Lets us rebuild chronological order
        # from SEAT-INDEXED lastCards even with total table-frame loss.
        self._next_leader = None
        self.last_trick_source = None      # 'cardOrder' | 'seatIndex' | 'partial'
        # `round` increments at the deal (observed 13->2 with round n->n+1),
        # making it a far more reliable hand boundary than phase edges.
        self._synced_round = None
        self._sentinel = object()          # never equal to any round value
        self._reset_for_round = self._sentinel
        self.hand_id = 0                   # non-consuming hand counter
        # declarer_has_played_trump derived locally (server field observed
        # stale across hand boundaries).
        self._declarer_trump_seen = False
        self._score_anomaly_logged = False
        self._applied_combos = set()    # (seat, value) already folded in
        self._swap = None          # (who, seven_id, top_id) once a swap is seen
        self.combo_conflicts = 0        # decoded card found in OUR hand
        # True while playing out a hand we joined mid-way (past tricks are
        # unrecoverable). The auditor downgrades conservation violations to
        # INFO while this is set. Cleared at the next clean deal.
        self.degraded_hand = False

    # ------------------------------------------------------------------ events
    def consume_new_hand(self) -> bool:
        """One-shot flag: the agent zeroes its LSTM hidden state when True."""
        if self._new_hand_event:
            self._new_hand_event = False
            return True
        return False

    # ------------------------------------------------------------------ resets
    def _neutralize_fictional_deal(self):
        """env.reset() deals a fictional random deck locally and ~1/8 of the
        time (face-up Jack) self-finalizes bidding: ghost known_cards,
        fictional trump/declarer, 8-card hands. The live deal comes from the
        server, so wipe every artifact. bolts_by_team and dealer survive
        (matching env.reset semantics)."""
        e = self.state
        e.deck = []
        e.hands = [[] for _ in range(4)]
        e.face_up_card = None
        e.face_up_suit = None
        e.face_up_rank = None
        e.trump = None
        e.declarer = None
        e.declaring_team = None
        e.defending_team = None
        e.declarer_has_played_trump = False
        e.phase = "BIDDING"
        e.bidding_round = 1
        e.passes_in_round = 0
        e.tricks_played = 0
        e.current_trick = []
        e.trick_history = []
        e.last_trick = []
        e.graveyard = []
        e.tricks_won_by_team = [0, 0]
        e.raw_points_by_team = [0, 0]
        e.known_cards[:] = False
        e.impossible_cards[:] = False
        e.done = False

    def _start_new_hand(self, raw_state):
        self.state.reset()
        self._neutralize_fictional_deal()
        self._trick_plays = {}
        # ADOPT (don't process) whatever lastCards the server still carries
        # from the previous hand, so it can't be replayed into the fresh
        # graveyard / tricks_played counter on this same frame.
        self._known_last_cards_str = _last_cards_str(raw_state)
        # Per-hand belief bookkeeping. Leaving _swap set was what froze
        # face_up_card on the previous hand's card; leaving _applied_combos
        # set silently dropped a declaration whose (seat, value) had already
        # been seen in an earlier hand.
        self._swap = None
        self._applied_combos = set()
        self._new_hand_event = True
        self.hand_id += 1
        self._next_leader = None
        self.last_trick_source = None
        self._declarer_trump_seen = False

    # ------------------------------------------------------------------ scores
    @staticmethod
    def _decode_score_table(tbl):
        """-> (match_scores, bolts). Walks the cumulative rows, carrying the
        previous total forward across 'BT-N' cells and reading the bolt count
        straight off the marker (authoritative -- replaces the old inference
        from roundTotals.b, which never fired because b is the *string*
        'BT-N', not 0). Bolt count is taken mod 3 to match env's roll-over
        at the third bolt.

        THE THIRD BOLT ALSO COSTS 10 POINTS, and the penalty is invisible in
        scoreTable because the cell is the string 'BT-3' rather than a number.
        Carrying the previous total forward therefore overstates the bolted
        team by 10 until the next numeric row lands. Confirmed against a
        captured match:

            row 4  [9, 72]        -> team 0 on 9
            row 5  ['BT-3', 88]   -> cell is a string, true total is -1
            row 6  [2, 101]       -> roundTotals.b says +3, and -1 + 3 == 2

        Without the penalty the implied delta is 2 - 9 = -7, which is what the
        auditor flagged as 'deltas (-7,13) vs b(3,13) -> MISMATCH'. This is
        env.py's own rule (bolts += 1; at 3, subtract 10 and reset), so the
        counter was already right -- only the score was not.

        `n % 3 == 0` catches the penalty whether the platform keeps counting
        up (BT-3, BT-6, ...) or resets its label after every third."""
        scores, bolts = [0, 0], [0, 0]
        if not isinstance(tbl, list):
            return scores, bolts
        for row in tbl:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            for t in (0, 1):
                n = _bolt_marker(row[t])
                if n is not None:
                    bolts[t] = n % 3        # env zeroes the counter at 3
                    if n and n % 3 == 0:    # third bolt: -10, invisible above
                        scores[t] -= 10
                else:                        # score unchanged on a bolt row
                    scores[t] = _to_int(row[t], scores[t])
        return scores, bolts

    # ------------------------------------------------------------------ bolts
    @staticmethod
    def _parse_round_totals(raw_state):
        try:
            rt = raw_state.get("roundTotals", "[]")
            rt_str = rt if isinstance(rt, str) else json.dumps(rt)
            rt = json.loads(rt) if isinstance(rt, str) else rt
            if isinstance(rt, list) and len(rt) == 2:
                return rt_str, rt
        except Exception:
            pass
        return None, None

    # ------------------------------------------------------------------ tricks
    def _trick_winner(self, trick):
        """Chronological trick -> winning seat. Faithful mirror of
        env._evaluate_trick's comparison logic (reuses the env's own
        card_value, so the rank tables can never drift)."""
        if len(trick) != 4:
            return None
        led_suit = trick[0][1] // 8
        best_seat, best_rank, best_is_trump = None, -1, False
        for seat, card in trick:
            suit = card // 8
            is_trump = (self.state.trump is not None and suit == self.state.trump)
            _, rank = self.state.card_value(card, is_trump)
            if not is_trump and not best_is_trump and suit == led_suit:
                if rank > best_rank:
                    best_rank, best_seat = rank, seat
            elif is_trump and best_is_trump:
                if rank > best_rank:
                    best_rank, best_seat = rank, seat
            elif is_trump and not best_is_trump:
                best_is_trump, best_rank, best_seat = True, rank, seat
        return best_seat

    def _track_declarer_trump(self, trick):
        """env sets declarer_has_played_trump the moment the declarer plays a
        trump. Derived locally rather than taken from the server field, whose
        semantics differ (see the intersection note in sync())."""
        if self.state.declarer is None or self.state.trump is None:
            return
        for seat, card in trick:
            if seat == self.state.declarer and card // 8 == self.state.trump:
                self._declarer_trump_seen = True

    # ------------------------------------------------------------------ pinning
    def _pin_known(self, seat, card, exclusive=True):
        """Mark `card` as certainly held by `seat`. With `exclusive`, first
        clear the column so no stale holder survives (used by the seven-swap,
        where a card genuinely changes hands mid-deal)."""
        if seat is None or not (0 <= seat <= 3) or card is None:
            return False
        if card in self.state.graveyard:
            return False
        if any(c == card for _, c in self.state.current_trick):
            return False
        if (self.my_pos is not None and seat != self.my_pos
                and card in self.state.hands[self.my_pos]):
            return False            # it is in OUR hand: the claim is wrong
        if exclusive:
            self.state.known_cards[:, card] = False
        self.state.known_cards[seat, card] = True
        return True

    def apply_seven_swap(self, who, seven, top):
        """A player traded the 7 of trump for the face-up card.

        Server message: SWAP_SEVEN {who, swappedCard, topCard}. Afterwards
        `who` holds the face-up card and the card's previous holder -- the
        natural recipient, i.e. the declarer in a round-1 accept -- holds the
        7. Both are certainties, so both are pinned, and the pins are
        exclusive because ownership actually moved.

        Covers every case symmetrically: we swap with someone, someone swaps
        with us, or two opponents swap. Our own hand is resynced from the
        server either way; what needs inferring is the other side of the
        trade.
        """
        if who is None or not (0 <= who <= 3):
            return
        if seven is None and self.state.trump is not None:
            seven = self.state.trump * 8                 # rank 0 == the 7
        if top is None:
            top = self.state.face_up_card
        if seven is None or top is None or seven == top:
            return

        natural = self._natural_face_up_recipient()
        self._swap = (who, seven, top)
        self._pin_known(who, top)
        if natural is not None and natural != who:
            self._pin_known(natural, seven)
        print(f"[Sync] seven-swap: seat {who} took {ID_TO_ASCII.get(top)} "
              f"(face-up), seat {natural} received "
              f"{ID_TO_ASCII.get(seven)} (7 of trump)")

    def _natural_face_up_recipient(self):
        """Who would hold the face-up card with no swap: the declarer when it
        was accepted in round 1, otherwise the dealer."""
        if self.state.face_up_card is None or self.state.trump is None:
            return None
        if self.state.trump == self.state.face_up_card // 8:
            return self.state.declarer
        return self.state.dealer

    # ------------------------------------------------------------------ combos
    def apply_combination(self, seat, value):
        """Fold a declared combination into the belief state.

        Uses known_cards, the same channel and the same 1.0 semantics the
        face-up card uses in training. observation.py now excludes every
        known card from all three candidate rows, so one pin is enough: the
        holder reads 1.0 and the other two read 0.0, with the column summing
        to exactly 1.0. (Before that fix a pin leaked ~0.49 onto each other
        opponent, so five declared cards misallocated ~37% of their mass.)
        """
        if seat is None or seat < 0 or not value:
            return
        key = (seat, value)
        if key in self._applied_combos:
            return
        self._applied_combos.add(key)

        cards, unknown = combo.decode_field(value, ASCII_TO_ID)
        for type_id, ch in unknown:
            print(f"[Sync][WARN] unrecognised combination type {type_id} "
                  f"('{type_id}{ch}') from seat {seat}: ignored")
        if not cards:
            return

        # Self-check: a decoded card sitting in OUR hand proves the decoding
        # is wrong. Writing a false 1.0 into the belief matrix is far worse
        # than ignoring the declaration, so drop the whole thing.
        if seat == self.my_pos:
            return          # our own declaration echoed back: nothing to learn
        mine = set(self.state.hands[self.my_pos]) if self.my_pos is not None else set()
        clash = [c for c in cards if c in mine]
        if clash:
            self.combo_conflicts += 1
            print(f"[Sync][ERROR] combination '{value}' from seat {seat} "
                  f"decodes to cards {clash} that are in OUR hand -- "
                  f"decoding is wrong, declaration ignored")
            return

        gone = set(self.state.graveyard) | {c for _, c in self.state.current_trick}
        applied = [c for c in cards if c not in gone]
        applied = [c for c in applied if self._pin_known(seat, c, exclusive=False)]
        dropped = [c for c in cards if c not in applied]
        if dropped:
            print(f"[Sync] seat {seat} '{value}': "
                  f"{[ID_TO_ASCII[c] for c in dropped]} already played, "
                  f"not pinned")
        if applied:
            print(f"[Sync] seat {seat} declared '{value}' -> "
                  f"{[ID_TO_ASCII[c] for c in applied]} pinned in belief state")

    # ------------------------------------------------------------------ beliefs
    def _mark_voids(self, trick):
        """Exact mirror of env._handle_playing_action's inference, applied
        incrementally so mid-trick observations match training (the env marks
        voids at play time, not at trick end). Idempotent."""
        if len(trick) < 2:
            return
        led_suit = trick[0][1] // 8
        for seat, card in trick[1:]:
            suit = card // 8
            if suit != led_suit:
                self.state.impossible_cards[seat, led_suit * 8:led_suit * 8 + 8] = True
                if self.state.trump is not None and suit != self.state.trump:
                    t = self.state.trump
                    self.state.impossible_cards[seat, t * 8:t * 8 + 8] = True

    # ------------------------------------------------------------------ main
    def sync(self, raw_state: dict, my_player_id: str) -> int:
        players = raw_state.get("players", [])
        if not players:
            return None

        # 1. Identity
        self.my_pos = None
        for idx, p in enumerate(players):
            if p.get("id") == my_player_id or p.get("sessionId") == my_player_id:
                self.my_pos = idx
                break
        if self.my_pos is None:
            return None

        phase = raw_state.get("currentPhase", 0)
        prev_phase = self._prev_phase
        self._prev_phase = phase

        # 2. Hand lifecycle ------------------------------------------------
        # Arm the latch on transitions into deal/end phases OR on a `round`
        # bump. Fire it on the first frame of the new hand that is not an END
        # phase, i.e. AT the deal (phase 2) rather than waiting for bidding.
        # Waiting caused the live `hand n graveyard` violations: the server
        # deals the next hand while phase is still 13/2, so new cards met the
        # previous hand's graveyard.
        cur_round = raw_state.get("round", None)
        round_changed = (cur_round is not None
                         and self._synced_round is not None
                         and cur_round != self._synced_round)
        self._synced_round = cur_round

        edge = (prev_phase not in DEAL_PHASES and prev_phase not in END_PHASES)
        if round_changed or ((phase in DEAL_PHASES or phase in END_PHASES) and edge):
            self._pending_reset = True

        # A hand can be ABORTED at any point: LESS_THAN_14 and FOUR_OF_SEVEN
        # both cancel the deal outright, dropping the phase straight back to
        # the deal (10 -> 1 -> 2 observed live) while the server appends a
        # duplicate scoreTable row. The abort can equally fire during bidding
        # or the swap window, so ANY entry into a deal phase from a live one
        # counts. `round` happened to increment every observed time, but
        # nothing guarantees it, and a reused round number would let the
        # guard below suppress the reset and leak the aborted hand's
        # graveyard, trick count and beliefs into the fresh deal.
        if (phase in DEAL_PHASES and prev_phase not in DEAL_PHASES
                and prev_phase not in END_PHASES):
            self._pending_reset = True
            self._reset_for_round = self._sentinel

        entering_bidding = phase in BID_PHASES and prev_phase not in BID_PHASES
        may_reset = (phase not in END_PHASES and phase != 0
                     and (cur_round is None or cur_round != self._reset_for_round))
        if (self._pending_reset and may_reset) or (entering_bidding and may_reset):
            joined_mid = phase >= 10 and bool(_last_cards_str(raw_state))
            if joined_mid:
                print("[Sync][WARN] Joined mid-hand: past tricks are "
                      "unrecoverable; beliefs unreliable until the next deal.")
            self._start_new_hand(raw_state)
            self._pending_reset = False
            self._reset_for_round = cur_round
            self.degraded_hand = joined_mid

        if phase in BID_PHASES and self.state.face_up_card is None:
            print("[Sync][CRITICAL] entered bidding with no face-up card "
                  f"(topCard={raw_state.get('topCard')!r}); the round-2 suit "
                  "mask cannot exclude the flipped suit")

        # Phase mapping
        if phase in BID_PHASES:
            self.state.phase = "BIDDING"
            self.state.bidding_round = 1 if phase == 6 else 2
        elif phase >= 10:
            self.state.phase = "PLAYING"
        else:
            self.state.phase = "PRE_GAME"
        self.state.done = phase in END_PHASES

        # 3. Direct mappings -----------------------------------------------
        self.state.dealer = raw_state.get("dealer", 0)
        self.state.current_player = raw_state.get("activePlayer", 0)

        # LIVE-OBSERVED: `trump` and `declarer` carry the PREVIOUS hand's
        # values right through the deal and bidding, only updating when a bid
        # is accepted (phase 6 showed trump=2/declarer=0 from the last round,
        # flipping to trump=4/declarer=3 at phase 8). Training guarantees the
        # opposite invariant -- env transitions to PLAYING the instant a bid
        # lands, so phase == "BIDDING" always implies trump is None and
        # declarer is None. Taken literally the server values put a phantom
        # trump in obs feature 3 and a phantom declarer in feature 4 for
        # every bidding decision, so they are suppressed until the deal and
        # bidding are over.
        declarer = raw_state.get("declarer", None)
        bidding_open = phase in DEAL_PHASES or phase in BID_PHASES
        self.state.declarer = (None if bidding_open
                             else (declarer if (declarer is not None and declarer >= 0)
                                   else None))

        # Server flag kept only as a degraded-mode fallback (see
        # _track_declarer_trump). Primary source is our own trick history.
        self._server_trump_flag = bool(raw_state.get("trumpWasPlayed", False))

        trump_val = raw_state.get("trump", -1)
        self.state.trump = (None if bidding_open
                          else ((trump_val - 1) if 1 <= trump_val <= 4 else None))

        # LIVE-OBSERVED: `topCard` is stale from the previous hand until the
        # deal completes, AND it changes meaning after a seven-swap -- the
        # server rewrites it from the flipped card to the 7 the declarer
        # received (observed "v" -> "E"). Keep following it until a swap is
        # seen, then freeze: everything downstream (bidding legality, the
        # face-up observation feature, the swap's own bookkeeping) needs the
        # ORIGINAL flipped card. The rewritten value is recoverable anyway,
        # since the 7 of trump is just trump * 8.
        top_card_char = raw_state.get("topCard", "")
        swap_seat = raw_state.get("swapSeven", -1)
        top_id = ASCII_TO_ID.get(top_card_char)
        if top_id is not None:
            if phase < SWAP_PHASE:
                # Deal and bidding: follow the server. topCard is stale from
                # the previous hand for the first frames, but it settles at
                # the real flipped card by the time cards are dealt (phase 5)
                # and long before we ever act (phase 6).
                self.state.face_up_card = top_id
                self.state.face_up_suit = top_id // 8
            elif self.state.face_up_card is None:
                # Joined mid-hand: adopt whatever is there. If a swap already
                # happened this is the 7, not the flipped card -- degraded,
                # and flagged as such.
                self.state.face_up_card = top_id
                self.state.face_up_suit = top_id // 8
            # From phase 8 on the value is FROZEN. A swap makes the server
            # rewrite topCard to the 7 the declarer received, and everything
            # downstream (bidding legality, the face-up feature, the swap
            # bookkeeping) needs the ORIGINAL flipped card.

        # Seven-swap fallback: the SWAP_SEVEN broadcast is the primary
        # source (bot_agent forwards it), but a dropped message would lose
        # the information, so the state field is honoured too. Both the
        # swapped card (7 of trump) and the taken card (face-up) are
        # derivable from `who` alone.
        # LIVE-CONFIRMED: swapSeven is the SEAT INDEX of the player who
        # swapped (-1 when nobody did). Primary source is the SWAP_SEVEN
        # broadcast; this recovers the information if that message is lost.
        # Gated on phase >= 8: like trump, declarer and topCard, swapSeven
        # carries the PREVIOUS hand's value through the deal. Honouring it
        # early re-armed the swap right after the reset, which then froze
        # face_up_card on the stale topCard for the whole hand -- corrupting
        # bidding legality (an illegal suit pick that the server rejected
        # until our turn timed out), the phase-8 checks, and the belief pins.
        if (phase >= SWAP_PHASE and phase not in END_PHASES
                and isinstance(swap_seat, int) and 0 <= swap_seat <= 3
                and self._swap is None):
            self.apply_seven_swap(swap_seat, None, None)

        # Deferred to section 6b: the pin guard compares candidate cards
        # against our own hand, which is only accurate after the hands have
        # been resynced (a swap changes our hand in the very same frame).
        entered_play = (phase >= 10 and not self.state.done
                        and (prev_phase < 10 or prev_phase in END_PHASES))

        # 4. Trick reconstruction (per-seat merge; wipe/bundle-safe) --------
        # cardPlayed/cardOrder are ABSENT for players yet to act; cardOrder is
        # 1-based within the trick.
        table = []
        for i, p in enumerate(players):
            ch = p.get("cardPlayed", "")
            if ch in ASCII_TO_ID:
                table.append((p.get("cardOrder", 99), i, ASCII_TO_ID[ch]))
        table.sort(key=lambda x: x[0])

        grave = set(self.state.graveyard)
        for order, seat, card in table:
            if card in grave:          # stale table remnant of a resolved trick
                continue
            self._trick_plays[seat] = (order, card)

        # 5. Trick completion — consumed BEFORE rebuilding current_trick ----
        # LIVE-CONFIRMED: lastCards is SEAT-INDEXED (lastCards[i] == the card
        # played by players[i]), NOT chronological. Chronological order is the
        # seat order rotated by the trick leader, and leaders chain:
        # trick 1 -> declarer, trick n+1 -> winner of trick n. That lets us
        # rebuild full order+seats even if every table frame was dropped.
        last_str = _last_cards_str(raw_state)
        if last_str and last_str != self._known_last_cards_str:
            completed = {ASCII_TO_ID[c] for c in last_str if c in ASCII_TO_ID}

            by_seat = {}
            if len(last_str) == 4:
                for seat, ch in enumerate(last_str):
                    if ch in ASCII_TO_ID:
                        by_seat[seat] = ASCII_TO_ID[ch]

            finished = sorted(
                (order, seat, card)
                for seat, (order, card) in self._trick_plays.items()
                if card in completed
            )

            if len(finished) == 4:
                # Primary: observed cardOrder (authoritative when we saw the
                # table fill up).
                self.state.last_trick = [(seat, card) for _, seat, card in finished]
                self.last_trick_source = "cardOrder"
            elif len(by_seat) == 4 and self._next_leader is not None:
                # Fallback: rotate seat-indexed lastCards by the known leader.
                L = self._next_leader
                self.state.last_trick = [((L + k) % 4, by_seat[(L + k) % 4])
                                       for k in range(4)]
                self.last_trick_source = "seatIndex"
            else:
                # Last resort: whatever partial evidence exists.
                self.state.last_trick = [(seat, card) for _, seat, card in finished]
                self.last_trick_source = "partial"

            # Void inference reads trick[0] as the LEADER. On the "partial"
            # path the order is not trustworthy, so marking voids there
            # attributes the wrong suits to the wrong seat -- and
            # impossible_cards is never re-derived, so it poisons the belief
            # matrix for the rest of the hand. Only the two ordered paths
            # feed it. (_track_declarer_trump is order-independent.)
            if self.last_trick_source in ("cardOrder", "seatIndex"):
                self._mark_voids(self.state.last_trick)
            elif self.state.last_trick:
                print("[Sync][WARN] last trick reconstructed from partial "
                      "evidence; skipping void inference to avoid poisoning "
                      "the belief state")
            self._track_declarer_trump(self.state.last_trick)

            # Chain the leader for the next trick.
            winner = self._trick_winner(self.state.last_trick)
            if winner is not None:
                self._next_leader = winner

            # Graveyard sourced from lastCards itself (never a partial cache),
            # so content stays complete even if table frames were lost.
            for card in completed:
                if card not in self.state.graveyard:
                    self.state.graveyard.append(card)
                self.state.known_cards[:, card] = False

            self.state.tricks_played += 1
            self._trick_plays = {
                seat: (order, card)
                for seat, (order, card) in self._trick_plays.items()
                if card not in completed
            }
            self._known_last_cards_str = last_str

        # Current trick = surviving plays, chronological by cardOrder
        cur = sorted((o, s, c) for s, (o, c) in self._trick_plays.items())
        self.state.current_trick = [(s, c) for _, s, c in cur]
        self._mark_voids(self.state.current_trick)             # mid-trick, live
        self._track_declarer_trump(self.state.current_trick)
        if self._next_leader is None and self.state.current_trick:
            self._next_leader = self.state.current_trick[0][0]
        for seat, card in self.state.current_trick:
            self.state.known_cards[seat, card] = False         # env clears on play

        # declarer_has_played_trump: locally derived (mirrors env), with the
        # server flag as fallback only when we joined mid-hand and genuinely
        # cannot know the history.
        # LIVE-DECODED: the server flag is NOT stale (it resets every hand and
        # flips mid-hand), but it means something LATER than env's rule -- it
        # stayed False for 5 tricks after the declarer had played a trump,
        # so it most likely tracks trump being *led*. Our local derivation is
        # therefore MORE PERMISSIVE than the server, which is the dangerous
        # direction: a trump lead the server rejects would loop the
        # re-dispatch. Use the intersection -- never more permissive than the
        # server, never more permissive than training. (Only affects
        # non-declarers; env exempts the declarer from the restriction.)
        self.state.declarer_has_played_trump = bool(
            self._declarer_trump_seen and self._server_trump_flag
        ) or (self._server_trump_flag if self.degraded_hand else False)

        # 6. Hands ----------------------------------------------------------
        # Payload-confirmed: opponents' cards are hidden; numCards is sent
        # instead. Only the LENGTH feeds the actor (belief row mass:
        # remaining = len(hand) - known.sum()). Contents only reach the
        # critic-side global obs, unused at inference.
        # Once the hand is scored (phase 13/14) the server starts dealing the
        # NEXT hand into `cards`. Freeze the finished hand instead of letting
        # new cards contaminate it; the reset at the deal rebuilds everything.
        for i, p in enumerate(players if not self.state.done else []):
            cards_field = p.get("cards") or ""
            if cards_field:
                self.state.hands[i] = [ASCII_TO_ID[c] for c in cards_field
                                     if c in ASCII_TO_ID]
            elif i == self.my_pos:
                self.state.hands[i] = []
            else:
                n = p.get("numCards")
                if n is None:  # tertiary fallback: infer from trick progress
                    if self.state.phase == "BIDDING":
                        n = 5
                    elif self.state.phase == "PLAYING":
                        played_now = any(s == i for s, _ in self.state.current_trick)
                        n = 8 - self.state.tricks_played - (1 if played_now else 0)
                    else:
                        n = 0
                self.state.hands[i] = list(range(max(0, int(n))))  # dummy ids

        # A seat cannot hold more certainties than cards. This goes wrong when
        # a whole trick's frames are lost: pins are only cleared for cards seen
        # in `lastCards` or on the live table, so a pinned card that was
        # actually played stays pinned while numCards shrinks. The row then
        # carries more belief mass than the hand can hold. At least one pin is
        # provably wrong and we cannot tell which, so drop them all -- a false
        # 1.0 is worse than no pin (same rule apply_combination follows).
        for i in range(4):
            n_pins = int(self.state.known_cards[i].sum())
            if n_pins > len(self.state.hands[i]):
                print(f"[Sync][WARN] seat {i} has {n_pins} pinned cards but "
                      f"only {len(self.state.hands[i])} in hand (dropped trick "
                      f"frames?); clearing its pins rather than feeding the "
                      f"model a contradiction")
                self.state.known_cards[i, :] = False

        # known_cards: face-up recipient, edge-triggered on entry into play.
        # (prev in END_PHASES covers a dropped-bidding jump 14 -> 10.)
        # NOT re-run every frame: the flag is cleared once the card is played
        # and must never be resurrected.
        if entered_play:
            top_id = ASCII_TO_ID.get(top_card_char)
            natural = self._natural_face_up_recipient()
            if top_id is not None:
                if self._swap is not None:
                    who, seven, top = self._swap
                    self._pin_known(who, top)
                    if natural is not None and natural != who:
                        self._pin_known(natural, seven)
                elif (self.my_pos is not None
                      and top_id in self.state.hands[self.my_pos]
                      and self.state.trump == top_id // 8):
                    # Round-1 accept only: phase 8 (and therefore any swap)
                    # exists solely on that path, so a round-2 pick can never
                    # be mistaken for a swap just because we happen to hold
                    # the flipped card.
                    if natural != self.my_pos:
                        self.apply_seven_swap(self.my_pos, None, top_id)
                    else:
                        self._pin_known(self.my_pos, top_id)
                elif natural == self.my_pos and self.state.trump is not None \
                        and self.state.trump * 8 in self.state.hands[self.my_pos]:
                    # We were the natural recipient but hold the 7 instead:
                    # someone swapped with us. We cannot see who, so assert
                    # nothing about the face-up card rather than guess.
                    print("[Sync] seven-swap: an opponent took the face-up "
                          "card from us; holder unknown, left unpinned")
                else:
                    self._pin_known(natural, top_id)
            # Anchor the leader chain: the declarer leads the first trick.
            if self._next_leader is None and self.state.declarer is not None:
                self._next_leader = self.state.declarer

        # 6b. Declared combinations ------------------------------------------
        # Sourced from the per-player `combinations` field so a dropped
        # SHOW_COMBINATION message cannot lose the information; bot_agent
        # also forwards the live messages. Both paths are deduplicated.
        for i, p in enumerate(players):
            declared = p.get("combinations") or ""
            if declared and i != self.my_pos:
                self.apply_combination(i, declared)

        # 7. Match scores + bolts -------------------------------------------
        # scoreTable rows are CUMULATIVE (row deltas == roundTotals.b) and
        # col0/col1 are the position-parity teams (user-confirmed), matching
        # observation.py's team_us = abs_id % 2.
        try:
            raw_tbl = raw_state.get("scoreTable", "[]")
            score_table = (json.loads(raw_tbl)
                           if isinstance(raw_tbl, str) and raw_tbl.strip()
                           else (raw_tbl if isinstance(raw_tbl, list) else []))
            if not score_table:
                self.match_scores = [0, 0]
                self.state.bolts_by_team = [0, 0]        # fresh match
            else:
                scores, bolts = self._decode_score_table(score_table)
                self.match_scores = scores
                self.state.bolts_by_team = bolts
        except Exception as exc:
            if not self._score_anomaly_logged:
                self._score_anomaly_logged = True
                print(f"[Sync][WARN] scoreTable parse failed "
                      f"({type(exc).__name__}: {exc}); match_scores held at "
                      f"{self.match_scores}")

        # 8. Live raw points -------------------------------------------------
        # Payload-confirmed: roundTotals.p is the PREVIOUS round's summary
        # (sums to exactly 162 mid-hand). Live trick captures are the
        # per-player `points` fields; team = array-index parity, matching
        # observation.py's team_us = abs_id % 2. Gated to play so bidding
        # observations keep raw = 0 like training.
        # NOTE (verify via auditor): assumed to be trick captures only; if the
        # site adds shown-combination points here, values can exceed the
        # training cap of 162 (out-of-distribution).
        if self.state.phase == "PLAYING" and len(players) == 4:
            pts = [_to_int(pl.get("points", 0)) for pl in players]
            self.state.raw_points_by_team[0] = pts[0] + pts[2]
            self.state.raw_points_by_team[1] = pts[1] + pts[3]

        return self.my_pos