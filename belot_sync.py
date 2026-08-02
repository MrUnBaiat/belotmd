import json
import re
import numpy as np
from env import BelotEnv

ASCII_TO_ID = {
    'y': 0, 'z': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7,
    'A': 8, 'B': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15,
    'C': 16, 'D': 17, 'm': 18, 'n': 19, 'o': 20, 'p': 21, 'q': 22, 'r': 23,
    'E': 24, 'F': 25, 's': 26, 't': 27, 'u': 28, 'v': 29, 'w': 30, 'x': 31
}

ID_TO_ASCII = {v: k for k, v in ASCII_TO_ID.items()}

DEAL_PHASES = (0, 1, 2, 3, 4, 5)   # ready / cut / deal
BID_PHASES  = (6, 7)               # bidding round 1 / round 2
END_PHASES  = (13, 14)             # hand scored / finished


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
        self.env = BelotEnv()
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
        self._swap_warned = False
        self._new_hand_event = False
        # Leader chain: first trick led by the declarer, every later trick by
        # the previous trick's winner. Lets us rebuild chronological order
        # from SEAT-INDEXED lastCards even with total table-frame loss.
        self._next_leader = None
        self.last_trick_source = None      # 'cardOrder' | 'seatIndex' | 'partial'
        # `round` increments at the deal (observed 13->2 with round n->n+1),
        # making it a far more reliable hand boundary than phase edges.
        self._synced_round = None
        self._reset_for_round = object()   # sentinel: never equal to a round
        self.hand_id = 0                   # non-consuming hand counter
        # declarer_has_played_trump derived locally (server field observed
        # stale across hand boundaries).
        self._declarer_trump_seen = False
        self._score_anomaly_logged = False
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
        e = self.env
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
        self.env.reset()
        self._neutralize_fictional_deal()
        self._trick_plays = {}
        # ADOPT (don't process) whatever lastCards the server still carries
        # from the previous hand, so it can't be replayed into the fresh
        # graveyard / tricks_played counter on this same frame.
        self._known_last_cards_str = _last_cards_str(raw_state)
        self._swap_warned = False
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
        at the third bolt."""
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
        _get_card_value, so the rank tables can never drift)."""
        if len(trick) != 4:
            return None
        led_suit = trick[0][1] // 8
        best_seat, best_rank, best_is_trump = None, -1, False
        for seat, card in trick:
            suit = card // 8
            is_trump = (self.env.trump is not None and suit == self.env.trump)
            _, rank = self.env._get_card_value(card, is_trump)
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
        if self.env.declarer is None or self.env.trump is None:
            return
        for seat, card in trick:
            if seat == self.env.declarer and card // 8 == self.env.trump:
                self._declarer_trump_seen = True

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
                self.env.impossible_cards[seat, led_suit * 8:led_suit * 8 + 8] = True
                if self.env.trump is not None and suit != self.env.trump:
                    t = self.env.trump
                    self.env.impossible_cards[seat, t * 8:t * 8 + 8] = True

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

        # Phase mapping
        if phase in BID_PHASES:
            self.env.phase = "BIDDING"
            self.env.bidding_round = 1 if phase == 6 else 2
        elif phase >= 10:
            self.env.phase = "PLAYING"
        else:
            self.env.phase = "PRE_GAME"
        self.env.done = phase in END_PHASES

        # 3. Direct mappings -----------------------------------------------
        self.env.dealer = raw_state.get("dealer", 0)
        self.env.current_player = raw_state.get("activePlayer", 0)

        declarer = raw_state.get("declarer", None)
        self.env.declarer = declarer if (declarer is not None and declarer >= 0) else None

        # Server flag kept only as a degraded-mode fallback (see
        # _track_declarer_trump). Primary source is our own trick history.
        self._server_trump_flag = bool(raw_state.get("trumpWasPlayed", False))

        trump_val = raw_state.get("trump", -1)
        self.env.trump = (trump_val - 1) if 1 <= trump_val <= 4 else None

        top_card_char = raw_state.get("topCard", "")
        if top_card_char in ASCII_TO_ID:
            self.env.face_up_card = ASCII_TO_ID[top_card_char]
            self.env.face_up_suit = self.env.face_up_card // 8

        # known_cards: face-up recipient, edge-triggered on entry into play.
        # (prev in END_PHASES covers a dropped-bidding jump 14 -> 10.)
        # NOT re-run every frame: the flag is cleared once the card is played
        # and must never be resurrected.
        if (phase >= 10 and not self.env.done
                and (prev_phase < 10 or prev_phase in END_PHASES)):
            if top_card_char in ASCII_TO_ID and self.env.trump is not None:
                top_id = ASCII_TO_ID[top_card_char]
                if self.env.trump == top_id // 8:
                    recipient = self.env.declarer          # accepted in round 1
                else:
                    recipient = raw_state.get("dealer", None)  # round-2 pick
                if recipient is not None and recipient >= 0:
                    self.env.known_cards[recipient, top_id] = True
            # Anchor the leader chain: the declarer leads the first trick.
            if self._next_leader is None and self.env.declarer is not None:
                self._next_leader = self.env.declarer

        # swapSeven: unmodeled platform feature (trump-7 exchange). If it ever
        # fires, the face-up card's ownership is no longer what our inference
        # says -> neutralize the known flag rather than assert a wrong one.
        swap = raw_state.get("swapSeven", -1)
        if swap is not None and swap >= 0 and self.env.face_up_card is not None:
            if not self._swap_warned:
                print("[Sync][WARN] swapSeven active; clearing face-up known_cards.")
                self._swap_warned = True
            self.env.known_cards[:, self.env.face_up_card] = False

        # 4. Trick reconstruction (per-seat merge; wipe/bundle-safe) --------
        # cardPlayed/cardOrder are ABSENT for players yet to act; cardOrder is
        # 1-based within the trick.
        table = []
        for i, p in enumerate(players):
            ch = p.get("cardPlayed", "")
            if ch in ASCII_TO_ID:
                table.append((p.get("cardOrder", 99), i, ASCII_TO_ID[ch]))
        table.sort(key=lambda x: x[0])

        grave = set(self.env.graveyard)
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
                self.env.last_trick = [(seat, card) for _, seat, card in finished]
                self.last_trick_source = "cardOrder"
            elif len(by_seat) == 4 and self._next_leader is not None:
                # Fallback: rotate seat-indexed lastCards by the known leader.
                L = self._next_leader
                self.env.last_trick = [((L + k) % 4, by_seat[(L + k) % 4])
                                       for k in range(4)]
                self.last_trick_source = "seatIndex"
            else:
                # Last resort: whatever partial evidence exists.
                self.env.last_trick = [(seat, card) for _, seat, card in finished]
                self.last_trick_source = "partial"

            self._mark_voids(self.env.last_trick)   # backstop for dropped frames
            self._track_declarer_trump(self.env.last_trick)

            # Chain the leader for the next trick.
            winner = self._trick_winner(self.env.last_trick)
            if winner is not None:
                self._next_leader = winner

            # Graveyard sourced from lastCards itself (never a partial cache),
            # so content stays complete even if table frames were lost.
            for card in completed:
                if card not in self.env.graveyard:
                    self.env.graveyard.append(card)
                self.env.known_cards[:, card] = False

            self.env.tricks_played += 1
            self._trick_plays = {
                seat: (order, card)
                for seat, (order, card) in self._trick_plays.items()
                if card not in completed
            }
            self._known_last_cards_str = last_str

        # Current trick = surviving plays, chronological by cardOrder
        cur = sorted((o, s, c) for s, (o, c) in self._trick_plays.items())
        self.env.current_trick = [(s, c) for _, s, c in cur]
        self._mark_voids(self.env.current_trick)             # mid-trick, live
        self._track_declarer_trump(self.env.current_trick)
        if self._next_leader is None and self.env.current_trick:
            self._next_leader = self.env.current_trick[0][0]
        for seat, card in self.env.current_trick:
            self.env.known_cards[seat, card] = False         # env clears on play

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
        self.env.declarer_has_played_trump = bool(
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
        for i, p in enumerate(players if not self.env.done else []):
            cards_field = p.get("cards") or ""
            if cards_field:
                self.env.hands[i] = [ASCII_TO_ID[c] for c in cards_field
                                     if c in ASCII_TO_ID]
            elif i == self.my_pos:
                self.env.hands[i] = []
            else:
                n = p.get("numCards")
                if n is None:  # tertiary fallback: infer from trick progress
                    if self.env.phase == "BIDDING":
                        n = 5
                    elif self.env.phase == "PLAYING":
                        played_now = any(s == i for s, _ in self.env.current_trick)
                        n = 8 - self.env.tricks_played - (1 if played_now else 0)
                    else:
                        n = 0
                self.env.hands[i] = list(range(max(0, int(n))))  # dummy ids

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
                self.env.bolts_by_team = [0, 0]        # fresh match
            else:
                scores, bolts = self._decode_score_table(score_table)
                self.match_scores = scores
                self.env.bolts_by_team = bolts
        except Exception:
            pass

        # 8. Live raw points -------------------------------------------------
        # Payload-confirmed: roundTotals.p is the PREVIOUS round's summary
        # (sums to exactly 162 mid-hand). Live trick captures are the
        # per-player `points` fields; team = array-index parity, matching
        # observation.py's team_us = abs_id % 2. Gated to play so bidding
        # observations keep raw = 0 like training.
        # NOTE (verify via auditor): assumed to be trick captures only; if the
        # site adds shown-combination points here, values can exceed the
        # training cap of 162 (out-of-distribution).
        if self.env.phase == "PLAYING" and len(players) == 4:
            pts = [_to_int(pl.get("points", 0)) for pl in players]
            self.env.raw_points_by_team[0] = pts[0] + pts[2]
            self.env.raw_points_by_team[1] = pts[1] + pts[3]

        return self.my_pos