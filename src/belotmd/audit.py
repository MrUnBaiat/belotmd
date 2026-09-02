"""
audit.py — verification layer for the live bot.

Four capabilities:
  1. FrameRecorder  — append every raw STATE frame to a JSONL file, so any
     bug seen live can be replayed offline, deterministically, forever.
  2. Auditor.check  — hard invariants on the synchronized state after every
     sync. Violations are logged, never raised: the bot keeps playing while
     you collect evidence.
  3. Auditor probes — automatic evidence collection for the still-unverified
     server semantics (lastCards ordering, trumpWasPlayed meaning, score
     column mapping, points-include-combos, combination flow).
  4. replay()       — run a recorded JSONL through a fresh StateSynchronizer
     and report every violation with its frame number.

Usage (live) is wired up by `bot.py` whenever audit mode is on.

Usage (offline):

    python -c "from belotmd.audit import replay; replay('frames.jsonl')"

For a second opinion that shares no code with the synchronizer at all, see
tools/frame_inspector.py.
"""

import json
import time
import numpy as np

from .game import combinations as combo
from .game.belief import belief_matrix
from .platform.protocol import ASCII_TO_ID, CHAR_TO_CARD, ID_TO_ASCII
from .platform.sync import _bolt_marker, _last_cards_str, _to_int


class FrameRecorder:
    def __init__(self, path="frames.jsonl"):
        self._f = open(path, "a", buffering=1, encoding="utf-8")

    def record(self, raw_state, my_player_id):
        self._f.write(json.dumps(
            {"t": time.time(), "pid": my_player_id, "state": raw_state}
        ) + "\n")


class Auditor:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.frame_no = 0
        self.violations = 0
        # probe state
        self._probed_last_cards = ""
        self._trump_flag_prev = False
        self._trick_log = []            # [(trick_no, [(seat, card), ...])]
        self._prev_score_rows = None
        self._last_raw = [0, 0]
        self._expected_leader = None
        self._trump_bid_sampled = False
        self._trump_divergences = set()
        self._prev_rt_str = None
        self._last_team_points = None
        self._points_cap_flagged = False
        self._hand_id = None
        self._short_hand = None
        self._pending_special = None
        self._combo_conflicts = 0
        self._claim_syntax = set()
        self._swap_seen = set()
        self._in_swap_window = False
        self._swap_windows = 0
        self._swap_window_had_seven = False
        self._last_top = ""
        self._last_top_hand = None
        self._swap_window_t0 = 0.0
        self._swap_window_swapped = False
        self._combo_flagged = set()
        self._declined_claims = {}
        self._me = None

    # ------------------------------------------------------------------
    def _emit(self, kind, msg):
        if kind == "VIOLATION":
            self.violations += 1
        if self.verbose:
            print(f"[AUDIT:{kind}] f{self.frame_no} | {msg}")

    # ------------------------------------------------------------------
    def check(self, sync, raw_state):
        self.frame_no += 1
        # Reset per-hand state on the sync engine's own hand counter; the
        # leader chain does not survive a deal (live false positives).
        self._me = sync.my_pos
        hid = getattr(sync, "hand_id", None)
        if hid != self._hand_id:
            self._hand_id = hid
            self._expected_leader = None
            self._pending_special = None
            self._probed_last_cards = ""
            self._trick_log = []
            self._trump_bid_sampled = False
        env = sync.state
        me = sync.my_pos
        if me is None:
            return
        phase = raw_state.get("currentPhase", 0)
        players = raw_state.get("players", [])

        # Once phase hits 13/14 the server deals the NEXT hand into `cards`
        # while this hand's graveyard is still live: any overlap there is
        # an artifact of that window, not a sync bug.
        scored = env.done or phase in (12, 13, 14)
        grave = set(env.graveyard)
        trick_cards = {c for _, c in env.current_trick}
        my_hand = set(env.hands[me])

        # ---- hard invariants -----------------------------------------
        if len(env.graveyard) != 4 * env.tricks_played:
            self._emit("VIOLATION",
                       f"graveyard {len(env.graveyard)} != 4*tricks "
                       f"{env.tricks_played} (dropped trick frame?)")
        if my_hand & grave and not scored:
            self._emit("VIOLATION", f"my hand ∩ graveyard: {my_hand & grave}")
        if trick_cards & grave and not scored:
            self._emit("VIOLATION", f"trick ∩ graveyard: {trick_cards & grave}")
        if my_hand & trick_cards and not scored:
            self._emit("VIOLATION", f"my hand ∩ trick: {my_hand & trick_cards}")

        if getattr(sync, "combo_conflicts", 0) != self._combo_conflicts:
            self._combo_conflicts = sync.combo_conflicts
            self._emit("VIOLATION",
                       "a declared combination decoded to a card in OUR hand "
                       "-- combination decoding is wrong (see [Sync][ERROR])")

        for pidx in range(4):
            for card in ([] if scored else np.flatnonzero(env.known_cards[pidx])):
                if int(card) in grave:
                    self._emit("VIOLATION",
                               f"known_cards ghost: seat {pidx} card "
                               f"{int(card)} already in graveyard")

        # ---- training invariants ------------------------------------
        # env transitions to PLAYING the instant a bid is accepted, so
        # phase == "BIDDING" implies trump is None and declarer is None.
        # The server does NOT honour this: it keeps the previous hand's
        # values through the deal and bidding.
        if env.phase == "BIDDING":
            if env.trump is not None:
                self._emit("VIOLATION",
                           f"trump={env.trump} during BIDDING (training "
                           f"guarantees None) -- stale server field leaking "
                           f"into obs feature 3")
            if env.declarer is not None:
                self._emit("VIOLATION",
                           f"declarer={env.declarer} during BIDDING "
                           f"(training guarantees None) -- stale server "
                           f"field leaking into obs feature 4")

        # ---- server cross-checks -------------------------------------
        if env.phase == "PLAYING" and not scored:
            total = len(my_hand) + len(env.graveyard) + len(env.current_trick)
            complete = True
            for i, p in enumerate(players):
                if i == me:
                    continue
                n = p.get("numCards")
                if n is None:
                    complete = False
                    continue
                total += int(n)
                if len(env.hands[i]) != int(n):
                    self._emit("VIOLATION",
                               f"hand-size desync seat {i}: env "
                               f"{len(env.hands[i])} vs numCards {n}")
            if complete and total != 32:
                kind = ("INFO(degraded)"
                        if getattr(sync, "degraded_hand", False) else "VIOLATION")
                self._emit(kind,
                           f"card conservation: accounted {total}/32"
                           + (" (expected while mid-hand-joined)"
                              if kind != "VIOLATION" else ""))

        # ASCII map vs server debug fields (free oracle)
        for i, p in enumerate(players):
            ch, dbg = p.get("cardPlayed"), p.get("card_played")
            if ch and dbg and dbg != "None" and CHAR_TO_CARD.get(ch) != dbg:
                self._emit("VIOLATION",
                           f"ASCII map mismatch seat {i}: '{ch}' -> "
                           f"{CHAR_TO_CARD.get(ch)} but server says {dbg}")

        # ---- belief and legality checks at our decision points --------
        # These validate what the SYNCHRONIZER produced, so they are framed in
        # terms of the state and the shared belief matrix rather than any
        # particular agent's observation encoding.
        if env.current_player == me and phase in (6, 7, 10) and not env.done:
            mask = env.get_legal_actions()
            if not mask.any() and env.hands[me]:
                # An empty mask with an EMPTY hand is legitimate: the server
                # reports phase 10 with activePlayer still set for one more
                # frame after the last card of the last trick. Flagging that
                # was pure noise, and noise masks real signal.
                self._emit("VIOLATION", "empty legal mask on our turn")

            bel = belief_matrix(env, me)
            if np.isnan(bel).any():
                self._emit("VIOLATION", "NaN in the belief matrix")
            if (bel < -1e-6).any():
                self._emit("VIOLATION", "negative belief mass")
            for r, rel in enumerate([1, 2, 3]):
                pidx = (me + rel) % 4
                dead = [c for c in grave if bel[r, c] > 1e-6]
                if dead:
                    self._emit("VIOLATION",
                               f"belief>0 for graveyard cards {dead} (rel {rel})")
                if bel[r].sum() > len(env.hands[pidx]) + 0.01:
                    self._emit("VIOLATION",
                               f"belief row {rel} mass {bel[r].sum():.2f} > "
                               f"hand size {len(env.hands[pidx])}")
                if len(env.hands[pidx]) > 0 and bel[r].sum() < 1e-6:
                    self._emit("VIOLATION",
                               f"belief row {rel} collapsed to zero "
                               f"(hand size {len(env.hands[pidx])})")

        # ---- probes for unverified semantics -------------------------
        self._probe_last_cards_order(sync, raw_state)
        self._probe_trump_flag(sync, raw_state)
        self._probe_score_columns(sync, raw_state)
        self._probe_points_semantics(sync, raw_state)
        self._probe_combo_flow(raw_state)
        self._probe_swap_window(sync, raw_state)
        self._probe_top_card_rewrite(sync, raw_state)
        sw = raw_state.get("swapSeven", -1)
        if isinstance(sw, int) and sw >= 0 and (self._hand_id, sw) not in self._swap_seen:
            self._swap_seen.add((self._hand_id, sw))
            self._emit("PROBE",
                       f"state swapSeven={sw} "
                       + (f"-> reads as seat {sw}" if sw <= 3 else
                          "-> OUT OF SEAT RANGE: field is not a seat index, "
                          "the SWAP_SEVEN message is the only reliable source"))
        if (env.done and env.tricks_played < 8
                and self._short_hand != self._hand_id):
            self._short_hand = self._hand_id
            src = (f"after {self._pending_special[0]} by seat "
                   f"{self._pending_special[1]}" if self._pending_special
                   else "with no announcement seen")
            self._emit("INFO",
                       f"hand ended early: {env.tricks_played}/8 tricks, "
                       f"{len(env.graveyard)} cards in the graveyard, {src}. "
                       f"No further actions are taken; the deal resets next.")
        # snapshot AFTER probes so the score probe sees the pre-reset value
        self._last_raw = list(env.raw_points_by_team)
        if env.phase == 'PLAYING':
            self._last_team_points = list(env.raw_points_by_team)

    # ------------------------------------------------------------------
    def _probe_last_cards_order(self, sync, raw_state):
        """LIVE-CONFIRMED (5 unambiguous samples, 0 contradictions):
        lastCards is SEAT-INDEXED. This is now a VALIDATOR, not a probe --
        it asserts the property still holds and cross-checks the two
        independent reconstruction paths (cardOrder vs leader-chain)."""
        lc = _last_cards_str(raw_state)
        if not lc or lc == self._probed_last_cards or len(sync.state.last_trick) != 4:
            return
        self._probed_last_cards = lc
        self._trick_log.append((sync.state.tricks_played, list(sync.state.last_trick)))

        seat_map = dict(sync.state.last_trick)
        by_seat = "".join(ID_TO_ASCII[seat_map[s]] for s in sorted(seat_map))
        if by_seat != lc:
            self._emit("VIOLATION",
                       f"lastCards seat-indexing broken: server '{lc}' vs "
                       f"reconstructed-by-seat '{by_seat}' "
                       f"(source={sync.last_trick_source})")

        # Cross-check: chronological order must be the seat order rotated by
        # the leader, and the leader must be the previous trick's winner.
        leader = sync.state.last_trick[0][0]
        expect = [((leader + k) % 4) for k in range(4)]
        if [s for s, _ in sync.state.last_trick] != expect:
            self._emit("VIOLATION",
                       f"trick seat sequence {[s for s, _ in sync.state.last_trick]} "
                       f"is not clockwise from leader {leader}")
        # Anchor check: the declarer leads trick 1. This is the only thing
        # the seat-index fallback depends on, and it is the one place the
        # forced-trump ("BIZON") hands could differ, since they skip bidding.
        if (sync.state.tricks_played == 1 and sync.state.declarer is not None
                and not getattr(sync, 'degraded_hand', False)):
            # (skipped on a mid-hand join: our trick counter starts at
            #  the join point, so 'trick 1' is not the real first trick)
            if leader != sync.state.declarer:
                self._emit("VIOLATION",
                           f"trick 1 led by seat {leader}, but declarer is "
                           f"{sync.state.declarer} -- leader-chain anchor wrong; "
                           f"the seatIndex fallback would misorder tricks")
        if self._expected_leader is not None and leader != self._expected_leader:
            self._emit("VIOLATION",
                       f"leader chain broken: expected {self._expected_leader} "
                       f"(prev trick winner), observed {leader}")
        self._expected_leader = sync._trick_winner(sync.state.last_trick)
        self._emit("INFO",
                   f"trick {sync.state.tricks_played} ok "
                   f"(leader {leader}, winner {self._expected_leader}, "
                   f"src {sync.last_trick_source})")

    def _probe_trump_flag(self, sync, raw_state):
        """The server flag never flipped False->True in a full live hand even
        though trump was played at trick 2 => it is suspected STALE across
        hand boundaries. Definitive test: sample it during BIDDING, before a
        single card of the new hand exists. True there proves staleness."""
        flag = bool(raw_state.get("trumpWasPlayed", False))
        if sync.state.phase == "BIDDING" and not self._trump_bid_sampled:
            self._trump_bid_sampled = True
            self._emit("PROBE",
                       f"trumpWasPlayed during BIDDING = {flag} -> "
                       + ("STALE across hands (server field unusable; local "
                          "derivation in use)" if flag else
                          "resets correctly (field trustworthy)"))
        if sync.state.phase != "BIDDING":
            self._trump_bid_sampled = False

        # Compare server flag against our local derivation every play frame.
        if sync.state.phase == "PLAYING" and flag != sync.state.declarer_has_played_trump:
            key = (sync.state.tricks_played, flag, sync.state.declarer_has_played_trump)
            if key not in self._trump_divergences:
                self._trump_divergences.add(key)
                self._emit("PROBE",
                           f"trick {sync.state.tricks_played}: server "
                           f"trumpWasPlayed={flag} vs locally-derived "
                           f"declarer_has_played_trump="
                           f"{sync.state.declarer_has_played_trump} "
                           f"(declarer={sync.state.declarer})")
        self._trump_flag_prev = flag

    def _probe_score_columns(self, sync, raw_state):
        """Validate the cumulative table against roundTotals.b, decoding BT-N
        bolt markers (a bolt row scores 0 and carries the total forward).
        Naive int('BT-1') = -1 is what produced the phantom -73 deltas."""
        try:
            raw = raw_state.get("scoreTable", "[]")
            st = (json.loads(raw) if isinstance(raw, str) and raw.strip()
                  else raw)
        except Exception:
            st = None
        # A new match clears the table, and the server sends it as the EMPTY
        # STRING rather than "[]". Returning early there left _prev_score_rows
        # stale at the previous match's row count, so the growth test
        # (rows == prev + 1) failed for the first hand of every subsequent
        # match and that hand was silently never cross-checked. An
        # unreadable or absent table means zero rows, not "no information".
        rows = len(st) if isinstance(st, list) else 0
        if not isinstance(st, list):
            self._prev_score_rows = rows
            return
        if self._prev_score_rows is not None and rows == self._prev_score_rows + 1:
            cur, _ = sync._decode_score_table(st)
            prv, _ = sync._decode_score_table(st[:-1])
            deltas = [cur[0] - prv[0], cur[1] - prv[1]]
            _, rt = sync._parse_round_totals(raw_state)
            if deltas == [0, 0]:
                # LESS_THAN_14 voids the deal: the server appends a DUPLICATE
                # cumulative row and leaves roundTotals on the previous hand,
                # so comparing against b() here is meaningless.
                self._emit("INFO",
                           f"hand cancelled (LESS_THAN_14): duplicate score "
                           f"row, totals stay ({cur[0]},{cur[1]})")
                self._prev_score_rows = rows
                return
            if rt is not None:
                expect, tags = [], []
                for t in (0, 1):
                    b = rt[t].get("b")
                    n = _bolt_marker(b)
                    expect.append(0 if n is not None else _to_int(b))
                    tags.append(f"BOLT#{n}" if n is not None else str(_to_int(b)))
                verdict = ("CONSISTENT" if deltas == expect else
                           "MISMATCH -- scoring model needs revisiting")
                self._emit("PROBE",
                           f"round scored: cumulative deltas "
                           f"({deltas[0]},{deltas[1]}) vs b({tags[0]},{tags[1]})"
                           f" -> {verdict}; totals now "
                           f"({cur[0]},{cur[1]}), bolts {sync.state.bolts_by_team}")
        self._prev_score_rows = rows

    def _probe_points_semantics(self, sync, raw_state):
        """Training caps raw hand points at 162 (trick captures + pasledu).
        If the site folds combination points into per-player `points`, our
        raw_points features go out of distribution. Two checks: (a) live cap,
        (b) at hand end, team-summed points vs roundTotals.p."""
        players = raw_state.get("players", [])
        if len(players) != 4:
            return
        if sync.state.phase == "PLAYING":
            tot = sum(_to_int(p.get("points", 0)) for p in players)
            if tot > 162 and not self._points_cap_flagged:
                self._points_cap_flagged = True
                self._emit("PROBE",
                           f"per-player points sum to {tot} > 162 training cap "
                           f"-> combination points likely folded into `points`; "
                           f"raw_points features are out of distribution")
        rt_str, rt = sync._parse_round_totals(raw_state)
        if rt is None or rt_str == self._prev_rt_str:
            return
        if self._prev_rt_str is not None:
            p0, p1 = _to_int(rt[0].get("p", 0)), _to_int(rt[1].get("p", 0))
            c0, c1 = _to_int(rt[0].get("c", 0)), _to_int(rt[1].get("c", 0))
            # LIVE-VERIFIED over 10 consecutive hands: p0 + p1 == 162 exactly,
            # with combination points reported separately in `c`. That is the
            # training env's raw-points semantics (tricks + 10 pasledu), so
            # per-player `points` -> raw_points_by_team is parity-correct.
            verdict = ("162 OK (tricks+pasledu, combos excluded)"
                       if p0 + p1 == 162 else
                       f"*** p0+p1 == {p0 + p1}, expected 162 -- raw-points "
                       f"semantics changed; features may be out of distribution")
            self._emit("PROBE",
                       f"hand scored: p({p0},{p1}) c({c0},{c1}) -> {verdict}")
        self._prev_rt_str = rt_str

    def _probe_top_card_rewrite(self, sync, raw_state):
        """After a swap the server rewrites topCard from the flipped card to
        the 7 the declarer received. belot_sync freezes the pre-swap value;
        this reports each rewrite so a change in that behaviour is visible."""
        top = raw_state.get("topCard", "")
        hid = getattr(sync, "hand_id", None)
        if not top or (top == self._last_top and hid == self._last_top_hand):
            return
        # Only a rewrite WITHIN one hand is swap bookkeeping; a change across
        # a hand boundary is simply the next deal's flipped card.
        prev = self._last_top if hid == self._last_top_hand else ""
        self._last_top, self._last_top_hand = top, hid
        if prev and sync.state.face_up_card is not None:
            frozen = ID_TO_ASCII.get(sync.state.face_up_card)
            if top != frozen:
                self._emit("PROBE",
                           f"topCard rewritten '{prev}' -> '{top}' while the "
                           f"face-up card stays frozen at '{frozen}' "
                           f"(swap bookkeeping); swapSeven="
                           f"{raw_state.get('swapSeven')}")

    def _probe_swap_window(self, sync, raw_state):
        """CONFIRMED: phase 8 opens after EVERY round-1 accept regardless of
        whether anyone can swap, then waits on a timeout. So it carries no
        per-player information and the eligibility test is purely local.

        Two things are still worth measuring. (a) phase 8 must imply a
        round-1 accept, i.e. trump == the face-up suit -- a violation would
        break the swap bookkeeping. (b) how long the window actually stalls
        when nobody swaps, which is the real cost of declining a swap
        because the face-up card is going to our partner.
        """
        phase = raw_state.get("currentPhase")
        env = sync.state
        if phase != 8:
            if self._in_swap_window and phase is not None and phase > 8:
                self._in_swap_window = False
                dt = time.time() - self._swap_window_t0
                self._emit("INFO",
                           f"phase 8 window #{self._swap_windows} closed after "
                           f"{dt:.1f}s "
                           + ("(a swap occurred)" if self._swap_window_swapped
                              else ("(no swap; we did not hold the 7 of trump)"
                                    if not self._swap_window_had_seven else
                                    "(no swap; WE held the 7 -- either the "
                                    "partner rule declined it or our "
                                    "SWAP_SEVEN payload was not accepted)")))
            return
        if raw_state.get("swapSeven", -1) not in (-1, None):
            self._swap_window_swapped = True
        if env.trump is not None and sync.my_pos is not None:
            if env.trump * 8 in env.hands[sync.my_pos]:
                self._swap_window_had_seven = True
        if self._in_swap_window:
            return
        self._in_swap_window = True
        self._swap_windows += 1
        self._swap_window_t0 = time.time()
        self._swap_window_had_seven = False
        self._swap_window_swapped = False

        if env.face_up_card is not None and env.trump is not None:
            if env.trump != env.face_up_card // 8:
                self._emit("VIOLATION",
                           f"phase 8 opened with trump {env.trump} != face-up "
                           f"suit {env.face_up_card // 8}: phase 8 should only "
                           f"follow a round-1 accept, and the swap "
                           f"bookkeeping assumes it")
        seven = env.trump * 8 if env.trump is not None else None
        hold = (seven is not None and sync.my_pos is not None
                and seven in env.hands[sync.my_pos])
        self._emit("PROBE",
                   f"phase 8 window #{self._swap_windows}: face-up="
                   f"{ID_TO_ASCII.get(env.face_up_card)} declarer="
                   f"{env.declarer} | "
                   + ("the face-up card IS the 7 of trump -- no swap is "
                      "possible for anyone this hand"
                      if seven is not None and seven == env.face_up_card else
                      f"we {'HOLD' if hold else 'do NOT hold'} the 7 of "
                      f"trump ({ID_TO_ASCII.get(seven)})"))

    def note_special(self, msg_type, data):
        """Hand-altering server events. None of them need special handling in
        the sync layer -- what matters is that the hand ends or is cancelled
        cleanly -- but they change what the following frames MEAN, so they are
        recorded and correlated with the phase path the hand actually took."""
        who = data.get("who") if isinstance(data, dict) else None
        effect = {
            "LESS_THAN_14":  "deal CANCELLED, scoreboard unchanged",
            "FOUR_OF_SEVEN": "deal CANCELLED (four 7s), scoreboard unchanged",
            "FOUR_OF_EIGHT": "combinations DISABLED except bella; hand continues",
            "WIN_ALL_HANDS": "fast-forward: claimant takes every remaining "
                             "trick, points DO score",
            "SURRENDER_BT":  "fast-forward: claimant concedes and takes a "
                             "bolt, points DO score",
            "BIZON":         "forced trump; bidding and the swap window are "
                             "skipped",
        }.get(msg_type, "unknown effect")
        self._pending_special = (msg_type, who)
        self._emit("INFO", f"{msg_type} by seat {who} -- {effect}")

    def note_combination(self, seat, value, source):
        """Record a declaration and check the wire syntax of the non-card
        claim types, whose trailing char is believed to be a filler 'a'."""
        for type_id, ch in combo.parse_field(value or ""):
            if type_id in combo.CLAIM_COMBOS:
                key = (type_id, ch)
                if key in self._claim_syntax:
                    continue
                self._claim_syntax.add(key)
                name = combo.TYPE_NAMES.get(type_id, str(type_id))
                verdict = ("filler 'a' as expected"
                           if ch == combo.CLAIM_FILLER else
                           f"trailing char is '{ch}', NOT 'a' -- our outgoing "
                           f"token {combo.claim_token(type_id)} may be wrong")
                self._emit("PROBE",
                           f"claim declaration {type_id}{ch} ({name}) via "
                           f"{source}, seat {seat}: {verdict}")

    def _probe_combo_flow(self, raw_state):
        """Report what the server offers OUR seat, once per hand per value.

        Point combinations plus LESS_THAN_14 and BELOT_COMBO are declared
        automatically; WIN_ALL_HANDS and SURRENDER_BT are declined by policy.
        The declines are counted, because WIN_ALL_HANDS is a claim on every
        remaining trick and how often it is offered is worth knowing.
        """
        for i, p in enumerate(raw_state.get("players", [])):
            for field in ("combinationsCanShow", "combinations"):
                self.note_combination(i, p.get(field), f"state.{field}")
            v = p.get("combinationsCanShow") or ""
            if not v or i != self._me:
                continue
            key = (self._hand_id, v)
            if key in self._combo_flagged:
                continue
            self._combo_flagged.add(key)
            # Mirror the agent's own policy so the preview cannot
            # disagree with what is actually sent.
            declared = combo.declarable(v, ASCII_TO_ID)
            declined = [f"{t}{c}" for t, c in combo.parse_field(v)
                        if t in combo.NEVER_DECLARE]
            note = f"offered '{v}'"
            if declared:
                note += f" -> declaring {declared}"
            for tok in declined:
                self._declined_claims[tok] = self._declined_claims.get(tok, 0) + 1
            if declined:
                names = [combo.TYPE_NAMES.get(int(t[:-1]), t) for t in declined]
                note += (f" -> DECLINING {declined} ({', '.join(names)}) by "
                         f"policy; {sum(self._declined_claims.values())} such "
                         f"offers declined so far")
            self._emit("PROBE", note)





# ----------------------------------------------------------------------
def replay(path, my_player_id=None, verbose=True):
    """Deterministically re-run recorded frames through a fresh synchronizer.
    Iterate on the sync layer offline instead of burning live games."""
    from .platform.sync import StateSynchronizer

    sync = StateSynchronizer()
    auditor = Auditor(verbose=verbose)
    frames = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = my_player_id or rec.get("pid")
            sync.sync(rec["state"], pid)
            auditor.check(sync, rec["state"])
            frames += 1
    print(f"[replay] {frames} frames, {auditor.violations} violations")
    return auditor


if __name__ == "__main__":       # python -m belotmd.audit [frames.jsonl]
    import sys
    replay(sys.argv[1] if len(sys.argv) > 1 else "frames.jsonl")