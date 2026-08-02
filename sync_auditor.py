"""
sync_auditor.py — verification layer for the live Belot bot.

Three capabilities:
  1. FrameRecorder  — append every raw STATE frame to a JSONL file, so any
     bug seen live can be replayed offline, deterministically, forever.
  2. Auditor.check  — hard invariants on the synchronized env + observation
     after every sync. Violations are logged, never raised: the bot keeps
     playing while you collect evidence.
  3. Auditor probes — automatic evidence collection for the still-unverified
     server semantics (lastCards ordering, trumpWasPlayed meaning, score
     column mapping, points-include-combos, combination flow).
  4. replay()       — run a recorded JSONL through a fresh StateSynchronizer
     and report every violation with its frame number.

Usage (live), in bot_agent.on_state_update right after sync():

    if AUDIT:
        self.recorder.record(raw_state, my_player_id)
        self.auditor.check(self.sync_engine, raw_state)

Usage (offline):

    python -c "import sync_auditor; sync_auditor.replay('frames.jsonl')"
"""

import json
import time
import numpy as np

from observation import build_observation
from belot_sync import (ASCII_TO_ID, ID_TO_ASCII, _last_cards_str,
                        _to_int, _bolt_marker)
from client import CHAR_TO_CARD

# Local-obs layout offsets (must mirror observation.py exactly):
# hand 0:32 | faceup 32:64 | trump 64:69 | declarer 69:74 | phase 74:77 |
# cur trick 77:185 | stats 185:191 | dealer 191:195 | last trick 195:339 |
# BELIEF 339:435 | trick# 435:443 | mask 443:481 | graveyard 481:513
BELIEF_SLICE = slice(339, 435)


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
        self._combo_flagged = set()

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
        hid = getattr(sync, "hand_id", None)
        if hid != self._hand_id:
            self._hand_id = hid
            self._expected_leader = None
            self._probed_last_cards = ""
            self._trick_log = []
            self._trump_bid_sampled = False
        env = sync.env
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

        for pidx in range(4):
            for card in ([] if scored else np.flatnonzero(env.known_cards[pidx])):
                if int(card) in grave:
                    self._emit("VIOLATION",
                               f"known_cards ghost: seat {pidx} card "
                               f"{int(card)} already in graveyard")

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

        # ---- observation-level checks at our decision points ---------
        if env.current_player == me and phase in (6, 7, 10) and not env.done:
            obs, gobs, mask = build_observation(env, me, sync.match_scores)
            if np.isnan(obs).any() or np.isnan(gobs).any():
                self._emit("VIOLATION", "NaN in observation")
            if not mask.any():
                self._emit("VIOLATION", "empty legal mask on our turn")

            bel = obs[BELIEF_SLICE].reshape(3, 32)
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
        if (env.done and 0 < env.tricks_played < 8
                and self._short_hand != self._hand_id):
            self._short_hand = self._hand_id
            self._emit('INFO',
                       f'hand ended after {env.tricks_played} tricks '
                       f'(WIN_ALL_HANDS claim) -- expected, no more '
                       f'actions are taken in this hand')
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
        if not lc or lc == self._probed_last_cards or len(sync.env.last_trick) != 4:
            return
        self._probed_last_cards = lc
        self._trick_log.append((sync.env.tricks_played, list(sync.env.last_trick)))

        seat_map = dict(sync.env.last_trick)
        by_seat = "".join(ID_TO_ASCII[seat_map[s]] for s in sorted(seat_map))
        if by_seat != lc:
            self._emit("VIOLATION",
                       f"lastCards seat-indexing broken: server '{lc}' vs "
                       f"reconstructed-by-seat '{by_seat}' "
                       f"(source={sync.last_trick_source})")

        # Cross-check: chronological order must be the seat order rotated by
        # the leader, and the leader must be the previous trick's winner.
        leader = sync.env.last_trick[0][0]
        expect = [((leader + k) % 4) for k in range(4)]
        if [s for s, _ in sync.env.last_trick] != expect:
            self._emit("VIOLATION",
                       f"trick seat sequence {[s for s, _ in sync.env.last_trick]} "
                       f"is not clockwise from leader {leader}")
        # Anchor check: the declarer leads trick 1. This is the only thing
        # the seat-index fallback depends on, and it is the one place the
        # forced-trump ("BIZON") hands could differ, since they skip bidding.
        if (sync.env.tricks_played == 1 and sync.env.declarer is not None
                and not getattr(sync, 'degraded_hand', False)):
            # (skipped on a mid-hand join: our trick counter starts at
            #  the join point, so 'trick 1' is not the real first trick)
            if leader != sync.env.declarer:
                self._emit("VIOLATION",
                           f"trick 1 led by seat {leader}, but declarer is "
                           f"{sync.env.declarer} -- leader-chain anchor wrong; "
                           f"the seatIndex fallback would misorder tricks")
        if self._expected_leader is not None and leader != self._expected_leader:
            self._emit("VIOLATION",
                       f"leader chain broken: expected {self._expected_leader} "
                       f"(prev trick winner), observed {leader}")
        self._expected_leader = sync._trick_winner(sync.env.last_trick)
        self._emit("INFO",
                   f"trick {sync.env.tricks_played} ok "
                   f"(leader {leader}, winner {self._expected_leader}, "
                   f"src {sync.last_trick_source})")

    def _probe_trump_flag(self, sync, raw_state):
        """The server flag never flipped False->True in a full live hand even
        though trump was played at trick 2 => it is suspected STALE across
        hand boundaries. Definitive test: sample it during BIDDING, before a
        single card of the new hand exists. True there proves staleness."""
        flag = bool(raw_state.get("trumpWasPlayed", False))
        if sync.env.phase == "BIDDING" and not self._trump_bid_sampled:
            self._trump_bid_sampled = True
            self._emit("PROBE",
                       f"trumpWasPlayed during BIDDING = {flag} -> "
                       + ("STALE across hands (server field unusable; local "
                          "derivation in use)" if flag else
                          "resets correctly (field trustworthy)"))
        if sync.env.phase != "BIDDING":
            self._trump_bid_sampled = False

        # Compare server flag against our local derivation every play frame.
        if sync.env.phase == "PLAYING" and flag != sync.env.declarer_has_played_trump:
            key = (sync.env.tricks_played, flag, sync.env.declarer_has_played_trump)
            if key not in self._trump_divergences:
                self._trump_divergences.add(key)
                self._emit("PROBE",
                           f"trick {sync.env.tricks_played}: server "
                           f"trumpWasPlayed={flag} vs locally-derived "
                           f"declarer_has_played_trump="
                           f"{sync.env.declarer_has_played_trump} "
                           f"(declarer={sync.env.declarer})")
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
            return
        if not isinstance(st, list):
            return
        rows = len(st)
        if self._prev_score_rows is not None and rows == self._prev_score_rows + 1:
            cur, _ = sync._decode_score_table(st)
            prv, _ = sync._decode_score_table(st[:-1])
            deltas = [cur[0] - prv[0], cur[1] - prv[1]]
            _, rt = sync._parse_round_totals(raw_state)
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
                           f"({cur[0]},{cur[1]}), bolts {sync.env.bolts_by_team}")
        self._prev_score_rows = rows

    def _probe_points_semantics(self, sync, raw_state):
        """Training caps raw hand points at 162 (trick captures + pasledu).
        If the site folds combination points into per-player `points`, our
        raw_points features go out of distribution. Two checks: (a) live cap,
        (b) at hand end, team-summed points vs roundTotals.p."""
        players = raw_state.get("players", [])
        if len(players) != 4:
            return
        if sync.env.phase == "PLAYING":
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

    def _probe_combo_flow(self, raw_state):
        for i, p in enumerate(raw_state.get("players", [])):
            v = p.get("combinationsCanShow") or ""
            if v and (i, v) not in self._combo_flagged:
                self._combo_flagged.add((i, v))
                self._emit("PROBE",
                           f"combinationsCanShow='{v}' for seat {i} — sniff "
                           f"the message the human client sends here; the bot "
                           f"has no combo action and may need an auto-reply.")


# ----------------------------------------------------------------------
def replay(path, my_player_id=None, verbose=True):
    """Deterministically re-run recorded frames through a fresh synchronizer.
    Iterate on belot_sync.py offline instead of burning live games."""
    from belot_sync import StateSynchronizer
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


if __name__ == "__main__":
    import sys
    replay(sys.argv[1] if len(sys.argv) > 1 else "frames.jsonl")