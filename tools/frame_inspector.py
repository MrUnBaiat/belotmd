"""
frame_inspector.py — offline forensics on a recorded frames.jsonl.

Pure payload analysis: does NOT import belot_sync / env / torch, so it runs
anywhere and can never be confused by a synchronizer bug. That independence is
the whole point — it is the second opinion.

    python frame_inspector.py                 # defaults to frames.jsonl
    python frame_inspector.py frames.jsonl
    python frame_inspector.py frames.jsonl -q # findings only

Exit code is 1 if any FINDING was raised, so it can gate a run.

Changes from the previous version, every one driven by a real capture:

  §1   reconciles each hand's scoreTable delta against roundTotals.b, and
       knows the THIRD BOLT COSTS 10 POINTS. This is the check that catches
       the "deltas (-7,13) vs b(3,13)" class of error; the old version could
       not see it because it only listed the BT cells.
  §2b  the seven-swap probe now requires swapSeven >= 0 or phase >= 8. The old
       one fired on every deal (topCard stays stale until phase 5) and in a
       capture that contained a REAL swap it was buried among 8 false hits.
  §3   adds card conservation — the same card appearing twice in one deal.
  §5   no longer claims "the model has no combo action"; it does, and it
       withholds some offers deliberately.
  §6   inventories phases against a KNOWN set and shouts about anything else.
       An unknown phase is how the SURRENDER_BT path (10 -> 12 -> 13) hid.
  NEW  session splitting: FrameRecorder appends, so one file can hold several
       matches and diffing across the seam produces nonsense.
"""
import json
import sys
from collections import Counter, OrderedDict

CHARS = ("yzabcdef" "ABghijkl" "CDmnopqr" "EFstuvwx")
CARD_ID = {c: i for i, c in enumerate(CHARS)}
SUITS = ["diamonds", "hearts", "clubs", "spades"]
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]

KNOWN_PHASES = {
    0: "LOBBY", 1: "ABORT", 2: "CUT_DECK", 5: "DEAL_5", 6: "TRUMP_CHOOSE_1",
    7: "TRUMP_CHOOSE_2", 8: "SWAP_WINDOW", 9: "DEAL_3", 10: "PLAYING",
    11: "TRICK_RESOLVE", 12: "CLAIM_RESOLVE", 13: "HAND_END", 14: "MATCH_END",
}
END_PHASES = (12, 13, 14)
FINDINGS = []


def finding(section, msg):
    FINDINGS.append((section, msg))
    print(f"  *** FINDING [{section}] {msg}")


def name(ch):
    i = CARD_ID.get(ch)
    return f"{RANKS[i % 8]}_{SUITS[i // 8]}" if i is not None else f"?{ch}"


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"  [skip line {n}: {e}]")
    return out


def jparse(v, default):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return v if v is not None else default


def bolt_marker(cell):
    """'BT-3' / 'bt_3' / 'BT3' -> 3.  Anything else -> None."""
    if not isinstance(cell, str):
        return None
    s = cell.strip().upper().replace("_", "-")
    if not s.startswith("BT"):
        return None
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else 0


def h(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def split_sessions(recs):
    """FrameRecorder opens frames.jsonl in APPEND mode, so one file can hold
    several matches. A boundary is a scoreTable that SHRINKS (a fresh match
    starts with an empty table and only ever grows) or a long wall-clock gap
    (the process was restarted).

    NOT gameStartTime. It looked like the obvious key and it is not stable
    within a single match -- observed live:

        lobby, gathering players   gameStartTime 1785924621
        lobby, more players        gameStartTime 1785924641
        phase 2, deal begins       gameStartTime 1785924653
        phase 14, match over       gameStartTime 1785925141

    Using it split one ordinary match into three "sessions": a 10-frame lobby
    fragment, the match, and a 2-frame MATCH_END fragment. The fragment then
    held a full 7-row scoreTable but only one roundTotals, which the row/b
    pairing below mis-aligned into a bogus MISMATCH. Two bugs feeding each
    other."""
    sessions, cur = [], []
    prev_t = prev_rows = None
    for r in recs:
        st = r.get("state", {})
        t = r.get("t")
        rows = len(jparse(st.get("scoreTable", "[]"), []) or [])
        boundary = (
            (prev_t is not None and t is not None and t - prev_t > 600)
            or (prev_rows is not None and rows < prev_rows)
        )
        if boundary and cur:
            sessions.append(cur)
            cur = []
        cur.append(r)
        prev_t, prev_rows = t, rows
    if cur:
        sessions.append(cur)
    return sessions


# ---------------------------------------------------------------- sections
def sec_scoring(recs, base):
    h("1. scoreTable — cell types, bolts, and delta vs roundTotals.b")
    seen, unknown = OrderedDict(), []
    for i, r in enumerate(recs):
        raw = r["state"].get("scoreTable")
        if raw in (None, ""):
            continue
        # key on the canonical form, but KEEP the raw value: `raw` is already a
        # JSON string, so json.dumps(raw) double-encodes it and jparse then
        # hands back a str instead of the table.
        seen.setdefault(json.dumps(raw), raw)
        for row in jparse(raw, []) or []:
            for cell in row if isinstance(row, list) else []:
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    continue
                if bolt_marker(cell) is None and cell not in ("", None):
                    unknown.append((base + i, cell))
    print(f"  distinct scoreTable values: {len(seen)}")
    if seen:
        print(f"  final: {list(seen.values())[-1]}")
    if unknown:
        finding("1", f"{len(unknown)} non-numeric cell(s) that are NOT bolt "
                     f"markers -- new format: {unknown[:3]}")

    table = jparse(list(seen.values())[-1], []) if seen else []
    if not isinstance(table, list) or not table:
        print("  (no table to reconcile)")
        return

    bs, prev_key = [], None
    for r in recs:
        raw = r["state"].get("roundTotals")
        if raw in (None, ""):
            continue
        key = json.dumps(raw)
        if key == prev_key:
            continue
        prev_key = key
        rt = jparse(raw, [])
        if isinstance(rt, list) and len(rt) == 2:
            bs.append((rt[0].get("b"), rt[1].get("b")))

    # roundTotals.b is published once per hand, alongside the row. If the
    # capture starts mid-match we have FEWER b's than rows, and the b's we do
    # have belong to the LAST rows -- so align from the end. Pairing from the
    # front produced a false MISMATCH on a mid-match fragment.
    offset = len(table) - len(bs)
    if offset < 0:
        print(f"  (more roundTotals ({len(bs)}) than table rows "
              f"({len(table)}) -- alignment unreliable, skipping the "
              f"delta cross-check)")
        bs, offset = [], len(table)

    print(f"\n  {'hand':>4}  {'row':<22} {'cumulative':>14} {'delta':>12} "
          f"{'roundTotals.b':>16}")
    scores, bolts, bad = [0, 0], [0, 0], 0
    for i, row in enumerate(table):
        if not isinstance(row, list) or len(row) < 2:
            continue
        prev = list(scores)
        for t in (0, 1):
            n = bolt_marker(row[t])
            if n is not None:
                bolts[t] = n % 3
                if n and n % 3 == 0:
                    # Confirmed live: the third bolt costs 10 and the penalty
                    # is invisible because the cell is the string 'BT-3'.
                    scores[t] -= 10
            else:
                try:
                    scores[t] = int(row[t])
                except (TypeError, ValueError):
                    pass
        delta = (scores[0] - prev[0], scores[1] - prev[1])
        j = i - offset
        b = bs[j] if 0 <= j < len(bs) else (None, None)
        flag = ""
        for t in (0, 1):
            if isinstance(b[t], (int, float)) and b[t] != delta[t]:
                flag, bad = "  <-- MISMATCH", bad + 1
        print(f"  {i+1:>4}  {str(row):<22} {str(scores):>14} {str(delta):>12} "
              f"{str(b):>16}{flag}")
    checked = min(len(bs), len(table))
    if bad:
        finding("1", f"{bad} hand(s) where the scoreTable delta disagrees with "
                     f"roundTotals.b -- the scoring model is wrong somewhere")
    elif checked:
        print(f"\n  {checked} of {len(table)} hand(s) cross-checked against "
              f"roundTotals.b, all reconcile (incl. the -10 third-bolt "
              f"penalty)")
    else:
        print(f"\n  no roundTotals in this session -- table walked but not "
              f"cross-checked")
    print(f"  final cumulative {scores}, bolt counters {bolts}")


def sec_stale(recs, base):
    h("2. trump / declarer staleness through the deal and bidding")
    stale = fresh = 0
    first = []
    for i, r in enumerate(recs):
        st = r["state"]
        if st.get("currentPhase") not in (2, 3, 4, 5, 6, 7):
            continue
        t, d = st.get("trump", -1), st.get("declarer", -1)
        if (isinstance(t, int) and 1 <= t <= 4) or (isinstance(d, int) and d >= 0):
            stale += 1
            if len(first) < 2:
                first.append(f"f{base+i} phase {st.get('currentPhase')}: "
                             f"trump={t} declarer={d}")
        else:
            fresh += 1
    for s in first:
        print(f"  {s}")
    print(f"  {stale} deal/bidding frames carry a trump or declarer, "
          f"{fresh} do not")
    if stale:
        print("  -> stale as documented; both must stay suppressed until "
              "phase 8+, or obs features 3 and 4 are wrong for every bid.")


def sec_swap(recs, base):
    h("2b. seven-swap (topCard rewrite) -- gated, not every deal")
    prev_top, hits, deals = None, 0, 0
    for i, r in enumerate(recs):
        st = r["state"]
        top, ph = st.get("topCard", ""), st.get("currentPhase", -1)
        sw = st.get("swapSeven", -1)
        if top and prev_top and top != prev_top:
            if (isinstance(sw, int) and sw >= 0) or ph in (8, 9):
                hits += 1
                print(f"  f{base+i} phase {ph}: topCard '{prev_top}' "
                      f"({name(prev_top)}) -> '{top}' ({name(top)}) "
                      f"swapSeven={sw}   <-- REAL swap")
            else:
                deals += 1
        if top:
            prev_top = top
    print(f"  {hits} genuine rewrite(s); {deals} ordinary deal flips ignored")
    print("  (topCard does NOT clear at phase 2, so it changes on every deal "
          "at phase 5.")
    print("   Only swapSeven >= 0, or a change inside the swap window "
          "(phase 8/9), is a swap.)")
    if hits:
        print("  -> the flip is replaced by the 7 the declarer received; the "
              "PRE-swap value must stay frozen as face_up_card.")

    h("2c. swapSeven field")
    vals = Counter(r["state"].get("swapSeven", -1) for r in recs)
    print(f"  observed values: {dict(vals)}")
    bad = [v for v in vals if not isinstance(v, int) or v < -1 or v > 3]
    if bad:
        finding("2c", f"values outside seat range: {bad} -- not a seat index")
    else:
        print("  all -1 or a valid seat index (0-3)")


def sec_tricks(recs, base):
    h("3. lastCards vs the table -- seat indexing and card conservation")
    ok = bad = 0
    table, prev_lc, graveyard, dupes = {}, None, [], []
    for i, r in enumerate(recs):
        st = r["state"]
        if st.get("currentPhase") in (2, 5):
            graveyard = []
        for s, p in enumerate(st.get("players", [])):
            ch = p.get("cardPlayed")
            if ch and ch in CARD_ID:
                table[s] = ch
        lc = st.get("lastCards") or ""
        lc = "".join(lc) if isinstance(lc, list) else lc
        if len(lc) == 4 and lc != prev_lc:
            prev_lc = lc
            if len(table) == 4 and set(table.values()) == set(lc):
                by_seat = "".join(table[s] for s in sorted(table))
                if by_seat == lc:
                    ok += 1
                else:
                    bad += 1
                    finding("3", f"f{base+i} lastCards={lc} but seat-order="
                                 f"{by_seat} -- NOT seat-indexed")
            for c in lc:
                if c in graveyard:
                    dupes.append((base + i, c))
                graveyard.append(c)
            table = {}
    print(f"  conclusive trick samples: {ok} consistent, {bad} mismatched")
    if ok and not bad:
        print("  -> SEAT-INDEXED confirmed.")
    if dupes:
        finding("3", f"{len(dupes)} card(s) appeared twice inside one deal, "
                     f"e.g. {dupes[:3]} -- a trick was double-counted")
    else:
        print("  no card appeared twice within a deal")


def sec_points(recs, base):
    h("4. roundTotals.p -- do trick points still total 162?")
    prev, shown = None, 0
    for i, r in enumerate(recs):
        raw = r["state"].get("roundTotals")
        if raw in (None, ""):
            continue
        key = json.dumps(raw)
        if key == prev:
            continue
        prev = key
        rt = jparse(raw, [])
        if not (isinstance(rt, list) and len(rt) == 2):
            continue
        p0, p1 = rt[0].get("p", 0), rt[1].get("p", 0)
        c0, c1 = rt[0].get("c", 0), rt[1].get("c", 0)
        shown += 1
        if p0 + p1 != 162:
            finding("4", f"f{base+i} p({p0},{p1}) sums to {p0+p1}, not 162 -- "
                         f"the points semantics changed")
        else:
            print(f"  f{base+i:>4} p({p0},{p1}) c({c0},{c1}) -> 162 OK "
                  f"(tricks+pasledu; combos live in c)")
    if not shown:
        print("  (no roundTotals in this session)")


def sec_cancelled(recs, base):
    h("4b. cancelled hands (LESS_THAN_14 voids the deal)")
    prev_rows, dup = None, 0
    for i, r in enumerate(recs):
        tbl = jparse(r["state"].get("scoreTable", "[]"), [])
        if not isinstance(tbl, list) or not tbl:
            continue
        n = len(tbl)
        if (prev_rows is not None and n == prev_rows + 1 and n >= 2
                and tbl[-2] == tbl[-1]):
            dup += 1
            print(f"  f{base+i} round {r['state'].get('round')}: duplicate row "
                  f"{tbl[-1]} -> hand voided, match score unchanged")
        prev_rows = n
    if dup:
        print(f"  {dup} cancelled hand(s). roundTotals stays on the PREVIOUS "
              f"hand, so a b() comparison there is meaningless. A reset must "
              f"be forced on the 10 -> 1/2 abort even if `round` is reused.")
    else:
        print("  none in this recording")


def sec_combos(recs, base):
    h("5. combinations offered to us vs declared")
    my_pid = recs[0].get("pid")
    # Count OFFER WINDOWS (contiguous runs), not frames. combinationsCanShow
    # persists for a run of frames and the raw frame count says nothing about
    # how many times we were actually asked.
    windows, widths, cur, run = [], [], None, 0
    theirs, ours_declared = Counter(), set()
    for r in recs:
        for p in r["state"].get("players", []):
            is_me = str(p.get("id")) == str(my_pid)
            v = p.get("combinationsCanShow") or ""
            if is_me:
                if v == cur:
                    run += 1 if v else 0
                else:
                    if cur:
                        windows.append(cur)
                        widths.append(run)
                    cur, run = v, 1 if v else 0
            elif v:
                theirs[v] += 1
            d = p.get("combinations") or ""
            if d and is_me:
                ours_declared.add(d)
    if cur:
        windows.append(cur)
        widths.append(run)
    print(f"  our seat ({my_pid}) OFFER WINDOWS: "
          f"{dict(Counter(windows)) or 'never'}")
    print(f"  our seat DECLARED (distinct)   : "
          f"{sorted(ours_declared) or 'never'}")
    print(f"  opponents were offered         : {dict(theirs) or 'never'}")
    if widths:
        print(f"  offer window width, frames: min={min(widths)} "
              f"max={max(widths)} ({widths})")
        if min(widths) <= 1:
            finding("5", f"an offer appeared in a SINGLE frame "
                         f"(widths {widths}). combinationsCanShow is not a "
                         f"latched field -- miss that one frame and the "
                         f"declaration is lost silently. Declarations are "
                         f"worth 20-200 points each.")
    offers = set(windows)
    claims = sorted({k.split('|')[-1] for k in offers if k[:1] in "6789"})
    if claims:
        print(f"  claim-type offers: {claims} "
              f"(6=LESS_THAN_14 7=BELOT 8=WIN_ALL 9=SURRENDER_BT)")
        print("  these are withheld by policy -- declining is intentional, "
              "not a stall.")
    never = sorted(offers - ours_declared)
    if never:
        print(f"  offered but never declared: {never}")
        print("  check AUTO_DECLARE switches (four 7s/8s, WIN_ALL, "
              "SURRENDER_BT are deliberately withheld).")


def sec_lifecycle(recs, base):
    h("6. phase inventory, transitions and session boundaries")
    trans, prev, phases = Counter(), None, Counter()
    for r in recs:
        ph = r["state"].get("currentPhase")
        phases[ph] += 1
        if prev is not None and ph != prev:
            trans[(prev, ph)] += 1
        prev = ph
    for ph, n in sorted(phases.items(), key=lambda kv: (kv[0] is None, kv[0])):
        known = ph in KNOWN_PHASES
        print(f"  phase {str(ph):>3} {KNOWN_PHASES.get(ph, 'UNKNOWN'):<14} x{n}"
              f"{'' if known else '   <-- UNKNOWN'}")
        if not known:
            finding("6", f"phase {ph} is not in the known set -- every "
                         f"phase-gated branch needs reviewing for it")
    print()
    for (a, b), n in sorted(trans.items()):
        note = ""
        if a == 5 and b == 9:
            note = "   (BIZON: bidding skipped entirely)"
        elif b == 12:
            note = "   (claim resolution -- must count as an END phase)"
        elif a == 14:
            note = "   (match over -> lobby: a 4005 here is a reshuffle, "
            note += "not a mid-match violation)"
        print(f"  {a:>3} -> {b:<3} x{n}{note}")


def run(recs, base, quiet):
    sec_scoring(recs, base)
    if not quiet:
        sec_stale(recs, base)
    sec_swap(recs, base)
    sec_tricks(recs, base)
    sec_points(recs, base)
    sec_cancelled(recs, base)
    sec_combos(recs, base)
    sec_lifecycle(recs, base)


def main(path="frames.jsonl", quiet=False):
    recs = load(path)
    if not recs:
        print(f"No frames in {path}")
        return 0
    print(f"Loaded {len(recs)} frames from {path}")
    sessions = split_sessions(recs)
    if len(sessions) > 1:
        print(f"Detected {len(sessions)} sessions in this file (the recorder "
              f"appends). Analysing each independently -- diffing across the "
              f"seam would produce nonsense.")
    off = 0
    for n, s in enumerate(sessions, 1):
        if len(sessions) > 1:
            print("\n" + "#" * 72)
            print(f"# SESSION {n}/{len(sessions)}  ({len(s)} frames, "
                  f"f{off}..f{off + len(s) - 1})")
            print("#" * 72)
        run(s, off, quiet)
        off += len(s)

    h("SUMMARY")
    if FINDINGS:
        print(f"  {len(FINDINGS)} finding(s):")
        for sec, msg in FINDINGS:
            print(f"    [{sec}] {msg}")
        return 1
    print("  no mismatches detected in this recording")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sys.exit(main(args[0] if args else "frames.jsonl", "-q" in sys.argv))