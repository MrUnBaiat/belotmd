"""
inspect_frames.py — offline forensics on a recorded frames.jsonl.

Pure payload analysis: does NOT import belot_sync/env/torch, so it runs
anywhere and can never be confused by a synchronizer bug. Answers the
questions the live session raised.

Usage:
    python inspect_frames.py                 # defaults to frames.jsonl
    python inspect_frames.py frames.jsonl
"""
import json
import sys
from collections import Counter, OrderedDict

CHARS = ("yzabcdef" "ABghijkl" "CDmnopqr" "EFstuvwx")
CARD_ID = {c: i for i, c in enumerate(CHARS)}
SUITS = ["diamonds", "hearts", "clubs", "spades"]
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]


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


def h(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(path="frames.jsonl"):
    recs = load(path)
    if not recs:
        print(f"No frames in {path}")
        return
    print(f"Loaded {len(recs)} frames from {path}")

    # ---------------- 1. THE CRASH: non-numeric scoreTable ---------------
    h("1. scoreTable cells  (source of the live TypeError)")
    seen, anomalies = OrderedDict(), []
    for i, r in enumerate(recs):
        raw = r["state"].get("scoreTable")
        if raw is None:
            continue
        seen.setdefault(json.dumps(raw), i)
        tbl = jparse(raw, [])
        for ri, row in enumerate(tbl if isinstance(tbl, list) else []):
            for ci, cell in enumerate(row if isinstance(row, list) else []):
                if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                    anomalies.append((i, ri, ci, cell, raw))
    print(f"distinct scoreTable values: {len(seen)}")
    for v, first in list(seen.items())[-4:]:
        print(f"  first@f{first}: {v}")
    bolts = [(i, ri, ci, c) for i, ri, ci, c, _ in anomalies
             if isinstance(c, str) and c.upper().startswith("BT")]
    other = [a for a in anomalies if a not in
             [(i, ri, ci, c, r) for i, ri, ci, c, r in anomalies
              if isinstance(c, str) and c.upper().startswith("BT")]]
    if bolts:
        rows = sorted({(ri, ci, c) for _, ri, ci, c in bolts})
        print(f"\n  BT markers (DECODED: team bolted, gained 0 that round, "
              f"N = bolt number; cumulative carries forward):")
        for ri, ci, c in rows:
            print(f"    row {ri} col {ci} = {c!r}")
        print("  -> handled by belot_sync._bolt_marker (naive int() gives -1!)")
    if other:
        print(f"\n  *** {len(other)} UNRECOGNISED non-numeric cell(s) — new format:")
        for i, ri, ci, cell, raw in other[:5]:
            st = recs[i]["state"]
            print(f"    f{i} row{ri} col{ci} = {cell!r}  table={raw}")
            print(f"        phase={st.get('currentPhase')} round={st.get('round')} "
                  f"roundTotals={st.get('roundTotals')}")
    if not anomalies:
        print("  all cells numeric in this recording")

    # ---------------- 2. stale trump / declarer during bidding -----------
    h("2. trump / declarer staleness through the deal and bidding")
    stale = fresh = 0
    for i, r in enumerate(recs):
        st = r["state"]
        ph = st.get("currentPhase")
        if ph not in (2, 3, 4, 5, 6, 7):
            continue
        t, d = st.get("trump", -1), st.get("declarer", -1)
        if (isinstance(t, int) and 1 <= t <= 4) or (isinstance(d, int) and d >= 0):
            stale += 1
            if stale <= 3:
                print(f"  f{i:>4} phase {ph}: trump={t} declarer={d} "
                      f"(training guarantees BOTH unset while bidding)")
        else:
            fresh += 1
    print(f"  {stale} deal/bidding frames carry a trump or declarer, "
          f"{fresh} do not")
    if stale:
        print("  -> CONFIRMED stale. belot_sync suppresses both until phase 8+,\n"
              "     otherwise obs features 3 and 4 are wrong for every bid.")

    # ---------------- 2b. topCard rewritten by a swap --------------------
    h("2b. topCard rewrite (seven-swap bookkeeping)")
    prev_top, prev_round, hits = None, None, 0
    for i, r in enumerate(recs):
        st = r["state"]
        top, rnd = st.get("topCard", ""), st.get("round")
        if rnd == prev_round and top and prev_top and top != prev_top:
            sw = st.get("swapSeven", -1)
            hits += 1
            print(f"  f{i:>4} round {rnd}: topCard '{prev_top}' -> '{top}' "
                  f"with swapSeven={sw} (phase {st.get('currentPhase')})")
        if top:
            prev_top = top
        prev_round = rnd
    if not hits:
        print("  no mid-hand rewrite in this recording")
    else:
        print("  -> the flipped card is replaced by the 7 the declarer got;\n"
              "     belot_sync freezes the pre-swap value.")

    # ---------------- 2c. swapSeven semantics ----------------------------
    h("2c. swapSeven field")
    vals = Counter()
    for r in recs:
        v = r["state"].get("swapSeven", -1)
        vals[v] += 1
    print(f"  observed values: {dict(vals)}")
    bad = [v for v in vals if not isinstance(v, int) or v < -1 or v > 3]
    if bad:
        print(f"  *** values outside seat range: {bad} -- NOT a seat index!")
    else:
        print("  all values are -1 or a valid seat index (0-3), "
              "consistent with 'seat that swapped'")

    h("3. lastCards vs table cards  (seat-indexing holds?)")
    # The server wipes the table in the same frame it publishes lastCards, so
    # compare against the LAST SEEN table rather than the current one.
    ok = bad = 0
    table = {}
    prev_lc = None
    for r in recs:
        st = r["state"]
        for i, p in enumerate(st.get("players", [])):
            ch = p.get("cardPlayed")
            if ch and ch in CARD_ID:
                table[i] = ch
        lc = st.get("lastCards") or ""
        lc = "".join(lc) if isinstance(lc, list) else lc
        if len(lc) == 4 and lc != prev_lc:
            prev_lc = lc
            if len(table) == 4 and set(table.values()) == set(lc):
                by_seat = "".join(table[s2] for s2 in sorted(table))
                if by_seat == lc:
                    ok += 1
                else:
                    bad += 1
                    print(f"  MISMATCH: lastCards={lc} seat-order={by_seat}")
            table = {}
    print(f"  conclusive samples: {ok} consistent, {bad} mismatched")
    if ok and not bad:
        print("  -> SEAT-INDEXED confirmed.")
    elif not ok:
        print("  -> inconclusive here; the live auditor validates every trick.")

    # ---------------- 4. points vs roundTotals (combo leakage) -----------
    h("4. per-player points vs roundTotals.p  (are combos folded in?)")
    prev_rt, last_pts = None, None
    for i, r in enumerate(recs):
        st = r["state"]
        rt_raw = st.get("roundTotals")
        pls = st.get("players", [])
        if st.get("currentPhase") == 10 and len(pls) == 4:
            last_pts = [int(p.get("points", 0) or 0) for p in pls]
        if rt_raw is None:
            continue
        key = json.dumps(rt_raw)
        if prev_rt is not None and key != prev_rt and last_pts:
            rt = jparse(rt_raw, [])
            if isinstance(rt, list) and len(rt) == 2:
                t0, t1 = last_pts[0] + last_pts[2], last_pts[1] + last_pts[3]
                p0, p1 = rt[0].get("p", 0), rt[1].get("p", 0)
                c0, c1 = rt[0].get("c", 0), rt[1].get("c", 0)
                # The invariant that settles it: trick points always total
                # 162 (incl. the 10-point pasledu); combos live in `c`.
                v = ("162 OK -> tricks+pasledu, combos excluded "
                     "(training-compatible)" if p0 + p1 == 162 else
                     f"*** p0+p1={p0 + p1} != 162 -- semantics changed!")
                lag = (p0 - t0) + (p1 - t1)
                print(f"  f{i:>4} p({p0},{p1}) c({c0},{c1}) "
                      f"b({rt[0].get('b')},{rt[1].get('b')}) -> {v}")
                if lag:
                    print(f"        (tracked ({t0},{t1}) trails by {lag}: "
                          f"final trick + pasledu land after the last "
                          f"PLAYING frame — expected)")
        prev_rt = key

    # ---------------- 4b. cancelled hands (LESS_THAN_14) -----------------
    h("4b. cancelled hands (LESS_THAN_14 voids the deal)")
    prev_rows, dup = None, 0
    for i, r in enumerate(recs):
        st = r["state"]
        tbl = jparse(st.get("scoreTable", "[]"), [])
        if not isinstance(tbl, list) or not tbl:
            continue
        n = len(tbl)
        if prev_rows is not None and n == prev_rows + 1 and n >= 2:
            a, b = tbl[-2], tbl[-1]
            if isinstance(a, list) and isinstance(b, list) and a == b:
                dup += 1
                print(f"  f{i:>4} round {st.get('round')}: duplicate row {b} "
                      f"-> hand voided, match score unchanged")
        prev_rows = n
    if dup:
        print(f"\n  {dup} cancelled hand(s). The server appends a duplicate")
        print("  cumulative row and leaves roundTotals on the PREVIOUS hand,")
        print("  so a b() comparison there is meaningless. belot_sync forces a")
        print("  reset on the 10 -> 1 abort even if `round` is reused.")
    else:
        print("  none in this recording")

    # ---------------- 5. combinations -------------------------------------
    h("5. combination fields  (does the bot ever need to respond?)")
    mine = Counter(); theirs = Counter()
    my_pid = recs[0].get("pid")
    for r in recs:
        for p in r["state"].get("players", []):
            v = p.get("combinationsCanShow") or ""
            if not v:
                continue
            (mine if str(p.get("id")) == str(my_pid) else theirs)[v] += 1
    print(f"  our seat  ({my_pid}): {dict(mine) or 'never'}")
    print(f"  opponents          : {dict(theirs) or 'never'}")
    if mine:
        print("  *** We were offered a declaration. The model has no combo "
              "action, so we forfeit those points; sniff the browser's "
              "WebSocket for the message type if you want them.")
    else:
        print("  We were never offered one in this recording -> no stall risk "
              "observed; opponents' declarations only affect their score.")

    # ---------------- 6. lifecycle / frame health -------------------------
    h("6. phase transitions & frame health")
    trans, prev = Counter(), None
    swaps = 0
    for r in recs:
        st = r["state"]
        ph = st.get("currentPhase")
        if prev is not None and ph != prev:
            trans[(prev, ph)] += 1
        prev = ph
        if (st.get("swapSeven", -1) or -1) >= 0:
            swaps += 1
    for (a, b), n in sorted(trans.items()):
        print(f"  {a:>3} -> {b:<3}  x{n}")
    print(f"  swapSeven active in {swaps} frame(s)")
    if any(a >= 10 and b >= 10 for (a, b) in trans if a != b):
        print("  note: play-phase-to-play-phase transitions present")
    if not any(b in (6, 7) for (_, b) in trans):
        print("  *** no transition INTO bidding was ever observed -> the "
              "hand-reset latch (_pending_reset) is doing real work here.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "frames.jsonl")