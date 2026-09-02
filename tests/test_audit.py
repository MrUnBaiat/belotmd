"""
test_audit.py — the verification layer's own bookkeeping.
"""

from belotmd.audit import Auditor


class _FakeSync:
    """Just enough of a StateSynchronizer for _probe_score_columns."""
    def __init__(self):
        from belotmd.platform.sync import StateSynchronizer
        self._real = StateSynchronizer()
        self.state = self._real.state

    def _decode_score_table(self, tbl):
        return self._real._decode_score_table(tbl)

    def _parse_round_totals(self, raw):
        return self._real._parse_round_totals(raw)


def _frame(score_table, b=(10, 6)):
    return {
        "scoreTable": score_table,
        "roundTotals": ('[{"p":100,"c":0,"b":%d},{"p":62,"c":0,"b":%d}]'
                        % (b[0], b[1])),
    }


def test_a_cleared_score_table_resets_the_row_counter():
    """A new match clears scoreTable, and belot.md sends it as the EMPTY
    STRING rather than "[]". If that leaves the previous match's row count
    stale, the growth test never matches again and the first hand of every
    subsequent match goes silently un-cross-checked."""
    a = Auditor(verbose=False)
    sync = _FakeSync()

    # Match 1: two hands scored.
    a._probe_score_columns(sync, _frame("[[10,6]]"))
    a._probe_score_columns(sync, _frame("[[10,6],[20,12]]"))
    assert a._prev_score_rows == 2

    # Match ends: the table comes back as "".
    a._probe_score_columns(sync, _frame(""))
    assert a._prev_score_rows == 0, "an empty table must read as zero rows"

    # Match 2, first hand: the growth test must fire again.
    a._probe_score_columns(sync, _frame("[[10,6]]"))
    assert a._prev_score_rows == 1


def test_an_unparseable_score_table_also_resets_rather_than_sticking():
    a = Auditor(verbose=False)
    sync = _FakeSync()
    a._probe_score_columns(sync, _frame("[[10,6],[20,12]]"))
    a._probe_score_columns(sync, _frame("not json at all"))
    assert a._prev_score_rows == 0


def test_the_third_bolt_costs_ten_and_the_probe_knows_it():
    """A bolt row scores 0 -- except the third, which also costs 10. The
    penalty is invisible in scoreTable (the cell is the string "BT-3"), so
    _decode_score_table applies it. If the probe still expects 0, every third
    bolt reports a MISMATCH against our own correct model. Observed live:
    row [108, "BT-3"] with b "BT-3", delta (18, -10)."""
    a = Auditor(verbose=True)
    sync = _FakeSync()
    emitted = []
    a._emit = lambda kind, msg: emitted.append((kind, msg))

    table = '[[90,62]]'
    a._probe_score_columns(sync, _frame(table))
    a._probe_score_columns(
        sync,
        {"scoreTable": '[[90,62],[108,"BT-3"]]',
         "roundTotals": '[{"p":109,"c":20,"b":18},{"p":53,"c":0,"b":"BT-3"}]'},
    )
    probes = [m for k, m in emitted if k == "PROBE" and "round scored" in m]
    assert probes, "the growth probe did not fire"
    assert "CONSISTENT" in probes[-1], probes[-1]
    assert "MISMATCH" not in probes[-1]


def test_a_first_or_second_bolt_still_expects_zero():
    a = Auditor(verbose=True)
    sync = _FakeSync()
    emitted = []
    a._emit = lambda kind, msg: emitted.append((kind, msg))

    a._probe_score_columns(sync, _frame('[[10,6]]'))
    a._probe_score_columns(
        sync,
        {"scoreTable": '[[10,6],[26,"BT-1"]]',
         "roundTotals": '[{"p":118,"c":0,"b":16},{"p":44,"c":0,"b":"BT-1"}]'},
    )
    probes = [m for k, m in emitted if k == "PROBE" and "round scored" in m]
    assert "CONSISTENT" in probes[-1], probes[-1]
