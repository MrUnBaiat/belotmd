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
