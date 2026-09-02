"""
test_frame_inspector.py — the independent forensics tool.

frame_inspector deliberately shares no code with the synchronizer, which is
what makes it a useful second opinion. That also means its own arithmetic has
nothing checking it, so the alignment rule gets a test of its own: it once
reported 12 mismatches in an 11-row table that was entirely correct.
"""

import importlib.util
import pathlib
import sys

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "frame_inspector.py"
_spec = importlib.util.spec_from_file_location("frame_inspector", _PATH)
fi = importlib.util.module_from_spec(_spec)
sys.modules["frame_inspector"] = fi
_spec.loader.exec_module(fi)


def _frames(score_tables, round_totals):
    """One frame per (scoreTable, roundTotals) step."""
    return [{"t": float(i), "pid": "me",
             "state": {"scoreTable": st, "roundTotals": rt, "players": []}}
            for i, (st, rt) in enumerate(zip(score_tables, round_totals))]


def test_a_cancelled_deal_does_not_shift_every_earlier_row(capsys):
    """A cancelled deal (LESS_THAN_14 / four 7s) appends a DUPLICATE cumulative
    row and publishes no new roundTotals, so a `b` is missing from the MIDDLE
    of the table. Aligning from the end shifted all eight rows before it.
    Captured live: an 11-row table, one cancellation at row 8, everything
    reconciling -- reported as 12 mismatches."""
    rows = [
        [[1, 17]],
        [[1, 17], [17, "BT-1"]],
        [[1, 17], [17, "BT-1"], [27, 23]],
        [[1, 17], [17, "BT-1"], [27, 23], [59, 59]],
        [[1, 17], [17, "BT-1"], [27, 23], [59, 59], [59, 59]],   # cancelled
        [[1, 17], [17, "BT-1"], [27, 23], [59, 59], [59, 59], [77, "BT-2"]],
    ]
    bs = [
        (1, 17), (16, "BT-1"), (10, 6), (32, 36),
        (32, 36),            # unchanged across the cancellation
        (18, "BT-2"),
    ]
    import json
    sts = [json.dumps(r) for r in rows]
    rts = ['[{"p":0,"c":0,"b":%s},{"p":0,"c":0,"b":%s}]'
           % (json.dumps(a), json.dumps(b)) for a, b in bs]

    fi.sec_scoring(_frames(sts, rts), 0)
    out = capsys.readouterr().out

    assert "MISMATCH" not in out, out
    assert "1 cancelled deal(s) at row(s) [5]" in out
    assert "all reconcile" in out


def test_the_third_bolt_penalty_is_applied_when_walking_the_table(capsys):
    import json
    sts = [json.dumps([[90, 62]]),
           json.dumps([[90, 62], [108, "BT-3"]])]
    rts = ['[{"p":0,"c":0,"b":90},{"p":0,"c":0,"b":62}]',
           '[{"p":0,"c":0,"b":18},{"p":0,"c":0,"b":"BT-3"}]']
    fi.sec_scoring(_frames(sts, rts), 0)
    out = capsys.readouterr().out
    assert "MISMATCH" not in out, out
    # third bolt: 62 - 10 == 52, and the counter rolls over to 0
    assert "[108, 52]" in out
    assert "bolt counters [0, 0]" in out
