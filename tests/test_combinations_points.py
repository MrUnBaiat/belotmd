"""Combination points as belot.md pays them, and the state field that carries
them to an agent.

The platform bolts the declaring team on trick points PLUS combination points
and pays 16 + all combinations/10 (docs/PLATFORM_NOTES.md §6-7). An agent that
scores the hand that way needs, from trick 3, each seat's settled declarations.
"""
from belotmd.game import combinations as combo
from belotmd.platform.protocol import ASCII_TO_ID
from belotmd.platform.sync import StateSynchronizer

MY_ID = "1000003"


def P(pid, **kw):
    d = {"id": pid, "connected": True}
    d.update(kw)
    return d


def test_points_by_type():
    pts = lambda v: combo.points(v, ASCII_TO_ID)       # noqa: E731
    assert pts("1c") == (20, 0)                         # three-run
    assert pts("2l") == (50, 0)                         # four-run
    assert pts("3d") == (100, 0)                        # five-run
    assert pts("5q") == (0, 20)                         # bella
    assert pts("5q|1c") == (20, 20)
    assert pts("4x") == (100, 0)                        # four aces
    assert pts("4c") == (200, 0)                        # four jacks
    assert pts("4a") == (150, 0)                        # four nines
    assert pts("4y") == (0, 0)                          # four sevens: cancels, 0
    assert pts("4z") == (0, 0)                          # four eights: silences, 0


def test_claims_and_junk_score_nothing():
    for v in ("", None, "6a", "7a", "8a", "9a", "junk", "1", "x"):
        assert combo.points(v, ASCII_TO_ID) == (0, 0)


def test_team_points_by_seat_parity():
    fields = ["", "2l|5k", "", "1t"]
    assert combo.team_points(fields, ASCII_TO_ID) == (0, 90)
    assert combo.team_points(["5q", "", "3d", ""], ASCII_TO_ID) == (120, 0)
    assert combo.team_points(["", "", "", ""], ASCII_TO_ID) == (0, 0)


def test_bela_declared_finds_the_seat():
    assert combo.bela_declared(["", "2l|5k", "", "1t"]) == 1
    assert combo.bela_declared(["1c", "", "", ""]) is None


def _play_frame(combinations, rnd=3):
    return {
        "currentPhase": 10, "dealer": 2, "activePlayer": 2,
        "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": True,
        "swapSeven": -1, "scoreTable": "[[14,4],[25,11]]",
        "roundTotals": '[{"p":116,"c":0,"b":11},{"p":46,"c":20,"b":7}]',
        "round": rnd, "lastCards": "", "targetScore": 101,
        "players": [
            P("1000000", points=0, numCards=6, combinations=combinations[0]),
            P("1000001", points=0, numCards=6, combinations=combinations[1]),
            P("1000002", points=0, numCards=6, combinations=combinations[2]),
            P(MY_ID, points=0, cards="yCDrne", combinations=combinations[3]),
        ],
    }


def test_state_carries_the_field_during_play_and_clears_at_the_deal():
    sync = StateSynchronizer()
    assert sync.state.combinations == ["", "", "", ""]

    sync.sync(_play_frame(["", "2l|5k", "", "1t"]), MY_ID)
    assert sync.state.phase == "PLAYING"
    assert sync.state.combinations == ["", "2l|5k", "", "1t"]
    assert combo.team_points(sync.state.combinations, ASCII_TO_ID) == (0, 90)

    # the server settles the contest: a cleared field is mirrored, not cached
    sync.sync(_play_frame(["", "2l|5k", "", ""]), MY_ID)
    assert sync.state.combinations == ["", "2l|5k", "", ""]

    # the next deal starts empty
    bidding = _play_frame(["", "2l|5k", "", ""], rnd=4)
    bidding.update(currentPhase=6, trump=-1, declarer=-1, dealer=3,
                   activePlayer=0, topCard="k")
    for p in bidding["players"]:
        p["numCards"] = 5
    bidding["players"][3]["cards"] = "ABCDE"
    sync.sync(bidding, MY_ID)
    assert sync.state.phase == "BIDDING"
    assert sync.state.combinations == ["", "", "", ""]


def test_auditor_holds_the_field_to_the_published_totals():
    """At each hand's summary the auditor compares the last play frame's field
    with the published per-team c, so a change in what the field means shows
    up in the session log rather than in the agent's scoring."""
    from belotmd.audit import Auditor

    sync, auditor = StateSynchronizer(), Auditor(verbose=False)
    probes = []
    auditor._emit = lambda kind, msg: probes.append((kind, msg))

    first = _play_frame(["", "2l|5k", "", "1t"])
    sync.sync(first, MY_ID)
    auditor.check(sync, first)                    # first summary: only remembered

    # the hand is scored: team 1's 90 is published exactly as the field said
    scored = _play_frame(["", "2l|5k", "", "1t"], rnd=4)
    scored["currentPhase"] = 13
    scored["roundTotals"] = '[{"p":100,"c":0,"b":10},{"p":62,"c":90,"b":15}]'
    sync.sync(scored, MY_ID)
    auditor.check(sync, scored)
    assert any("combinations field OK" in m for k, m in probes if k == "PROBE")

    # ...and a summary the field cannot explain is called out
    sync2, auditor2 = StateSynchronizer(), Auditor(verbose=False)
    probes2 = []
    auditor2._emit = lambda kind, msg: probes2.append((kind, msg))
    sync2.sync(first, MY_ID)
    auditor2.check(sync2, first)
    off = dict(scored)
    off["roundTotals"] = '[{"p":100,"c":50,"b":15},{"p":62,"c":90,"b":15}]'
    sync2.sync(off, MY_ID)
    auditor2.check(sync2, off)
    assert any("combinations field says c(0,90)" in m for k, m in probes2 if k == "PROBE")
