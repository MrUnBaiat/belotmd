"""Seats played by belot.md's own bot.

The platform replaces a player whose turn times out and often hands the seat
back a few hands later. Nothing is broadcast for either, so the per-frame `bot`
flag is the only trace: `state.bot_seats` must mirror it on every frame, and the
auditor must log each change for another seat -- once, not once per frame.
"""
from belotmd.audit import Auditor
from belotmd.platform.sync import StateSynchronizer

MY_ID = "1000003"          # seat 3: seat 1 is our partner, seats 0 and 2 opponents


def P(pid, **kw):
    d = {"id": pid, "connected": True}
    d.update(kw)
    return d


def _frame(bots=(), phase=10, rnd=3):
    players = [P("1000000", points=0, numCards=6),
               P("1000001", points=0, numCards=6),
               P("1000002", points=0, numCards=6),
               P(MY_ID, points=0, cards="yCDrne")]
    for s in bots:
        players[s]["bot"] = True
    return {
        "currentPhase": phase, "dealer": 2, "activePlayer": 2,
        "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": True,
        "swapSeven": -1, "scoreTable": "[[14,4],[25,11]]",
        "roundTotals": '[{"p":116,"c":0,"b":11},{"p":46,"c":20,"b":7}]',
        "round": rnd, "lastCards": "", "targetScore": 101, "players": players,
    }


def test_the_flag_is_mirrored_every_frame_and_cleared_on_hand_back():
    sync = StateSynchronizer()
    assert sync.state.bot_seats == [False, False, False, False]

    sync.sync(_frame(bots=(0,)), MY_ID)
    assert sync.state.bot_seats == [True, False, False, False]

    sync.sync(_frame(bots=(0, 1)), MY_ID)
    assert sync.state.bot_seats == [True, True, False, False]

    sync.sync(_frame(bots=()), MY_ID)             # both humans are back
    assert sync.state.bot_seats == [False, False, False, False]


def test_the_auditor_logs_each_change_once():
    sync, aud = StateSynchronizer(), Auditor(verbose=False)
    lines = []
    aud._emit = lambda kind, msg: lines.append((kind, msg))

    def feed(frame):
        sync.sync(frame, MY_ID)
        aud.check(sync, frame)

    def bot_lines():
        return [m for k, m in lines if k == "INFO"
                and ("platform bot" in m or "its human" in m)]

    feed(_frame())                                # first sighting: nothing to report
    assert bot_lines() == []

    feed(_frame(bots=(2,)))
    assert len(bot_lines()) == 1
    assert "seat 2 (opponent) is now played by the platform bot" in bot_lines()[0]

    feed(_frame(bots=(2,)))                       # unchanged: no repeat
    assert len(bot_lines()) == 1

    feed(_frame(bots=(1, 2)))
    assert "seat 1 (partner) is now played by the platform bot" in bot_lines()[1]

    feed(_frame(bots=(1,)))
    assert "seat 2 (opponent) is back with its human" in bot_lines()[2]
    assert len(bot_lines()) == 3


def test_a_different_player_in_the_seat_is_not_a_takeover():
    """Keyed by player id: a new table whose seat-0 player is already a bot, or
    a lobby rotation, must not read as a flip."""
    sync, aud = StateSynchronizer(), Auditor(verbose=False)
    lines = []
    aud._emit = lambda kind, msg: lines.append(msg)

    frame = _frame()
    sync.sync(frame, MY_ID)
    aud.check(sync, frame)

    other_table = _frame(bots=(0,), rnd=4)
    other_table["players"][0]["id"] = "2000000"   # someone new, already a bot
    sync.sync(other_table, MY_ID)
    aud.check(sync, other_table)
    assert not any("platform bot" in m for m in lines)
