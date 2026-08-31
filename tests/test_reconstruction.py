"""
test_reconstruction.py — regression test for the LIVE-CONFIRMED findings.

Covers:
  * lastCards is SEAT-INDEXED -> chronological order rebuilt by rotating on
    the leader, with the leader chained from the previous trick's winner.
    Simulated with EVERY table frame dropped (no cardPlayed/cardOrder at all),
    which is the worst case the old code could not survive.
  * scoreTable cells that are strings (observed live; crashed observation.py).
  * declarer_has_played_trump derived locally instead of from the server's
    stale trumpWasPlayed flag.

Run:  python test_reconstruction.py
"""
import numpy as np
from belotmd.platform.sync import StateSynchronizer
from belotmd.platform.protocol import ASCII_TO_ID
from belotmd.audit import Auditor
from belotmd.game.belief import belief_matrix

MY_ID = "me"
ME = 3                      # my seat
DECLARER = 0
TRUMP_SERVER = 3            # 3 -> clubs -> env trump index 2
TRUMP = 2


def frame(phase, last_cards, num_cards, my_cards, score_table,
          trump_flag=True, declarer=DECLARER, table=None):
    """Build a payload. `table` is optional {seat: (char, order)}; when None
    the trick table is entirely absent — the dropped-frame worst case."""
    players = []
    for i in range(4):
        p = {"id": MY_ID if i == ME else f"opp{i}", "position": i, "points": 0}
        if i == ME:
            p["cards"] = my_cards
        else:
            p["numCards"] = num_cards[i]
        if table and i in table:
            ch, order = table[i]
            p["cardPlayed"] = ch
            p["cardOrder"] = order
        players.append(p)
    return {
        "currentPhase": phase, "dealer": 3, "activePlayer": 0,
        "topCard": "r", "trump": TRUMP_SERVER, "declarer": declarer,
        "trumpWasPlayed": trump_flag, "swapSeven": -1,
        "scoreTable": score_table,
        "roundTotals": '[{"p":80,"c":0,"b":8},{"p":82,"c":0,"b":8}]',
        "lastCards": last_cards, "players": players,
    }


def test_trick_reconstruction():
    """Regression test for the live-confirmed reconstruction findings:
    seat-indexed lastCards, string scoreTable cells, locally derived
    declarer_has_played_trump."""
    sync = StateSynchronizer()
    auditor = Auditor(verbose=True)
    env = sync.state
    SCORE = "[[14,4],[25,11]]"

    # ---- Enter play. No trick yet. Server flag is stale-True. ------------
    sync.sync(frame(10, "", [8, 8, 8, 0], "zdmABghi", SCORE), MY_ID)
    auditor.check(sync, frame(10, "", [8, 8, 8, 0], "zdmABghi", SCORE))
    assert sync.consume_new_hand() is True
    assert sync.degraded_hand is False           # no lastCards -> clean deal
    assert env.trump == TRUMP and env.declarer == DECLARER
    assert sync._next_leader == DECLARER          # leader chain anchored
    # Server says True, but the declarer has played nothing: local derivation
    # must override the stale flag.
    assert env.declarer_has_played_trump is False

    # ---- Trick 1: 0 leads A_d, 1 K_d, 2 7_d, 3 8_d. Seat 0 wins. --------
    # Seat-indexed lastCards: seat0='f', seat1='e', seat2='y', seat3='z'
    f1 = frame(10, "feyz", [7, 7, 7, 0], "dmABghi", SCORE)
    sync.sync(f1, MY_ID); auditor.check(sync, f1)
    assert env.tricks_played == 1
    assert sync.last_trick_source == "seatIndex"          # rebuilt with no table
    assert env.last_trick == [(0, 7), (1, 6), (2, 0), (3, 1)]
    assert sorted(env.graveyard) == [0, 1, 6, 7]
    assert sync._next_leader == 0                          # A_d wins
    assert env.declarer_has_played_trump is False

    # ---- Trick 2: 0 leads 10_d, 1 9_d, 2 RUFFS 7_c, 3 Q_d. Seat 2 wins. --
    f2 = frame(10, "baCd", [6, 6, 6, 0], "mABghi", SCORE)
    sync.sync(f2, MY_ID); auditor.check(sync, f2)
    assert env.last_trick == [(0, 3), (1, 2), (2, 16), (3, 5)]
    assert sync._next_leader == 2                          # trump ruff wins
    assert env.impossible_cards[2, 0:8].all()              # seat 2 void diamonds
    assert not env.impossible_cards[1].any()               # seat 1 followed
    assert env.declarer_has_played_trump is False          # declarer still hasn't

    # ---- Trick 3: 2 leads 8_c, 3 9_c, 0 Q_c (DECLARER plays trump), 1 J_c
    f3 = frame(10, "poDm", [5, 5, 5, 0], "ABghi", SCORE)
    sync.sync(f3, MY_ID); auditor.check(sync, f3)
    assert env.last_trick == [(2, 17), (3, 18), (0, 21), (1, 20)]  # rotated by leader 2
    assert sync._next_leader == 1                          # J of trump wins
    assert env.declarer_has_played_trump is True           # derived locally

    # ---- Cross-check: identical trick WITH table frames must agree -------
    sync2 = StateSynchronizer()
    sync2.sync(frame(10, "", [8, 8, 8, 0], "zdmABghi", SCORE), MY_ID)
    tbl = {0: ("f", 1), 1: ("e", 2), 2: ("y", 3), 3: ("z", 4)}
    sync2.sync(frame(10, "feyz", [7, 7, 7, 0], "dmABghi", SCORE, table=tbl), MY_ID)
    assert sync2.last_trick_source == "cardOrder"
    assert sync2.state.last_trick == [(0, 7), (1, 6), (2, 0), (3, 1)]  # same result

    # ---- String scoreTable cell: the live crash, now coerced -------------
    bad = frame(10, "poDm", [5, 5, 5, 0], "ABghi", '[[14,4],["25","11"]]')
    sync.sync(bad, MY_ID); auditor.check(sync, bad)
    assert sync.match_scores == [25, 11]
    assert all(isinstance(v, int) for v in sync.match_scores)

    weird = frame(10, "poDm", [5, 5, 5, 0], "ABghi", '[[14,4],[25,"-10B"]]')
    sync.sync(weird, MY_ID)
    assert sync.match_scores == [25, -10]

    # A string cell must never reach an agent as a string: that TypeError
    # is what crashed the bot live. The synchronizer coerces defensively.
    assert all(isinstance(v, int) for v in sync.match_scores)
    assert not np.isnan(belief_matrix(env, ME)).any()

    print("\n" + "=" * 60)
    print(f"RECONSTRUCTION TEST PASSED  ({auditor.violations} violations)")
    print("=" * 60)
    assert auditor.violations == 0


if __name__ == "__main__":
    test_trick_reconstruction()
