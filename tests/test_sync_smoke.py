"""
smoke_test.py — offline validation of belot_sync v2 + sync_auditor.

Frame 1 is a REAL captured belot.md payload (mid-hand join scenario), with the
player ids replaced by synthetic ones -- the last digit is the seat index. Only
the ids were changed; every field the synchronizer reads is as it arrived.
Frames 2-4 are synthetic continuations exercising: mid-trick void inference,
trick completion via lastCards, points/score decoding, new-hand reset, and
bolt detection from the roundTotals `b` field.

Run:  python smoke_test.py     (expects "SMOKE TEST PASSED")
"""
import numpy as np
from belotmd.platform.sync import StateSynchronizer
from belotmd.platform.protocol import ASCII_TO_ID
from belotmd.audit import Auditor
from belotmd.game.belief import belief_matrix

MY_ID = "1000003"

def P(id, **kw):
    d = {"id": id, "connected": True}
    d.update(kw)
    return d

# ---- Frame 1: the real captured payload (trimmed to fields we consume) ----
FRAME1 = {
    "currentPhase": 10, "dealer": 2, "activePlayer": 2,
    "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": True,
    "swapSeven": -1,
    "scoreTable": "[[14,4],[25,11]]",
    "roundTotals": '[{"p":116,"c":0,"b":11},{"p":46,"c":20,"b":7}]',
    "round": 3, "lastCards": "xFtu", "targetScore": 101,
    "players": [
        P("1000000",  points=57, cardPlayed="s", cardOrder=1, numCards=5,
          card_played="9_spades"),
        P("1000001", points=0,  cardPlayed="v", cardOrder=2, numCards=5,
          card_played="Q_spades"),
        P("1000002", points=0,  numCards=6, card_played="None"),
        P(MY_ID,     points=0,  cards="yCDrne", card_played="None"),
    ],
}

# ---- Frame 2: seats 2 and 3 complete the trick on the table ----
# Seat 2 follows with K_spades ('w'); we (seat 3) hold no spades and no
# hearts (trump) -> discard 7_diamonds ('y'). Mid-trick voids must appear.
FRAME2 = {
    "currentPhase": 10, "dealer": 2, "activePlayer": 2,
    "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": True,
    "swapSeven": -1,
    "scoreTable": "[[14,4],[25,11]]",
    "roundTotals": '[{"p":116,"c":0,"b":11},{"p":46,"c":20,"b":7}]',
    "round": 3, "lastCards": "xFtu", "targetScore": 101,
    "players": [
        P("1000000",  points=57, cardPlayed="s", cardOrder=1, numCards=5),
        P("1000001", points=0,  cardPlayed="v", cardOrder=2, numCards=5),
        P("1000002", points=0,  cardPlayed="w", cardOrder=3, numCards=5),
        P(MY_ID,     points=0,  cardPlayed="y", cardOrder=4, cards="CDrne"),
    ],
}

# ---- Frame 3: server resolves the trick. K_spades wins -> seat 2 (+7 pts),
# table wiped, lastCards replaced ----
FRAME3 = {
    "currentPhase": 10, "dealer": 2, "activePlayer": 2,
    "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": True,
    "swapSeven": -1,
    "scoreTable": "[[14,4],[25,11]]",
    "roundTotals": '[{"p":116,"c":0,"b":11},{"p":46,"c":20,"b":7}]',
    "round": 3, "lastCards": "svwy", "targetScore": 101,
    "players": [
        P("1000000",  points=57, numCards=5),
        P("1000001", points=0,  numCards=5),
        P("1000002", points=7,  numCards=5),
        P(MY_ID,     points=0,  cards="CDrne"),
    ],
}

# ---- Frame 4: next hand's bidding. Round 3 ended with the declaring team
# (seat 2 -> team 0) bolting: b=0 for team 0, defenders take 16. ----
FRAME4 = {
    "currentPhase": 6, "dealer": 3, "activePlayer": 0,
    "topCard": "k", "trump": -1, "declarer": -1, "trumpWasPlayed": False,
    "swapSeven": -1,
    "scoreTable": "[[14,4],[25,11],[\"BT-1\",27]]",
    "roundTotals": '[{"p":70,"c":0,"b":0},{"p":92,"c":0,"b":16}]',
    "round": 4, "lastCards": "svwy", "targetScore": 101,
    "players": [
        P("1000000",  points=0, numCards=5),
        P("1000001", points=0, numCards=5),
        P("1000002", points=0, numCards=5),
        P(MY_ID,     points=0, cards="ABCDE"),
    ],
}


def test_sync_smoke():
    """Replay four frames (one real capture + three synthetic continuations)
    through the synchronizer and assert the auditor sees nothing wrong."""
    sync = StateSynchronizer()
    auditor = Auditor(verbose=True)
    env = sync.state

    # ================= Frame 1: real payload, mid-hand join ==============
    my_pos = sync.sync(FRAME1, MY_ID)
    auditor.check(sync, FRAME1)

    assert my_pos == 3
    assert sync.consume_new_hand() is True          # reset event fired once
    assert sync.consume_new_hand() is False
    assert sync.degraded_hand is True               # joined mid-hand
    assert env.phase == "PLAYING"
    assert env.trump == 1                           # trump 2 -> hearts
    assert env.face_up_card == 27                   # 't' = 10_spades
    assert env.declarer == 2 and env.dealer == 2 and env.current_player == 2
    assert env.declarer_has_played_trump is True
    assert env.current_trick == [(0, 26), (1, 29)]  # 9S then QS, by cardOrder
    assert sorted(env.hands[3]) == sorted(ASCII_TO_ID[c] for c in "yCDrne")
    assert [len(h) for h in env.hands] == [5, 5, 6, 6]   # numCards fallback
    assert sync.match_scores == [25, 11]            # LAST cumulative row
    assert env.raw_points_by_team == [57, 0]        # live per-player points
    assert env.known_cards[2, 27]                   # trump!=topsuit -> dealer
    assert env.tricks_played == 0                   # lastCards ADOPTED
    assert len(env.graveyard) == 0
    assert env.bolts_by_team == [0, 0]              # history never scored

    assert env.current_player != my_pos              # not our turn
    assert not np.isnan(belief_matrix(env, my_pos)).any()

    # Idempotency: identical frame is a strict no-op
    sync.sync(FRAME1, MY_ID)
    assert env.tricks_played == 0 and env.current_trick == [(0, 26), (1, 29)]
    assert sync.consume_new_hand() is False

    # ================= Frame 2: trick fills, mid-trick voids =============
    sync.sync(FRAME2, MY_ID)
    auditor.check(sync, FRAME2)

    assert env.current_trick == [(0, 26), (1, 29), (2, 30), (3, 0)]
    # We discarded off-suit and off-trump -> void in spades AND hearts, NOW
    # (the training env marks this at play time, before the trick resolves):
    assert env.impossible_cards[3, 24:32].all()     # spades void
    assert env.impossible_cards[3, 8:16].all()      # trump (hearts) void
    assert not env.impossible_cards[2].any()        # seat 2 followed suit
    assert len(env.hands[3]) == 5

    # ================= Frame 3: trick resolves via lastCards =============
    sync.sync(FRAME3, MY_ID)
    auditor.check(sync, FRAME3)

    assert env.tricks_played == 1
    assert sorted(env.graveyard) == [0, 26, 29, 30]
    assert env.last_trick == [(0, 26), (1, 29), (2, 30), (3, 0)]  # chrono+seats
    assert env.current_trick == []                  # partitioned, not clobbered
    assert env.raw_points_by_team == [64, 0]        # 57 + 7
    assert env.known_cards[2, 27]                   # 10S not played yet: kept

    # ================= Frame 4: new hand + bolt detection ================
    sync.sync(FRAME4, MY_ID)
    auditor.check(sync, FRAME4)

    assert sync.consume_new_hand() is True          # LSTM reset event
    assert sync.degraded_hand is False              # clean deal
    assert env.phase == "BIDDING" and env.bidding_round == 1
    assert env.tricks_played == 0 and len(env.graveyard) == 0
    assert not env.known_cards.any() and not env.impossible_cards.any()
    assert env.trump is None and env.declarer is None
    assert env.face_up_card == ASCII_TO_ID["k"]
    assert [len(h) for h in env.hands] == [5, 5, 5, 5]
    assert env.raw_points_by_team == [0, 0]         # bidding: zero like training
    assert sync.match_scores == [25, 27]
    assert env.bolts_by_team == [1, 0]              # from the BT-1 marker
    assert sync.match_scores == [25, 27]            # bolt row carries 25 forward
    assert sync._known_last_cards_str == "svwy"     # adopted, not replayed

    # Idempotent: re-reading the same table cannot double-count
    sync.sync(FRAME4, MY_ID)
    assert env.bolts_by_team == [1, 0]
    assert sync.match_scores == [25, 27]

    # Team parity: seats {0,2} are team 0 and {1,3} are team 1, so from
    # seat 3 the bolt belongs to the opposing team.
    us, them = my_pos % 2, 1 - (my_pos % 2)
    assert env.bolts_by_team[them] == 1
    assert env.bolts_by_team[us] == 0

    print("\n" + "=" * 60)
    print(f"SMOKE TEST PASSED  ({auditor.violations} auditor violations)")
    print("=" * 60)
    assert auditor.violations == 0, "auditor flagged unexpected violations"


if __name__ == "__main__":
    test_sync_smoke()
