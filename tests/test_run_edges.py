"""
test_run_edges.py — what a declared run says about the cards it does NOT contain.

The platform reports a run maximally: it always shows the longest one in the
hand. So a run that could have been extended would have been, and the ranks
immediately outside it are provably not held.

One exception, and it is the whole subtlety: the enum stops at a five-run, so
a run of six or more is reported as its TOP FIVE. That keeps the upper edge
(the top card is still the highest, or a longer run would have been reported
ending higher) and destroys the lower one (a six-run looks exactly like a
five-run with a card underneath it).

    3-run   above absent    below absent
    4-run   above absent    below absent
    5-run   above absent    below UNKNOWN
    four of a kind / bella  --  nothing

Card ids are suit * 8 + rank, ranks 0=7 … 7=A. Spades is suit 3:
    E=7 F=8 s=9 t=10 u=J v=Q w=K x=A   ->  ids 24..31
"""

import numpy as np
import pytest

from belotmd.game import combinations as combo
from belotmd.platform.protocol import ASCII_TO_ID, ID_TO_ASCII
from belotmd.platform.sync import StateSynchronizer

S7, S8, S9, S10, SJ, SQ, SK, SA = range(24, 32)


def _ex(value):
    return combo.excluded_field(value, ASCII_TO_ID)


# ------------------------------------------------------------- the rule

def test_a_three_run_rules_out_both_neighbours():
    """The worked example: J, Q, K of spades means no 10 and no ace."""
    assert _ex("1w") == [S10, SA]


def test_a_four_run_rules_out_both_neighbours():
    assert _ex("2w") == [S9, SA]          # 10,J,Q,K -> no 9, no ace


def test_a_five_run_keeps_only_the_upper_edge():
    """A six-run is reported as its top five, so the card below may well be
    held. The card above is still safe."""
    assert _ex("3w") == [SA]              # 9..K -> no ace; the 8 is unknown


def test_nothing_sits_above_the_ace():
    assert _ex("3x") == []                # 10..A five-run: no upper, no lower
    assert _ex("1x") == [SJ]              # Q,K,A three-run -> no jack


def test_nothing_sits_below_the_seven():
    assert _ex("1s") == [S10]             # 7,8,9 -> only the 10 is ruled out


@pytest.mark.parametrize("value", ["4u", "5w", "6a", "7a", "8a", "9a"])
def test_non_runs_imply_nothing(value):
    """Four of a kind and bella are not runs: holding a neighbouring rank
    would not have changed what was declared."""
    assert _ex(value) == []


def test_a_malformed_run_is_refused_rather_than_guessed():
    """A wrong exclusion is a FALSE VOID, which silently removes a real
    candidate from every belief and every sampled world."""
    assert _ex("3F") == []                # five-run ending at the 8: impossible
    assert _ex("1E") == []                # three-run ending at the 7
    assert _ex("1?") == []                # unknown card char


def test_several_declarations_combine():
    got = _ex("1w|1s")                    # J,Q,K and 7,8,9 in the same suit
    assert got == [S10, SA]               # the 10 is ruled out by both


# ------------------------------------------- reaching the belief state

def _sync_at_play(my_hand="yzab", others=6):
    """A synchronizer parked in a play phase, with us at seat 3."""
    sync = StateSynchronizer()
    frame = {
        "currentPhase": 10, "activePlayer": 0, "dealer": 0, "round": 1,
        "topCard": "", "trump": 4, "declarer": 0, "swapSeven": -1,
        "lastCards": "", "scoreTable": "", "roundTotals": "",
        "timeleft": 250, "totalTime": 250,
        "players": [
            {"id": "0", "position": 0, "numCards": others},
            {"id": "1", "position": 1, "numCards": others},
            {"id": "2", "position": 2, "numCards": others},
            {"id": "me", "position": 3, "cards": my_hand,
             "numCards": len(my_hand)},
        ],
    }
    sync.sync(frame, "me")
    return sync


def test_a_declared_run_writes_voids_into_the_belief_state():
    sync = _sync_at_play()
    sync.apply_combination(1, "1w")       # seat 1 shows J, Q, K of spades

    held = np.flatnonzero(sync.state.known_cards[1]).tolist()
    barred = np.flatnonzero(sync.state.impossible_cards[1]).tolist()
    assert held == [SJ, SQ, SK]
    assert S10 in barred and SA in barred


def test_the_voids_are_only_about_the_declaring_seat():
    sync = _sync_at_play()
    sync.apply_combination(1, "1w")
    for other in (0, 2, 3):
        assert not sync.state.impossible_cards[other, S10]
        assert not sync.state.impossible_cards[other, SA]


def test_a_five_run_writes_only_the_upper_void():
    sync = _sync_at_play()
    sync.apply_combination(2, "3w")       # 9..K of spades
    barred = np.flatnonzero(sync.state.impossible_cards[2]).tolist()
    assert SA in barred
    assert S8 not in barred, "a six-run looks identical; the 8 may be held"


def test_a_rejected_decoding_writes_no_voids():
    """If the cards land in OUR hand the decoding is wrong, and so are the
    exclusions. Both must be dropped together."""
    sync = _sync_at_play(my_hand="uvw")   # we hold J, Q, K of spades
    sync.apply_combination(1, "1w")       # seat 1 cannot also hold them
    assert sync.combo_conflicts == 1
    assert not sync.state.impossible_cards[1].any()
    assert not sync.state.known_cards[1].any()


def test_an_exclusion_never_overrides_a_certainty():
    sync = _sync_at_play()
    sync.state.known_cards[1, SA] = True          # we know they hold the ace
    sync.apply_combination(1, "1w")               # ...which implies they do not
    assert sync.state.known_cards[1, SA], "the pin must win"
    assert not sync.state.impossible_cards[1, SA]


def test_voids_survive_into_the_sampler_and_the_belief_matrix():
    """The point of the inference: fewer worlds, sharper probabilities."""
    from belotmd.game.belief import belief_matrix, constraints
    sync = _sync_at_play()
    sync.apply_combination(1, "1w")

    c = constraints(sync.state, 3)
    assert S10 in c.voids[1] and SA in c.voids[1]
    assert 1 not in c.candidates(S10), "seat 1 can no longer be dealt the 10"

    # belief_matrix rows are relative to the observer: left, partner, right.
    bel = belief_matrix(sync.state, 3)
    row = (1 - 3) % 4 - 1                         # seat 1 seen from seat 3
    assert bel[row, SK] == 1.0, "the run itself is a certainty"
    assert bel[row, S10] == 0.0, "and its edges carry no mass at all"
