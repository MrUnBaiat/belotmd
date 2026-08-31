"""
test_belief.py — the belief matrix over opponents' hands.

This is the SDK's answer to the only interesting question in an
imperfect-information game, so its guarantees are worth pinning down:
certainties read 1.0, dead cards read 0.0, and a pinned card is excluded from
every other candidate row rather than double-counted.
"""

import numpy as np
import pytest

from belotmd.game.belief import belief_matrix, can_hold, unseen_cards
from belotmd.game.state import BelotState


@pytest.fixture
def state():
    """A mid-play state with known hand sizes and nothing else asserted."""
    s = BelotState()
    s.phase = "PLAYING"
    s.trump, s.declarer, s.dealer = 2, 1, 3
    s.current_player, s.done, s.tricks_played = 0, False, 0
    s.hands = [[0, 1, 2, 3]] + [list(range(4, 8)) for _ in range(3)]
    s.graveyard, s.current_trick = [], []
    s.known_cards[:] = False
    s.impossible_cards[:] = False
    return s


def test_shape_and_dtype(state):
    bel = belief_matrix(state, 0)
    assert bel.shape == (3, 32)
    assert bel.dtype == np.float32


def test_our_own_cards_are_never_attributed_to_anyone(state):
    bel = belief_matrix(state, 0)
    assert bel[:, state.hands[0]].sum() == 0.0


def test_a_pin_reads_one_for_its_holder_and_zero_elsewhere(state):
    """The exclusion that matters: without it the two other opponents each
    pick up ~0.49 of phantom mass on a card somebody else provably holds."""
    state.known_cards[2, 20] = True          # seat 2 == our partner, row 1

    bel = belief_matrix(state, 0)
    assert bel[1, 20] == 1.0
    assert bel[0, 20] == 0.0
    assert bel[2, 20] == 0.0
    assert bel[:, 20].sum() == pytest.approx(1.0)


def test_graveyard_cards_carry_no_mass(state):
    state.graveyard = [10, 11, 12, 13]
    bel = belief_matrix(state, 0)
    assert bel[:, state.graveyard].sum() == 0.0


def test_cards_on_the_table_carry_no_mass(state):
    state.current_trick = [(1, 14), (2, 15)]
    bel = belief_matrix(state, 0)
    assert bel[:, [14, 15]].sum() == 0.0


def test_rows_sum_to_the_unknown_part_of_each_hand(state):
    """Rows are exact by construction; columns are only approximate."""
    bel = belief_matrix(state, 0)
    for row, seat in enumerate([1, 2, 3]):
        expected = len(state.hands[seat]) - state.known_cards[seat].sum()
        assert bel[row].sum() == pytest.approx(expected, abs=1e-4)


def test_a_void_removes_that_suit_for_that_seat_only(state):
    hearts = list(range(8, 16))
    state.impossible_cards[1, hearts] = True     # seat 1 is void in hearts
    bel = belief_matrix(state, 0)
    assert bel[0, hearts].sum() == 0.0
    assert bel[1, hearts].sum() > 0.0            # seat 2 unaffected


def test_mass_is_never_negative_or_above_one(state):
    state.known_cards[1, 21] = True
    state.impossible_cards[3, list(range(8, 16))] = True
    bel = belief_matrix(state, 0)
    assert (bel >= 0.0).all()
    assert (bel <= 1.0).all()


def test_unseen_pool_excludes_everything_accounted_for(state):
    state.graveyard = [10, 11]
    state.current_trick = [(1, 12)]
    pool = unseen_cards(state, 0)
    for card in state.hands[0] + state.graveyard + [12]:
        assert card not in pool


def test_can_hold_respects_pins_and_voids(state):
    state.known_cards[1, 21] = True
    state.impossible_cards[2, 22] = True
    state.graveyard = [23]

    assert can_hold(state, 1, 21) is True       # its holder
    assert can_hold(state, 2, 21) is False      # pinned to someone else
    assert can_hold(state, 2, 22) is False      # void
    assert can_hold(state, 2, 23) is False      # already played
    assert can_hold(state, 2, 24) is True       # unconstrained
