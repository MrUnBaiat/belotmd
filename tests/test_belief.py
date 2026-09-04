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


# ------------------------------------------------------- determinization

def _mid_hand(cards_each=5, trump=2):
    """A well-formed mid-hand state: four equal hands plus a graveyard that
    accounts for every remaining card."""
    s = BelotState()
    s.phase = "PLAYING"
    s.trump, s.declarer, s.dealer = trump, 1, 3
    s.current_player, s.done = 0, False
    s.tricks_played = 8 - cards_each
    s.hands = [list(range(p * cards_each, (p + 1) * cards_each)) for p in range(4)]
    dealt = 4 * cards_each
    s.graveyard = list(range(dealt, 32))
    s.current_trick = []
    s.known_cards[:] = False
    s.impossible_cards[:] = False
    return s


def _assert_valid(state, seat, hands):
    from belotmd.game.belief import Infeasible  # noqa: F401
    flat = [c for h in hands for c in h]
    assert len(flat) == len(set(flat)), "a card was dealt twice"
    assert hands[seat] == list(state.hands[seat]), "our own hand was changed"
    for p in range(4):
        assert len(hands[p]) == len(state.hands[p])
        for c in hands[p]:
            assert c not in state.graveyard
            assert not state.impossible_cards[p, c], f"void violated: {p},{c}"
        for c in np.flatnonzero(state.known_cards[p]):
            assert int(c) in hands[p], f"pin violated: {p},{c}"


def test_determinization_produces_a_legal_world():
    from belotmd.game.belief import sample_determinization
    s = _mid_hand()
    rng = np.random.default_rng(0)
    for _ in range(50):
        _assert_valid(s, 0, sample_determinization(s, 0, rng))


def test_determinization_honours_pins_and_voids():
    from belotmd.game.belief import sample_determinization
    s = _mid_hand()
    s.known_cards[2, 10] = True             # partner provably holds card 10
    hearts = [c for c in range(8, 16) if c not in s.graveyard]
    s.impossible_cards[1, hearts] = True    # left is void in hearts
    rng = np.random.default_rng(1)
    for _ in range(50):
        hands = sample_determinization(s, 0, rng)
        _assert_valid(s, 0, hands)
        assert 10 in hands[2]
        assert not (set(hands[1]) & set(hearts))


def test_determinization_explores_rather_than_repeating():
    """PIMC is worthless if every sampled world is the same world."""
    from belotmd.game.belief import sample_determinization
    s = _mid_hand()
    rng = np.random.default_rng(2)
    seen = {tuple(tuple(h) for h in sample_determinization(s, 0, rng))
            for _ in range(40)}
    assert len(seen) > 20, f"only {len(seen)} distinct worlds in 40 samples"


def test_determinization_reports_contradictions_instead_of_guessing():
    """A dead-end silently patched would produce an ILLEGAL world, which is
    worse than no world: the search would evaluate a position that cannot
    exist. Contradictory constraints must raise."""
    from belotmd.game.belief import Infeasible, sample_determinization
    s = _mid_hand()
    # Make every unplaced card impossible for all three opponents.
    for p in (1, 2, 3):
        s.impossible_cards[p, :] = True
    with pytest.raises(Infeasible):
        sample_determinization(s, 0, np.random.default_rng(3), max_attempts=4)


def test_determinization_rejects_an_inconsistent_state():
    from belotmd.game.belief import Infeasible, sample_determinization
    s = _mid_hand()
    s.graveyard = []          # hand sizes no longer account for 32 cards
    with pytest.raises(Infeasible):
        sample_determinization(s, 0, np.random.default_rng(4))


# ------------------------------------------------ the constraints bundle

def test_hand_sizes_are_real_even_though_contents_are_not():
    """Opponent hand CONTENTS are placeholders; only the lengths are real.
    Live, hands[0] can read [0,1,2,3,4,5] -- which looks exactly like the
    7 through Q of diamonds, and is not."""
    from belotmd.game.belief import hand_sizes
    s = _mid_hand(cards_each=5)
    assert hand_sizes(s) == (5, 5, 5, 5)


def test_constraints_bundle_never_exposes_placeholder_hands():
    from belotmd.game.belief import constraints
    s = _mid_hand()
    s.known_cards[2, 10] = True
    c = constraints(s, 0)
    assert c.own_hand == tuple(s.hands[0])
    assert c.pins[2] == (10,)
    assert 10 not in c.pool, "a pinned card is not still up for grabs"
    assert not set(c.pool) & set(c.own_hand)
    assert not set(c.pool) & set(c.played)
    assert c.is_consistent()
    assert sum(c.need(p) for p in c.others) == len(c.pool)


def test_check_accepts_a_world_the_sdk_sampler_produced():
    from belotmd.game.belief import constraints, sample_determinization
    s = _mid_hand()
    s.known_cards[2, 10] = True
    c = constraints(s, 0)
    rng = np.random.default_rng(5)
    for _ in range(20):
        c.check(sample_determinization(s, 0, rng))


@pytest.mark.parametrize("break_it,reason", [
    (lambda h, c: h[c.others[0]].append(h[c.others[1]][0]), "more than one seat"),
    (lambda h, c: h[c.seat].pop(), "own hand was altered"),
    (lambda h, c: h[c.others[0]].__setitem__(0, c.played[0]), "played card"),
])
def test_check_rejects_illegal_worlds(break_it, reason):
    """A sampler written elsewhere should be held to the same standard, so the
    validator has to actually catch things."""
    from belotmd.game.belief import Infeasible, constraints, sample_determinization
    s = _mid_hand()
    c = constraints(s, 0)
    hands = [list(h) for h in sample_determinization(s, 0, np.random.default_rng(6))]
    break_it(hands, c)
    with pytest.raises(Infeasible) as exc:
        c.check(hands)
    assert reason in str(exc.value)


def test_check_rejects_a_violated_pin():
    from belotmd.game.belief import Infeasible, constraints, sample_determinization
    s = _mid_hand()
    s.known_cards[2, 10] = True
    c = constraints(s, 0)
    hands = [list(h) for h in sample_determinization(s, 0, np.random.default_rng(7))]
    hands[2].remove(10)
    hands[2].append(next(x for x in hands[1]))
    hands[1].remove(hands[2][-1])
    hands[1].append(10)
    with pytest.raises(Infeasible) as exc:
        c.check(hands)
    assert "provably holds" in str(exc.value)
