"""
test_rules.py — the legal-action rules, pinned.

`BelotState.get_legal_actions()` is now the single rulebook shared by the live
bot and by the trainer that produces agents for it. If the two ever disagree,
a model trains against one set of rules and plays against another, and the
failure shows up live as a move the server refuses -> the turn times out ->
belot.md hands the seat to its own bot for the rest of the session.

So these cases are deliberately explicit rather than generated: each one is a
rule of Belot, written out, with the expected mask spelled in full.

Card ids are `suit * 8 + rank`:
    suits  0 diamonds  1 hearts  2 clubs  3 spades
    ranks  0=7 1=8 2=9 3=10 4=J 5=Q 6=K 7=A
"""

import numpy as np
import pytest

from belotmd.game.actions import (ACTION_ACCEPT, ACTION_PASS, ACTION_SUIT_BASE)
from belotmd.game.state import BelotState

DIAMONDS, HEARTS, CLUBS, SPADES = 0, 1, 2, 3


def card(suit, rank):
    return suit * 8 + rank


# Trump is clubs throughout, so these are the interesting cards.
C7, C8, C9, C10, CJ, CQ, CK, CA = [card(CLUBS, r) for r in range(8)]
D7, D8, D9, D10, DJ, DQ, DK, DA = [card(DIAMONDS, r) for r in range(8)]
H7 = card(HEARTS, 0)
S7 = card(SPADES, 0)


def playing(hand, trick=(), *, seat=0, trump=CLUBS, declarer=1,
            declarer_played_trump=True):
    s = BelotState()
    s.phase = "PLAYING"
    s.trump, s.declarer, s.dealer = trump, declarer, 3
    s.current_player, s.done, s.tricks_played = seat, False, 0
    s.declarer_has_played_trump = declarer_played_trump
    s.hands = [[] for _ in range(4)]
    s.hands[seat] = list(hand)
    s.current_trick = [tuple(t) for t in trick]
    s.graveyard = []
    return s


def bidding(*, bidding_round, seat, dealer, face_up_suit=HEARTS):
    s = BelotState()
    s.phase = "BIDDING"
    s.bidding_round = bidding_round
    s.current_player, s.dealer = seat, dealer
    s.face_up_suit = face_up_suit
    s.done = False
    return s


def legal(state):
    return set(np.flatnonzero(state.get_legal_actions()).tolist())


# --------------------------------------------------------------- bidding

def test_round_one_offers_pass_and_accept_only():
    assert legal(bidding(bidding_round=1, seat=0, dealer=3)) == {
        ACTION_PASS, ACTION_ACCEPT}


def test_round_two_offers_every_suit_except_the_one_turned_up():
    s = bidding(bidding_round=2, seat=0, dealer=3, face_up_suit=HEARTS)
    assert legal(s) == {ACTION_PASS,
                        ACTION_SUIT_BASE + DIAMONDS,
                        ACTION_SUIT_BASE + CLUBS,
                        ACTION_SUIT_BASE + SPADES}


def test_the_dealer_may_not_pass_out_round_two():
    """Somebody has to name a trump, and by round 2 it is the dealer's job."""
    s = bidding(bidding_round=2, seat=3, dealer=3, face_up_suit=HEARTS)
    assert ACTION_PASS not in legal(s)
    assert legal(s) == {ACTION_SUIT_BASE + DIAMONDS,
                        ACTION_SUIT_BASE + CLUBS,
                        ACTION_SUIT_BASE + SPADES}


# ---------------------------------------------------------------- leading

def test_trump_may_not_be_led_before_the_declarer_has_played_one():
    s = playing([D7, C7], seat=0, declarer=1, declarer_played_trump=False)
    assert legal(s) == {D7}


def test_but_a_hand_of_nothing_but_trump_may_lead_it():
    s = playing([C7, C8], seat=0, declarer=1, declarer_played_trump=False)
    assert legal(s) == {C7, C8}


def test_and_the_declarer_is_never_restricted():
    s = playing([D7, C7], seat=1, declarer=1, declarer_played_trump=False)
    assert legal(s) == {D7, C7}


def test_once_the_declarer_has_played_trump_anyone_may_lead_it():
    s = playing([D7, C7], seat=0, declarer=1, declarer_played_trump=True)
    assert legal(s) == {D7, C7}


# -------------------------------------------------------------- following

def test_you_must_follow_suit():
    s = playing([D7, D8, C7], trick=[(1, DQ)])
    assert legal(s) == {D7, D8}


def test_void_in_the_led_suit_you_must_ruff():
    s = playing([H7, C7], trick=[(1, DQ)])
    assert legal(s) == {C7}


def test_void_in_both_you_may_discard_anything():
    s = playing([H7, S7], trick=[(1, DQ)])
    assert legal(s) == {H7, S7}


def test_overruffing_is_compulsory_when_possible():
    """An opponent ruffed with the 8; we hold the 7 and the 9. In trump the
    order is 7 < 8 < Q < K < 10 < A < 9 < J, so only the 9 beats it."""
    s = playing([H7, C7, C9], trick=[(1, DQ), (2, C8)])
    assert legal(s) == {C9}


def test_but_a_trump_you_cannot_beat_may_still_be_played():
    s = playing([H7, C7], trick=[(1, DQ), (2, C8)])
    assert legal(s) == {C7}


def test_overruffing_applies_when_trump_itself_was_led():
    """The rule is easy to miss: following a led trump you must still beat
    the highest trump on the table if you can."""
    s = playing([D7, C7, C9], trick=[(1, C8)])
    assert legal(s) == {C9}


def test_following_a_led_trump_you_cannot_beat_frees_the_rest_of_it():
    s = playing([D7, C7], trick=[(1, C8)])
    assert legal(s) == {C7}


def test_holding_the_led_suit_you_never_ruff_instead():
    s = playing([D7, C9], trick=[(1, DQ), (2, C8)])
    assert legal(s) == {D7}


# ----------------------------------------------------------------- misc

def test_a_finished_hand_offers_nothing():
    s = playing([D7, C7])
    s.done = True
    assert legal(s) == set()


def test_the_mask_is_always_the_full_action_space():
    assert playing([D7]).get_legal_actions().shape == (38,)
    assert bidding(bidding_round=1, seat=0, dealer=3).get_legal_actions().shape == (38,)


# ----------------------------------------------------------- card values

@pytest.mark.parametrize("rank,points", [
    (0, 0), (1, 0), (2, 14), (3, 10), (4, 20), (5, 3), (6, 4), (7, 11)])
def test_trump_card_points(rank, points):
    assert BelotState().card_value(card(CLUBS, rank), is_trump=True)[0] == points


@pytest.mark.parametrize("rank,points", [
    (0, 0), (1, 0), (2, 0), (3, 10), (4, 2), (5, 3), (6, 4), (7, 11)])
def test_plain_card_points(rank, points):
    assert BelotState().card_value(card(CLUBS, rank), is_trump=False)[0] == points


def test_trump_order_is_seven_eight_queen_king_ten_ace_nine_jack():
    s = BelotState()
    order = [C7, C8, CQ, CK, C10, CA, C9, CJ]
    ranks = [s.card_value(c, is_trump=True)[1] for c in order]
    assert ranks == sorted(ranks), "trump hierarchy is wrong"


def test_plain_order_is_seven_eight_nine_jack_queen_king_ten_ace():
    s = BelotState()
    order = [D7, D8, D9, DJ, DQ, DK, D10, DA]
    ranks = [s.card_value(c, is_trump=False)[1] for c in order]
    assert ranks == sorted(ranks), "plain hierarchy is wrong"


def test_a_full_deck_of_points_is_152():
    """152 in cards; the extra 10 for the last trick makes the familiar 162."""
    s = BelotState()
    total = sum(s.card_value(c, is_trump=(c // 8 == CLUBS))[0] for c in range(32))
    assert total == 152
