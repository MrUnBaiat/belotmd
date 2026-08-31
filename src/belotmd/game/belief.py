"""
belief.py — who probably holds what.

Belot is an imperfect-information game: you see your own eight cards and
nothing else. What you *can* know about the other three hands accumulates as a
hand is played, and `belotmd.platform.sync` records all of it on the state:

    graveyard, current_trick   cards that are gone
    known_cards[seat]          cards that seat provably HOLDS
                               (declared combinations, the seven-swap, the
                               face-up card)
    impossible_cards[seat]     cards that seat provably CANNOT hold
                               (suit and trump voids, inferred when a player
                               failed to follow)

`belief_matrix()` turns those constraints into a probability per (opponent,
card). Use it directly for a heuristic agent, as a feature for a learned one,
or as the sampling distribution for determinization in a search agent.
"""

import numpy as np


def belief_matrix(state, seat):
    """Probability that each opponent holds each card, as a (3, 32) array.

    Rows are ordered **relative to `seat`**: left (seat+1), partner (seat+2),
    right (seat+3). Columns are card ids.

    Semantics:
      * a card in `known_cards[p]` reads exactly 1.0 for p;
      * a card that is gone (played, in our hand, or the face-up card during
        bidding) reads 0.0 for everyone;
      * everything else is spread over the opponents who could still hold it.

    The spread is a **single normalisation pass**: columns are divided among
    the candidates, then each row is rescaled to that opponent's remaining
    hand size. Rows therefore sum exactly to the number of unknown cards in
    that hand; columns are approximate. A card only one opponent can hold
    reads ~0.90 rather than 1.00 — that is the intended behaviour of one pass,
    not a bug to iterate away.
    """
    unseen = np.ones(32, dtype=bool)
    unseen[state.hands[seat]] = False
    unseen[state.graveyard] = False
    for _, card in state.current_trick:
        unseen[card] = False
    if state.phase == "BIDDING" and state.face_up_card is not None:
        unseen[state.face_up_card] = False

    # A card known to sit in one specific hand is not an open candidate for
    # anybody else. Without this line the holder reads 1.0 (set below) while
    # the other two opponents still receive ~0.49 of phantom mass on the same
    # card -- the column sums to ~1.97 instead of 1.0 -- and their genuine
    # candidates are diluted to compensate. It matters most exactly when the
    # information is richest: a declared 5-card run pins five cards at once.
    unseen[state.known_cards.any(axis=0)] = False

    others = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]

    W = np.zeros((3, 32), dtype=np.float32)
    for i, p in enumerate(others):
        candidates = unseen & ~state.impossible_cards[p] & ~state.known_cards[p]
        W[i, candidates] = 1.0

    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    P = W / col_sums

    remaining = np.array(
        [len(state.hands[p]) - state.known_cards[p].sum() for p in others],
        dtype=np.float32,
    )
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    P = P * (remaining[:, np.newaxis] / row_sums[:, np.newaxis])
    P = np.clip(P, 0.0, 1.0)

    out = np.zeros((3, 32), dtype=np.float32)
    for i, p in enumerate(others):
        out[i, state.known_cards[p]] = 1.0
        unresolved = ~state.known_cards[p]
        out[i, unresolved] = P[i, unresolved]
    return out


def unseen_cards(state, seat):
    """Card ids whose location is unknown to `seat` — the pool a search agent
    deals out when sampling a determinization."""
    unseen = np.ones(32, dtype=bool)
    unseen[state.hands[seat]] = False
    unseen[state.graveyard] = False
    for _, card in state.current_trick:
        unseen[card] = False
    if state.phase == "BIDDING" and state.face_up_card is not None:
        unseen[state.face_up_card] = False
    return np.flatnonzero(unseen).tolist()


def can_hold(state, seat, card):
    """Could `seat` be holding `card`, given everything currently known?"""
    if state.known_cards[seat, card]:
        return True
    if state.impossible_cards[seat, card]:
        return False
    if card in state.graveyard or any(c == card for _, c in state.current_trick):
        return False
    return not state.known_cards[:, card].any()
