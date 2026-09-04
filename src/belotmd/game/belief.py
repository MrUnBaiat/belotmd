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

from dataclasses import dataclass
from typing import Dict, Tuple

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


def hand_sizes(state):
    """How many cards each seat holds, as a 4-tuple.

    Use this, never ``len(state.hands[p])`` for an opponent. The contents of
    an opponent's ``hands`` entry are PLACEHOLDER ids of the right length --
    live, ``hands[0]`` may read ``[0, 1, 2, 3, 4, 5]``, which looks exactly
    like the 7 through Q of diamonds and is not. The lengths are real because
    the server publishes ``numCards``; the contents never are.
    """
    return tuple(len(h) for h in state.hands)


@dataclass(frozen=True)
class Constraints:
    """Everything known about the hidden state, in one object.

    Built by :func:`constraints`. This is the complete input to *any*
    determinization sampler -- the SDK's own, or one you bring. If you write
    your own, read it from here rather than off the state, so you cannot
    accidentally consume the placeholder hands (see :func:`hand_sizes`).

        seat        the observing seat
        own_hand    that seat's real cards
        hand_sizes  cards held, per seat (authoritative)
        pool        unplaced cards: location unknown, not pinned to anyone
        pins        seat -> cards it provably HOLDS
        voids       seat -> cards it provably CANNOT hold
        played      graveyard plus whatever is on the table
        degraded    True if we joined mid-hand and earlier tricks were never
                    observed, so the belief state is incomplete rather than
                    merely uncertain. Worth refusing to search on.
    """

    seat: int
    own_hand: Tuple[int, ...]
    hand_sizes: Tuple[int, int, int, int]
    pool: Tuple[int, ...]
    pins: Dict[int, Tuple[int, ...]]
    voids: Dict[int, Tuple[int, ...]]
    played: Tuple[int, ...]
    degraded: bool

    @property
    def others(self):
        return tuple((self.seat + r) % 4 for r in (1, 2, 3))

    def need(self, seat):
        """How many POOL cards `seat` still has to be dealt."""
        return self.hand_sizes[seat] - len(self.pins.get(seat, ()))

    def candidates(self, card):
        """Which seats could be holding `card`, given the voids."""
        return tuple(p for p in self.others if card not in self.voids.get(p, ()))

    def is_consistent(self):
        """Do the constraints admit any deal at all, on a cheap count check?"""
        return (all(self.need(p) >= 0 for p in self.others)
                and len(self.pool) == sum(self.need(p) for p in self.others))

    def check(self, hands):
        """Validate a candidate world. Raises `Infeasible` with the reason.

        Provided so a sampler written elsewhere can be held to exactly the
        same standard as the built-in one -- call it in your tests, or behind
        a debug flag in production.
        """
        if len(hands) != 4:
            raise Infeasible(f"expected 4 hands, got {len(hands)}")
        flat = [c for h in hands for c in h]
        if len(flat) != len(set(flat)):
            raise Infeasible("a card was dealt to more than one seat")
        if list(hands[self.seat]) != list(self.own_hand):
            raise Infeasible("the observing seat's own hand was altered")
        for p in range(4):
            if len(hands[p]) != self.hand_sizes[p]:
                raise Infeasible(
                    f"seat {p} holds {len(hands[p])} cards, "
                    f"expected {self.hand_sizes[p]}")
            for c in hands[p]:
                if c in self.played:
                    raise Infeasible(f"seat {p} was dealt played card {c}")
                if c in self.voids.get(p, ()):
                    raise Infeasible(f"seat {p} was dealt {c}, which it "
                                     f"provably cannot hold")
            for c in self.pins.get(p, ()):
                if c not in hands[p]:
                    raise Infeasible(f"seat {p} provably holds {c}, "
                                     f"but it was dealt elsewhere")


def constraints(state, seat):
    """Bundle everything known about the hidden state, from `seat`'s view."""
    played = tuple(sorted(set(state.graveyard)
                          | {c for _, c in state.current_trick}))
    pins, voids = {}, {}
    for p in range(4):
        pins[p] = tuple(int(c) for c in np.flatnonzero(state.known_cards[p]))
        voids[p] = tuple(int(c) for c in np.flatnonzero(state.impossible_cards[p]))
    pinned = {c for p in range(4) if p != seat for c in pins[p]}
    pool = tuple(c for c in unseen_cards(state, seat) if c not in pinned)
    return Constraints(
        seat=seat,
        own_hand=tuple(state.hands[seat]),
        hand_sizes=hand_sizes(state),
        pool=pool,
        pins={p: v for p, v in pins.items() if p != seat},
        voids={p: v for p, v in voids.items() if p != seat},
        played=played,
        degraded=bool(getattr(state, "beliefs_degraded", False)),
    )


class Infeasible(RuntimeError):
    """No deal satisfies the constraints. See `sample_determinization`."""


def sample_determinization(state, seat, rng=None, max_attempts=64):
    """Sample one full deal consistent with everything `seat` can observe.

    This is deliberately **constraint satisfaction only** — it answers "is
    this world legal", not "is this world likely". Returns a list of four card
    lists indexed by seat; `seat`'s own entry is its real hand.

    The division of labour with a search agent: this function hands you one
    legal world, and how many you draw, how you weight them and what you do
    with them stays yours. If you want a different sampling *distribution* —
    importance-weighted by `belief_matrix`, say — build it on `unseen_cards`
    and `can_hold` instead and use this as an oracle to check yours against.

    Every constraint the synchronizer has established is honoured:

      * hand sizes match `numCards` as reported by the server;
      * a card in `known_cards[p]` always goes to p;
      * a card in `impossible_cards[p]` never goes to p;
      * played cards and our own cards are never dealt out.

    Cards are placed most-constrained-first — the card with the fewest
    remaining candidate seats goes next — because a naive shuffle-and-fill
    dead-ends surprisingly often once voids accumulate late in a hand, and a
    dead-end that is silently patched produces an *illegal* world, which is
    worse than no world at all. Ties are broken randomly and candidate seats
    are chosen with probability proportional to their remaining capacity, so
    repeated calls explore the space rather than returning the same deal.

    Raises `Infeasible` if no consistent deal is found in `max_attempts`
    tries. That is a real signal, not a nuisance: it means the constraints
    contradict each other, which points at a sync bug rather than bad luck.
    """
    rng = rng or np.random.default_rng()
    c = constraints(state, seat)

    if any(c.need(p) < 0 for p in c.others):
        raise Infeasible(
            f"a seat is pinned more cards than it holds: "
            f"{ {p: (len(c.pins.get(p, ())), c.hand_sizes[p]) for p in c.others} }")
    if not c.is_consistent():
        raise Infeasible(
            f"{len(c.pool)} unplaced cards but seats need "
            f"{sum(c.need(p) for p in c.others)}")

    for _ in range(max_attempts):
        capacity = {p: c.need(p) for p in c.others}
        deal = {p: list(c.pins.get(p, ())) for p in c.others}
        remaining = list(c.pool)
        rng.shuffle(remaining)
        ok = True

        while remaining:
            # Recompute candidates each round: capacities shrink as we go.
            best, best_options = None, None
            for card in remaining:
                options = [p for p in c.candidates(card) if capacity[p] > 0]
                if not options:
                    ok = False
                    break
                if best_options is None or len(options) < len(best_options):
                    best, best_options = card, options
                    if len(options) == 1:
                        break            # cannot do better than forced
            if not ok:
                break

            weights = np.array([capacity[p] for p in best_options], dtype=float)
            chosen = int(rng.choice(best_options, p=weights / weights.sum()))
            deal[chosen].append(best)
            capacity[chosen] -= 1
            remaining.remove(best)

        if ok and all(v == 0 for v in capacity.values()):
            hands = [None] * 4
            hands[seat] = list(c.own_hand)
            for p in c.others:
                hands[p] = sorted(deal[p])
            return hands

    raise Infeasible(
        f"no consistent deal for seat {seat} after {max_attempts} attempts "
        f"(pool={len(c.pool)}, "
        f"need={ {p: c.need(p) for p in c.others} }) -- the constraints may contradict"
    )
