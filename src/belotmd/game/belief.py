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


class UniformSampler:
    """Exactly uniform draws over EVERY deal consistent with the constraints.

    Given only hand sizes, pins and voids, every consistent completion of a
    random deal is equally likely -- so that is the distribution a search that
    samples worlds needs, and anything else tilts its view of where the cards
    are. This counts the completions exactly and draws each one with equal
    probability.

    The count is a small dynamic programme: pool cards are placed one at a
    time, and the state is the three seats' remaining capacities -- at most 24
    cards by 9 x 9 x 9 states, built once. A draw then places each card with
    probability proportional to the number of completions that remain, which is
    exactly uniform over whole deals, in ~24 table lookups.

    The capacity-weighted greedy this replaced was uniform only without voids.
    Measured on real recorded positions at tricks 3-6 -- where voids are the
    rule and where the search runs -- it put 8-9% of (card, seat) probabilities
    more than 3 standard errors from the exact value (0.27% expected), some by
    35 (research/v11 x32 in the BelotAgents repository).

    Build one per decision and draw as many worlds as the clock allows;
    `sample_determinization` builds one per call.
    """

    def __init__(self, cons):
        self.cons = cons
        self.others = cons.others
        if any(cons.need(p) < 0 for p in self.others):
            raise Infeasible(
                f"a seat is pinned more cards than it holds: "
                f"{ {p: (len(cons.pins.get(p, ())), cons.hand_sizes[p]) for p in self.others} }")
        if not cons.is_consistent():
            raise Infeasible(
                f"{len(cons.pool)} unplaced cards but seats need "
                f"{sum(cons.need(p) for p in self.others)}")
        # Any order gives the same distribution; most-constrained first keeps
        # the reachable capacity states few.
        self.cards = tuple(sorted(cons.pool, key=lambda card: (len(cons.candidates(card)), card)))
        self.allowed = tuple(
            tuple(j for j, p in enumerate(self.others) if card not in cons.voids.get(p, ()))
            for card in self.cards)
        self.need = tuple(cons.need(p) for p in self.others)
        self._memo = {}
        self.worlds = self._count(0, self.need)
        if not self.worlds:
            raise Infeasible(
                f"no deal satisfies the constraints for seat {cons.seat} "
                f"(pool={len(cons.pool)}, need={dict(zip(self.others, self.need))}) "
                f"-- they contradict each other, which points at a sync bug")

    def _count(self, i, caps):
        """Completions placing cards[i:] into seats with capacities `caps`."""
        if i == len(self.cards):
            return 1                  # capacities sum to the cards left: all zero
        key = (i, caps)
        hit = self._memo.get(key)
        if hit is None:
            hit = 0
            for j in self.allowed[i]:
                if caps[j]:
                    hit += self._count(i + 1, caps[:j] + (caps[j] - 1,) + caps[j + 1:])
            self._memo[key] = hit
        return hit

    def draw(self, rng):
        """One deal, as four card lists indexed by seat; ours is our real hand."""
        caps = self.need
        deal = {p: list(self.cons.pins.get(p, ())) for p in self.others}
        for i, card in enumerate(self.cards):
            options = []
            for j in self.allowed[i]:
                if caps[j]:
                    nxt = caps[:j] + (caps[j] - 1,) + caps[j + 1:]
                    options.append((j, nxt, self._count(i + 1, nxt)))
            r = int(rng.integers(sum(w for _, _, w in options)))
            for j, nxt, w in options:
                if r < w:
                    break
                r -= w
            deal[self.others[j]].append(card)
            caps = nxt
        hands = [None] * 4
        hands[self.cons.seat] = list(self.cons.own_hand)
        for p in self.others:
            hands[p] = sorted(deal[p])
        return hands


def uniform_sampler(state, seat):
    """A `UniformSampler` over everything `seat` can observe. Build it once per
    decision, then `draw(rng)` a world per determinization."""
    return UniformSampler(constraints(state, seat))


def sample_determinization(state, seat, rng=None, max_attempts=None):
    """Sample one full deal consistent with everything `seat` can observe,
    **uniformly** over all such deals.

    Returns a list of four card lists indexed by seat; `seat`'s own entry is
    its real hand. Every constraint the synchronizer has established is
    honoured:

      * hand sizes match `numCards` as reported by the server;
      * a card in `known_cards[p]` always goes to p;
      * a card in `impossible_cards[p]` never goes to p;
      * played cards and our own cards are never dealt out.

    Each consistent deal is equally likely -- the right prior when all that is
    known is those hard constraints -- see `UniformSampler`, which this builds
    per call. Drawing many worlds for one decision, build a sampler once with
    `uniform_sampler(state, seat)` instead. If you want a different
    distribution -- weighted by what the opponents' play suggests, say -- build
    it on `unseen_cards` and `can_hold`, and use this as the reference.

    Raises `Infeasible` when no deal satisfies the constraints. That is a real
    signal, not a nuisance: they contradict each other, which points at a sync
    bug rather than bad luck. `max_attempts` is accepted for compatibility and
    ignored: the draw is exact, so it never retries.
    """
    rng = rng or np.random.default_rng()
    return uniform_sampler(state, seat).draw(rng)
