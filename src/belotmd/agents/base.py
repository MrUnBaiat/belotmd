"""
base.py — the seam between the platform bridge and whatever is deciding.

`bot.py` knows the rules of belot.md: when it is our turn, when to declare a
combination, when the seven-swap window is open, when the server has quietly
handed our seat to its own bot. It does not know or care how a card is chosen.
That is an `Agent`.

An agent sees a reconstructed `BelotState`, our seat index, the running match
score, and an authoritative legal-action mask. It returns one index from the
38-slot action space (see `belotmd.game.actions`). Nothing about neural
networks, observation vectors or tensors appears in this contract — those are
private to whichever agent wants them.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """What `bot.py` requires of a decision-maker."""

    name: str

    def reset(self) -> None:
        """A new hand is starting.

        Agents carrying state across turns (a recurrent policy's hidden state,
        a card-counting table) must clear it here. Called at every hand
        boundary, which the synchronizer detects itself and which is robust to
        dropped frames. Stateless agents can no-op.
        """

    def act(self, state, seat: int, match_scores, legal_mask: np.ndarray) -> int:
        """Choose an action for `seat`.

        `state`       a `BelotState` reconstructed from live server frames.
                      Treat as read-only.
        `seat`        our absolute seat index, 0-3.
        `match_scores` running [team0, team1] match score.
        `legal_mask`  length-38 mask of currently permitted actions. It is
                      AUTHORITATIVE: it may already be narrower than
                      `state.get_legal_actions()` because the server refused
                      an action earlier this turn. Returning a masked-out
                      index gets the move rejected and, eventually, the seat
                      taken over by the platform bot.

        Return an action index with `legal_mask[action]` true.

        IMPORTANT: **`state.hands` is only real for `seat`.** belot.md never reveals
        another player's cards, so the other three entries are placeholder ids
        with the correct *length* and meaningless contents. An agent that
        reads them is looking at fiction.

        What is genuinely known about the other hands lives in
        `state.known_cards`, `state.impossible_cards`, `state.graveyard` and
        `state.current_trick`. `belotmd.game.belief` turns those into a
        probability per (opponent, card), or into a pool of unplaced cards to
        deal out if you are sampling determinizations for a search.
        """

    def snapshot(self):
        """Capture whatever `act()` would mutate, before a turn begins.

        belot.md occasionally refuses a move, and `bot.py` retries the turn
        with the offending action masked out. A retry must resume from the
        state the agent was in BEFORE the turn, so that internal state
        advances once per game decision rather than once per network message —
        which is how the recurrent policy was trained. Stateless agents return
        None.
        """
        return None

    def restore(self, snapshot) -> None:
        """Undo `act()`'s state change, before retrying the same turn."""


# --------------------------------------------------------------- registry
#
# Agents arrive from two places: `register()` for anything defined in this
# process, and the "belotmd.agents" entry-point group for separately installed
# packages. Both resolve lazily, so a plugin that needs torch, a checkpoint or
# a search engine costs nothing until someone actually asks for it.

ENTRY_POINT_GROUP = "belotmd.agents"

_BUILDERS = {}


def register(name):
    """Decorator: register a zero-or-more-kwargs agent factory under `name`."""
    def wrap(factory):
        _BUILDERS[name] = factory
        return factory
    return wrap


def _entry_points():
    """Installed packages advertising an agent, as {name: EntryPoint}."""
    from importlib import metadata

    try:
        eps = metadata.entry_points()
        # Python 3.10+ has select(); 3.9 returns a plain dict of groups.
        group = (eps.select(group=ENTRY_POINT_GROUP)
                 if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, []))
    except Exception:                                  # pragma: no cover
        return {}
    return {ep.name: ep for ep in group}


def available():
    """Every agent name we can build, from both sources, sorted."""
    return sorted(set(_BUILDERS) | set(_entry_points()))


def get_agent(name, **kwargs):
    """Build the named agent. Extra kwargs are passed to its factory.

    An agent registered in-process wins over an entry point of the same name,
    so a program can always override a plugin it has installed.
    """
    if name in _BUILDERS:
        return _BUILDERS[name](**kwargs)

    entry = _entry_points().get(name)
    if entry is not None:
        try:
            factory = entry.load()
        except Exception as exc:
            raise ImportError(
                f"the {name!r} agent is installed but failed to load "
                f"({entry.value}): {type(exc).__name__}: {exc}"
            ) from exc
        return factory(**kwargs)

    names = available()
    raise ValueError(
        f"unknown agent {name!r}; available: {', '.join(names) or '(none)'}.\n"
        f"Agents from other packages are discovered through the "
        f"{ENTRY_POINT_GROUP!r} entry-point group -- check that the package "
        f"providing {name!r} is installed in this environment."
    )
