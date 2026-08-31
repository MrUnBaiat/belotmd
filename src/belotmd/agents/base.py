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
                      Read-only as far as the agent is concerned.
        `seat`        our absolute seat index, 0-3.
        `match_scores` running [team0, team1] match score.
        `legal_mask`  length-38 mask of currently permitted actions. It is
                      AUTHORITATIVE: it may already be narrower than
                      `state.get_legal_actions()` because the server refused
                      an action earlier this turn. Returning a masked-out
                      index will get the move rejected and, eventually, the
                      seat taken over by the platform bot.

        Return an action index with `legal_mask[action]` true.
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
# Agents are registered lazily by name so that `--agent random` never imports
# torch, and a missing checkpoint or missing torch install produces a clear
# message instead of an ImportError at startup.

_BUILDERS = {}


def register(name):
    """Decorator: register a zero-or-more-kwargs agent factory under `name`."""
    def wrap(factory):
        _BUILDERS[name] = factory
        return factory
    return wrap


def available():
    """Registered agent names, sorted."""
    return sorted(_BUILDERS)


def get_agent(name, **kwargs):
    """Build the named agent. Extra kwargs go to its factory."""
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown agent {name!r}; available: {', '.join(available())}"
        )
    return _BUILDERS[name](**kwargs)
