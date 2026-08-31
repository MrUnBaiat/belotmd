"""
random_agent.py — uniform play over the legal actions.

Depends on nothing but numpy, so the bot runs end to end with no checkpoint
and no torch installed. Useful as a baseline, as a smoke test of the bridge
and synchronizer in isolation from the policy, and as the worked example for
`docs/` on writing your own agent.
"""

import numpy as np

from .base import register


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        """Stateless: nothing to clear between hands."""

    def snapshot(self):
        return None

    def restore(self, snapshot):
        """Stateless: nothing to undo."""

    def act(self, state, seat, match_scores, legal_mask):
        legal = np.flatnonzero(legal_mask)
        if legal.size == 0:
            raise ValueError("act() called with an empty legal mask")
        return int(self._rng.choice(legal))


@register("random")
def _build(seed=None, **_ignored):
    return RandomAgent(seed=seed)
