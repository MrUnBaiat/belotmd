"""
test_agents.py — the Agent seam.

These guard the property that makes the bot model-agnostic: the registry and
the default agent must work with nothing installed beyond numpy, and no agent
may ever return an action the mask forbids.
"""

import numpy as np
import pytest

from belotmd.agents import available, get_agent
from belotmd.game import actions
from belotmd.game.state import BelotState


def test_registry_lists_both_agents():
    assert "random" in available()
    assert "ppo" in available()


def test_unknown_agent_names_the_alternatives():
    with pytest.raises(ValueError) as exc:
        get_agent("nope")
    assert "random" in str(exc.value)


def test_random_agent_needs_no_torch():
    """The whole point of the fallback: importable and usable on its own."""
    agent = get_agent("random", seed=0)
    assert agent.name == "random"
    # Stateless contract: reset/snapshot/restore must all be safe no-ops.
    agent.reset()
    agent.restore(agent.snapshot())


def test_random_agent_respects_the_mask():
    agent = get_agent("random", seed=1234)
    state = BelotState()
    mask = state.get_legal_actions().astype(np.int8)
    assert mask.any(), "fixture produced no legal actions"

    for _ in range(500):
        action = agent.act(state, state.current_player, [0, 0], mask)
        assert mask[action], f"returned masked-out action {action}"


def test_random_agent_honours_a_narrowed_mask():
    """A retry hands the agent a mask with the refused action removed."""
    agent = get_agent("random", seed=7)
    state = BelotState()
    mask = state.get_legal_actions().astype(np.int8)
    only = int(np.flatnonzero(mask)[0])
    narrowed = np.zeros_like(mask)
    narrowed[only] = 1

    assert all(agent.act(state, 0, [0, 0], narrowed) == only for _ in range(20))


def test_empty_mask_raises_rather_than_guessing():
    agent = get_agent("random")
    with pytest.raises(ValueError):
        agent.act(BelotState(), 0, [0, 0], np.zeros(38, dtype=np.int8))


def test_action_space_constants_match_the_state():
    assert BelotState().action_space_size == actions.ACTION_SPACE_SIZE == 38
    assert actions.suit_of(actions.ACTION_SUIT_BASE + 2) == 2
    assert actions.suit_of(actions.ACTION_PASS) is None
