"""
test_agents.py — the Agent seam.

These guard the property that makes the bot model-agnostic: the registry and
the default agent must work with nothing installed beyond numpy, and no agent
may ever return an action the mask forbids.
"""

import numpy as np
import pytest

from belotmd.agents import available, get_agent, register
from belotmd.agents.base import _BUILDERS
from belotmd.game import actions
from belotmd.game.state import BelotState


def test_registry_lists_the_bundled_agent():
    assert "random" in available()


def test_unknown_agent_names_the_alternatives_and_the_entry_point_group():
    with pytest.raises(ValueError) as exc:
        get_agent("nope")
    message = str(exc.value)
    assert "random" in message
    assert "belotmd.agents" in message, "should point at the plugin mechanism"


def test_in_process_registration_beats_an_entry_point():
    """A program must always be able to override a plugin it has installed."""
    sentinel = object()
    register("test-override")(lambda **kw: sentinel)
    try:
        assert get_agent("test-override") is sentinel
        assert "test-override" in available()
    finally:
        _BUILDERS.pop("test-override", None)


def test_agent_kwargs_reach_the_factory():
    seen = {}
    register("test-kwargs")(lambda **kw: seen.update(kw) or object())
    try:
        get_agent("test-kwargs", checkpoint="w.pt", depth="3")
        assert seen == {"checkpoint": "w.pt", "depth": "3"}
    finally:
        _BUILDERS.pop("test-kwargs", None)


def test_random_agent_is_dependency_light():
    """The bundled agent must work with numpy alone."""
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
