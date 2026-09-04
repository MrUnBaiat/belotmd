"""
belotmd — an SDK for building bots that play Belot on belot.md.

The platform has no API. It speaks msgpack over Colyseus WebSockets and
publishes a partial, frequently stale view of the table. This library does the
work of turning that stream into a consistent game state, and gives you one
place to plug in a decision-maker.

Quick start::

    from belotmd import Agent, Config, LiveBelotBot, register

    @register("greedy")
    def build(**kw):
        return MyAgent()

    LiveBelotBot(Config.from_env(agent="greedy")).run()   # await this

Or from the shell, with an agent installed from another package::

    belot-bot --agent ppo

The three layers, lowest first:

    belotmd.game       Belot rules. No belot.md, no network, no ML.
                       BelotState, the action space, combination decoding,
                       and the belief matrix over opponents' hands.

    belotmd.platform   Everything belot.md-specific: the Node bridge, the
                       client, the wire protocol, and the state synchronizer
                       that reconstructs a BelotState from raw frames.

    belotmd.agents     The seam. Implement `Agent`, register it, run it.

`belotmd.audit` records every frame and asserts invariants alongside a live
session; `belotmd.audit.replay` re-runs a recording offline, deterministically.

See docs/PLATFORM_NOTES.md for how belot.md actually behaves, and
docs/WRITING_AN_AGENT.md for the agent contract.
"""

from .agents.base import Agent, available, get_agent, register
from .config import Config
from .game.actions import (ACTION_ACCEPT, ACTION_PASS, ACTION_SPACE_SIZE,
                           ACTION_SUIT_BASE, CARD_ACTIONS, SUIT_ACTIONS)
from .game.belief import (Constraints, Infeasible, belief_matrix, can_hold,
                          constraints, hand_sizes, sample_determinization,
                          unseen_cards)
from .game.state import BelotState

__version__ = "1.0.0"

__all__ = [
    # the seam
    "Agent", "register", "get_agent", "available",
    # game
    "BelotState", "belief_matrix", "unseen_cards", "can_hold",
    "sample_determinization", "Infeasible",
    "Constraints", "constraints", "hand_sizes",
    "ACTION_SPACE_SIZE", "CARD_ACTIONS", "ACTION_PASS", "ACTION_ACCEPT",
    "ACTION_SUIT_BASE", "SUIT_ACTIONS",
    # running a session
    "Config", "LiveBelotBot",
    "__version__",
]


def __getattr__(name):
    """`LiveBelotBot` is resolved on demand.

    Importing it pulls in `websockets`, which a consumer using only the rules
    and belief layers has no reason to need.
    """
    if name == "LiveBelotBot":
        from .bot import LiveBelotBot
        return LiveBelotBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
