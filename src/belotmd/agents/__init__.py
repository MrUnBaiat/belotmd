"""
Agent registry.

`random` is imported eagerly (numpy only). `ppo` is registered through a thin
shim that defers the torch import until the agent is actually built, so the
package stays importable — and the bot stays runnable — without torch.
"""

from .base import Agent, available, get_agent, register
from . import random_agent  # noqa: F401  (registers "random")


@register("ppo")
def _build_ppo(**kwargs):
    try:
        from .ppo.agent import PPOAgent
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "the 'ppo' agent needs PyTorch: pip install -e \".[ppo]\"\n"
            "Or run with --agent random, which has no such dependency."
        ) from exc
    return PPOAgent(**kwargs)


__all__ = ["Agent", "available", "get_agent", "register"]
