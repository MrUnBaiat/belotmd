"""
actions.py — the 38-slot action space, shared by every agent.

It is a property of the game, not of any model or of the wire protocol:

    0..31   play the card with that id (suit * 8 + rank)
    32      pass
    33      accept the face-up card as trump   (bidding round 1 only)
    34..37  name a suit as trump               (bidding round 2); suit = a - 34

An agent returns one of these indices; `bot.py` translates it into the
corresponding server message.
"""

ACTION_SPACE_SIZE = 38

CARD_ACTIONS     = range(0, 32)
ACTION_PASS      = 32
ACTION_ACCEPT    = 33
ACTION_SUIT_BASE = 34
SUIT_ACTIONS     = range(ACTION_SUIT_BASE, ACTION_SUIT_BASE + 4)


def suit_of(action):
    """Suit index named by a round-2 bid action, else None."""
    return action - ACTION_SUIT_BASE if action in SUIT_ACTIONS else None


def describe(action, id_to_ascii=None):
    """Human-readable action name, for logs."""
    if action in CARD_ACTIONS:
        ch = (id_to_ascii or {}).get(action)
        return f"PLAY {action}" + (f" ({ch})" if ch else "")
    if action == ACTION_PASS:
        return "PASS"
    if action == ACTION_ACCEPT:
        return "ACCEPT FACE-UP"
    if action in SUIT_ACTIONS:
        return f"CHOOSE SUIT {suit_of(action)}"
    return f"UNKNOWN({action})"
