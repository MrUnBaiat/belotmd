"""
protocol.py — the belot.md wire vocabulary, in one place.

Everything here describes the *platform*, not the bot and not any particular
agent: the card charset, the phase enum, the room close codes, and the shared
38-slot action space. See docs/PLATFORM_NOTES.md for how each was established.
"""

# --------------------------------------------------------------- card codes
# The server identifies cards by a single ASCII character. The mapping to the
# internal id is `suit * 8 + rank`, with suits ordered diamonds, hearts, clubs,
# spades and ranks ordered 7 < 8 < 9 < 10 < J < Q < K < A (natural order, which
# is also the order runs are scored in).
ASCII_TO_ID = {
    'y': 0,  'z': 1,  'a': 2,  'b': 3,  'c': 4,  'd': 5,  'e': 6,  'f': 7,
    'A': 8,  'B': 9,  'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15,
    'C': 16, 'D': 17, 'm': 18, 'n': 19, 'o': 20, 'p': 21, 'q': 22, 'r': 23,
    'E': 24, 'F': 25, 's': 26, 't': 27, 'u': 28, 'v': 29, 'w': 30, 'x': 31,
}

ID_TO_ASCII = {v: k for k, v in ASCII_TO_ID.items()}

SUIT_NAMES = ["diamonds", "hearts", "clubs", "spades"]
RANK_NAMES = ["7", "8", "9", "10", "J", "Q", "K", "A"]

# Human-readable card name per wire char, for logs.
CHAR_TO_CARD = {
    ch: f"{RANK_NAMES[cid % 8]}_{SUIT_NAMES[cid // 8]}"
    for ch, cid in ASCII_TO_ID.items()
}

# The server numbers suits from 1; the internal representation from 0.
SUIT_TO_INT = {name: i + 1 for i, name in enumerate(SUIT_NAMES)}


# -------------------------------------------------------------- phase enum
# Authoritative, read out of the site's own gameplay.js.
NOT_STARTED           = 0
STARTED               = 1
PUSH_CARDS            = 2
AFTER_PUSH_CARD       = 3
ANIMATION_FIRST_DEAL  = 4
DEAL_CARDS_1          = 5
TRUMP_CHOOSE_1        = 6
TRUMP_CHOOSE_2        = 7
SWAP_SEVEN            = 8
DEAL_CARDS_2          = 9
PLAY                  = 10
HAND_TAKE             = 11
WIN_ALL_HANDS         = 12
ROUND_ENDED           = 13
GAME_ENDED            = 14

DEAL_PHASES = (NOT_STARTED, STARTED, PUSH_CARDS, AFTER_PUSH_CARD,
               ANIMATION_FIRST_DEAL, DEAL_CARDS_1)   # ready / cut / first deal
BID_PHASES  = (TRUMP_CHOOSE_1, TRUMP_CHOOSE_2)       # bidding round 1 / round 2
SWAP_PHASE  = SWAP_SEVEN                             # round-1 accept only
END_PHASES  = (ROUND_ENDED, GAME_ENDED)              # hand scored / match over


# ------------------------------------------------------------- leave codes
# Colyseus room close codes used by belot.md.
LEAVE_OTHER_SESSION    = 4001   # our account opened the table elsewhere
LEAVE_KICKED           = 4002   # removed from the table
LEAVE_TABLE_REMOVED    = 4003   # table dissolved
LEAVE_I_LEFT           = 4004   # we left
LEAVE_POSITION_CHANGED = 4005   # lobby-only: seats shuffle as players gather

LEAVE_NAMES = {
    LEAVE_OTHER_SESSION: "OTHER_SESSION", LEAVE_KICKED: "KICKED",
    LEAVE_TABLE_REMOVED: "TABLE_REMOVED", LEAVE_I_LEFT: "I_LEFT",
    LEAVE_POSITION_CHANGED: "POSITION_CHANGED",
}
# 4005 is the only recoverable code, and it can only occur in the LOBBY while
# the server is still gathering players and seats shuffle. Once a match is
# under way it cannot happen, so there is never accumulated hand state to
# invalidate -- a plain rejoin is enough.


# ------------------------------------------------------------ action space
# Re-exported for convenience; the action space is a property of the game, so
# it is defined in belotmd.game.actions.
from ..game.actions import (ACTION_ACCEPT, ACTION_PASS,  # noqa: E402,F401
                            ACTION_SPACE_SIZE, ACTION_SUIT_BASE, CARD_ACTIONS,
                            SUIT_ACTIONS)
