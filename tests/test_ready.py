"""
test_ready.py — the READY handshake, across tables.

A table will not start until every seat is ready. Getting this wrong does not
raise, does not log, and does not end the session: the bot simply sits at a
table forever, having never announced itself.
"""

import asyncio

import pytest

from belotmd.bot import LiveBelotBot
from belotmd.config import Config


class StubClient:
    """Records outgoing messages; joins are simulated by the test."""

    def __init__(self):
        self.readies = 0
        self.sent = []
        self.sessions_played = 1      # the bridge reports joins; tests bump it

    async def send_ready(self, value=True):
        self.readies += 1
        self.sent.append(("READY", value))

    async def cut_deck(self, n):
        self.sent.append(("PUSH_CARD", n))

    async def pass_turn(self):
        self.sent.append(("PASS", None))

    async def bid_trump(self, s):
        self.sent.append(("TRUMP_CHOOSE", s))

    async def play_card_char(self, c):
        self.sent.append(("PLAY_CARD", c))

    async def show_combination(self, v):
        self.sent.append(("SHOW_COMBINATION", v))

    async def swap_seven(self):
        self.sent.append(("SWAP_SEVEN", None))


def _bot():
    """A bot with the network stubbed out, but its real state machine."""
    from belotmd.agents import get_agent
    from belotmd.platform.sync import StateSynchronizer

    bot = LiveBelotBot.__new__(LiveBelotBot)
    bot.config = Config(cookies="x", audit=False, agent="random")
    bot.client = StubClient()
    bot.sync_engine = StateSynchronizer()
    bot.agent = get_agent("random", seed=0)
    bot.audit = False
    bot._room_seq = 0
    bot._reset_room_state()
    return bot


def _lobby_frame(ready=False, phase=0):
    return {
        "currentPhase": phase, "activePlayer": -1, "dealer": 0, "round": 0,
        "topCard": "", "trump": -1, "declarer": -1, "swapSeven": -1,
        "lastCards": "", "scoreTable": "", "roundTotals": "",
        "timeleft": 0, "totalTime": 0,
        "players": [
            {"id": "1", "position": 0, "ready": True, "numCards": 0},
            {"id": "2", "position": 1, "ready": True, "numCards": 0},
            {"id": "3", "position": 2, "ready": True, "numCards": 0},
            {"id": "me", "position": 3, "ready": ready, "numCards": 0,
             "cards": ""},
        ],
    }


def test_we_announce_ourselves_at_a_new_table():
    bot = _bot()
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    assert bot.client.readies == 1


def test_we_stop_once_the_server_says_we_are_ready():
    """Otherwise every lobby frame re-sends it."""
    bot = _bot()
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    for _ in range(5):
        asyncio.run(bot.on_state_update(_lobby_frame(ready=True), "me"))
    assert bot.client.readies == 1


def test_a_second_table_still_gets_a_ready():
    """THE BUG: both tables begin at phase 0, so a dedup keyed on the phase
    suppresses exactly the READY the new table is waiting for. The bot sits
    there forever and nothing in the log says why.

    It bites on the ORDINARY path, not just an unlucky one: a match ends
    13 -> 14 -> 0, so the flag is left on 0 and the very next table -- which
    also starts at 0 -- is silently skipped.
    """
    bot = _bot()

    # First table: announce ourselves, play, watch the match end.
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    asyncio.run(bot.on_state_update(_lobby_frame(ready=True), "me"))
    for phase in (13, 14, 0):
        asyncio.run(bot.on_state_update(_lobby_frame(ready=False, phase=phase),
                                        "me"))
    first_table = bot.client.readies
    assert first_table >= 1

    # The table dissolves and the bridge joins another one, which also begins
    # at phase 0.
    bot.client.sessions_played += 1
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))

    assert bot.client.readies > first_table, (
        "no READY was sent at the new table -- it will never start")


def test_readies_are_not_flooded_while_the_server_catches_up():
    """Several frames arrive before the server reflects our READY. Retrying is
    correct; retrying on every frame is not."""
    bot = _bot()
    for _ in range(20):
        asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    assert bot.client.readies == 1


def test_a_lost_ready_is_retried():
    """The other half of the same property: if the server never acknowledges,
    we must try again rather than wait forever."""
    bot = _bot()
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    bot._ready_sent_ts -= bot.READY_RESEND_S + 0.1        # pretend a second passed
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))
    assert bot.client.readies == 2


def test_joining_a_new_table_clears_every_per_table_flag():
    """The same class of bug as the READY one: state that describes the table
    we were at is meaningless at the next, and carrying it across is how a
    rejoin goes quietly wrong."""
    bot = _bot()
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))

    # Dirty every per-table field.
    bot.last_action_turn_id = "PHASE_10_P3_HAND_5_TRICK_0"
    bot._turn_decision = ("x", None, {1, 2})
    bot._declared = {(1, "5q")}
    bot._pending_play = (1, 0, 7)
    bot._rejected_plays = 3
    bot.seat_bot_controlled = True
    bot._swap_hand, bot._swap_sent, bot._swap_skip = 1, True, True

    bot.client.sessions_played += 1
    asyncio.run(bot.on_state_update(_lobby_frame(ready=False), "me"))

    assert bot.last_action_turn_id is None
    assert bot._turn_decision is None
    assert bot._declared == set()
    assert bot._pending_play is None
    assert bot._rejected_plays == 0
    assert bot.seat_bot_controlled is False
    assert (bot._swap_hand, bot._swap_sent, bot._swap_skip) == (None, False, False)
