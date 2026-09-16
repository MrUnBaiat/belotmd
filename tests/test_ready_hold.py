"""
test_ready_hold.py — the seats must be right before we announce ourselves.

A match starts the moment all four players are READY. Our bot readies up as
soon as it sits down, which is correct alone and wrong as a pair: whoever
happens to be sitting opposite becomes our partner for the whole match, and two
of our own accounts playing AGAINST each other is the exact opposite of the
point.

So a paired bot holds READY until its partner is opposite, and the account that
created the table rotates the seats until that is true. Nobody can start the
match while we hold, so holding costs nothing.
"""

import asyncio

import pytest

from belotmd.bot import LiveBelotBot
from belotmd.config import Config

PARTNER = "MyOtherAccount"
ME = "me"


class StubClient:
    """Records what the bot would have sent."""

    def __init__(self):
        self.readies = 0
        self.rotations = 0
        self.sessions_played = 1

    async def send_ready(self, value=True):
        self.readies += 1

    async def change_players_position(self):
        self.rotations += 1


def _bot(partner=PARTNER, table_mode="create"):
    """A real bot, real state machine, no network."""
    from belotmd.agents import get_agent
    from belotmd.platform.sync import StateSynchronizer

    bot = LiveBelotBot.__new__(LiveBelotBot)
    bot.config = Config(cookies="x", audit=False, agent="random",
                        partner=partner or "", table_mode=table_mode)
    bot.client = StubClient()
    bot.sync_engine = StateSynchronizer()
    bot.agent = get_agent("random", seed=0)
    bot.audit = False
    bot._room_seq = 0
    bot.partner = (partner or "").strip().casefold() or None
    bot.is_host = table_mode == "create"
    bot._reset_room_state()
    return bot


def _lobby(names, ready=False, phase=0):
    """A lobby frame. `names` is the four seats; None is an empty one."""
    players = []
    for seat, name in enumerate(names):
        if name is None:
            players.append({"position": seat})          # nobody there
        elif name == ME:
            players.append({"id": ME, "name": "OurAccount", "position": seat,
                            "ready": ready, "numCards": 0, "cards": ""})
        else:
            players.append({"id": f"p{seat}", "name": name, "position": seat,
                            "ready": True, "numCards": 0})
    return {
        "currentPhase": phase, "activePlayer": -1, "dealer": 0, "round": 0,
        "topCard": "", "trump": -1, "declarer": -1, "swapSeven": -1,
        "lastCards": "", "scoreTable": "", "roundTotals": "",
        "timeleft": 0, "totalTime": 0, "players": players,
    }


def _feed(bot, frame, times=1):
    for _ in range(times):
        asyncio.run(bot.on_state_update(frame, ME))
    return bot.client


# ------------------------------------------------------------- holding READY
def test_a_bot_playing_alone_still_readies_immediately():
    """Every existing single-account run depends on this."""
    client = _feed(_bot(partner=None, table_mode="lobby"),
                   _lobby(["a", "b", "c", ME]))
    assert client.readies == 1


def test_we_wait_rather_than_start_without_our_partner():
    client = _feed(_bot(), _lobby(["a", "b", "c", ME]))
    assert client.readies == 0, "readying up here starts a match with strangers"


def test_we_wait_while_our_partner_sits_beside_us():
    """Adjacent is an OPPONENT. This is the case that silently ruins the run:
    all four seats are full and the table is one READY from starting."""
    client = _feed(_bot(), _lobby(["a", "b", PARTNER, ME]))
    assert client.readies == 0


def test_we_ready_up_as_soon_as_our_partner_is_opposite():
    client = _feed(_bot(), _lobby(["a", PARTNER, "c", ME]))
    assert client.readies == 1


def test_the_partner_is_recognised_whatever_the_casing():
    """The name is typed into a config file by hand."""
    client = _feed(_bot(partner=PARTNER.upper()),
                   _lobby(["a", PARTNER, "c", ME]))
    assert client.readies == 1


def test_readies_are_still_not_flooded_once_the_seats_are_right():
    client = _feed(_bot(), _lobby(["a", PARTNER, "c", ME]), times=10)
    assert client.readies == 1


# ---------------------------------------------------------- fixing the seats
def test_the_host_rotates_the_seats_when_they_are_wrong():
    client = _feed(_bot(table_mode="create"), _lobby(["a", "b", PARTNER, ME]))
    assert client.rotations == 1
    assert client.readies == 0, "not until it has worked"


def test_the_guest_waits_instead_of_rotating():
    """Only the table's creator can move players around."""
    client = _feed(_bot(table_mode="join"), _lobby(["a", "b", PARTNER, ME]))
    assert client.rotations == 0
    assert client.readies == 0


def test_nothing_is_rotated_until_the_table_is_full():
    """An empty seat is about to be filled by whoever sits down next, which
    changes the arrangement anyway."""
    client = _feed(_bot(), _lobby(["a", None, PARTNER, ME]))
    assert client.rotations == 0


def test_nothing_is_rotated_before_our_partner_arrives():
    client = _feed(_bot(), _lobby(["a", "b", "c", ME]))
    assert client.rotations == 0


def test_the_seats_are_left_alone_once_they_are_right():
    client = _feed(_bot(), _lobby(["a", PARTNER, "c", ME]), times=5)
    assert client.rotations == 0


def test_rotations_are_debounced():
    """Each one drops the other three players (close code 4005) and they
    rejoin. Sending one per frame would keep the table in a permanent
    reshuffle."""
    client = _feed(_bot(), _lobby(["a", "b", PARTNER, ME]), times=20)
    assert client.rotations == 1


def test_rotating_gives_up_rather_than_shuffling_the_table_forever():
    """The payload is unverified. If it does something other than what we
    think, this stops instead of spinning."""
    bot = _bot()
    bot.ROTATE_RETRY_S = 0.0                      # no debounce, for the test
    client = _feed(bot, _lobby(["a", "b", PARTNER, ME]), times=50)
    assert client.rotations == LiveBelotBot.MAX_ROTATIONS


def test_a_new_table_starts_its_rotation_count_again():
    bot = _bot()
    bot.ROTATE_RETRY_S = 0.0
    _feed(bot, _lobby(["a", "b", PARTNER, ME]), times=50)

    bot.client.sessions_played += 1               # the bridge joined another
    bot.ROTATE_RETRY_S = 0.0
    _feed(bot, _lobby(["a", "b", PARTNER, ME]))
    assert bot.client.rotations == LiveBelotBot.MAX_ROTATIONS + 1


# ------------------------------------------------------------------ logging
def test_the_seat_layout_is_reported_by_role_not_by_name(capsys):
    """Which seats are ours is the whole question; the strangers at the table
    are nobody's business, here or in a recording."""
    _feed(_bot(), _lobby(["StrangerOne", "StrangerTwo", PARTNER, ME]))
    out = capsys.readouterr().out

    assert "partner" in out and "us" in out
    assert "StrangerOne" not in out and "StrangerTwo" not in out
    assert PARTNER not in out


@pytest.mark.parametrize("phase", [0, 13, 14])
def test_the_hold_applies_at_every_idle_phase(phase):
    """A table hosts many matches: between them everyone readies up again, and
    the host can reseat. The hold has to cover those too, not just a fresh
    table."""
    client = _feed(_bot(), _lobby(["a", "b", PARTNER, ME], phase=phase))
    assert client.readies == 0
