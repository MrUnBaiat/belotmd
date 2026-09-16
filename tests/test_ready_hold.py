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


def _bot(partner=PARTNER, table_mode="create", rotation_probe=0):
    """A real bot, real state machine, no network."""
    from belotmd.agents import get_agent
    from belotmd.platform.sync import StateSynchronizer

    bot = LiveBelotBot.__new__(LiveBelotBot)
    bot.config = Config(cookies="x", audit=False, agent="random",
                        partner=partner or "", table_mode=table_mode,
                        rotation_probe=rotation_probe)
    bot.client = StubClient()
    bot.sync_engine = StateSynchronizer()
    bot.agent = get_agent("random", seed=0)
    bot.audit = False
    bot._room_seq = 0
    bot.partner = (partner or "").strip().casefold() or None
    bot.is_host = table_mode == "create"
    bot.rotation_probe = rotation_probe
    bot._reset_room_state()
    bot.ROTATE_PROBE_GAP_S = 0.0          # no waiting between probe rotations
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


# ----------------------------------------------------- the rotation probe
#
# CHANGE_PLAYERS_POSITION is the only message we send whose payload and effect
# are unverified. A table where the seats come out right by luck never
# exercises it, so the probe rotates deliberately and logs what moved.

def test_the_probe_rotates_even_when_the_seats_are_already_right():
    """The point of the experiment: without this, a correct layout means the
    message is never sent and we learn nothing."""
    bot = _bot(rotation_probe=4)
    client = _feed(bot, _lobby(["a", PARTNER, "c", ME]), times=50)

    assert client.rotations == 4, "exactly the requested number, then stop"
    assert bot._probe_done == 4


def test_the_probe_holds_ready_until_it_has_finished():
    """A match starting mid-experiment would end it early and seat us at
    whatever the last rotation happened to produce."""
    bot = _bot(rotation_probe=4)
    client = _feed(bot, _lobby(["a", PARTNER, "c", ME]), times=2)

    assert client.rotations == 2 and client.readies == 0


def test_play_resumes_normally_once_the_probe_is_done():
    client = _feed(_bot(rotation_probe=2), _lobby(["a", PARTNER, "c", ME]),
                   times=50)
    assert client.rotations == 2
    assert client.readies == 1, "after the experiment, behave as usual"


def test_the_guest_never_probes():
    """Only the table's creator can move players around."""
    client = _feed(_bot(table_mode="join", rotation_probe=4),
                   _lobby(["a", PARTNER, "c", ME]), times=20)
    assert client.rotations == 0


def test_the_probe_waits_for_our_partner_to_arrive():
    """With nobody of ours to watch move, a rotation tells us nothing."""
    client = _feed(_bot(rotation_probe=4), _lobby(["a", "b", "c", ME]),
                   times=20)
    assert client.rotations == 0


def test_the_probe_starts_over_at_a_new_table():
    bot = _bot(rotation_probe=2)
    _feed(bot, _lobby(["a", PARTNER, "c", ME]), times=20)
    bot.client.sessions_played += 1                  # the bridge joined another
    _feed(bot, _lobby(["a", PARTNER, "c", ME]), times=20)
    assert bot.client.rotations == 4


def test_the_probe_reports_movement_without_naming_anyone(capsys):
    """The log has to show a seat CHANGING occupant, which needs identity --
    so strangers get stable letters instead of names."""
    bot = _bot(rotation_probe=1)
    _feed(bot, _lobby(["StrangerOne", PARTNER, "StrangerTwo", ME]))
    out = capsys.readouterr().out

    assert "PROBE" in out and "rotation 1/1" in out
    assert "0=A" in out and "2=B" in out, "strangers appear as letters"
    assert "1=partner" in out and "3=us" in out
    assert "StrangerOne" not in out and "StrangerTwo" not in out
    assert PARTNER not in out


def test_a_stranger_keeps_the_same_letter_while_we_sit_there():
    """Otherwise 'A moved from 0 to 1' would be unreadable."""
    bot = _bot(rotation_probe=0)
    _feed(bot, _lobby(["StrangerOne", "StrangerTwo", PARTNER, ME]))
    first = dict(bot._seat_labels)
    _feed(bot, _lobby(["StrangerOne", "StrangerTwo", PARTNER, ME]))
    assert bot._seat_labels == first and len(first) == 2


# ------------------------------------------------- the measured seat rule
#
# Captured live on 2026-09-16: the sender stays put and every other seat's
# occupant moves one place forward. Empty seats take part, so this is a
# positional cycle over the three non-sender chairs.

def _rotate_once(layout):
    """The rule as measured, with the sender at seat 0."""
    out = list(layout)
    for seat in (1, 2, 3):
        out[(seat % 3) + 1] = layout[seat]
    return tuple(out)


CAPTURED = [
    (("us", "A", "B", "partner"), ("us", "partner", "A", "B")),
    (("us", "partner", "A", "empty"), ("us", "empty", "partner", "A")),
    (("us", "C", "partner", "A"), ("us", "A", "C", "partner")),
    (("us", "A", "C", "partner"), ("us", "partner", "A", "C")),
]


@pytest.mark.parametrize("before,after", CAPTURED)
def test_the_rule_reproduces_what_the_platform_did(before, after):
    """These four transitions are what the server actually produced. If the
    rule we reason from ever stops explaining them, everything below is
    wrong."""
    assert _rotate_once(before) == after


def test_three_rotations_return_the_table_to_where_it_started():
    """The period is 3, not 4 -- our own seat is the fixed point."""
    start = ("us", "A", "B", "partner")
    once = _rotate_once(start)
    assert _rotate_once(_rotate_once(once)) == start
    assert once != start


@pytest.mark.parametrize("my_pos", [0, 1, 2, 3])
def test_the_rotation_count_is_arithmetic_not_a_search(my_pos):
    need = LiveBelotBot._rotations_needed
    assert need(my_pos, (my_pos + 2) % 4) == 0, "already opposite: do nothing"
    assert need(my_pos, (my_pos + 1) % 4) == 1
    assert need(my_pos, (my_pos + 3) % 4) == 2, "never more than two"


def test_the_count_agrees_with_applying_the_rule():
    """The arithmetic and the measured behaviour have to be the same thing."""
    for partner_seat in (1, 2, 3):
        layout = ["us", "X", "Y", "Z"]
        layout[partner_seat] = "partner"
        current = tuple(layout)
        for _ in range(LiveBelotBot._rotations_needed(0, partner_seat)):
            current = _rotate_once(current)
        assert current[2] == "partner", f"from seat {partner_seat}"


def test_a_table_that_will_not_come_right_is_left_alone(capsys):
    """Two rotations always suffice. A third means the rule no longer holds,
    and shuffling a table full of people on a broken assumption is worse than
    stopping: every rotation ejects the other three players."""
    bot = _bot()
    bot.ROTATE_RETRY_S = 0.0
    client = _feed(bot, _lobby(["a", "b", PARTNER, ME]), times=40)

    assert client.rotations == LiveBelotBot.MAX_ROTATIONS == 3
    assert client.readies == 0, "must not start a mis-seated match either"
    out = capsys.readouterr().out
    assert "not behaving as measured" in out


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
