"""
test_tables.py — creating a table, and finding our partner's.

Two accounts have to meet at one table. The host creates it; the guest finds it
in the lobby by the host's username and sits down. Nothing here talks to the
network: the create payload and the lookup are pure, and the client's side of
the two-step join is driven against a fake bridge.
"""

import asyncio
import json

import pytest

from belotmd.platform.client import BelotClient
from belotmd.platform.tables import (CREATE_TABLE_BODY, creator_name,
                                     find_table, is_joinable, table_id_of)

# The captured lobby entry for a table one of our accounts created, as a second
# account sees it. `creator` is [level, username, account id].
HOST = "Mocasin123"
OURS = {"id": "new-44709391", "regula_14": 1, "type": 4, "colors": 1,
        "masa": 101, "players": "3", "time": 4, "pass": False,
        "creator": ["4", HOST, "104325"], "full": 0}
SOMEONE_ELSE = {"id": "new-44700000", "pass": False, "full": 0,
                "creator": ["2", "SomebodyElse", "999"]}
FULL = dict(OURS, id="new-44711111", full=1)


# ------------------------------------------------------------- the payload
def test_the_create_payload_is_the_captured_one_verbatim():
    """It goes to the platform's own form. Reordering or renaming a field is
    not ours to do, and `14puncte` is not even a valid identifier."""
    assert CREATE_TABLE_BODY == (
        "createNewTable=1&gametable_type=4&gametable_level=0&miza=50"
        "&gametable_color=1&gametable_points=101&14puncte=on&password=")


def test_the_table_is_open_to_anyone():
    """The two other seats are for humans: no password, and no rating floor,
    or the table serves no purpose."""
    assert "password=" in CREATE_TABLE_BODY
    assert not CREATE_TABLE_BODY.split("password=")[1]
    assert "gametable_level=0" in CREATE_TABLE_BODY


# -------------------------------------------------------------- the lookup
def test_our_partners_table_is_found_by_the_host_name():
    assert find_table([SOMEONE_ELSE, OURS], creator=HOST) is OURS


def test_the_host_name_is_matched_case_and_space_insensitively():
    """It is typed into a config file by hand."""
    assert find_table([OURS], creator=f"  {HOST.lower()} ") is OURS


def test_a_table_can_also_be_named_outright():
    assert find_table([SOMEONE_ELSE, OURS], table_id="new-44709391") is OURS


def test_the_new_prefix_survives():
    """`enterGame` takes the id as the lobby spells it -- it is not a
    number."""
    assert table_id_of(OURS) == "new-44709391"


def test_an_absent_table_is_not_an_error():
    """The guest asks before the host has finished creating; that is the
    normal case, not a failure."""
    assert find_table([SOMEONE_ELSE], creator=HOST) is None
    assert find_table([], creator=HOST) is None
    assert find_table(None, creator=HOST) is None


def test_a_full_table_is_not_joinable():
    assert is_joinable(OURS)
    assert not is_joinable(FULL)
    assert not is_joinable(None)


def test_a_lookup_with_no_criteria_is_a_mistake():
    with pytest.raises(ValueError):
        find_table([OURS])


def test_creator_name_handles_the_shapes_seen():
    assert creator_name(OURS) == HOST
    assert creator_name({"creator": "PlainName"}) == "PlainName"
    assert creator_name({}) is None
    assert creator_name({"creator": []}) is None


# --------------------------------------------------- what the client sends
class FakeBridge:
    """Replays scripted events, records what the client sends."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.script:
            raise StopAsyncIteration
        return json.dumps(self.script.pop(0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def of(self, action):
        return [m for m in self.sent if m.get("action") == action]


async def _noop(*a, **kw):
    pass


def _drive(script, **kwargs):
    """Drive the REAL `connect()` loop against a scripted bridge.

    Deliberately not a reimplementation of the lobby handling: the point is to
    exercise the branch that decides which table we sit at, since sitting at
    the wrong one is the failure this whole feature exists to avoid.
    """
    client = BelotClient(cookies="x", **kwargs)
    bridge = FakeBridge(script)
    client._start_node_daemon = lambda: None

    async def fake_daemon(uri):
        return bridge
    client._await_daemon = fake_daemon

    async def fake_send(action_type, payload):
        pass
    client._send = fake_send

    slept = []

    async def no_sleep(d):
        slept.append(d)

    async def go():
        orig, asyncio.sleep = asyncio.sleep, no_sleep
        try:
            await client.connect(on_state_callback=_noop)
        finally:
            asyncio.sleep = orig

    asyncio.run(go())
    return bridge, slept


def test_the_lobby_mode_is_unchanged():
    """Every existing run depends on this: no table field at all."""
    bridge, _ = _drive([])
    assert bridge.of("CONNECT") == [
        {"action": "CONNECT", "cookies": "x", "avoidLast": False}]


def test_the_host_asks_for_a_table_to_be_created():
    bridge, _ = _drive([], table_mode="create")
    request = bridge.of("CONNECT")[0]["table"]
    assert request["mode"] == "create"
    assert request["body"] == CREATE_TABLE_BODY


def test_the_guest_looks_at_the_lobby_before_joining_anything():
    """It must not fall back to the lobby pick: sitting at a stranger's table
    is exactly the outcome the pair exists to avoid."""
    bridge, _ = _drive([], table_mode="join", table_creator=HOST)
    assert bridge.of("LOBBY"), "should have asked what is listed"
    assert not bridge.of("CONNECT"), "must not join anything unseen"


def test_the_guest_joins_the_table_its_partner_created():
    bridge, _ = _drive(
        [{"event": "LOBBY", "mese": [SOMEONE_ELSE, OURS]}],
        table_mode="join", table_creator=HOST)

    joined = bridge.of("CONNECT")
    assert len(joined) == 1
    assert joined[0]["table"] == {"mode": "join", "tableId": "new-44709391"}


def test_the_guest_waits_and_asks_again_while_the_table_is_missing():
    """The host may still be creating it. A brief wait, then look again."""
    bridge, slept = _drive(
        [{"event": "LOBBY", "mese": [SOMEONE_ELSE]},
         {"event": "LOBBY", "mese": [SOMEONE_ELSE, OURS]}],
        table_mode="join", table_creator=HOST, rejoin_delay_s=5, join_poll_s=2)

    assert slept == [2], "the watch cadence, not the between-matches delay"
    assert len(bridge.of("LOBBY")) == 2, "should have looked again"
    assert bridge.of("CONNECT")[0]["table"]["tableId"] == "new-44709391"


def test_a_guest_never_falls_into_the_long_wait():
    """THE BUG, seen live: the host deleted a table while the guest was
    sitting down, so the guest's join failed with an error carrying no code --
    and slept for 300s. The host meanwhile churned through five tables looking
    for a partner that was unconscious. In join mode the table we want is
    created, filled and deleted on a timescale of seconds; nothing here
    deserves the five-minute wait."""
    bridge, slept = _drive(
        [{"event": "ERROR",
          "message": "Table new-1 vanished while we were sitting down."}],
        table_mode="join", table_creator=HOST,
        rejoin_delay_s=5, retry_delay_s=300)

    assert slept == [2], (
        "a failed join is a spent look: back on the watch cadence, and "
        "nowhere near the five-minute wait")


def test_a_vanished_table_is_told_apart_from_an_empty_lobby():
    """The bridge now labels it, so even the lobby pick retries promptly."""
    bridge, slept = _drive(
        [{"event": "ERROR", "code": "TABLE_NOT_FOUND",
          "message": "Table new-1 vanished while we were sitting down."}],
        rejoin_delay_s=5, retry_delay_s=300)

    assert slept == [5]


def test_an_empty_lobby_still_waits():
    """The long wait is for exactly one thing -- there is nothing to join --
    and a single-agent run must keep behaving that way."""
    bridge, slept = _drive(
        [{"event": "ERROR", "message": "No public open tables available."}],
        rejoin_delay_s=5, retry_delay_s=300)

    assert slept == [300]


def test_the_guest_looks_again_after_two_seconds():
    """A freshly created table is taken by strangers within a few seconds, so
    the watch has to be quicker than they are."""
    bridge, slept = _drive(
        [{"event": "LOBBY", "mese": [SOMEONE_ELSE]},
         {"event": "LOBBY", "mese": [SOMEONE_ELSE, OURS]}],
        table_mode="join", table_creator=HOST,
        join_poll_s=2, join_max_polls=20, pair_restart_pause_s=10)

    assert slept == [2]
    assert bridge.of("CONNECT")[0]["table"]["tableId"] == "new-44709391"


def test_a_table_that_filled_up_is_not_worth_watching():
    """It can never become ours now. Stand down, let the host delete it, and
    start the next attempt together rather than hammering the lobby."""
    bridge, slept = _drive([{"event": "LOBBY", "mese": [FULL]}],
                           table_mode="join", table_id="new-44711111",
                           join_poll_s=2, pair_restart_pause_s=10)

    assert slept == [10], "the pause both sides take before trying again"
    assert not bridge.of("CONNECT")


def test_the_guest_stands_down_once_its_budget_is_spent():
    bridge, slept = _drive(
        [{"event": "LOBBY", "mese": []} for _ in range(3)],
        table_mode="join", table_creator=HOST,
        join_poll_s=2, join_max_polls=3, pair_restart_pause_s=10)

    assert slept == [2, 2, 10], "three looks, then stand down"


def test_joining_restores_the_full_budget():
    """Otherwise a long-lived pairing would exhaust its looks and stand down
    in the middle of a perfectly good run of tables."""
    bridge, slept = _drive(
        [{"event": "LOBBY", "mese": []},
         {"event": "LOBBY", "mese": [OURS]},
         {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
         {"event": "LOBBY", "mese": []},
         {"event": "LOBBY", "mese": []}],
        table_mode="join", table_creator=HOST,
        join_poll_s=2, join_max_polls=2, pair_restart_pause_s=10)

    assert slept == [2, 2, 10], "the count restarted after the join"


def test_the_poll_settings_reach_the_client():
    """The knobs live on Config and are used by the client -- easy to add in
    one place and never wire up."""
    from belotmd.bot import LiveBelotBot
    from belotmd.config import Config

    bot = LiveBelotBot(config=Config(
        cookies="x", audit=False, agent="random", table_mode="join",
        table_creator=HOST, join_poll_s=1.5, join_max_polls=7,
        pair_restart_pause_s=11))

    assert bot.client.join_poll_s == 1.5
    assert bot.client.join_max_polls == 7
    assert bot.client.pair_restart_pause_s == 11.0


def test_a_rotated_guest_goes_straight_back_to_its_table():
    """THE BUG, seen live: the host rotates the seats, the guest is dropped
    (4005) and asks the lobby -- which reports the table FULL, because its own
    seat is one of the taken ones. The guest then stood down from its own
    table, and the host deleted a correctly seated table 30s later."""
    from belotmd.platform.protocol import LEAVE_POSITION_CHANGED

    bridge, _ = _drive(
        [{"event": "LOBBY", "mese": [OURS]},
         {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
         {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED}],
        table_mode="join", table_creator=HOST)

    joins = bridge.of("CONNECT")
    assert len(joins) == 2, "it must ask for the table again"
    assert joins[1]["table"] == {"mode": "join", "tableId": "new-44709391"}
    assert len(bridge.of("LOBBY")) == 1, (
        "asking the lobby again is what went wrong -- our own seat reads as "
        "taken")


def test_the_rejoin_note_is_used_once():
    """A later drop for some other reason must go back through the lobby."""
    from belotmd.platform.protocol import (LEAVE_POSITION_CHANGED,
                                           LEAVE_TABLE_REMOVED)

    bridge, _ = _drive(
        [{"event": "LOBBY", "mese": [OURS]},
         {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
         {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED},
         {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
         {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED}],
        table_mode="join", table_creator=HOST)

    assert len(bridge.of("LOBBY")) == 2, "the table is gone; look for it again"


def test_the_guest_does_not_squeeze_into_a_full_table():
    bridge, _ = _drive([{"event": "LOBBY", "mese": [FULL]}],
                       table_mode="join", table_id="new-44711111")
    assert not bridge.of("CONNECT")
