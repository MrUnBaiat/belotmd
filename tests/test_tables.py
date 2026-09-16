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
    """The host may still be creating it. A short wait, then look again."""
    bridge, slept = _drive(
        [{"event": "LOBBY", "mese": [SOMEONE_ELSE]},
         {"event": "LOBBY", "mese": [SOMEONE_ELSE, OURS]}],
        table_mode="join", table_creator=HOST, rejoin_delay_s=5)

    assert slept == [5], "short wait, not the five-minute empty-lobby one"
    assert len(bridge.of("LOBBY")) == 2, "should have looked again"
    assert bridge.of("CONNECT")[0]["table"]["tableId"] == "new-44709391"


def test_the_guest_does_not_squeeze_into_a_full_table():
    bridge, _ = _drive([{"event": "LOBBY", "mese": [FULL]}],
                       table_mode="join", table_id="new-44711111")
    assert not bridge.of("CONNECT")
