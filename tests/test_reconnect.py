"""
test_reconnect.py — staying on a table, and getting back to one.

A long unattended run spends most of its life *between* tables: matches end,
tables dissolve, hosts shuffle seats, and sometimes the lobby is simply empty.
Observed live: the bot played a few matches, hit "No public open tables
available.", and then sat idle forever waiting for frames from a room it had
never entered.

These drive the real recovery paths in `BelotClient` against a fake bridge, so
the delays and the rejoin decisions are exercised rather than assumed.
"""

import asyncio
import json

import pytest

from belotmd.platform.client import BelotClient
from belotmd.platform.protocol import (LEAVE_I_LEFT, LEAVE_KICKED,
                                       LEAVE_OTHER_SESSION,
                                       LEAVE_POSITION_CHANGED,
                                       LEAVE_TABLE_REMOVED, RECOVER_LATER,
                                       RECOVER_NOW, RECOVER_SOON, RECOVER_STOP,
                                       leave_recovery)


# --------------------------------------------------------------- policy
# The platform's own enum (read out of gameplay.js):
#   4001 OTHER_SESSION  4002 KICKED  4003 TABLE_REMOVED
#   4004 I_LEFT         4005 POSITION_CHANGED

@pytest.mark.parametrize("code,expected,why", [
    (LEAVE_OTHER_SESSION, RECOVER_STOP,
     "rejoining would kick the other session, which kicks us back, forever"),
    (LEAVE_KICKED, RECOVER_SOON,
     "safe to look again quickly because the bridge now skips that table"),
    (LEAVE_TABLE_REMOVED, RECOVER_SOON,
     "the normal end of every match -- go find another table"),
    (LEAVE_I_LEFT, RECOVER_STOP,
     "we asked to leave"),
    (LEAVE_POSITION_CHANGED, RECOVER_NOW,
     "nobody was removed; only seat indices moved, so resync at once"),
])
def test_leave_code_policy(code, expected, why):
    assert leave_recovery(code) == expected, why


def test_an_unknown_code_keeps_an_overnight_run_alive():
    assert leave_recovery(4099) == RECOVER_LATER


# ----------------------------------------------------------- the loop
class FakeBridge:
    """Stands in for bridge.js: replays a script, records what we send."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.joins = 0

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg.get("action") == "CONNECT":
            self.joins += 1

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.script:
            raise StopAsyncIteration
        return json.dumps(self.script.pop(0))


async def _drive(script, **kwargs):
    """Run the client's message loop over a scripted bridge."""
    client = BelotClient(cookies="x", **kwargs)
    bridge = FakeBridge(script)
    slept = []

    async def no_sleep(d):
        slept.append(d)

    orig = asyncio.sleep
    asyncio.sleep = no_sleep          # keep the tests instant
    try:
        client._queue = asyncio.Queue()
        await client._request_table(bridge)
        async for raw in bridge:
            msg = json.loads(raw)
            event = msg.get("event")
            if event == "CONNECTED":
                client._in_room = True
                client.sessions_played += 1
            elif event == "ERROR":
                if not client._in_room:
                    if not await client._recover(
                            bridge, RECOVER_LATER, "join failed"):
                        break
            elif event == "LEAVE":
                client._in_room = False
                if not await client._recover(
                        bridge, leave_recovery(msg.get("code")), "left"):
                    break
    finally:
        asyncio.sleep = orig
    return bridge, slept


def test_no_open_tables_retries_instead_of_sitting_idle():
    """The bug this fixes: an ERROR before we ever entered a room left the bot
    waiting forever for frames that could not arrive."""
    script = [
        {"event": "ERROR", "message": "No public open tables available."},
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
    ]
    bridge, slept = asyncio.run(_drive(script, retry_delay_s=300))
    assert bridge.joins == 2, "should have asked for another table"
    assert slept == [300], "should have waited the long delay first"


def test_being_kicked_looks_elsewhere_rather_than_walking_back_in():
    """The lobby pick takes the FIRST open table, and the one that ejected us
    has a free seat again -- so it is a prime candidate to be handed straight
    back. The bridge is told to skip it, which is what makes a short pause
    safe rather than a kick loop five times faster."""
    ws, slept, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_KICKED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ], rejoin_delay_s=5)
    assert ws.joins == 2
    assert slept == [5], "short pause, not the five-minute one"
    rejoin = [m for m in ws.sent if m.get("action") == "CONNECT"][-1]
    assert rejoin["avoidLast"] is True, "must not be offered that table again"


def test_only_a_kick_asks_the_bridge_to_skip_a_table():
    """A table dissolving at the end of a match is not a reason to avoid it."""
    ws, _, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ])
    rejoin = [m for m in ws.sent if m.get("action") == "CONNECT"][-1]
    assert rejoin["avoidLast"] is False


def test_a_table_dissolving_goes_straight_back_out():
    """TABLE_REMOVED is how every match ends -- it is not a failure."""
    script = [
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ]
    bridge, slept = asyncio.run(_drive(script, rejoin_delay_s=5))
    assert bridge.joins == 2
    assert slept == [5], "short delay, not the five-minute one"


def test_a_seat_shuffle_rejoins_with_no_delay():
    script = [
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED},
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
    ]
    bridge, slept = asyncio.run(_drive(script))
    assert bridge.joins == 2
    assert slept == [], "nobody was removed; no reason to wait"


@pytest.mark.parametrize("code", [LEAVE_OTHER_SESSION, LEAVE_I_LEFT])
def test_terminal_codes_stop_the_session(code):
    script = [
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": code},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ]
    bridge, slept = asyncio.run(_drive(script))
    assert bridge.joins == 1, "must not rejoin"
    assert slept == []


def test_once_disables_every_recovery():
    script = [
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ]
    bridge, slept = asyncio.run(_drive(script, reconnect=False))
    assert bridge.joins == 1
    assert slept == []


def test_sessions_are_counted_across_tables():
    script = [
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r3", "playerId": "p"},
    ]
    bridge, _ = asyncio.run(_drive(script))
    assert bridge.joins == 3


# ------------------------------------- the real connect() loop, end to end
class _FakeWS(FakeBridge):
    """A FakeBridge that also satisfies `async with`."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _run_connect(script, **kwargs):
    """Drive BelotClient.connect() itself, with the daemon stubbed out."""
    client = BelotClient(cookies="x", **kwargs)
    ws = _FakeWS(script)
    client._start_node_daemon = lambda: None

    async def fake_daemon(uri):
        return ws
    client._await_daemon = fake_daemon

    sent_types = []

    async def fake_send(action_type, payload):
        sent_types.append(action_type)
    client._send = fake_send

    slept = []
    orig = asyncio.sleep

    async def no_sleep(d):
        slept.append(d)

    async def go():
        asyncio.sleep = no_sleep
        try:
            await client.connect(on_state_callback=_noop)
        finally:
            asyncio.sleep = orig

    asyncio.run(go())
    return ws, slept, sent_types


async def _noop(*a, **kw):
    pass


def test_connect_itself_retries_when_the_lobby_is_empty():
    """End to end through the real loop -- not a reimplementation of it."""
    ws, slept, sent = _run_connect([
        {"event": "ERROR", "message": "No public open tables available."},
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_TABLE_REMOVED},
        {"event": "CONNECTED", "roomId": "r2", "playerId": "p"},
    ], retry_delay_s=300, rejoin_delay_s=5)

    assert ws.joins == 3, "one initial join plus two recoveries"
    assert slept == [300, 5], "long wait for an empty lobby, short after a match"
    assert sent.count("BOT_ACTIVATION") == 2, "must deactivate on every join"


def test_connect_stops_on_a_terminal_code():
    ws, slept, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "LEAVE", "code": LEAVE_OTHER_SESSION},
    ])
    assert ws.joins == 1
    assert slept == []


def test_a_seat_shuffle_before_the_match_is_routine():
    """4005 happens between joining a table and the match starting, when the
    host rotates players to set up teams. Nobody is removed."""
    ws, slept, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "STATE", "data": {"currentPhase": 0}},      # still the lobby
        {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED},
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
    ])
    assert ws.joins == 2
    assert slept == [], "no reason to wait; nothing was lost"


def test_a_seat_shuffle_after_the_cut_is_flagged_as_impossible(capsys):
    """The match is underway from the deck cut (phase 2), not the first bid.
    A shuffle there cannot happen -- if it ever does, it must be loud."""
    ws, _, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "STATE", "data": {"currentPhase": 2}},      # cut: underway
        {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED},
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
    ])
    out = capsys.readouterr().out
    assert "VIOLATION" in out
    assert "not supposed to be possible" in out
    assert ws.joins == 2, "still resyncs rather than ending the run"


def test_the_match_started_flag_survives_a_cancelled_deal():
    """A cancelled deal drops back through phases 1-2; that is not a return
    to the lobby, so a later shuffle must still read as impossible."""
    ws, _, _ = _run_connect([
        {"event": "CONNECTED", "roomId": "r", "playerId": "p"},
        {"event": "STATE", "data": {"currentPhase": 10}},
        {"event": "STATE", "data": {"currentPhase": 1}},      # deal cancelled
        {"event": "STATE", "data": {"currentPhase": 2}},
        {"event": "LEAVE", "code": LEAVE_POSITION_CHANGED},
    ])
    assert ws.joins == 2
