"""
test_bridge_isolation.py — one bridge per client, and proving it is ours.

The bridge used to listen on a hardcoded 8765. That is fine for one bot and
quietly dangerous for two: the second daemon cannot bind, so the second client
connects to the FIRST one's bridge and is served another account's STATE
frames -- hand included -- while sending its own cookies down the same socket.

Two accounts now play the same table as partners, so this is the exact accident
that must be impossible. Each client picks its own port and proves the daemon
that answers is the one it started.
"""

import asyncio
import json
import os
import shutil
import subprocess

import pytest

from belotmd.platform.client import BelotClient


# ------------------------------------------------------------ the daemon
class FakePopen:
    """Stands in for `node bridge.js`: alive, and remembers its argv."""

    def __init__(self, argv):
        self.argv = argv

    def poll(self):
        return None

    def terminate(self):
        pass


@pytest.fixture
def spawned(monkeypatch):
    calls = []

    def fake_popen(argv, *a, **kw):
        calls.append(argv)
        return FakePopen(argv)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_the_daemon_is_told_its_port_and_token(spawned):
    client = BelotClient(cookies="x")
    client._start_node_daemon()

    argv = spawned[0]
    assert argv[0] == "node"
    assert argv[1].endswith("bridge.js")
    assert argv[2] == str(client.bridge_port)
    assert argv[3] == client.bridge_token
    assert client.bridge_port != 0, "a port must have been chosen"


def test_two_clients_never_share_a_port_or_a_token(spawned):
    """The whole point: two accounts on one machine must not meet."""
    a, b = BelotClient(cookies="a"), BelotClient(cookies="b")
    a._start_node_daemon()
    b._start_node_daemon()

    assert a.bridge_port != b.bridge_port
    assert a.bridge_token != b.bridge_token
    assert len(spawned) == 2, "each client starts its own daemon"


def test_an_explicit_port_is_respected(spawned):
    client = BelotClient(cookies="x", bridge_port=9999)
    client._start_node_daemon()
    assert spawned[0][2] == "9999"


# ------------------------------------------------------- the handshake
class FakeWS:
    """A bridge that says `first` (or nothing at all) and can be closed."""

    def __init__(self, first=None, silent=False):
        self.first = first
        self.silent = silent
        self.closed = False

    async def recv(self):
        if self.silent:
            await asyncio.sleep(60)          # never answers
        return json.dumps(self.first)

    async def close(self):
        self.closed = True


def _verify(ws, client=None):
    client = client or BelotClient(cookies="x")
    client.bridge_port = 1234
    client.HELLO_TIMEOUT_S = 0.05
    asyncio.run(client._verify_bridge(ws))
    return client


def test_our_own_daemon_is_accepted():
    client = BelotClient(cookies="x")
    ws = FakeWS({"event": "HELLO", "token": client.bridge_token})
    _verify(ws, client)
    assert not ws.closed, "our own bridge must not be hung up on"


def test_another_clients_daemon_is_refused():
    """The 8765 collision, reproduced: the daemon answering is somebody
    else's. Continuing would send this account's cookies down it."""
    ws = FakeWS({"event": "HELLO", "token": "some-other-clients-token"})
    with pytest.raises(RuntimeError, match="not ours"):
        _verify(ws)
    assert ws.closed, "must hang up before sending anything"


def test_a_daemon_that_never_identifies_itself_is_refused():
    """An old bridge.js, or an unrelated program on the port."""
    ws = FakeWS(silent=True)
    with pytest.raises(RuntimeError, match="did not identify"):
        _verify(ws)
    assert ws.closed


def test_a_first_frame_that_is_not_a_hello_is_refused():
    ws = FakeWS({"event": "STATE", "data": {}})
    with pytest.raises(RuntimeError, match="not ours"):
        _verify(ws)
    assert ws.closed


def test_the_handshake_guards_the_connect_path(monkeypatch):
    """_await_daemon must not hand back a socket it has not checked."""
    client = BelotClient(cookies="x")
    client.bridge_port = 4321
    client.HELLO_TIMEOUT_S = 0.05
    client.DAEMON_TIMEOUT_S = 0.2
    client.node_process = FakePopen(["node"])

    ws = FakeWS({"event": "HELLO", "token": "not-ours"})

    async def fake_connect(uri):
        return ws

    monkeypatch.setattr("belotmd.platform.client.websockets.connect", fake_connect)
    with pytest.raises(RuntimeError, match="not ours"):
        asyncio.run(client._await_daemon("ws://127.0.0.1:4321"))


# ------------------------------------------------------- the real thing
BRIDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "belotmd", "platform", "bridge.js")
MODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "node_modules")

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isdir(MODULES),
    reason="needs node and an installed node_modules")


@needs_node
def test_two_real_bridges_each_answer_only_their_own_client():
    """End to end, with two actual daemons: the property the fakes above
    describe has to hold for the file we ship."""
    import websockets

    a, b = BelotClient(cookies="a"), BelotClient(cookies="b")

    async def go():
        a._start_node_daemon()
        b._start_node_daemon()
        try:
            # Each client's own daemon answers with its own token: if the two
            # had shared a bridge, one of these would raise.
            ws_a = await a._await_daemon(f"ws://127.0.0.1:{a.bridge_port}")
            ws_b = await b._await_daemon(f"ws://127.0.0.1:{b.bridge_port}")
            await ws_a.close()

            # A second agent on B's port is refused rather than served. The
            # refusal comes INSTEAD of the HELLO, not after it: nothing about
            # this bridge is revealed to a client it will not serve.
            async with websockets.connect(f"ws://127.0.0.1:{b.bridge_port}") as intruder:
                first = json.loads(await intruder.recv())
            await ws_b.close()
            return first
        finally:
            a.close()
            b.close()

    first = asyncio.run(go())
    assert first["event"] == "ERROR", "a second Python agent must be refused"
    assert a.bridge_port != b.bridge_port
