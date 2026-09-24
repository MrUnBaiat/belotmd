"""
test_bridge_lifetime.py -- the bridge lives exactly as long as its bot.

A supervisor stops a bot at the end of the MATCH by sending a break (Windows,
CTRL_BREAK_EVENT to the bot's process group) or an interrupt (a terminal's
Ctrl-C reaches the whole foreground group). The bot traps it and plays on to
the match's end. The bridge used to share the bot's group, so the same signal
killed it on the spot (exit 0xC000013A): the socket dropped and the table was
abandoned mid-match, which belot.md penalises.

So the bridge now has a process group of its own, and a pipe ties its life to
the bot's instead -- which must also hold when the bot is killed outright and
never gets to call close(), or a daemon would be left seated at the table.
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from belotmd.platform.client import BelotClient


# --------------------------------------------------------------- the spawn
class FakeLifeline:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePopen:
    def __init__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        self.stdin = FakeLifeline() if kw.get("stdin") is subprocess.PIPE else None
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


@pytest.fixture
def spawned(monkeypatch):
    calls = []

    def fake_popen(argv, *a, **kw):
        proc = FakePopen(argv, **kw)
        calls.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_the_bridge_gets_a_process_group_of_its_own(spawned):
    BelotClient(cookies="x")._start_node_daemon()
    kw = spawned[0].kw
    if os.name == "nt":
        assert kw.get("creationflags", 0) & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kw.get("start_new_session") is True


def test_the_bridge_is_tied_to_the_bot_by_a_pipe(spawned):
    BelotClient(cookies="x")._start_node_daemon()
    proc = spawned[0]
    assert proc.kw.get("stdin") is subprocess.PIPE
    assert proc.argv[4] == BelotClient.LIFELINE_ARG, "bridge.js must be told to watch it"


def test_close_ends_the_bridge_and_its_lifeline(spawned):
    client = BelotClient(cookies="x")
    client._start_node_daemon()
    client.close()
    assert spawned[0].terminated
    assert spawned[0].stdin.closed


def test_close_tolerates_a_process_without_a_pipe():
    client = BelotClient(cookies="x")

    class Bare:
        terminated = False

        def terminate(self):
            self.terminated = True

    client.node_process = Bare()
    client.close()
    assert client.node_process.terminated


# ------------------------------------------------------------- the real thing
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = os.path.join(ROOT, "node_modules")

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isdir(MODULES),
    reason="needs node and an installed node_modules")

# The bot's side, as a supervisor sees it: a process that traps the stop signal
# (as play_online does) and has started a real bridge through the SDK.
BOT = """
import signal, sys, time
from belotmd.platform.client import BelotClient
for name in ("SIGINT", "SIGBREAK"):
    sig = getattr(signal, name, None)
    if sig is not None:
        signal.signal(sig, lambda s, f: print("TRAPPED", s, flush=True))
client = BelotClient(cookies="x")
client._start_node_daemon()
print("BRIDGE", client.bridge_port, client.bridge_token, client.node_process.pid, flush=True)
while True:
    time.sleep(0.2)
"""


def _start_bot(tmp_path):
    script = tmp_path / "bot.py"
    script.write_text(BOT)
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    bot = subprocess.Popen([sys.executable, "-u", str(script)], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, **group)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = bot.stdout.readline()
        if line.startswith("BRIDGE"):
            _, port, token, pid = line.split()
            return bot, int(port), token, int(pid)
        if not line and bot.poll() is not None:
            break
    bot.kill()
    pytest.fail("the bot never started its bridge")


def _hello(port, timeout=15.0):
    """The bridge's token if it answers on `port` within `timeout`, else None."""
    import websockets

    async def once():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            return json.loads(await asyncio.wait_for(ws.recv(), 5))

    deadline = time.monotonic() + timeout
    while True:
        try:
            return asyncio.run(once()).get("token")
        except (OSError, asyncio.TimeoutError, Exception):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)


def _gone(port, timeout=15.0):
    """True once nothing accepts connections on `port` any more."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(0.25)
    return False


def _stop_signal(bot):
    """What play_pair (Windows) or a terminal's Ctrl-C (POSIX) sends the bot."""
    if os.name == "nt":
        bot.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(bot.pid, signal.SIGINT)


def _cleanup(bot, bridge_pid):
    if bot.poll() is None:
        bot.kill()
    bot.wait(timeout=10)
    try:
        os.kill(bridge_pid, signal.SIGTERM)
    except OSError:
        pass
    bot.stdout.close()


@needs_node
def test_a_stop_signal_for_the_bot_does_not_reach_its_bridge(tmp_path):
    bot, port, token, bridge_pid = _start_bot(tmp_path)
    try:
        assert _hello(port) == token, "the bridge never came up"
        _stop_signal(bot)
        time.sleep(1.5)
        assert bot.poll() is None, "the bot must trap the stop and play on"
        assert _hello(port, timeout=5) == token, (
            "the bridge died with the bot's stop signal -- the table would be "
            "abandoned mid-match")
    finally:
        _cleanup(bot, bridge_pid)


@needs_node
def test_the_bridge_exits_when_its_bot_is_killed(tmp_path):
    """No close(), no terminate(): only the pipe can end the bridge here."""
    bot, port, token, bridge_pid = _start_bot(tmp_path)
    try:
        assert _hello(port) == token, "the bridge never came up"
        bot.kill()
        bot.wait(timeout=10)
        assert _gone(port), "the bridge outlived its bot -- it would stay seated"
    finally:
        _cleanup(bot, bridge_pid)
