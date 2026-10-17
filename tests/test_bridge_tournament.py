"""
test_bridge_tournament.py -- the bridge's side of a tournament, end to end.

The shipped bridge.js runs under node with `fetch` replaced (fake_fetch.mjs):
every request it makes is logged and answered from a script, and none reaches
belot.md. What is checked is what only the bridge can get wrong:

  * the three tournament calls are the page's own form POSTs;
  * other players' names, which the page sends, never reach Python;
  * a tournament seat that is not there yet is an error -- never a fall back
    to the public lobby, which would sit an enrolled account at a stranger's
    table while its tournament goes on.
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import websockets

from belotmd.platform.client import BelotClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BRIDGE = ROOT / "src" / "belotmd" / "platform" / "bridge.js"
FAKE_FETCH = HERE / "fake_fetch.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (ROOT / "node_modules").is_dir(),
    reason="needs node and an installed node_modules")

PAGE = "https://belot.md/tournaments.php"
NAMES = ("SomePlayerName", "AnotherPlayer")

ENROLLED = {"playerState": "enrolled",
            "tournamentInfo": {"id": "1", "numPlayers": 4, "maxPlayers": 16}}
BRACKET = {"playerState": "playing", "phase": "1/4", "isAtTable": True,
           "progress": {"1/4": [{"playerIds": ["1", "2", "3", "4"],
                                 "score": [0, 0], "winner": -1}]},
           "playersInfo": {"1": {"player": NAMES[0], "status": "5"},
                           "2": {"player": NAMES[1], "status": "6"}},
           "latestTournaments": [{"id": "7", "players": {
               "9": {"player": NAMES[0], "status": "5"}}}]}


def _bridge(tmp_path, answers, requests):
    """Start the real bridge, send `requests` one by one, return (replies,
    the requests the bridge made)."""
    answers_file = tmp_path / "answers.json"
    answers_file.write_text(json.dumps(answers), encoding="utf-8")
    log = tmp_path / "requests.jsonl"
    log.write_text("", encoding="utf-8")
    port, token = BelotClient._free_port(), "t" * 32
    env = dict(os.environ, FAKE_FETCH_ANSWERS=str(answers_file),
               FAKE_FETCH_LOG=str(log))
    out = open(tmp_path / "bridge.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["node", "--import", FAKE_FETCH.as_uri(), str(BRIDGE), str(port), token],
        env=env, stdout=out, stderr=subprocess.STDOUT)

    async def talk():
        for _ in range(200):
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{port}",
                                              ping_interval=None)
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            raise AssertionError("the bridge never started listening")
        async with ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 10))
            assert hello == {"event": "HELLO", "token": token}
            replies = []
            for request in requests:
                await ws.send(json.dumps(request))
                replies.append(await asyncio.wait_for(ws.recv(), 20))
            return replies

    try:
        raw = asyncio.run(talk())
    finally:
        proc.terminate()
        proc.wait(10)
        out.close()
    made = [json.loads(line) for line in
            log.read_text(encoding="utf-8").splitlines() if line]
    return raw, made


def _tour(op, **extra):
    return {"action": "TOURNAMENT", "op": op, "cookies": "PHPSESSID=x", **extra}


def test_the_three_calls_are_the_pages_own_form_posts(tmp_path):
    answers = [["getState=1", 200, json.dumps(ENROLLED)],
               ["enroll=", 200, json.dumps({"success": True})],
               ["withdraw=1", 200, json.dumps({"success": True})]]
    raw, made = _bridge(tmp_path, answers,
                        [_tour("state"), _tour("enroll", id="1"),
                         _tour("withdraw")])
    replies = [json.loads(r) for r in raw]

    assert [r["event"] for r in replies] == ["TOURNAMENT"] * 3
    assert [r["op"] for r in replies] == ["state", "enroll", "withdraw"]
    assert replies[0]["data"] == ENROLLED
    assert replies[1]["data"] == {"success": True}

    assert [m["url"] for m in made] == [PAGE] * 3
    assert [m["method"] for m in made] == ["POST"] * 3
    assert [m["body"] for m in made] == ["getState=1", "enroll=1", "withdraw=1"]
    # URLSearchParams is what makes fetch send it as a form; a plain string
    # goes as text/plain, and PHP then reads no fields at all.
    assert all(m["form"] for m in made)
    for m in made:
        assert m["headers"]["Cookie"] == "PHPSESSID=x"
        assert m["headers"]["X-Requested-With"] == "XMLHttpRequest"


def test_other_players_names_never_leave_the_bridge(tmp_path):
    raw, _ = _bridge(tmp_path, [["getState=1", 200, json.dumps(BRACKET)]],
                     [_tour("state")])
    reply = json.loads(raw[0])
    assert "playersInfo" not in reply["data"]
    assert "latestTournaments" not in reply["data"]
    assert not [n for n in NAMES if n in raw[0]], "a name reached Python"
    # The control: everything else arrives intact, ids and all -- the client
    # needs its own id in the bracket to find its table.
    assert reply["data"]["progress"] == BRACKET["progress"]
    assert reply["data"]["isAtTable"] is True


def test_a_page_that_is_not_json_is_an_error_without_its_text(tmp_path):
    """A logged-out session gets the site's HTML, which can carry names."""
    page = f"<html><body>Bine ai venit, {NAMES[0]}</body></html>"
    raw, _ = _bridge(tmp_path, [["getState=1", 200, page]], [_tour("state")])
    reply = json.loads(raw[0])
    assert "data" not in reply
    assert "not JSON" in reply["error"]
    assert NAMES[0] not in raw[0]


def test_an_unknown_call_is_refused_without_a_request(tmp_path):
    raw, made = _bridge(tmp_path, [], [_tour("deleteEverything")])
    assert "unknown tournament call" in json.loads(raw[0])["error"]
    assert made == []


NO_GAME = [["gameplay_new.php", 200, "<html>no game here</html>"],
           ["gameTables.php", 200, json.dumps({"mese": []})]]


def test_a_tournament_seat_never_falls_back_to_the_lobby(tmp_path):
    """THE ACCIDENT: gameplay_new.php has no room on it yet, and the bridge
    goes looking for any open public table instead."""
    raw, made = _bridge(tmp_path, NO_GAME,
                        [{"action": "CONNECT", "cookies": "PHPSESSID=x",
                          "table": {"mode": "tournament"}}])
    reply = json.loads(raw[0])
    assert reply["event"] == "ERROR"
    assert reply["code"] == "TOURNAMENT_NO_ROOM"
    urls = [m["url"] for m in made]
    assert urls == ["https://belot.md/gameplay_new.php"]


def test_without_the_tournament_mode_the_lobby_is_asked(tmp_path):
    """The control for the test above: the same empty page sends an ordinary
    CONNECT to the lobby, so a lobby request would have been seen there."""
    _, made = _bridge(tmp_path, NO_GAME,
                      [{"action": "CONNECT", "cookies": "PHPSESSID=x"}])
    assert "https://belot.md/gameTables.php" in [m["url"] for m in made]
