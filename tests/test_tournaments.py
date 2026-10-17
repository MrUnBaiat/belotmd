"""
test_tournaments.py -- playing one belot.md tournament.

A tournament is sixteen players, random teams of two, three knockout rounds.
The site seats us at each round's table by itself; we only enrol, wait, sit
where it says, and withdraw once we are out. The page answers that drive all
of it were captured from a tournament played by hand on 2026-10-17 (shapes
only: every id below is synthetic).

Three things must never happen, and each has a test that would see it:
  * an enrolled account sitting down at a public table -- the lobby is never
    asked while a tournament is on;
  * a stop that walks out of the bracket mid-tournament, leaving a partner the
    draw gave us; only the waiting room is left at once;
  * a second tournament in one run.
"""

import asyncio
import json

import pytest

from belotmd.bot import LiveBelotBot
from belotmd.config import Config
from belotmd.platform import tournaments as tour
from belotmd.platform.client import BelotClient, TournamentError
from belotmd.platform.protocol import GAME_TYPE_TOURNAMENT

ME = "1000003"                                    # our in-room player id
OURS = [ME, "2000001", "2000002", "2000003"]      # we sit at index 0: team 0
OTHER_TABLES = [[f"30{i}000{j}" for j in range(4)] for i in range(3)]
OTHER_IDS = {p for ids in OTHER_TABLES for p in ids} | set(OURS[1:])
OK = {"success": True}


# ------------------------------------------------------------- page answers
def _slot(id_, min_status=5, n=0, cap=16, stake="200"):
    """One entry of `tournaments`, as the list shows it."""
    return {"id": str(id_), "stake": stake, "prizes": [1000, 400],
            "numPlayers": n, "maxPlayers": cap, "minStatus": min_status,
            "timeWaiting": 0}


def _offer(*slots):
    return {"playerState": "none", "tournaments": list(slots)}


def _waiting(n=1):
    return {"playerState": "enrolled",
            "tournamentInfo": {"id": "1", "stake": "200", "prizes": [1000, 400],
                               "numPlayers": n, "maxPlayers": 16,
                               "minStatus": 5, "timeWaiting": 0,
                               "estTimeStart": 145},
            "suitGameStake": 10}


def _table(ids, score=(0, 0), winner=-1):
    return {"playerIds": list(ids), "score": list(score), "winner": winner,
            "gameTableId": 46000000}


def _bracket(phase="1/4", at_table=False, eliminated=False, ours=None,
             semi=None, final=None, winners=None):
    """`playerState: playing`. Finished tables carry their scores as strings,
    exactly as captured."""
    progress = {"1/4": [_table(ids) for ids in OTHER_TABLES]
                + [ours if ours is not None else _table(OURS)]}
    if semi is not None:
        progress["semifinal"] = semi
    if final is not None:
        progress["final"] = final
    state = {"playerState": "playing", "phase": phase, "progress": progress,
             "isEliminated": eliminated, "prizes": [1000, 400, 20],
             "tournamentId": "900001", "duration": 3, "suitGameStake": 10}
    if at_table:
        state["isAtTable"] = True
    if winners is not None:
        state["winners"] = winners
    return state


LOST_QF = _table(OURS, score=("101", "111"), winner=1)
WON_QF = _table(OURS, score=("103", "50"), winner=0)
SEMI = [_table([ME, "2000004", "2000005", "2000006"]),
        _table(["4000001", "4000002", "4000003", "4000004"])]


def _seated(*phases, leave=4004):
    """What one CONNECT meets: the room, a frame per phase, then the close."""
    return ([{"event": "CONNECTED", "roomId": "room-1", "playerId": ME}]
            + [{"event": "STATE", "data": {"currentPhase": p}} for p in phases]
            + ([{"event": "LEAVE", "code": leave}] if leave else []))


# --------------------------------------------------------------- the site
class Site:
    """tournaments.php and the game server, as the client meets them through
    its bridge: every request gets the next answer the script has for it.
    When the script runs out nothing more is said, and the session ends --
    so no test here can poll for ever."""

    def __init__(self, states, *, enroll=OK, withdraw=OK, seats=(),
                 on_state=None):
        self.states = list(states)
        self.enroll, self.withdraw = enroll, withdraw
        self.seats = list(seats)
        self.on_state = on_state          # hook(n, state) as the n-th is served
        self.sent, self.events, self.served = [], [], 0
        self.client = None

    async def send(self, raw):
        request = json.loads(raw)
        self.sent.append(request)
        self.events.extend(self._answer(request))

    def _answer(self, request):
        if request["action"] == "TOURNAMENT":
            op = request["op"]
            if op == "state":
                if not self.states:
                    return []
                state = self.states.pop(0)
                if self.on_state:
                    self.on_state(self.served, state)
                self.served += 1
                if isinstance(state, str):                 # a failed call
                    return [{"event": "TOURNAMENT", "op": op, "error": state}]
                return [{"event": "TOURNAMENT", "op": op, "data": state}]
            answer = self.enroll if op == "enroll" else self.withdraw
            return [{"event": "TOURNAMENT", "op": op, "data": answer}]
        if request["action"] == "CONNECT":
            return self.seats.pop(0) if self.seats else []
        return []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return json.dumps(self.events.pop(0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def steps(self):
        """The requests, in order, in a form a test can read."""
        out = []
        for r in self.sent:
            if r["action"] == "TOURNAMENT":
                out.append(r["op"] + (f":{r['id']}" if "id" in r else ""))
            elif r["action"] == "CONNECT":
                out.append("CONNECT:" + (r.get("table") or {}).get("mode", "-"))
            else:
                out.append(r["action"])
        return out


async def _noop(*a, **kw):
    pass


def _drive(site, raises=None, stop=False, my_id=None, **kwargs):
    """Drive the REAL `connect()` loop against the site. -> (client, sleeps)

    `sleeps` lists (requests sent so far, seconds) for every pause, so a test
    can say which request waited and for how long. `my_id`: our player id is
    already known -- we sat at a table earlier in this session."""
    kwargs.setdefault("table_mode", "tournament")
    client = BelotClient(cookies="x", **kwargs)
    client.stop_requested = stop
    client.my_player_id = my_id
    site.client = client
    client._start_node_daemon = lambda: None

    async def fake_daemon(uri):
        return site
    client._await_daemon = fake_daemon

    async def fake_send(action_type, payload):
        pass
    client._send = fake_send

    slept = []

    async def no_sleep(d):
        slept.append((len(site.sent), d))

    async def go():
        orig, asyncio.sleep = asyncio.sleep, no_sleep
        try:
            await client.connect(on_state_callback=_noop)
        finally:
            asyncio.sleep = orig

    if raises is None:
        asyncio.run(go())
    else:
        with pytest.raises(raises) as caught:
            asyncio.run(go())
        client.raised = caught.value
    return client, slept


# ----------------------------------------------------- choosing a tournament
def test_only_a_tournament_for_exactly_our_status_is_chosen():
    """Not one we cannot enter (6), nor one meant for weaker players (4)."""
    offer = [_slot(1, min_status=4, n=15), _slot(2, min_status=6, n=15),
             _slot(3, min_status=5, n=2)]
    assert tour.pick_tournament(offer, 5)["id"] == "3"


def test_the_fullest_tournament_is_chosen_since_it_starts_soonest():
    offer = [_slot(1, n=2), _slot(2, n=9), _slot(3, n=4)]
    assert tour.pick_tournament(offer, 5)["id"] == "2"


def test_a_full_tournament_is_passed_over():
    offer = [_slot(1, n=16), _slot(2, n=3)]
    assert tour.pick_tournament(offer, 5)["id"] == "2"
    assert tour.pick_tournament([_slot(1, n=16)], 5) is None
    assert tour.offers_status([_slot(1, n=16)], 5), "full is not absent"


def test_ties_go_to_the_lowest_id_so_the_choice_is_stable():
    offer = [_slot(7, n=3), _slot(2, n=3), _slot(10, n=3)]
    assert tour.pick_tournament(offer, 5)["id"] == "2"


def test_numbers_sent_as_strings_are_read_as_numbers():
    offer = [{"id": "4", "minStatus": "5", "numPlayers": "8", "maxPlayers": "16"},
             {"id": "5", "minStatus": 5, "numPlayers": 3, "maxPlayers": 16}]
    assert tour.pick_tournament(offer, 5)["id"] == "4"


def test_nothing_on_offer_is_not_an_error_here():
    assert tour.pick_tournament([], 5) is None
    assert tour.pick_tournament(None, 5) is None
    assert not tour.offers_status([_slot(1, min_status=6)], 5)


# ------------------------------------------------------- reading the bracket
def test_our_table_is_found_by_our_player_id_and_scored_for_our_team():
    """The captured shape: a finished table, scores as strings, winner 1.
    We sit at index 0, so we are team 0 and lost 101-111."""
    state = _bracket(phase="semifinal", eliminated=True, ours=LOST_QF)
    assert tour.our_table(state, ME) == {"stage": "1/4", "us": 101,
                                         "them": 111, "won": False}


def test_the_team_follows_the_seat_index():
    """Indexes 0+2 against 1+3, the seat pairing of the room itself."""
    ids = ["2000001", ME, "2000002", "2000003"]           # index 1: team 1
    state = _bracket(ours=_table(ids, score=("101", "111"), winner=1))
    assert tour.our_table(state, ME) == {"stage": "1/4", "us": 111,
                                         "them": 101, "won": True}


def test_a_table_still_being_played_has_no_winner():
    assert tour.our_table(_bracket(), ME)["won"] is None


def test_every_stage_we_reached_is_kept():
    state = _bracket(phase="semifinal", ours=WON_QF, semi=SEMI)
    tables = tour.our_tables(state, ME)
    assert tables["1/4"] == {"us": 103, "them": 50, "won": True}
    assert tables["semifinal"]["won"] is None
    assert tour.our_table(state, ME)["stage"] == "semifinal"


def test_without_our_id_or_a_place_in_the_bracket_there_is_no_table():
    assert tour.our_table(_bracket(), None) is None
    assert tour.our_table(_bracket(), "9999999") is None
    assert tour.our_tables({"playerState": "playing"}, ME) == {}


def test_the_placing_comes_from_the_winners_list():
    """Indexes 0-1 won, 2-3 came second."""
    won = _bracket(phase="finished", winners=["5", ME, "6", "7"])
    second = _bracket(phase="finished", winners=["5", "6", ME, "7"])
    out = _bracket(phase="finished", winners=["5", "6", "7", "8"])
    assert tour.placing(won, ME) == "won"
    assert tour.placing(second, ME) == "second"
    assert tour.placing(out, ME) is None
    assert tour.placing(won, None) is None


def test_out_means_eliminated_or_finished():
    assert tour.is_out(_bracket(eliminated=True))
    assert tour.is_out(_bracket(phase="finished"))
    assert not tour.is_out(_bracket(phase="semifinal"))


# ------------------------------------------------------ the whole tournament
def test_a_tournament_from_enrolment_to_elimination():
    """The run captured by hand, end to end: enrol, wait, sit at the table
    the site gives us, lose the quarterfinal, withdraw from the bracket."""
    site = Site(
        [_offer(_slot(1, n=3)), _waiting(4), _waiting(16),
         _bracket(at_table=True),
         _bracket(phase="semifinal", eliminated=True, ours=LOST_QF),
         {"playerState": "none", "tournaments": [_slot(1)]}],
        seats=[_seated(0, 10, 14)])
    client, _ = _drive(site)

    assert site.steps() == ["state", "enroll:1", "state", "state", "state",
                            "CONNECT:tournament", "state", "withdraw",
                            "state"]
    assert client.tournament_over == "eliminated in the 1/4 (101-111)"
    assert client.tournament["stages"]["1/4"] == {"us": 101, "them": 111,
                                                  "won": False}
    assert client.tournament["id"] == "900001"
    assert client.tournament["slot"] == "1"
    assert client.tournament["withdrew"] is True
    assert client.fatal is None


def test_a_tournament_run_never_asks_the_lobby():
    """THE ACCIDENT: an enrolled account sitting down at a stranger's public
    table while its tournament goes on without it."""
    site = Site(
        [_offer(_slot(1)), _waiting(), _bracket(at_table=True),
         _bracket(), _bracket(phase="semifinal", at_table=True, ours=WON_QF,
                              semi=SEMI),
         _bracket(phase="semifinal", eliminated=True, ours=WON_QF,
                  semi=[_table(SEMI[0]["playerIds"], ("90", "102"), 1),
                        SEMI[1]]),
         _offer()],
        seats=[_seated(0, 10, 14), _seated(0, 10, 14, leave=4003)])
    client, _ = _drive(site)
    steps = site.steps()
    assert "LOBBY" not in steps
    assert [s for s in steps if s.startswith("CONNECT")] == [
        "CONNECT:tournament"] * 2
    assert client.tournament_over == "eliminated in the semifinal (90-102)"
    assert client.tournament["stages"]["1/4"]["won"] is True


def test_winning_the_tournament_is_reported_as_such():
    final = [_table([ME, "2000007", "2000008", "2000009"], ("120", "80"), 0)]
    site = Site(
        [_bracket(phase="final", at_table=True, ours=WON_QF, semi=SEMI,
                  final=[_table(final[0]["playerIds"])]),
         _bracket(phase="finished", ours=WON_QF, semi=SEMI, final=final,
                  winners=["2000008", ME, "4000001", "4000002"]),
         _offer()],
        seats=[_seated(0, 10, 14)])
    client, _ = _drive(site)
    assert client.tournament_over == "won the tournament"
    assert client.tournament["stages"]["final"]["won"] is True


def test_coming_second_is_reported_as_such():
    site = Site([_bracket(phase="finished", ours=WON_QF,
                          winners=["5", "6", ME, "7"]), _offer()])
    client, _ = _drive(site, my_id=ME)
    assert client.tournament_over == "second in the tournament"


def test_waiting_polls_at_the_sites_own_pace():
    """The site's page asks every 3 s. Faster is rude; slower risks the turn
    clock on a table that has already started."""
    site = Site([_offer(_slot(1)), _waiting(2), _waiting(3), _waiting(4)])
    _, slept = _drive(site)
    pauses = [d for _, d in slept]
    assert pauses and all(d == tour.POLL_S == 3.0 for d in pauses)


def test_one_request_at_a_time():
    """Every answer leads to exactly one next request: two polling chains
    would mean two enrolments, or a withdrawal racing a seat."""
    site = Site([_offer(_slot(1)), _waiting(), _bracket(at_table=True),
                 _bracket(eliminated=True, ours=LOST_QF), _offer()],
                seats=[_seated(0, 10, 14)])
    _drive(site)
    assert site.steps().count("enroll:1") == 1
    assert site.steps().count("withdraw") == 1


def test_other_players_never_reach_the_log(capsys):
    """The bracket carries every player's id. Only our own table is read,
    and nothing about anyone else is printed."""
    site = Site([_offer(_slot(1)), _waiting(), _bracket(at_table=True),
                 _bracket(phase="semifinal", eliminated=True, ours=LOST_QF,
                          semi=SEMI), _offer()],
                seats=[_seated(0, 10, 14)])
    client, _ = _drive(site)
    out = capsys.readouterr().out
    assert client.tournament_over
    assert not [p for p in OTHER_IDS if p in out]
    assert not [p for p in OTHER_IDS if p in json.dumps(client.tournament)]


# ------------------------------------------------------------------ stopping
def _stop_when(n):
    def hook(served, state):
        if served == n:
            hook.site.client.stop_requested = True
    return hook


def test_a_stop_in_the_waiting_room_withdraws_at_once():
    hook = _stop_when(2)                 # as the second waiting answer arrives
    site = Site([_offer(_slot(1)), _waiting(2), _waiting(3), _offer()],
                on_state=hook)
    hook.site = site
    client, _ = _drive(site)
    assert site.steps() == ["state", "enroll:1", "state", "state", "withdraw",
                            "state"]
    assert client.tournament_over == "withdrew from the waiting room before the start"
    assert not [s for s in site.steps() if s.startswith("CONNECT")]


def test_without_a_stop_the_waiting_room_is_simply_polled():
    """The control for the test above: the same answers, no stop, no
    withdrawal."""
    site = Site([_offer(_slot(1)), _waiting(2), _waiting(3)])
    _drive(site)
    assert "withdraw" not in site.steps()


def test_a_stop_in_the_bracket_waits_until_we_are_out():
    """Walking out mid-tournament abandons the partner the draw gave us, and
    belot.md penalises leaving matches. The next round is still played."""
    hook = _stop_when(1)                 # stop while waiting between rounds
    site = Site([_bracket(phase="1/4", ours=WON_QF),
                 _bracket(phase="1/4", ours=WON_QF),
                 _bracket(phase="semifinal", at_table=True, ours=WON_QF,
                          semi=SEMI),
                 _bracket(phase="semifinal", eliminated=True, ours=WON_QF,
                          semi=[_table(SEMI[0]["playerIds"], ("70", "101"), 1),
                                SEMI[1]]),
                 _offer()],
                seats=[_seated(0, 10, 14)], on_state=hook)
    hook.site = site
    client, _ = _drive(site)
    assert site.steps() == ["state", "state", "state", "CONNECT:tournament",
                            "state", "withdraw", "state"]
    assert client.tournament_over == "eliminated in the semifinal (70-101)"


def test_a_stop_before_enrolling_enrols_nowhere():
    site = Site([_offer(_slot(1))])
    client, _ = _drive(site, stop=True)
    assert site.steps() == ["state"]
    assert client.tournament_over == "stopped before enrolling"


# ------------------------------------------------------- one tournament only
def test_once_out_the_session_ends_rather_than_enrolling_again():
    """The list is on offer again as soon as we have withdrawn; a run plays
    one tournament."""
    site = Site([_bracket(eliminated=True, ours=LOST_QF),
                 _offer(_slot(1, n=15))])
    client, _ = _drive(site, my_id=ME)
    assert "enroll:1" not in site.steps()
    assert client.tournament_over == "eliminated in the 1/4 (101-111)"


def test_a_resumed_session_goes_back_to_its_tournament():
    """A supervisor restarting a crashed session mid-tournament: the seat is
    taken again..."""
    site = Site([_bracket(at_table=True)], seats=[_seated(0, 10, leave=None)])
    _drive(site, tournament_enroll=False)
    assert site.steps() == ["state", "CONNECT:tournament"]


def test_a_resumed_session_never_enrols_in_a_new_one():
    """...but finding itself in no tournament, it does not start another."""
    site = Site([_offer(_slot(1))])
    client, _ = _drive(site, tournament_enroll=False)
    assert site.steps() == ["state"]
    assert "may not enrol" in client.tournament_over


# --------------------------------------------------------------- the seat
def test_a_finished_table_is_not_taken_again():
    """Just after our match the site may still count us at that table. Our
    latest table already has a winner, so that seat is the past: wait for
    the next round's instead of walking back into a finished match."""
    site = Site([_bracket(at_table=True),
                 _bracket(at_table=True, ours=WON_QF),       # stale
                 _bracket(phase="semifinal", at_table=True, ours=WON_QF,
                          semi=SEMI)],
                seats=[_seated(0, 10, 14), _seated(0, leave=None)])
    _drive(site)
    assert site.steps() == ["state", "CONNECT:tournament", "state", "state",
                            "CONNECT:tournament"]


def test_a_stale_seat_is_declined_only_for_so_long():
    """A site that seated us before listing the next round must not keep us
    from a round that is starting: past the bound, the seat is taken."""
    stale = _bracket(at_table=True, ours=WON_QF)
    site = Site([stale] * (BelotClient.STALE_SEAT_POLLS + 1),
                seats=[_seated(14, leave=None)])
    _drive(site, my_id=ME)
    steps = site.steps()
    assert steps.count("state") == BelotClient.STALE_SEAT_POLLS + 1
    assert steps[-1] == "CONNECT:tournament"


def test_a_table_without_a_winner_is_taken():
    """The control: the same answer with the match still on is our seat."""
    site = Site([_bracket(at_table=True)], seats=[_seated(0, leave=None)])
    _drive(site)
    assert site.steps() == ["state", "CONNECT:tournament"]


def test_a_dropped_connection_goes_straight_back_to_the_match():
    """1006 mid-hand: the platform keeps the seat, and every second away is a
    turn its bot plays for us. Ask the tournament at once, and sit back down."""
    site = Site([_bracket(at_table=True), _bracket(at_table=True)],
                seats=[_seated(0, 10, leave=1006), _seated(10, leave=None)])
    client, slept = _drive(site)
    assert site.steps() == ["state", "CONNECT:tournament", "state",
                            "CONNECT:tournament"]
    assert not [d for n, d in slept if n == 2], "waited before asking again"
    assert client.resumed_match is True


def test_a_seat_that_cannot_be_taken_asks_the_tournament_again():
    """gameplay_new.php had no room on it yet: ask the tournament, never a
    lobby, and at the site's pace."""
    no_room = [{"event": "ERROR", "code": "TOURNAMENT_NO_ROOM",
                "message": "no room yet"}]
    site = Site([_bracket(at_table=True), _bracket(at_table=True)],
                seats=[no_room, _seated(0, leave=None)])
    _, slept = _drive(site)
    assert site.steps() == ["state", "CONNECT:tournament", "state",
                            "CONNECT:tournament"]
    assert (2, tour.POLL_S) in slept


def test_this_account_opened_elsewhere_ends_the_run():
    """4001: a browser on the tournament page goes to the table by itself.
    Taking the seat back would throw it out, and it us, for ever."""
    site = Site([_bracket(at_table=True), _bracket(at_table=True)],
                seats=[_seated(0, 10, leave=4001)])
    client, _ = _drive(site, raises=TournamentError)
    assert site.steps() == ["state", "CONNECT:tournament"]
    assert "opened somewhere else" in str(client.raised)


# ----------------------------------------------------------------- refusals
def test_no_tournament_for_our_status_is_an_error():
    site = Site([_offer(_slot(1, min_status=6), _slot(2, min_status=4))])
    client, _ = _drive(site, raises=TournamentError)
    assert site.steps() == ["state"]
    assert "minimum status 5" in str(client.raised)


def test_every_place_taken_is_a_wait_not_an_error():
    site = Site([_offer(_slot(1, n=16)), _offer(_slot(1, n=2))])
    client, _ = _drive(site)
    assert site.steps() == ["state", "state", "enroll:1", "state"]
    assert client.fatal is None


def test_a_refused_enrolment_ends_the_run():
    site = Site([_offer(_slot(1)), _offer(_slot(1))],
                enroll={"error": "no_status"})
    client, _ = _drive(site, raises=TournamentError)
    assert site.steps() == ["state", "enroll:1", "state"]
    assert "status is below" in str(client.raised)


def test_a_refusal_the_page_contradicts_is_no_refusal():
    """"Already enrolled", say: the page shows us in the waiting room, and
    that is what counts. Ending the run there would leave the account
    enrolled with nobody to play its seat."""
    site = Site([_offer(_slot(1)), _waiting(5), _waiting(6)],
                enroll={"error": "Esti deja inscris"})
    client, _ = _drive(site)
    assert client.fatal is None
    assert site.steps() == ["state", "enroll:1", "state", "state", "state"]


def test_an_enrolment_that_did_not_land_is_retried_a_few_times_only():
    """Accepted, yet the page lists us nowhere: enrol again, a few times."""
    site = Site([_offer(_slot(1))] * 6)
    client, _ = _drive(site, raises=TournamentError)
    assert site.steps().count("enroll:1") == BelotClient.MAX_ENROLLS
    assert "lists us in no tournament" in str(client.raised)


def test_a_dead_login_ends_the_run_quickly():
    """A logged-out cookie gets a page instead of JSON, every time."""
    site = Site(["tournaments.php answered HTTP 200 with 5120 bytes that "
                 "are not JSON"] * 10)
    client, _ = _drive(site, raises=TournamentError)
    assert len(site.steps()) == BelotClient.MAX_TOURNAMENT_FAILURES
    assert "cookie" in str(client.raised)


def test_failures_after_the_site_has_answered_are_waited_out():
    """The control: once it has answered, a failing call is a blip in a long
    wait -- giving up then would walk out of a tournament we are in."""
    site = Site([_offer(_slot(1)), _waiting()] + ["timeout"] * 8 + [_waiting()])
    client, _ = _drive(site)
    assert client.fatal is None
    assert site.steps()[-1] == "state"


def test_a_refused_withdrawal_is_retried_then_given_up():
    site = Site([_bracket(eliminated=True, ours=LOST_QF)] * 5,
                withdraw={"error": "nu"})
    client, _ = _drive(site, my_id=ME)
    assert site.steps().count("withdraw") == BelotClient.MAX_WITHDRAWS
    assert client.tournament_over == "eliminated in the 1/4 (101-111)"


# ------------------------------------------------- the other modes untouched
@pytest.mark.parametrize("mode", ["lobby", "create"])
def test_the_other_modes_never_ask_the_tournament(mode):
    site = Site([])
    _drive(site, table_mode=mode)
    assert not [s for s in site.steps() if s in ("state", "enroll", "withdraw")]
    assert [s for s in site.steps() if s.startswith("CONNECT")]


# ------------------------------------------------------------------ the bot
class StubClient:
    def __init__(self):
        self.sent = []
        self.sessions_played = 1

    async def send_ready(self, value=True):
        self.sent.append("READY")

    async def leave_table(self, restart=True):
        self.sent.append("LEAVE_TABLE")


def _bot(mode="tournament"):
    from belotmd.agents import get_agent
    from belotmd.platform.sync import StateSynchronizer

    bot = LiveBelotBot.__new__(LiveBelotBot)
    bot.config = Config(cookies="x", audit=False, agent="random",
                        table_mode=mode)
    bot.client = StubClient()
    bot.sync_engine = StateSynchronizer()
    bot.agent = get_agent("random", seed=0)
    bot.audit = False
    bot._room_seq = 0
    bot._reset_room_state()
    return bot


def _frame(phase, ready=False, game_type=GAME_TYPE_TOURNAMENT):
    frame = {
        "currentPhase": phase, "activePlayer": -1, "dealer": 0, "round": 0,
        "topCard": "", "trump": -1, "declarer": -1, "swapSeven": -1,
        "lastCards": "", "scoreTable": "", "roundTotals": "",
        "timeleft": 0, "totalTime": 0, "targetScore": 101,
        "players": [
            {"id": "1", "position": 0, "ready": True, "numCards": 0},
            {"id": "2", "position": 1, "ready": True, "numCards": 0},
            {"id": "3", "position": 2, "ready": True, "numCards": 0},
            {"id": "me", "position": 3, "ready": ready, "numCards": 0,
             "cards": ""},
        ],
    }
    if game_type:
        frame["gameType"] = game_type
    return frame


def _feed(bot, *frames):
    for frame in frames:
        asyncio.run(bot.on_state_update(frame, "me"))


@pytest.mark.parametrize("phase", [0, 13])
def test_a_tournament_match_is_readied_for_even_with_a_stop_pending(phase):
    """At an ordinary table a pending stop holds READY at 0. Here the match is
    the draw's: holding READY would abandon the partner it gave us."""
    bot = _bot()
    bot.stop_requested = True
    _feed(bot, _frame(phase))
    assert bot.client.sent == ["READY"]


def test_at_an_ordinary_table_a_pending_stop_still_holds_ready():
    """The control: nothing changed for the other modes."""
    bot = _bot(mode="lobby")
    bot.stop_requested = True
    _feed(bot, _frame(0, game_type="GAME_4"))
    assert bot.client.sent == []


def test_a_finished_tournament_match_is_left_not_readied_again():
    """The site offers only leaving once a tournament match is over; its
    page then says where we play next."""
    bot = _bot()
    _feed(bot, _frame(14), _frame(14), _frame(14))
    assert bot.client.sent == ["LEAVE_TABLE"]


def test_a_finished_ordinary_match_is_readied_for_the_next_one():
    """The control: an ordinary table plays match after match."""
    bot = _bot(mode="lobby")
    _feed(bot, _frame(14, game_type="GAME_4"))
    assert bot.client.sent == ["READY"]


def test_a_room_that_is_not_a_tournament_table_is_reported(capsys):
    bot = _bot()
    _feed(bot, _frame(0, game_type="GAME_4"), _frame(0, game_type="GAME_4"))
    out = capsys.readouterr().out
    assert out.count("not TOURNAMENT_4") == 1


def test_the_tournament_settings_reach_the_client():
    bot = LiveBelotBot(Config(cookies="x", audit=False, agent="random",
                              table_mode="tournament", tournament_min_status=6,
                              tournament_enroll=False))
    assert bot.client.table_mode == "tournament"
    assert bot.client.tournament_min_status == 6
    assert bot.client.tournament_enroll is False
