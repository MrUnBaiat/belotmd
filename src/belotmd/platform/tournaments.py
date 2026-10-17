"""
tournaments.py -- reading belot.md's tournament page.

A tournament is sixteen players in eight random teams, three knockout rounds
(1/4, semifinal, final), each a match to 101 at a table the site seats us at
by itself. Everything about it goes through `tournaments.php`:

    getState=1    where we stand (the site's own page polls it every 3 s)
    enroll=<id>   take a place in tournament <id>, an `id` from the list
    withdraw=1    leave the waiting room before the start, or the bracket
                  once we are out ("Continua spre turnee noi")

`getState` answers with a `playerState`:

    none       not in a tournament: `tournaments` lists what is on offer
    enrolled   in a waiting room: `tournamentInfo` counts who has arrived
    playing    in the bracket: `phase`, `isEliminated`, `progress`, and
               `isAtTable` once our table is ready -- the site's page then
               simply opens gameplay_new.php, whose room is ours

This module interprets those answers. It does no I/O, so every rule here is
tested directly. See docs/PLATFORM_NOTES.md, "Tournaments".
"""

# The site's own page polls `getState` this often (`tournamentsRefreshRate`).
POLL_S = 3.0

NONE = "none"
ENROLLED = "enrolled"
PLAYING = "playing"

# `phase` in the bracket. "finished" comes after the final.
STAGES = ("1/4", "semifinal", "final")
FINISHED = "finished"


def as_int(value, default=None):
    """The site sends numbers both as numbers and as strings ("200")."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def offers_status(tournaments, min_status):
    """Is any tournament for exactly `min_status` on the list, full or not?"""
    return any(isinstance(t, dict) and as_int(t.get("minStatus")) == min_status
               for t in tournaments or ())


def pick_tournament(tournaments, min_status):
    """The tournament to enrol in, or None.

    Only one whose minimum status is exactly `min_status`: the account's own
    status, so neither one it cannot enter nor one meant for weaker players.
    Among those with a free place, the fullest first -- it is the one that
    starts soonest, and the start is what we are waiting for -- then the
    lowest id, so the choice is the same every time the list is.
    """
    candidates = []
    for t in tournaments or ():
        if not isinstance(t, dict) or as_int(t.get("minStatus")) != min_status:
            continue
        if as_int(t.get("numPlayers"), 0) >= as_int(t.get("maxPlayers"), 16):
            continue
        candidates.append(t)
    if not candidates:
        return None
    return min(candidates, key=lambda t: (-as_int(t.get("numPlayers"), 0),
                                          as_int(t.get("id"), 1 << 30),
                                          str(t.get("id"))))


def our_tables(state, my_id):
    """Our table at each stage we have reached. -> {stage: {us, them, won}}

    `progress[stage]` lists the tables of that stage, each with its four
    `playerIds`, the two teams' `score` and the `winner` (0, 1, or -1 while
    it is being played). The ids are the in-room player ids, so ours is the
    one the room gave us, and teams pair indexes 0+2 against 1+3 -- the seat
    pairing of the room itself. `won` is None until the table has a winner.

    Only OUR table is read: the other ids in the bracket belong to other
    players, and nothing about them leaves this function.
    """
    found = {}
    if my_id is None:
        return found
    progress = state.get("progress") or {}
    me = str(my_id)
    for stage in STAGES:
        for table in progress.get(stage) or ():
            if not isinstance(table, dict):
                continue
            ids = [str(p) for p in (table.get("playerIds") or ())]
            if me not in ids:
                continue
            team = ids.index(me) % 2
            score = list(table.get("score") or ()) + [None, None]
            winner = as_int(table.get("winner"), -1)
            found[stage] = {"us": as_int(score[team]),
                            "them": as_int(score[1 - team]),
                            "won": None if winner not in (0, 1) else winner == team}
            break
    return found


def our_table(state, my_id):
    """Our table at the latest stage we have reached, with its `stage`, or
    None if the bracket does not show us (yet)."""
    tables = our_tables(state, my_id)
    for stage in reversed(STAGES):
        if stage in tables:
            return dict(tables[stage], stage=stage)
    return None


def placing(state, my_id):
    """'won' or 'second' once the tournament has finished, else None.

    `winners` lists four player ids: the two who won, then the two runners-up.
    """
    if my_id is None:
        return None
    winners = [str(w) for w in (state.get("winners") or ())]
    if str(my_id) not in winners:
        return None
    return "won" if winners.index(str(my_id)) < 2 else "second"


def is_out(state):
    """Is the tournament over for us -- eliminated, or finished altogether?"""
    return bool(state.get("isEliminated")) or state.get("phase") == FINISHED
