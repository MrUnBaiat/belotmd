"""
test_sync_isolation.py — one player's cards must never reach the other's state.

Two of our accounts now sit at the same table as partners, in two processes.
The platform itself is on our side here: a frame carries `cards` for the
receiving player only, and `numCards` for everyone else. But section 6 of
`sync` stored `cards` from ANY seat that had them, and `sync` accepted a frame
whose identity it could not find -- or was not given at all. So a misrouted
frame did not fail, it quietly wrote the other player's hand into our state and
then the agent was asked to act on it.

These tests pin the guarantee rather than the assumption.
"""

import numpy as np
import pytest

from belotmd.platform.sync import StateSynchronizer

# Our two accounts, partners: seats 1 and 3.
A, B = "acct-a", "acct-b"
A_SEAT, B_SEAT = 1, 3
# Deliberately NOT the lowest card ids: those decode to 0,1,2,3, which is
# exactly what a placeholder hand of four looks like, so a leak would be
# indistinguishable from correct behaviour.
A_HAND, B_HAND = "stuv", "wxEF"


def _frame(owner_id, owner_seat, hand, others=4, phase=10):
    """A play-phase frame as the server sends it to ONE player."""
    players = []
    for seat in range(4):
        pid = {A_SEAT: A, B_SEAT: B}.get(seat, f"human-{seat}")
        if seat == owner_seat:
            players.append({"id": owner_id, "position": seat,
                            "cards": hand, "numCards": len(hand)})
        else:
            players.append({"id": pid, "position": seat, "numCards": others})
    return {
        "currentPhase": phase, "activePlayer": 0, "dealer": 0, "round": 1,
        "topCard": "", "trump": 4, "declarer": 0, "swapSeven": -1,
        "lastCards": "", "scoreTable": "", "roundTotals": "",
        "timeleft": 250, "totalTime": 250, "players": players,
    }


def _frame_for_a():
    return _frame(A, A_SEAT, A_HAND)


def _frame_for_b():
    return _frame(B, B_SEAT, B_HAND)


def _is_placeholder(hand, n):
    """Section 6's stand-in for a hidden hand: n meaningless ids, not cards."""
    return hand == list(range(n))


def test_two_bots_at_one_table_never_see_each_others_hands():
    """The headline guarantee, driven the way the pair actually runs: each
    synchronizer only ever gets its own account's frames."""
    sync_a, sync_b = StateSynchronizer(), StateSynchronizer()

    for _ in range(3):
        assert sync_a.sync(_frame_for_a(), A) == A_SEAT
        assert sync_b.sync(_frame_for_b(), B) == B_SEAT

    real_a = sorted(sync_a.state.hands[A_SEAT])
    real_b = sorted(sync_b.state.hands[B_SEAT])
    assert real_a and real_b and real_a != real_b
    assert real_a != list(range(4)) and real_b != list(range(4)), (
        "these hands must not look like placeholders, or the test proves "
        "nothing")

    # Each sees the partner's seat as a count, never as cards.
    assert _is_placeholder(sync_a.state.hands[B_SEAT], 4)
    assert _is_placeholder(sync_b.state.hands[A_SEAT], 4)
    assert sorted(sync_a.state.hands[B_SEAT]) != real_b
    assert sorted(sync_b.state.hands[A_SEAT]) != real_a


def test_a_frame_meant_for_the_partner_is_refused(capsys):
    """THE BUG: B's id is in A's frame too, so nothing used to notice. A's
    hand was written straight into B's state and handed to B's agent."""
    sync_b = StateSynchronizer()
    sync_b.sync(_frame_for_b(), B)
    before = [list(h) for h in sync_b.state.hands]

    assert sync_b.sync(_frame_for_a(), B) is None, "must refuse the frame"

    out = capsys.readouterr().out
    assert "VIOLATION" in out and "another player" in out
    assert [list(h) for h in sync_b.state.hands] == before, (
        "a refused frame must change nothing")


def test_a_refused_frame_leaves_no_identity_behind():
    """`my_pos` is what the bot indexes every later decision by, so a refusal
    must not leave a stale seat pointing at someone else's cards."""
    sync_b = StateSynchronizer()
    sync_b.sync(_frame_for_b(), B)
    sync_b.sync(_frame_for_a(), B)
    assert sync_b.my_pos is None


@pytest.mark.parametrize("identity", [None, ""])
def test_a_frame_with_no_identity_is_refused(identity, capsys):
    """`p.get("sessionId") == None` matched the FIRST seat, so an empty
    identity silently made us player 0 -- with player 0's cards if it had
    any."""
    sync = StateSynchronizer()
    assert sync.sync(_frame_for_a(), identity) is None
    assert sync.my_pos is None
    assert "VIOLATION" in capsys.readouterr().out


def test_a_frame_from_a_table_we_are_not_at_is_refused(capsys):
    sync = StateSynchronizer()
    assert sync.sync(_frame_for_a(), "somebody-else") is None
    out = capsys.readouterr().out
    assert "VIOLATION" in out and "not at this table" in out


def test_other_seats_carry_no_certainties_from_a_plain_frame():
    """Beliefs about other seats may only come from public events -- a card
    played, a declaration shown. An ordinary frame must pin nothing."""
    sync_a = StateSynchronizer()
    sync_a.sync(_frame_for_a(), A)

    for seat in range(4):
        if seat == A_SEAT:
            continue
        assert int(sync_a.state.known_cards[seat].sum()) == 0, (
            f"seat {seat} gained certainties from a frame that only told us "
            f"how many cards it holds")
    assert not np.any(sync_a.state.known_cards[B_SEAT])
