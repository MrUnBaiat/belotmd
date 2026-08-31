"""
observation.py — the reference agent's observation encoding.

Turns a `BelotState` plus the running match score into the three arrays the
recurrent MAPPO policy consumes: a (513,) local observation, a (332,)
egocentric global observation for the critic, and a (38,) legal-action mask.

GROUND TRUTH. The layout here is a faithful, side-effect-free port of the
trainer's `BelotAECEnv.observe()`, and the checkpoint's weights depend on
every offset in it. Changing the layout silently invalidates the trained
model. A different architecture should bring its own encoder rather than
edit this one — see docs/PPO_AGENT.md for the field-by-field contract.
"""

import numpy as np


def build_observation(belot, abs_id, match_scores):
    """
    Build the local (513,), egocentric global (332,) observations and the
    legal-action mask (38,) for player `abs_id` in `belot`.

    `match_scores` is this env's running [team0, team1] match score.
    """
    team_us = abs_id % 2
    team_them = 1 - team_us

    # ====================== LOCAL OBSERVATION (513) ======================
    obs = np.zeros(513, dtype=np.float32)
    idx = 0

    # 1. Private Hand (32)
    for card in belot.hands[abs_id]:
        obs[idx + card] = 1.0
    idx += 32

    # 2. Face-Up Card (32)
    if belot.face_up_card is not None and belot.phase == "BIDDING":
        obs[idx + belot.face_up_card] = 1.0
    idx += 32

    # 3. Current Trump (5)
    if belot.trump is None:
        obs[idx] = 1.0
    else:
        obs[idx + 1 + belot.trump] = 1.0
    idx += 5

    # 4. Relative Declarer (5)
    if belot.declarer is None:
        obs[idx] = 1.0
    else:
        obs[idx + 1 + (belot.declarer - abs_id) % 4] = 1.0
    idx += 5

    # 5. Phase (3)
    if belot.phase == "BIDDING":
        obs[idx] = 1.0 if belot.bidding_round == 1 else 0.0
        if belot.bidding_round != 1:
            obs[idx + 1] = 1.0
    else:
        obs[idx + 2] = 1.0
    idx += 3

    # 6. Current Trick (108): Left(1), Partner(2), Right(3)
    for rel in [1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.current_trick):
            if p == abs_p:
                obs[idx + c] = 1.0
                obs[idx + 32 + seq_idx] = 1.0
        idx += 36

    # 7. Game Stats (6)
    obs[idx]     = match_scores[team_us] / 101.0
    obs[idx + 1] = match_scores[team_them] / 101.0
    obs[idx + 2] = belot.raw_points_by_team[team_us] / 162.0
    obs[idx + 3] = belot.raw_points_by_team[team_them] / 162.0
    obs[idx + 4] = belot.bolts_by_team[team_us] / 2.0
    obs[idx + 5] = belot.bolts_by_team[team_them] / 2.0
    idx += 6

    # 8. Relative Dealer (4)
    obs[idx + (belot.dealer - abs_id) % 4] = 1.0
    idx += 4

    # 9. Last Trick (144): Me(0), Left(1), Partner(2), Right(3)
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.last_trick):
            if p == abs_p:
                obs[idx + c] = 1.0
                obs[idx + 32 + seq_idx] = 1.0
        idx += 36

    # 10. Belief State Matrix (96)
    unseen = np.ones(32, dtype=bool)
    unseen[belot.hands[abs_id]] = False
    unseen[belot.graveyard] = False
    for p, c in belot.current_trick:
        unseen[c] = False
    if belot.phase == "BIDDING" and belot.face_up_card is not None:
        unseen[belot.face_up_card] = False
    # A card known to sit in one specific hand is not an open candidate for
    # anybody else. Without this line the holder gets belief 1.0 (set below)
    # while the other two opponents still receive ~0.49 of phantom mass on
    # the same card -- the column sums to ~1.97 instead of 1.0 -- and their
    # genuine candidates are diluted to compensate.
    unseen[belot.known_cards.any(axis=0)] = False

    other_players = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]

    W = np.zeros((3, 32), dtype=np.float32)
    for i, p in enumerate(other_players):
        valid = unseen & ~belot.impossible_cards[p] & ~belot.known_cards[p]
        W[i, valid] = 1.0

    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    P = W / col_sums

    remaining = np.array(
        [len(belot.hands[p]) - belot.known_cards[p].sum() for p in other_players],
        dtype=np.float32,
    )
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    P = P * (remaining[:, np.newaxis] / row_sums[:, np.newaxis])
    P = np.clip(P, 0.0, 1.0)

    for i, p in enumerate(other_players):
        belief = np.zeros(32, dtype=np.float32)
        belief[belot.known_cards[p]] = 1.0
        unresolved = ~belot.known_cards[p]
        belief[unresolved] = P[i, unresolved]
        obs[idx:idx + 32] = belief
        idx += 32

    # 11. Trick Number (8)
    obs[idx + min(belot.tricks_played, 7)] = 1.0
    idx += 8

    # Legal-action mask (real for the active player, zeros otherwise)
    if belot.current_player == abs_id and not belot.done:
        legal_mask = belot.get_legal_actions().astype(np.int8)
    else:
        legal_mask = np.zeros(38, dtype=np.int8)

    # 12. Valid Actions feature (38)
    obs[idx:idx + 38] = legal_mask.astype(np.float32)
    idx += 38

    # 13. Graveyard (32)
    for c in belot.graveyard:
        obs[idx + c] = 1.0
    idx += 32

    assert idx == 513, f"Local obs built {idx} features, expected 513"

    # ====================== GLOBAL OBSERVATION (332) ======================
    g_obs = np.zeros(332, dtype=np.float32)
    g = 0

    # 1. Relative Hands (128): Me, Left, Partner, Right
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for card in belot.hands[abs_p]:
            g_obs[g + card] = 1.0
        g += 32

    # 2. Graveyard (32)
    for c in belot.graveyard:
        g_obs[g + c] = 1.0
    g += 32

    # 3. Face-Up Card (32)
    if belot.phase == "BIDDING" and belot.face_up_card is not None:
        g_obs[g + belot.face_up_card] = 1.0
    g += 32

    # 4. Current Trump (5)
    if belot.trump is None:
        g_obs[g] = 1.0
    else:
        g_obs[g + 1 + belot.trump] = 1.0
    g += 5

    # 5. Relative Declarer (5)
    if belot.declarer is None:
        g_obs[g] = 1.0
    else:
        g_obs[g + 1 + (belot.declarer - abs_id) % 4] = 1.0
    g += 5

    # 6. Relative Dealer (4)
    g_obs[g + (belot.dealer - abs_id) % 4] = 1.0
    g += 4

    # 7. Phase (3)
    if belot.phase == "BIDDING":
        if belot.bidding_round != 1:
            g_obs[g + 1] = 1.0
        else:
            g_obs[g] = 1.0
    else:
        g_obs[g + 2] = 1.0
    g += 3

    # 8. Relative Current Trick (108): Left, Partner, Right
    for rel in [1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.current_trick):
            if p == abs_p:
                g_obs[g + c] = 1.0
                g_obs[g + 32 + seq_idx] = 1.0
        g += 36

    # 9. Relative Game Stats (6)
    g_obs[g]     = match_scores[team_us] / 101.0
    g_obs[g + 1] = match_scores[team_them] / 101.0
    g_obs[g + 2] = belot.raw_points_by_team[team_us] / 162.0
    g_obs[g + 3] = belot.raw_points_by_team[team_them] / 162.0
    g_obs[g + 4] = belot.bolts_by_team[team_us] / 2.0
    g_obs[g + 5] = belot.bolts_by_team[team_them] / 2.0
    g += 6

    # 10. Trick Number (8)
    g_obs[g + min(belot.tricks_played, 7)] = 1.0
    g += 8

    # 11. Declarer Has Played Trump (1)
    g_obs[g] = float(belot.declarer_has_played_trump)
    g += 1

    assert g == 332, f"Global obs built {g} features, expected 332"

    return obs, g_obs, legal_mask

# --------------------------------------------------------------- self-check
BELIEF_SLICE = slice(339, 435)


def verify_observation_contract(verbose=True):
    """Confirm this encoder excludes known cards from every candidate row.

    The synchronizer pins declared and swapped cards using `known_cards`
    alone. If the belief matrix does not exclude a pinned card from the other
    two candidate rows, each of them receives ~0.49 of phantom mass on a card
    somebody else provably holds, and the column sums to ~1.97 instead of 1.0.
    The belief state then degrades as declarations arrive instead of
    improving — silently. Cheap to assert, so assert it at startup.
    """
    from ...game.state import BelotState

    s = BelotState()
    s.phase = "PLAYING"
    s.trump, s.declarer, s.dealer = 2, 1, 3
    s.current_player, s.done, s.tricks_played = 0, False, 0
    s.hands = [[0, 1, 2, 3]] + [list(range(4, 8)) for _ in range(3)]
    s.graveyard, s.current_trick = [], []
    s.known_cards[:] = False
    s.impossible_cards[:] = False
    s.known_cards[2, 20] = True          # seat 2 provably holds card 20

    bel = build_observation(s, 0, [0, 0])[0][BELIEF_SLICE].reshape(3, 32)
    ok = bel[0, 20] == 0.0 and bel[2, 20] == 0.0
    if not ok:
        print("=" * 68)
        print("[Agent][CRITICAL] observation.py is missing the known-card "
              "exclusion.\n                  Pinned cards leak "
              f"{bel[0, 20]:.2f} of phantom mass onto each\n"
              "                  other opponent; beliefs will degrade as "
              "declarations arrive.")
        print("=" * 68)
    elif verbose:
        print("[Agent] observation belief exclusion verified "
              "(pins are exclusive).")
    return ok
