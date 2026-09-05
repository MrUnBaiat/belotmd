"""
combinations.py — decode belot.md declaration codes into concrete cards.

Wire format: "<type><card_char>", pipe-separated for multiple declarations,
e.g. "5q|1c". The card char is the HIGHEST (last) card of the combination.

Type ids are the platform's Ye enum. Two are pinned by observed traffic --
"1c" = 9,10,J of diamonds (3-run) and "5q" = Q,K of clubs (Bella) -- and the
declaration order in gameplay.js matches 1..9 exactly:

    1 TART            3-card run          20
    2 JUMATE_DE_SUTA  4-card run          50
    3 O_SUTA          5-card run         100
    4 PATRU_CARTI     four of a kind     150 (9s) / 200 (Js) / 0 (7s,8s) / 100
    5 BELA            Q+K of trump        20
    6 LESS_THAN_14    hand cancellation    0   <- not cards
    7 BELOT_COMBO     special           1010   <- not cards
    8 WIN_ALL_HANDS   claim the rest       0   <- not cards
    9 SURRENDER_BT    concede the hand     0   <- not cards

Runs use natural rank order (7<8<9<10<J<Q<K<A), which is exactly the training
card id's `rank = id % 8`, so a run ending at rank r covers r-n+1 .. r.

Types 6-9 carry a filler card char and MUST NOT be decoded into cards.
"""

TART = 1
JUMATE_DE_SUTA = 2
O_SUTA = 3
PATRU_CARTI = 4
BELA = 5
LESS_THAN_14 = 6
BELOT_COMBO = 7
WIN_ALL_HANDS = 8
SURRENDER_BT = 9

RUN_LENGTH = {TART: 3, JUMATE_DE_SUTA: 4, O_SUTA: 5}

# Four of a kind is scored by rank (gameplay.js): 9s = 150, Js = 200,
# 7s and 8s = 0, anything else = 100. The two zero-point ones are not
# scoring plays at all -- they change the hand:
#   four 7s (FOUR_OF_SEVEN) cancels the deal, exactly like LESS_THAN_14
#   four 8s (FOUR_OF_EIGHT) disables every combination except bella
RANK_SEVEN, RANK_EIGHT = 0, 1
FOUR_POINTS = {0: 0, 1: 0, 2: 150, 4: 200}          # rank -> points


def four_rank(type_id, card_char, ascii_to_id):
    """Rank index of a four-of-a-kind declaration, else None."""
    if type_id != PATRU_CARTI:
        return None
    card = ascii_to_id.get(card_char)
    return None if card is None else card % 8

TYPE_NAMES = {
    TART: "terta(20)", JUMATE_DE_SUTA: "50", O_SUTA: "100",
    PATRU_CARTI: "four-of-a-kind", BELA: "bella(20)",
    LESS_THAN_14: "less-than-14", BELOT_COMBO: "belot-combo",
    WIN_ALL_HANDS: "win-all-hands", SURRENDER_BT: "surrender-BT",
}

# Card combinations: free points, and the only types that decode to cards.
POINT_COMBOS = frozenset({TART, JUMATE_DE_SUTA, O_SUTA, PATRU_CARTI, BELA})
# Declarations that are not card sets. They carry a filler card char ('a').
CLAIM_COMBOS = frozenset({LESS_THAN_14, BELOT_COMBO, WIN_ALL_HANDS, SURRENDER_BT})

# Announce whenever offered: the point combinations, plus LESS_THAN_14 (bails
# out of a dead hand) and BELOT_COMBO (1010 points).
AUTO_DECLARE = POINT_COMBOS | {LESS_THAN_14, BELOT_COMBO}
# Never auto-fired. SURRENDER_BT concedes the hand outright and is offered
# precisely when losing; WIN_ALL_HANDS claims every remaining trick and is a
# judgement call the policy network cannot make.
NEVER_DECLARE = frozenset({WIN_ALL_HANDS, SURRENDER_BT})


def parse_field(value):
    """'5q|1c' -> [(5, 'q'), (1, 'c')]. Tolerates junk without raising."""
    out = []
    for token in (value or "").split("|"):
        token = token.strip()
        if len(token) < 2 or not token[:-1].isdigit():
            continue
        out.append((int(token[:-1]), token[-1]))
    return out


def decode(type_id, card_char, ascii_to_id):
    """-> sorted list of card ids the declaring player provably holds.

    Returns [] for non-card declarations and for anything malformed: a wrong
    guess here would write a false 1.0 into the belief matrix, so silence is
    strictly safer than a plausible reconstruction.
    """
    card = ascii_to_id.get(card_char)
    if card is None or type_id in CLAIM_COMBOS:
        return []

    suit, rank = card // 8, card % 8

    n = RUN_LENGTH.get(type_id)
    if n is not None:
        if rank - (n - 1) < 0:          # run would fall off the bottom
            return []
        return sorted(suit * 8 + r for r in range(rank - n + 1, rank + 1))

    if type_id == PATRU_CARTI:          # same rank, all four suits
        return sorted(s * 8 + rank for s in range(4))

    if type_id == BELA:                 # Q + K of that suit
        if rank != 6:                   # last card of Q,K is the King
            return []
        return sorted([suit * 8 + 5, suit * 8 + 6])

    return []


def decode_field(value, ascii_to_id):
    """'5q|1c' -> (cards, unknown_types). Cards are deduplicated."""
    cards, unknown = set(), []
    for type_id, ch in parse_field(value):
        got = decode(type_id, ch, ascii_to_id)
        if got:
            cards.update(got)
        elif type_id not in CLAIM_COMBOS:
            unknown.append((type_id, ch))
    return sorted(cards), unknown


# The platform reports a run MAXIMALLY -- it always shows the longest one the
# hand contains -- with a single exception: the enum stops at O_SUTA, so a run
# of six or more is reported as its TOP FIVE.
#
# That maximality is information beyond the cards themselves. If a declared run
# could have been extended, it would have been, so the ranks immediately
# outside it are provably NOT held:
#
#     3-run "1x"   card above absent    card below absent
#     4-run "2x"   card above absent    card below absent
#     5-run "3x"   card above absent    card below UNKNOWN  <- the exception
#
# The 5-run keeps the "above" half: the reported top card is still the highest
# of the run, or a longer run would have been reported ending higher. It loses
# the "below" half, because a six-run is exactly what a five-run looks like.
#
# Four of a kind and bella imply nothing: they are not runs, and holding the
# neighbouring rank would not have changed what was declared.
MAX_REPORTED_RUN = 5

RANK_ACE = 7


def excluded(type_id, card_char, ascii_to_id):
    """-> card ids the declarer provably does NOT hold, from a run's edges.

    Returns [] for anything that is not a run, and for a malformed token --
    a wrong exclusion is a false void, which quietly removes a real candidate
    from every belief and every sampled world.
    """
    n = RUN_LENGTH.get(type_id)
    if n is None:
        return []

    card = ascii_to_id.get(card_char)
    if card is None:
        return []
    suit, rank = card // 8, card % 8
    if rank - (n - 1) < 0:              # run falls off the bottom: malformed
        return []

    out = []
    if rank < RANK_ACE:                 # nothing sits above the ace
        out.append(suit * 8 + rank + 1)
    below = rank - n
    if below >= 0 and n < MAX_REPORTED_RUN:
        out.append(suit * 8 + below)
    return sorted(out)


def excluded_field(value, ascii_to_id):
    """'1x|2c' -> sorted card ids nobody declaring that could be holding."""
    out = set()
    for type_id, ch in parse_field(value):
        out.update(excluded(type_id, ch, ascii_to_id))
    return sorted(out)


def declarable(value, ascii_to_id=None, include_win_all=False,
               include_four_sevens=True, include_four_eights=False):
    """Which tokens from a combinationsCanShow field we should announce.

    The two four-of-a-kind ranks that carry an EFFECT rather than points get
    their own switches:

      four 7s (FOUR_OF_SEVEN)  cancels the deal, exactly like LESS_THAN_14.
                               Declared by default -- it is the escape hatch
                               from a hopeless hand.
      four 8s (FOUR_OF_EIGHT)  silences every combination except bella,
                               OURS INCLUDED. Declined by default: it is a
                               strategic call, not free points.

    Passing ascii_to_id is required for either switch to apply, since the
    rank lives in the card char.
    """
    allowed = AUTO_DECLARE | ({WIN_ALL_HANDS} if include_win_all else frozenset())
    allowed = allowed - {SURRENDER_BT}       # never, under any setting
    out = []
    for t, ch in parse_field(value):
        if t not in allowed:
            continue
        if t == PATRU_CARTI:
            # FAIL CLOSED. Without the map (or with an unknown card char) the
            # rank cannot be read, and the two effect-carrying ranks would
            # slip through: four 7s cancels the deal, four 8s silences every
            # combination except bella -- OURS INCLUDED. Withholding a
            # 100/150/200-point declaration is a far cheaper mistake.
            rank = four_rank(t, ch, ascii_to_id) if ascii_to_id is not None else None
            if rank is None:
                continue
            if rank == RANK_SEVEN and not include_four_sevens:
                continue
            if rank == RANK_EIGHT and not include_four_eights:
                continue                     # would cancel our own combos
        out.append(f"{t}{ch}")
    return out


def describe(value, ascii_to_id):
    """Human-readable summary of a declaration, for logs."""
    parts = []
    for t, ch in parse_field(value):
        name = TYPE_NAMES.get(t, str(t))
        if t == PATRU_CARTI:
            rank = four_rank(t, ch, ascii_to_id)
            if rank == RANK_SEVEN:
                name = "four 7s (CANCELS the deal)"
            elif rank == RANK_EIGHT:
                name = "four 8s (DISABLES all combinations but bella)"
            else:
                name = f"four of a kind ({FOUR_POINTS.get(rank, 100)} pts)"
        parts.append(f"{t}{ch}={name}")
    return ", ".join(parts)


# Claim declarations are sent as "<type>a" -- the trailing char is a filler.
CLAIM_FILLER = "a"


def claim_token(type_id):
    return f"{type_id}{CLAIM_FILLER}"