"""
tables.py — creating a table, and finding the one our partner created.

The lobby pick takes the first open public table, which is the right default
for one bot and useless for two: a pair has to meet at a table they choose.
Two accounts do that without talking to each other at all — the host creates a
table, and the guest recognises it in the lobby by the host's username.

Everything here is pure: no Node, no network, so the payload and the lookup can
be tested directly.
"""

# The create request, captured from the site's own form and sent VERBATIM.
# Field order and spelling are the platform's, not ours; `14puncte` is not a
# valid Python identifier and `miza` has no confirmed meaning, which is reason
# enough not to "tidy" any of it into a dict of keyword arguments.
#
#   gametable_type=4     4 players, two against two
#   gametable_level=0    minimum rating required to join; 0 lets anyone in
#   gametable_points=101 play to 101
#   14puncte=on          the 14-point rule
#   gametable_color=1    cosmetic
#   password=            open table: the other two seats are for humans
CREATE_TABLE_BODY = (
    "createNewTable=1"
    "&gametable_type=4"
    "&gametable_level=0"
    "&miza=50"
    "&gametable_color=1"
    "&gametable_points=101"
    "&14puncte=on"
    "&password="
)


def creator_name(table):
    """The username that created a lobby table, or None.

    Observed shape: `"creator": ["4", "<username>", "104325"]` — a level, the
    name, and an account id that is NOT the in-room player id, so the name is
    the only field that can be matched against something we configured.
    """
    creator = table.get("creator")
    if isinstance(creator, (list, tuple)):
        return str(creator[1]) if len(creator) >= 2 else None
    if isinstance(creator, str) and creator:
        return creator
    return None


def find_table(mese, *, table_id=None, creator=None):
    """Pick our own table out of a lobby listing.

    Unlike the lobby's "any open table" pick, this deliberately ignores the
    `pass` and `minStatus` filters: we are not shopping, we know which table we
    want. Returns the entry, or None if it is not listed (yet).
    """
    if not mese:
        return None
    if table_id is None and creator is None:
        raise ValueError("find_table needs a table_id or a creator")

    wanted = None if creator is None else creator.strip().casefold()
    for table in mese:
        if table_id is not None and str(table.get("id")) == str(table_id):
            return table
        if wanted is not None:
            name = creator_name(table)
            if name is not None and name.strip().casefold() == wanted:
                return table
    return None


def table_id_of(table):
    """The id to pass to `enterGame`. It is a string with a `new-` prefix
    (`"new-44709391"`), so it is never coerced to an int."""
    return None if table is None else str(table.get("id"))


def is_joinable(table):
    """A table with no free seat cannot be entered, however much we want it."""
    return bool(table) and not table.get("full")
