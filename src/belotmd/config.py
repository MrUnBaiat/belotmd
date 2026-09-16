"""
config.py — every runtime knob, in one documented place.

Values come from the process environment, falling back to a `.env` file in
the project root. `.env` is gitignored; `.env.example` documents the shape.
Nothing here has a hardcoded credential and nothing ever should.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root: src/belotmd/config.py -> src/belotmd -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _read_env_file(path):
    """Minimal KEY=VALUE parser, so `.env` support costs no dependency.

    Returns a dict and touches nothing global. Values are taken verbatim after
    the first `=` — belot cookies contain `=` and `;`, so no splitting or
    unquoting beyond stripping one matched pair of surrounding quotes.
    """
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _load_env_file(path=ENV_FILE):
    """Fold the default `.env` into the environment, which always wins — so
    `BELOT_AUDIT=0 belot-bot` behaves as expected regardless of the file.

    Only for the single-account path. Two accounts in one process cannot use
    it: `setdefault` means whichever ran first keeps its cookies, and the
    second silently plays as the first. That is what `env_file` below is for.
    """
    for key, value in _read_env_file(path).items():
        os.environ.setdefault(key, value)


def _flag(lookup, name, default):
    return lookup.get(name, "1" if default else "0") == "1"


@dataclass
class Config:
    """Resolved settings for one bot run."""

    # --- credentials ---
    cookies: str = ""

    # --- verification ---
    # Record every raw state frame and run the invariant auditor alongside
    # the bot. Cheap, and the recording is what makes the next bug
    # diagnosable. Leave on.
    audit: bool = True
    frames_path: str = "frames.jsonl"

    # --- declarations ---
    # Announce the combinations the server offers. Combination points land in
    # roundTotals.c and never in .p, so declaring cannot perturb any
    # observation feature — it is free match score.
    auto_declare: bool = True
    # Four 7s cancels the deal, exactly like LESS_THAN_14: the escape hatch
    # from a hopeless hand. On.
    declare_four_sevens: bool = True
    # Four 8s silences every combination except bella, OURS INCLUDED. That is
    # a strategic call, not free points. Off.
    declare_four_eights: bool = False
    # Claiming every remaining trick is a judgement the policy cannot make.
    # Off. (SURRENDER_BT concedes the hand and is never auto-fired at all.)
    declare_win_all: bool = False

    # --- seven-swap ---
    # Trade the 7 of trump for the face-up card in the phase-8 window. Only
    # reachable after a round-1 accept, where the face-up card is a trump
    # strictly better than the 7, so it is always a gain.
    auto_swap_seven: bool = True

    # --- agent ---
    # `random` is the only agent this package ships. Anything else comes from
    # a separately installed package advertising a "belotmd.agents" entry
    # point; name it here or with --agent.
    agent: str = "random"
    checkpoint: str = ""

    # --- which table ---
    # "lobby" joins the first open public table, which is right for one bot
    # and useless for a pair. Two of our accounts meet by having the host
    # "create" a table and the guest "join" the one that host made -- found in
    # the lobby by the host's username, so the two processes never talk.
    table_mode: str = "lobby"
    table_id: str = ""
    table_creator: str = ""

    # Our other account's username. Setting it means "we are playing as a
    # pair": the bot then withholds READY until that account is sitting
    # OPPOSITE us, because the match starts the moment all four players are
    # ready and a pair on opposite teams is the one outcome worth avoiding.
    partner: str = ""

    # Diagnostic: rotate the seats this many times even when they are ALREADY
    # right, logging the layout before and after each one. CHANGE_PLAYERS_POSITION
    # is the one message we send whose payload and effect are unverified, and a
    # table where the seats happen to be correct never exercises it. 0 = off.
    rotation_probe: int = 0

    # Diagnostic: abandon this many tables on purpose, as soon as our partner
    # has sat down and before any stranger joins, to prove the
    # leave-and-make-another path works without ejecting anyone. 0 = off.
    leave_probe: int = 0

    # --- staying on a table ---
    # Keep looking for tables instead of exiting after one session. A match
    # ending dissolves the table (TABLE_REMOVED), so without this the bot
    # plays one table's worth of matches and stops.
    reconnect: bool = True
    # Nothing to join right now: no open tables, or we were kicked. Waiting is
    # the only useful move, and hammering the lobby every few seconds is rude.
    retry_delay_s: float = 300.0
    # A match ended or the table dissolved -- go straight back out.
    rejoin_delay_s: float = 5.0

    # --- finding our partner's table ---
    # The guest watches the lobby for the host's table. Two seconds, because
    # on a busy lobby a freshly created table is taken by strangers within a
    # few: polling slower than they arrive is what makes the host delete table
    # after table. `gameTables.php` is cheap and the site tolerates it.
    join_poll_s: float = 2.0
    # Fruitless looks before giving up on this attempt. 20 x 2s = 40s, the
    # same window the host allows before abandoning a table nobody joined.
    join_max_polls: int = 20
    # After a failed pairing, BOTH sides sit still for this long before trying
    # again -- the host having deleted its table, the guest having stopped
    # looking. Restarting instantly just recreates the same race.
    pair_restart_pause_s: float = 10.0

    # --- timing ---
    # If the state has not advanced this long after we dispatched an action,
    # assume the server refused it and re-dispatch. A successful action always
    # changes the turn signature, so this can never double-fire a good move.
    redispatch_after_s: float = 3.0

    warnings: list = field(default_factory=list)

    # Which file the credentials came from, for error messages. Empty means
    # the default `.env` + environment path.
    env_file: str = ""

    @classmethod
    def from_env(cls, env_file=None, **overrides):
        """Build from the environment + `.env`, then apply CLI overrides.

        `env_file` names ONE account's file and is what makes two accounts
        safe. Its credentials come from that file and nowhere else: a
        `BELOT_COOKIES` left in the shell cannot decide which account a
        process plays as, and nothing is written into `os.environ`, so a
        second call with a different file is unaffected by the first. Other
        knobs still fall back to the environment.
        """
        if env_file is None:
            _load_env_file()
            lookup = dict(os.environ)
            cookies = lookup.get("BELOT_COOKIES", "").strip()
        else:
            from_file = _read_env_file(env_file)
            lookup = {**os.environ, **from_file}
            cookies = from_file.get("BELOT_COOKIES", "").strip()

        cfg = cls(
            cookies=cookies,
            audit=_flag(lookup, "BELOT_AUDIT", True),
            frames_path=lookup.get("BELOT_FRAMES", "frames.jsonl"),
            auto_declare=_flag(lookup, "BELOT_AUTO_DECLARE", True),
            declare_four_sevens=_flag(lookup, "BELOT_DECLARE_FOUR_SEVENS", True),
            declare_four_eights=_flag(lookup, "BELOT_DECLARE_FOUR_EIGHTS", False),
            declare_win_all=_flag(lookup, "BELOT_DECLARE_WIN_ALL", False),
            auto_swap_seven=_flag(lookup, "BELOT_AUTO_SWAP_SEVEN", True),
            agent=lookup.get("BELOT_AGENT", "random"),
            checkpoint=lookup.get("BELOT_CHECKPOINT", ""),
            table_mode=lookup.get("BELOT_TABLE_MODE", "lobby"),
            table_id=lookup.get("BELOT_TABLE_ID", ""),
            table_creator=lookup.get("BELOT_TABLE_CREATOR", ""),
            partner=lookup.get("BELOT_PARTNER", ""),
            rotation_probe=int(lookup.get("BELOT_ROTATION_PROBE", 0) or 0),
            leave_probe=int(lookup.get("BELOT_LEAVE_PROBE", 0) or 0),
            reconnect=_flag(lookup, "BELOT_RECONNECT", True),
            retry_delay_s=float(lookup.get("BELOT_RETRY_DELAY", 300)),
            rejoin_delay_s=float(lookup.get("BELOT_REJOIN_DELAY", 5)),
            join_poll_s=float(lookup.get("BELOT_JOIN_POLL", 2)),
            join_max_polls=int(lookup.get("BELOT_JOIN_MAX_POLLS", 20)),
            pair_restart_pause_s=float(lookup.get("BELOT_PAIR_RESTART_PAUSE", 10)),
            env_file=str(env_file or ""),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    def require_cookies(self):
        """Fail loudly and usefully rather than joining as nobody."""
        if not self.cookies:
            where = self.env_file or ENV_FILE
            raise SystemExit(
                f"BELOT_COOKIES is not set (looked in {where}).\n\n"
                "  cp .env.example .env\n"
                "  then paste your belot.md PHPSESSID and token into it.\n\n"
                "Get them from a logged-in browser: DevTools -> Application "
                "-> Cookies -> https://belot.md"
            )
        return self.cookies
