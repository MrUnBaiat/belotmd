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


def _load_env_file(path=ENV_FILE):
    """Minimal KEY=VALUE parser, so `.env` support costs no dependency.

    The process environment always wins, so `BELOT_AUDIT=0 belot-bot` behaves
    as expected regardless of what the file says. Values are taken verbatim
    after the first `=` — belot cookies contain `=` and `;`, so no splitting
    or unquoting beyond stripping one matched pair of surrounding quotes.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _flag(name, default):
    return os.environ.get(name, "1" if default else "0") == "1"


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

    # --- timing ---
    # If the state has not advanced this long after we dispatched an action,
    # assume the server refused it and re-dispatch. A successful action always
    # changes the turn signature, so this can never double-fire a good move.
    redispatch_after_s: float = 3.0

    warnings: list = field(default_factory=list)

    @classmethod
    def from_env(cls, **overrides):
        """Build from the environment + `.env`, then apply CLI overrides."""
        _load_env_file()
        cfg = cls(
            cookies=os.environ.get("BELOT_COOKIES", "").strip(),
            audit=_flag("BELOT_AUDIT", True),
            frames_path=os.environ.get("BELOT_FRAMES", "frames.jsonl"),
            auto_declare=_flag("BELOT_AUTO_DECLARE", True),
            declare_four_sevens=_flag("BELOT_DECLARE_FOUR_SEVENS", True),
            declare_four_eights=_flag("BELOT_DECLARE_FOUR_EIGHTS", False),
            declare_win_all=_flag("BELOT_DECLARE_WIN_ALL", False),
            auto_swap_seven=_flag("BELOT_AUTO_SWAP_SEVEN", True),
            agent=os.environ.get("BELOT_AGENT", "random"),
            checkpoint=os.environ.get("BELOT_CHECKPOINT", ""),
            reconnect=_flag("BELOT_RECONNECT", True),
            retry_delay_s=float(os.environ.get("BELOT_RETRY_DELAY", 300)),
            rejoin_delay_s=float(os.environ.get("BELOT_REJOIN_DELAY", 5)),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    def require_cookies(self):
        """Fail loudly and usefully rather than joining as nobody."""
        if not self.cookies:
            raise SystemExit(
                "BELOT_COOKIES is not set.\n\n"
                f"  cp .env.example .env      (looked in {ENV_FILE})\n"
                "  then paste your belot.md PHPSESSID and token into it.\n\n"
                "Get them from a logged-in browser: DevTools -> Application "
                "-> Cookies -> https://belot.md"
            )
        return self.cookies
