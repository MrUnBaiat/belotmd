"""
test_config_accounts.py — two accounts, two credential files, no crosstalk.

`_load_env_file` folded `.env` into `os.environ` with `setdefault`. For one bot
that is exactly right: the shell should be able to override the file. For two
it is a trap -- whichever account loaded first owns `BELOT_COOKIES` for the
whole process, so the second one silently plays as the first. Worse, a stray
`BELOT_COOKIES` in the shell would quietly decide both.

`from_env(env_file=...)` is the way out, and these pin its two properties: the
file decides the account, and nothing global is touched.
"""

import os

import pytest

from belotmd.config import ENV_FILE, Config, _read_env_file

COOKIES_A = "PHPSESSID=aaaaaaaa; token=AAAA;"
COOKIES_B = "PHPSESSID=bbbbbbbb; token=BBBB;"


def _env_file(tmp_path, name, cookies, extra=""):
    path = tmp_path / name
    path.write_text(f"BELOT_COOKIES={cookies}\n{extra}", encoding="utf-8")
    return path


def test_two_accounts_get_their_own_cookies(tmp_path):
    """THE BUG: with `setdefault`, the second call returned the first
    account's cookies and nothing said so."""
    a = _env_file(tmp_path, ".env.a", COOKIES_A)
    b = _env_file(tmp_path, ".env.b", COOKIES_B)

    cfg_a = Config.from_env(env_file=a)
    cfg_b = Config.from_env(env_file=b)

    assert cfg_a.cookies == COOKIES_A
    assert cfg_b.cookies == COOKIES_B, "the second account must not inherit"

    # Order must not matter either.
    assert Config.from_env(env_file=a).cookies == COOKIES_A


def test_reading_an_account_file_changes_nothing_global(tmp_path):
    """The other half: a second process-wide side effect would reintroduce the
    same bug through a different door."""
    before = dict(os.environ)
    Config.from_env(env_file=_env_file(tmp_path, ".env.a", COOKIES_A))
    assert dict(os.environ) == before


def test_a_stray_shell_cookie_cannot_pick_the_account(tmp_path, monkeypatch):
    """Running the pair from one shell must not let an exported cookie decide
    who both bots are."""
    monkeypatch.setenv("BELOT_COOKIES", "PHPSESSID=leftover; token=OLD;")
    cfg = Config.from_env(env_file=_env_file(tmp_path, ".env.a", COOKIES_A))
    assert cfg.cookies == COOKIES_A


def test_an_account_file_without_cookies_fails_loudly(tmp_path):
    path = tmp_path / ".env.empty"
    path.write_text("BELOT_AGENT=random\n", encoding="utf-8")

    cfg = Config.from_env(env_file=path)
    with pytest.raises(SystemExit) as exc:
        cfg.require_cookies()
    assert str(path) in str(exc.value), "must say which file it looked in"


def test_other_knobs_still_fall_back_to_the_environment(tmp_path, monkeypatch):
    """Only the credentials are file-only; the rest keeps working as before."""
    monkeypatch.setenv("BELOT_AGENT", "composite")
    cfg = Config.from_env(env_file=_env_file(tmp_path, ".env.a", COOKIES_A))
    assert cfg.agent == "composite"


def test_the_account_file_wins_for_ordinary_knobs_too(tmp_path, monkeypatch):
    monkeypatch.setenv("BELOT_AGENT", "random")
    cfg = Config.from_env(env_file=_env_file(tmp_path, ".env.a", COOKIES_A,
                                             extra="BELOT_AGENT=composite\n"))
    assert cfg.agent == "composite"


def test_overrides_still_win_over_the_file(tmp_path):
    cfg = Config.from_env(env_file=_env_file(tmp_path, ".env.a", COOKIES_A),
                          frames_path="sessions/frames_a.jsonl")
    assert cfg.frames_path == "sessions/frames_a.jsonl"


def test_the_single_account_path_is_unchanged(monkeypatch):
    """No `env_file`: the environment still decides, as every existing run
    depends on."""
    monkeypatch.setattr("belotmd.config._load_env_file", lambda *a, **kw: None)
    monkeypatch.setenv("BELOT_COOKIES", COOKIES_B)
    cfg = Config.from_env()
    assert cfg.cookies == COOKIES_B
    assert cfg.env_file == ""
    assert ENV_FILE.name == ".env"


# ------------------------------------------------------------- the parser
def test_a_missing_file_is_not_an_error(tmp_path):
    assert _read_env_file(tmp_path / "nope") == {}


def test_cookies_survive_verbatim(tmp_path):
    """Cookie strings contain `=` and `;`; splitting on them would corrupt
    the credential in a way that only shows up as a failed login."""
    path = tmp_path / ".env"
    path.write_text(f'BELOT_COOKIES="{COOKIES_A}"\n# a comment\n\n',
                    encoding="utf-8")
    assert _read_env_file(path)["BELOT_COOKIES"] == COOKIES_A
