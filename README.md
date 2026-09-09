# belotmd

An SDK for building bots that play Belot on [belot.md](https://belot.md).

The platform has no API. It speaks msgpack over Colyseus WebSockets, only ever
shows you your own cards, and publishes a state payload where half the fields
carry the *previous* hand's value until the moment they are recomputed. Get one
of them wrong and your move is refused, your turn times out, and the site hands
your seat to its own bot for the rest of the session.

This library handles all of that. You write the part that decides which card to
play.

```python
from belotmd import register
import numpy as np

@register("my-agent")
class MyAgent:
    def act(self, state, seat, match_scores, legal_mask):
        """`legal_mask` is authoritative — never return a masked-out index."""
        return int(np.random.choice(np.flatnonzero(legal_mask)))
```

```bash
belot-bot --agent my-agent
```

That is the whole contract. `act` is the only method you must write.

---

## What you get

**A reconstructed game state.** `sync.py` turns a stream of partial, stale
frames into a consistent `BelotState` — including the belief information no
single frame contains: suit voids inferred from play, cards proven by declared
combinations, the seven-swap, the face-up card's recipient.

**A belief layer.** `belief_matrix(state, seat)` gives you `P(opponent holds
card)` as a `(3, 32)` array, derived from everything provable. `unseen_cards()`
gives you the pool to deal from if you are sampling determinizations for a
search.

**The platform's rules, handled.** Auto-ready, the deck cut, which combinations
to announce and which are traps, the phase-8 seven-swap window, detecting that
the server silently ignored your move, detecting that your seat was taken.
None of it is your agent's problem.

**Verification that works offline.** Every frame is recorded; `replay()` re-runs
a recording through a fresh synchronizer deterministically, and an independent
inspector that shares no code with the sync layer gives a second opinion. Any
bug you hit live is reproducible forever.

**Documented protocol.** [docs/PLATFORM_NOTES.md](docs/PLATFORM_NOTES.md) is the
reverse-engineering write-up: card encoding, phase enum, message protocol, the
field trust table, trick reconstruction, scoring quirks, the seat-takeover
behaviour. Every claim marked CONFIRMED / DERIVED / ASSUMED.

## Install

Not on PyPI yet — clone and install in place:

```bash
git clone https://github.com/MrUnBaiat/belotmd
cd belotmd
pip install -e ".[live]"      # numpy + websockets
npm install                   # colyseus.js + ws, for the bridge
```

`pip install -e .` without the extra gives you the rules and belief layers with
numpy as the only dependency — enough to write and test an agent entirely
offline. The `live` extra adds what is needed to actually join a table, and the
bridge needs Node 18+.

Credentials go in `.env`:

```bash
cp .env.example .env
```

Log in to belot.md in a browser, open DevTools → Application → Cookies, and
paste `PHPSESSID` and `token`. The file is gitignored. The bot plays as that
account, so use one you are willing to run a bot on.

```bash
belot-bot --list-agents
belot-bot --agent random        # ships with the SDK; plays legal, plays badly
```

It keeps playing on its own. A match ending dissolves the table, so the bot
goes back to the lobby and finds another; an empty lobby or a kick means
waiting five minutes and looking again. `--once` plays a single table and
exits; `--retry-delay` and `--rejoin-delay` tune the two waits.

## Architecture

```
belot.md  (Colyseus / WebSocket, msgpack)
    │
    ▼
platform/bridge.js      Node daemon. Authenticates with cookies, finds or
    │                   joins a table, relays raw state over a local socket.
    ▼
platform/client.py      Message plumbing. Serialized, in-order delivery.
    │
    ▼
platform/sync.py        Raw frames -> a consistent BelotState, with belief.
    │
    ▼
bot.py                  Platform rules: turn detection, declarations, the
    │                   swap window, rejection handling, seat takeover.
    ▼
agents/                 Your code.

audit.py                Invariants on every frame; records to frames.jsonl.
tools/frame_inspector.py  Second opinion — imports nothing from sync.py.
```

Three layers, lowest first:

| Module | Knows about belot.md? | Dependencies |
|---|---|---|
| `belotmd.game` | no — pure Belot rules and belief | numpy |
| `belotmd.platform` | yes — bridge, protocol, sync | numpy, websockets |
| `belotmd.agents` | no — the seam | numpy |

The SDK never imports a machine-learning framework. An agent that needs one
brings it along.

## Writing an agent

The full guide is [docs/WRITING_AN_AGENT.md](docs/WRITING_AN_AGENT.md), with
worked examples for heuristic, search (ISMCTS / determinization) and learned
agents. The short version:

```python
def act(self, state, seat, match_scores, legal_mask) -> int:
    """Return an index into the 38-slot action space:
       0-31 play that card · 32 pass · 33 accept face-up · 34-37 name a suit.
       legal_mask is authoritative — never return a masked-out index."""
```

Plus `reset()` at hand boundaries, and `snapshot()`/`restore()` if you carry
state across turns and want a rejected move to replay cleanly. All three are
no-ops for a stateless agent.

**One thing to know before you start:** `state.hands` is only real for your own
seat. belot.md never reveals another player's cards, so the other three entries
are placeholders with the correct length and meaningless contents. What is
genuinely known lives in `state.known_cards`, `state.impossible_cards` and
`state.graveyard` — and in the belief matrix built from them.

### Shipping an agent as its own package

```toml
[project.entry-points."belotmd.agents"]
my-agent = "my_package.agent:build"
```

`pip install` it and `belot-bot --agent my-agent` works with no change to this
library. Entry points load lazily, so a heavy agent costs nothing until it is
selected.

## Verification

```bash
pytest                                    # 128 tests, no network needed
```

After a live session you also have a recording to replay:

```bash
python -c "from belotmd.audit import replay; replay('frames.jsonl')"
python tools/frame_inspector.py frames.jsonl
```

`replay` reports every invariant violation with its frame number.
`frame_inspector.py` deliberately imports nothing from the sync layer, so a bug
there cannot skew its verdict.

Recordings contain other players' usernames and account ids. `frames*.jsonl` is
gitignored; keep it that way.

## Scope

- This is a client library, not a strategy. The bundled `random` agent exists to
  prove the pipeline works end to end, not to win.
- Once belot.md hands your seat to its own bot, the seat is not reclaimed. That
  is deliberate, and it is logged loudly, so no play after that point is ever
  mistaken for your agent's.
- Declaring combinations is handled by rule, with the two traps (four 8s
  silences your own combinations, `SURRENDER_BT` concedes) off by default.
  See `.env.example`.

## Related

The reference learned agent — a recurrent MAPPO policy trained by self-play —
lives in its own repository and installs as a plugin. See
[docs/MIGRATION.md](docs/MIGRATION.md) for how that split works and what it
requires.

## License

MIT — see [LICENSE](LICENSE).
