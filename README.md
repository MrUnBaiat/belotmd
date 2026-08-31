# belot.md bot

A bot that sits down at a real table on [belot.md](https://belot.md) and plays
Belot against human opponents.

The platform has no API. It speaks msgpack over Colyseus WebSockets, publishes a
partial and frequently stale view of the table, and hands your seat to its own
bot the moment a turn times out. Most of this repository is the work of turning
that stream into a consistent game state that something can actually make
decisions from — and of proving, offline and repeatably, that the reconstruction
is correct.

The decision-making itself is a plug-in. The bundled agent is a recurrent MAPPO
policy trained by self-play elsewhere; a `RandomAgent` ships alongside it so the
bot runs end to end with no weights and no PyTorch.

---

## How it works

```
belot.md  (Colyseus / WebSocket, msgpack)
    │
    ▼
platform/bridge.js      Node daemon. Authenticates with browser cookies, finds
    │                   or joins a table, relays raw state over a local socket.
    ▼
platform/client.py      Message plumbing. Serialized, in-order delivery; no
    │                   game state of its own.
    ▼
platform/sync.py        The hard part. Raw frames -> a consistent BelotState,
    │                   including belief information no single frame contains:
    │                   voids, declared combinations, seven-swaps.
    ▼
bot.py                  Platform rules: when it is our turn, what to declare,
    │                   when the swap window is open, whether the server
    │                   actually accepted the card we sent.
    ▼
agents/                 The decision. Anything implementing `Agent`.
                        · random  — uniform over legal actions, no dependencies
                        · ppo     — the trained recurrent policy

audit.py                Runs alongside, asserting invariants on every frame and
                        recording each one to frames.jsonl for offline replay.
tools/frame_inspector.py  A second opinion that shares no code with the above.
```

Two documents carry the substance:

- **[docs/PLATFORM_NOTES.md](docs/PLATFORM_NOTES.md)** — how belot.md actually
  behaves. Card encoding, phase enum, message protocol, the field trust table,
  trick reconstruction, scoring quirks, the seven-swap, the seat-takeover
  behaviour. Every claim is marked CONFIRMED / DERIVED / ASSUMED.
- **[docs/PPO_AGENT.md](docs/PPO_AGENT.md)** — the reference agent's observation
  contract, for anyone reproducing or replacing it.

---

## Install

```bash
npm install                     # colyseus.js + ws, for the bridge
pip install -e ".[ppo]"         # omit [ppo] to skip PyTorch
```

Credentials come from a `.env` file:

```bash
cp .env.example .env
```

Log in to belot.md in a browser, open DevTools → Application → Cookies, and
paste `PHPSESSID` and `token` into `.env`. It is gitignored. The bot joins as
that account, so use one you are willing to run a bot on.

Weights are not in the repository. Download `latest_model.pt` from the
[Releases](../../releases) page into `checkpoints/`, or run without it:

```bash
belot-bot --agent random        # no weights, no torch, plays legal moves
belot-bot                       # the trained policy
```

Without a checkpoint the `ppo` agent prints a loud warning and plays with
random weights rather than failing silently.

## Usage

```bash
belot-bot --agent ppo --checkpoint checkpoints/latest_model.pt
belot-bot --list-agents
belot-bot --no-audit            # skip frame recording and invariant checks
```

Every flag has a `BELOT_*` environment variable; see `.env.example` for the
declaration and seven-swap switches.

## Writing your own agent

The seam is small on purpose. `bot.py` knows the rules of belot.md; an agent
knows how to choose:

```python
from belotmd.agents.base import register

class MyAgent:
    name = "greedy"

    def reset(self):
        """New hand. Clear anything carried across turns."""

    def snapshot(self):
        """State to restore if the server refuses our move. None if stateless."""
        return None

    def restore(self, snapshot):
        """Undo act()'s state change before retrying the same turn."""

    def act(self, state, seat, match_scores, legal_mask):
        # state is a BelotState: hands, trump, current_trick, graveyard,
        # known_cards, impossible_cards, ...
        # legal_mask is authoritative — never return a masked-out index.
        return int(legal_mask.argmax())

@register("greedy")
def _build(**kwargs):
    return MyAgent()
```

Import it, then `belot-bot --agent greedy`. Nothing in the bridge, the
synchronizer or the auditor needs to change — and nothing in your agent needs to
know that belot.md exists.

## Verification

The bot records every raw frame it sees, which makes any live bug reproducible
offline, deterministically, forever.

```bash
pytest                                                   # regression suite
python -c "from belotmd.audit import replay; replay('frames.jsonl')"
python tools/frame_inspector.py frames.jsonl             # independent second opinion
```

`replay` re-runs a recording through a fresh synchronizer and reports every
invariant violation with its frame number. `frame_inspector.py` deliberately
imports nothing from the sync layer, so a bug there cannot skew its verdict.

## Scope and honesty

- The seven-swap and the less-than-14 deal cancellation are handled by the live
  bridge, but the self-play simulator that trained the policy implements
  neither. The policy has never seen those mechanics; the bot applies them by
  rule. See [docs/PPO_AGENT.md §7](docs/PPO_AGENT.md).
- Declaring combinations is rule-driven, not learned. There is no combination
  action in the 38-slot action space; declared cards reach the policy as belief
  information instead.
- Once belot.md hands your seat to its own bot, the seat is not reclaimed. This
  is deliberate: it is logged loudly so that no play after that point is ever
  mistaken for the model's own.

## Layout

```
src/belotmd/
  platform/    bridge.js, client, wire protocol, state synchronizer
  game/        Belot rules, hand state, action space, combination decoding
  agents/      the Agent seam; random and ppo implementations
  bot.py       platform rules and orchestration
  audit.py     invariants, frame recording, offline replay
tools/         standalone forensics over a recording
tests/         regression suites built from real captured frames
docs/          platform notes and the reference agent's contract
```

## License

MIT.
