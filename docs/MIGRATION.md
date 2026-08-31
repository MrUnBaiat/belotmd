# Migrating the reference agent out of this repo

This SDK used to vendor a recurrent MAPPO agent — `observation.py`,
`policy.py` and the checkpoint loader — copied from the repository where the
model is trained. That copy is gone. This note records what moved, why, and
what the training repo has to do to pick it up.

## Why

The observation encoding is not a property of belot.md; it is part of the
trained checkpoint's serialization format. Whoever owns the weights owns the
encoder, or the two drift and the drift is silent.

The worse duplication was the rules. `game/state.py` here and `env.py` there
were the same file, forked. Both define `get_legal_actions()`. If they diverge,
the model trains against one rulebook and plays against another, and the
failure mode is a move the server refuses → the turn times out → belot.md
takes the seat for the rest of the session. That is the most expensive bug
class in the project, and a fork makes it inevitable rather than unlikely.

## What moved where

| Was | Now |
|---|---|
| `belotmd/agents/ppo/observation.py` | training repo — it is that model's contract |
| `belotmd/agents/ppo/policy.py` | training repo — it is that model's architecture |
| `belotmd/agents/ppo/agent.py` | training repo, as the `Agent` adapter |
| `docs/PPO_AGENT.md` | training repo |
| the belief-matrix computation | **stayed**, promoted to `belotmd.game.belief` |

The belief matrix moved *up* rather than out. It is derived entirely from
`known_cards`, `impossible_cards` and `graveyard` — all of which this SDK
reconstructs — so it belongs here, and every agent wants it, not just a
learned one. It was extracted verbatim: 1432 belief matrices computed from
real captured frames are byte-identical to the values the previous inline
implementation produced.

## What the training repo needs to do

### 1. Depend on the SDK

```toml
dependencies = ["belotmd>=1.0"]
```

### 2. Build the training environment on the shared rules

Instead of a forked `env.py`:

```python
from belotmd.game.state import BelotState

class BelotTrainingEnv(BelotState):
    """Self-play simulation on top of the shared rules."""

    def step(self, action): ...
    def _evaluate_trick(self): ...
    def _calculate_final_rewards(self): ...
```

`BelotState` provides `reset()`, `_finalize_bidding()`, `get_legal_actions()`
and `card_value()`. Everything the trainer adds is simulation: stepping,
resolving tricks, scoring, rewards — none of which the live bridge needs,
because the belot.md server is the authority on what happened.

**Verify before retraining.** Subclassing must not change `get_legal_actions()`
behaviour, or the checkpoint's assumptions shift under it. Replay a captured
`frames.jsonl` through both the old and new environments and diff the masks at
every decision point; they must be identical.

### 3. Use the shared belief matrix in the encoder

The `# 10. Belief State Matrix (96)` block in `observation.py` becomes:

```python
from belotmd.game.belief import belief_matrix

bel = belief_matrix(belot, abs_id)          # (3, 32), rows L/P/R
obs[idx:idx + 96] = bel.reshape(-1)
idx += 96
```

This is a drop-in replacement — verified byte-identical — and it removes the
last copy of that logic.

### 4. Adapt the agent

`agent.py` already implements the SDK's protocol; only the imports change:

```python
from belotmd.agents.base import Agent   # for typing, optional
from .observation import build_observation
from .policy import RecurrentMAPPOModel
```

Keep `snapshot()` / `restore()` returning and restoring the LSTM hidden state.
That is what makes a rejected move replay from the pre-turn state, so the
recurrence advances once per decision rather than once per network message —
matching how the policy was rolled out in training.

### 5. Advertise it as an entry point

```toml
[project.entry-points."belotmd.agents"]
ppo = "belot_agent.agent:PPOAgent"
```

After that, `belot-bot --agent ppo --agent-arg checkpoint=weights.pt` works
with nothing installed from this repo but the SDK itself. No glue repo, no
import wiring, no third project.

## Recording human data

The frame recorder in `belotmd.audit` is agent-agnostic and stays here. It
writes **raw server frames**, deliberately — not encoded observations — so
every recording survives a change to your observation layout. Today's captures
will still be trainable after the next architecture.

Two things the training repo's dataset builder needs to know:

1. **A completed hand is fully observable in retrospect.** You see only your
   own cards live, but `lastCards` is published per trick and is seat-indexed,
   so eight tricks reveal all thirty-two cards. After a hand ends you can
   reconstruct every player's true hand at every point in it — and therefore
   reconstruct each human opponent's actual decision context. Each recorded
   hand is three human trajectories, not zero.

2. **Filter contaminated spans.** After belot.md hands your seat to its own bot
   (`players[me].bot == true`, and the `[Bot][SEAT LOST]` banner in the log),
   the plays at your seat are neither yours nor a human's. Drop those spans.
   `tools/frame_inspector.py` already knows how to split a recording into
   matches.
