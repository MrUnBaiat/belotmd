# Writing an agent

An agent decides what to play. It does not talk to belot.md, track whose turn
it is, declare combinations, or handle the server refusing a move — the SDK
does all of that. You implement one method that matters.

## The contract

```python
class Agent(Protocol):
    name: str

    def reset(self) -> None: ...
    def act(self, state, seat, match_scores, legal_mask) -> int: ...
    def snapshot(self): ...
    def restore(self, snapshot) -> None: ...
```

| Method | When it is called | Stateless agents |
|---|---|---|
| `act` | on every turn that is yours | the only one you must write |
| `reset` | at every hand boundary | no-op |
| `snapshot` / `restore` | around a turn, to survive a rejected move | `return None` / no-op |

`act` returns an index into the 38-slot action space
(`belotmd.game.actions`): `0-31` play that card, `32` pass, `33` accept the
face-up card, `34-37` name a suit.

**`legal_mask` is authoritative.** It is sometimes *narrower* than
`state.get_legal_actions()`, because the server refused something earlier in
the same turn and the bot is retrying. Return a masked-out index and the move
is rejected; do it enough times and belot.md gives your seat to its own bot
for the rest of the session.

## What you can see

`state` is a `BelotState`:

```python
state.hands[seat]          your eight (then seven, six, …) cards
state.trump                0-3, or None during bidding
state.declarer             seat that chose trump
state.face_up_card         the flipped card
state.current_trick        [(seat, card), …] in play order
state.last_trick           the previous complete trick
state.graveyard            every card already played
state.tricks_played        0-8
state.raw_points_by_team   trick points captured so far
state.bolts_by_team        bolt counters
state.get_legal_actions()  the full legal mask
state.card_value(card, is_trump) -> (points, trick power)
```

### The one thing that will catch you out

**`state.hands` is only real for your own seat.** belot.md never reveals
another player's cards. The other three entries are placeholder ids with the
correct *length* and meaningless contents, because the length is genuinely
known (`numCards`) and the contents are not.

Read them and you are searching a fantasy. What is actually known lives in:

```python
state.known_cards[seat]       cards that seat provably HOLDS
state.impossible_cards[seat]  cards that seat provably CANNOT hold
```

`known_cards` is populated from declared combinations (a declared 5-card run
pins five cards at once), the seven-swap, and the face-up card.
`impossible_cards` is populated from voids — a player who fails to follow suit
cannot hold that suit.

Reconstructing those two arrays from a partial, frequently stale server feed is
most of what this SDK does.

## The belief layer

`belotmd.game.belief` turns those constraints into something usable:

```python
from belotmd.game.belief import belief_matrix, unseen_cards, can_hold

bel = belief_matrix(state, seat)   # (3, 32): P(opponent holds card)
                                   # rows are left, partner, right
pool = unseen_cards(state, seat)   # card ids whose location is unknown
can_hold(state, opponent, card)    # bool
```

`belief_matrix` is a single normalisation pass: rows sum exactly to each
opponent's unknown-card count, columns are approximate. A card only one
opponent could hold reads ~0.90, not 1.00. That is intended.

## Three worked shapes

### Heuristic

```python
from belotmd import register
import numpy as np

class HighestLegal:
    name = "highest"
    def reset(self): pass
    def snapshot(self): return None
    def restore(self, s): pass

    def act(self, state, seat, match_scores, legal_mask):
        cards = [a for a in np.flatnonzero(legal_mask) if a < 32]
        if not cards:
            return int(np.flatnonzero(legal_mask)[0])   # a bidding turn
        return max(cards, key=lambda c: state.card_value(
            c, is_trump=(c // 8 == state.trump))[0])

register("highest")(lambda **kw: HighestLegal())
```

### Search (ISMCTS / determinization)

You cannot search the real game, because you do not know the other hands. You
sample consistent worlds instead.

**Read the constraints from `constraints()`, never off `state.hands`.** An
opponent's `hands` entry holds placeholder ids of the right length — live it
can read `[0, 1, 2, 3, 4, 5]`, which looks exactly like the 7 through Q of
diamonds and is not:

```python
from belotmd import constraints

c = constraints(state, seat)
c.own_hand          # your real cards
c.hand_sizes        # (n0, n1, n2, n3) -- authoritative
c.pool              # unplaced cards, location unknown
c.pins[p]           # cards seat p provably HOLDS
c.voids[p]          # cards seat p provably CANNOT hold
c.played            # graveyard + table
c.degraded          # joined mid-hand: constraints are INCOMPLETE
c.need(p)           # how many pool cards p still takes
c.candidates(card)  # which seats could hold it
```

That is the complete input to any sampler. Bring your own, or use the one here:

```python
from belotmd import sample_determinization, Infeasible

hands = sample_determinization(state, seat, rng)   # four card lists, by seat
```

It places the most-constrained card first, because a naive shuffle-and-fill
dead-ends surprisingly often once voids accumulate — and a dead-end quietly
patched yields an *illegal* world, which is worse than no world at all.
Genuinely contradictory constraints raise `Infeasible` rather than being
guessed past.

**Using your own sampler is expected**, especially if you want a particular
distribution — importance-weighted by `belief_matrix`, say. Hold it to the
same standard with the validator:

```python
c.check(my_hands)        # raises Infeasible naming the violated constraint
```

**Mind the clock.** `state.deadline` is a `time.monotonic()` stamp by which the
action must be on the wire; `state.time_left_s` is the same as a duration. A
card gets 25s, a bid 12s, the seven-swap window 3s. Overrunning loses the
*seat*, not just the turn, so budget explicitly:

```python
import time

while time.monotonic() < state.deadline - self.safety_margin:
    self.one_more_world()
```

### Learned

Encode `state` into whatever your network expects, run it, mask the logits with
`legal_mask`, argmax. If your policy is recurrent, `reset()` zeroes the hidden
state and `snapshot`/`restore` make a retry replay from the pre-turn state so
the recurrence advances once per *decision* rather than once per network
message.

## Shipping it in your own package

Register through an entry point and the SDK finds it with no code change:

```toml
# your package's pyproject.toml
[project.entry-points."belotmd.agents"]
my-agent = "my_package.agent:build"
```

`build` is any callable returning an `Agent`; `--agent-arg KEY=VALUE` on the
command line becomes keyword arguments to it.

```bash
pip install my-agent-package
belot-bot --agent my-agent --agent-arg checkpoint=weights.pt
```

Entry points are loaded lazily, so a heavy agent costs nothing until it is
actually selected.

## Testing without a table

You do not need a live game — or even a network — to develop an agent:

```python
from belotmd import BelotState
state = BelotState()                      # a random dealt hand
mask = state.get_legal_actions()
agent.act(state, state.current_player, [0, 0], mask.astype("int8"))
```

And against real captured traffic, if you have a recording:

```python
from belotmd.platform.sync import StateSynchronizer
import json

sync = StateSynchronizer()
for line in open("frames.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    seat = sync.sync(rec["state"], rec["pid"])
    # sync.state is now exactly what the live bot would have handed your agent
```

That replay is deterministic and runs offline, which makes every bug you ever
hit live reproducible forever.
