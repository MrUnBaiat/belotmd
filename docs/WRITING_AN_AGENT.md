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
sample consistent worlds instead:

```python
def determinize(self, state, seat, rng):
    """One plausible full deal, consistent with everything known."""
    pool = [c for c in unseen_cards(state, seat)
            if not state.known_cards[:, c].any()]
    rng.shuffle(pool)

    hands = {seat: list(state.hands[seat])}
    for p in [(seat + i) % 4 for i in (1, 2, 3)]:
        hand = list(np.flatnonzero(state.known_cards[p]))   # certainties first
        need = len(state.hands[p]) - len(hand)
        # then fill from the pool, skipping cards p provably cannot hold
        take = [c for c in pool if not state.impossible_cards[p, c]][:need]
        for c in take:
            pool.remove(c)
        hands[p] = hand + take
    return hands
```

Sample *N* worlds, search each with your favourite algorithm, aggregate the
root statistics, return the best action. `belief_matrix` is a reasonable
importance-sampling weight if uniform determinization is too crude.

Note the constraint satisfaction is not guaranteed to succeed on a naive greedy
fill — with tight void constraints you may need to retry or use a matching
algorithm. Budget for that.

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
