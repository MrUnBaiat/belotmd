# belot.md — Platform Notes

How the belot.md game server actually behaves, reverse-engineered from live
traffic and from the site's own `gameplay.js`. Nothing here is about this bot's
internals; it is a description of the platform, and it would be equally true for
any client.

Every claim is marked with how it was established:

- **[CONFIRMED]** — verified against captured live traffic, usually many samples.
- **[DERIVED]** — read out of the site's `gameplay.js`.
- **[ASSUMED]** — plausible, never proven. Treat as a place bugs hide.

The single most useful section is [§4.1, the field trust table](#41-field-trust-table).
Most integration bugs are one entry in it being ignored.

---

## 1. Card encoding

Each card is one ASCII character:

```
id = suit * 8 + rank        suit: 0=diamonds 1=hearts 2=clubs 3=spades
                            rank: 0=7 1=8 2=9 3=10 4=J 5=Q 6=K 7=A

diamonds  y z a b c d e f     ids  0- 7
hearts    A B g h i j k l     ids  8-15
clubs     C D m n o p q r     ids 16-23
spades    E F s t u v w x     ids 24-31
```

Note the irregular casing: the 7 and 8 of **hearts, clubs and spades** are
uppercase (`A B` / `C D` / `E F`), but diamonds uses lowercase `y z` for the same
two ranks. Everything else is lowercase. **`A` is the 7 of hearts, not an ace**,
and `C` is the 7 of clubs. Always go through a lookup table; never pattern-match
on case.

Rank order is *natural* (7 < 8 < 9 < 10 < J < Q < K < A), which is exactly
`id % 8`. That matters for sequence combinations ([§7](#7-combinations-and-special-declarations)).
It is **not** the trick hierarchy:

```
trump      7 8 Q K 10 A 9 J
non-trump  7 8 9 J Q K 10 A
```

**Trump suit is 1-based on the wire**: `trump: 1|2|3|4` = diamonds / hearts /
clubs / spades. `-1` or `0` means unset.

---

## 2. Phase enum **[DERIVED from gameplay.js]**

```
 0 NOT_STARTED           8 SWAP_SEVEN          <- the seven-swap window
 1 STARTED               9 DEAL_CARDS_2        <- remaining 3 cards dealt
 2 PUSH_CARDS           10 PLAY
 3 AFTER_PUSH_CARD      11 HAND_TAKE           <- trick resolving
 4 ANIMATION_FIRST_DEAL 12 WIN_ALL_HANDS
 5 DEAL_CARDS_1         13 ROUND_ENDED
 6 TRUMP_CHOOSE_1       14 GAME_ENDED
 7 TRUMP_CHOOSE_2
```

### Observed transition paths **[CONFIRMED]**

```
normal hand    2 -> 5 -> 6 -> 8 -> 9 -> 10 <-> 11 -> 13 -> 2     (round-1 accept)
               2 -> 5 -> 6 -> 7 -> 9 -> 10 <-> 11 -> 13 -> 2     (round-2 pick)
forced trump   2 -> 5 -> 9 -> 10 ...                             (BIZON, skips bidding)
claim tricks            10 -> 12 -> 13 -> 2
deal cancelled          10 ->  1 ->  2                           (LESS_THAN_14 / four 7s)
match over              13 -> 14 -> 0
```

Consequences worth internalising:

- **Phase 8 exists only after a round-1 accept.** Round-2 picks go `7 -> 9`, and
  forced-trump hands go `5 -> 9`. Seven-swap logic gated on phase 8 therefore
  handles both exceptions with no special case.
- **Phase 8 opens after EVERY round-1 accept** and waits on a timeout (~2.5–3.5s),
  even when nobody can possibly swap. It is not a per-player offer. **[CONFIRMED]**
- A client only ever needs to act at phases **2** (cut), **6/7** (bid) and
  **10** (play).
- `10 <-> 11` oscillates once per trick; roughly 165 such transitions per match.

---

## 3. Transport and message protocol

Colyseus over WebSocket, msgpack-encoded.

### Frame layout **[CONFIRMED by decoding a capture]**

```
0x0d                        Colyseus ROOM_DATA
0xA0 | len, <type string>   msgpack fixstr, e.g. 0xAA "SWAP_SEVEN"
<msgpack payload>           omitted entirely if the client sent no payload
```

Useful when identifying a payload from a devtools string dump: each `<?>`
replacement char is one non-UTF8 byte. `<?>SWAP_SEVEN<?>` is a type plus
**exactly one** payload byte, which rules out a card char (`a1 45`, two bytes)
and an object (`81 ab ...`), leaving `{}` (0x80), `null` (0xc0) or `""` (0xa0).

### Client → server **[DERIVED]**

| Type | Payload | Notes |
|---|---|---|
| `READY` | `{"value": true}` | phases 0 / 13 / 14 |
| `PUSH_CARD` | `{"cardPushed": 15}` | deck cut, phase 2 |
| `PASS` | `{}` | bidding |
| `TRUMP_CHOOSE` | `<int 1-4>` | bare scalar |
| `PLAY_CARD` | `"<char>"` | bare scalar |
| `SHOW_COMBINATION` | `"<code>"` | bare scalar, e.g. `"1c"` |
| `SWAP_SEVEN` | `{}` | **[CONFIRMED]** — executed live; the server broadcast came back naming our seat |
| `BOT_ACTIVATION` | `"deactivate"` | **must** be sent on join, or the site's own bot plays your seat |

Also in the enum but unused: `CHAT_MESSAGE`, `REMOVE_PLAYER`,
`CHANGE_PLAYERS_POSITION`, `LEAVE_TABLE`, `IS_TYPING`.

### Server → client **[DERIVED + CONFIRMED]**

`PASS {who}`, `TRUMP_CHOSEN {who, trump}`, `SHOW_COMBINATION {who, value}`,
`SWAP_SEVEN {who, swappedCard, topCard}`, `BIZON {who}`, `LESS_THAN_14 {who}`,
`FOUR_OF_SEVEN`, `FOUR_OF_EIGHT`, `WIN_ALL_HANDS {who}`, `SURRENDER_BT`,
`BELOT_COMBO`, `CHAT_MESSAGE`, `SEND_GIFT`, `IS_TYPING`, `SERVER_MESSAGE`,
`NOTIFICATION`, `ERROR`, `PLAYERS_POSITION_CHANGED`.

**`who` is always a seat index into the `players` array**, never a player id.

### Room close codes **[DERIVED]**

`4001` OTHER_SESSION · `4002` KICKED · `4003` TABLE_REMOVED · `4004` I_LEFT ·
`4005` POSITION_CHANGED.

Meanings, from the platform's own enum plus what each looks like live:

| Code | What happened | Recoverable |
|---|---|---|
| `4001` OTHER_SESSION | the account opened the table somewhere else | **no** — rejoining kicks the other session, which kicks back, forever |
| `4002` KICKED | the host removed us from the table | yes — but the seat we vacated makes that table look *open* again, so a client that just re-queries the lobby is liable to walk straight back in. Skip it explicitly |
| `4003` TABLE_REMOVED | the table dissolved — **the normal end of every match**, and also what a host deleting a table before it starts looks like | yes, go and find another |
| `4004` I_LEFT | we asked to leave | no |
| `4005` POSITION_CHANGED | the host rotated players around the table to set up teams. **Nobody is removed** — only seat indices move. Happens **only between joining a table and the match starting** | yes, and rejoining is *necessary*: every seat-indexed belief now describes a different player |

`4003` is by far the most common in a long run, and it is not a failure. A
client that treats it as one plays a single table and stops.

`4005` is bounded: while a **hand is in progress** — from the deck cut
(phase 2) through to it being scored — the host can no longer rotate seats, so
it cannot occur.

Note the bound is on *now*, not on history. **One room connection hosts many
consecutive matches**: a match ends `13 -> 14 -> 0`, everyone readies up, and
the next one deals in the same room. So a client that latches "a match has
started here" gets it wrong from the second match onward and reports every
legal inter-match reseat as impossible.
That makes it a useful assertion. A `4005` after the cut would mean seat
indices moved underneath every belief we hold, and that the platform does not
behave the way this document claims. **[CONFIRMED by the platform's own
behaviour]**

Separately, a join can fail before any room exists — an empty lobby answers
with no open table at all. That is not a close code and arrives on a different
path, but it needs the same treatment: wait, then look again.

---

## 4. State payload

One frame, trimmed to the fields that matter. The player ids are synthetic
(last digit = seat); everything else is as it arrived:

```json
{
  "players": [
    {"id": "1000000",  "position": 0, "points": 57,
     "cardPlayed": "s", "cardOrder": 1, "numCards": 5, "combinations": ""},
    {"id": "1000001", "position": 1, "points": 0,
     "cardPlayed": "v", "cardOrder": 2, "numCards": 5, "combinations": ""},
    {"id": "1000002", "position": 2, "points": 0, "numCards": 6},
    {"id": "1000003", "position": 3, "points": 0,
     "cards": "yCDrne", "combinationsCanShow": "", "combinations": ""}
  ],
  "currentPhase": 10, "dealer": 2, "activePlayer": 2, "round": 3,
  "topCard": "t", "trump": 2, "declarer": 2, "trumpWasPlayed": true,
  "swapSeven": -1, "lastCards": "xFtu", "targetScore": 101,
  "scoreTable": "[[14,4],[25,11]]",
  "roundTotals": "[{\"p\":116,\"c\":0,\"b\":11},{\"p\":46,\"c\":20,\"b\":7}]",
  "winAllHandsInfo": "", "cardPushed": 15, "timeleft": 250
}
```

- `scoreTable`, `roundTotals` and `winAllHandsInfo` arrive as **JSON strings**,
  not objects. Parse defensively — cells are not always the type you expect.
- `position` is viewer-relative, but `dealer` / `activePlayer` / `declarer` /
  `who` all index the same array, so **array index is the seat everywhere**.
  Turn order is increasing array index, wrapping. **[CONFIRMED]**
- Team parity: columns 0/1 of `scoreTable` are seats {0,2} and {1,3}. **[CONFIRMED]**
- **Only your own hand is visible** (`cards`). Opponents expose `numCards` only.
- `cardPlayed` / `cardOrder` are **absent** for players who have not yet acted;
  `cardOrder` is 1-based within the trick.

### 4.2 The turn clock **[CONFIRMED]**

`timeleft` and `totalTime` are in **deciseconds**. Measured against wall clock
across several countdowns one unit is 0.107s, and the seven-swap window
(`totalTime` 30) closed after the 2.9-3.0s a client logged.

The budget depends on the phase:

| Phase | `totalTime` | Seconds |
|---|---|---|
| 2 cut, 5 deal | 50 | 5.0 |
| 6 / 7 bidding | 120 | 12.0 |
| 8 seven-swap | 30 | 3.0 |
| 10 **play a card** | 250 | **25.0** |

`timeleft` is the remaining budget for **the seat that must act**, not for you
specifically. It is republished on every frame and counts down.

This matters to any client that thinks for a variable amount of time.
Overrunning does not merely forfeit the turn: belot.md hands the seat to its own
bot **for the rest of the session** ([§10](#10-timeouts-and-the-platform-bot)).
Budget against the clock and leave margin for the round trip.

---

### 4.1 Field trust table

The server leaves many fields carrying the **previous hand's value** until the
moment they are recomputed. Every row marked ❌ has caused a real, costly bug.

| Field | Trust | Notes |
|---|---|---|
| `cards` (own) | ✅ | authoritative; also the receipt that a play landed |
| `numCards` | ✅ | the only source of opponent hand sizes |
| `cardPlayed` / `cardOrder` | ✅ | wiped the instant a trick resolves — cache it |
| `activePlayer`, `dealer` | ✅ | |
| `round` | ✅ | increments at the deal; the best hand-boundary signal |
| `points` (per player) | ✅ | live trick captures, **excludes** combinations |
| `lastCards` | ✅ but **seat-indexed** | see [§5](#5-trick-reconstruction) |
| `scoreTable` | ⚠️ | cumulative; cells can be the string `"BT-N"`; [§6](#6-scoring) |
| `roundTotals` | ⚠️ | describes the **previous completed** round, not the live one |
| `trumpWasPlayed` | ⚠️ | resets per hand, but means something later than a naive reading |
| `combinationsCanShow` | ⚠️ **[ASSUMED]** | never proven fresh; gate to phase ≥ 9 defensively |
| `trump` | ❌ **STALE** | previous hand's value through the whole deal and bidding |
| `declarer` | ❌ **STALE** | same |
| `topCard` | ❌ **STALE + REWRITTEN** | stale until phase 5; rewritten by a swap ([§8](#8-the-seven-swap)) |
| `swapSeven` | ❌ **STALE** | carries the previous hand's swapper through the deal |

**The rule:** during phases 0–7, only `cards`, `numCards`, `dealer`,
`activePlayer`, `round`, and `topCard` from phase 5 onward are trustworthy.

**What it costs to get wrong:** a stale `topCard` gives the wrong face-up suit →
a round-2 bid that is illegal → the server rejects it → the turn times out → the
platform bot takes the seat for the rest of the session ([§10](#10-timeouts-and-the-platform-bot)).

---

## 5. Trick reconstruction

`lastCards` is a 4-char string published when a trick resolves.

**It is SEAT-INDEXED, not chronological.** `lastCards[i]` is the card played by
`players[i]`. **[CONFIRMED — 165+ conclusive samples, 0 contradictions]**

Chronological order is seat order rotated by the trick leader, and leaders chain
deterministically:

```
trick 1 leader   = the declarer
trick n+1 leader = the winner of trick n
chronological    = [(L+0)%4, (L+1)%4, (L+2)%4, (L+3)%4]
```

That gives two independent reconstruction paths:

1. **primary** — observed `cardOrder` from table frames;
2. **fallback** — rotate the seat-indexed `lastCards` by the chained leader,
   which survives *total* table-frame loss.

Two traps, both confirmed live:

- The server **wipes the table in the same frame** it publishes `lastCards`, so a
  trick must be accumulated per seat as it fills, not read at completion.
- A single frame can carry a completed trick **and** the next trick's lead.
  Consume the completion first, then rebuild the current trick from what remains,
  or the cache gets clobbered.

---

## 6. Scoring

### 6.1 `scoreTable` — cumulative, with bolt markers

Rows are **cumulative** match scores, not per-round deltas. The match score is
the **last row**, never a column sum.

A cell can be the string `"BT-N"`, meaning *that team bolted this round, scored
0, and it was their N-th bolt.* The cumulative total **carries forward
unchanged** across a bolt row. **[CONFIRMED across four consecutive rows]**

```
[64,72] -> [87,"BT-1"] -> [93,84]     team1: 72 --(bolt, +0)--> 72, then b=12 -> 84 ✓
                                      team0: 64 -> 87 -> 93
[93,84] -> ["BT-1",100] -> [100,111]  team0: 93 --(bolt)--> 93, then b=7 -> 100 ✓
```

**Trap 1:** naive `int("BT-1")` regex-coercion yields **-1**, silently
corrupting the running score. Bolt markers must be parsed, not coerced.

**Trap 2:** the **third bolt also costs 10 points**, and that penalty is
invisible in `scoreTable` because the cell is a string rather than a number.
The platform keeps counting up (`BT-1`, `BT-2`, `BT-3`, ...) rather than
resetting the label, so the penalty is every marker where `n % 3 == 0`.
Carrying the previous total forward therefore overstates the bolted team by 10
until the next numeric row lands. **[CONFIRMED]**

```
row 4  [9, 72]        team 0 on 9
row 5  ['BT-3', 88]   string cell; the true total is -1
row 6  [2, 101]       roundTotals.b says +3, and -1 + 3 == 2 ✓
```

### 6.2 `roundTotals` — the *previous* round

`[{"p":..,"c":..,"b":..}, {...}]`, one entry per team.

- `p` = **trick points**, always summing to **exactly 162** (152 in tricks plus
  the 10-point *pasledu* for the last trick). **[CONFIRMED — 20+ hands]**
- `c` = combination points, **strictly separate from `p`**.
- `b` = match points awarded for that round, or `"BT-N"` on a bolt.

`p + p == 162` is the single best health check on the whole scoring model.

**Trap:** mid-hand, `roundTotals` still describes the *last completed* round — it
sums to 162 while only two tricks have been played. Live trick points must come
from the per-player `points` fields instead.

Round match points are `16 + combos/10` (verified: c=40 → b sums 20; c=20 → 18;
c=70 → 23).

### 6.3 Per-player `points`

Live trick captures for that player, **excluding** combination points.
**[CONFIRMED]** — in a hand scoring `c(120,0)`, the tracked team total matched
`p` exactly and ignored the 120.

Team raw points are `points[0]+points[2]` and `points[1]+points[3]`.

---

## 7. Combinations and special declarations

Wire format `"<type><card_char>"`, pipe-separated: `"5q|1c"`. The card char is
the **highest** card of the combination.

| id | Name | Points | Decodes to cards? |
|---|---|---|---|
| 1 | TART (3-run) | 20 | yes |
| 2 | JUMATE_DE_SUTA (4-run) | 50 | yes |
| 3 | O_SUTA (5-run) | 100 | yes |
| 4 | PATRU_CARTI (four of a kind) | 9s=150, Js=200, 7s/8s=0, else 100 | yes |
| 5 | BELA (Q+K of trump) | 20 | yes |
| 6 | LESS_THAN_14 | 0 | **no** |
| 7 | BELOT_COMBO | 1010 | **no** |
| 8 | WIN_ALL_HANDS | 0 | **no** |
| 9 | SURRENDER_BT | 0 | **no** |

Types 6–9 carry a **filler card char `'a'`** (`"6a"`, `"8a"`) which must never be
decoded into cards. Anchors: `"1c"` = 9,10,J of diamonds; `"5q"` = Q,K of clubs.
**[CONFIRMED]**

Runs use natural rank order, so a run of length *n* ending at rank *r* covers
ranks *r-n+1 .. r*. A run that would fall off the bottom of the suit is
malformed and should be discarded rather than guessed at.

### Runs are reported maximally **[DERIVED]**

The server always shows the **longest** run in the hand, which makes a
declaration say something about the cards it does *not* contain: if the run
could have been extended it would have been, so the ranks immediately outside
it are provably not held.

One exception, and it matters. The enum stops at `O_SUTA`, so a run of six or
more is reported as its **top five**. That preserves the upper edge — the
reported top card is still the highest, or a longer run would have been
reported ending higher — and destroys the lower one, because a six-run is
indistinguishable from a five-run with a card underneath it.

| Declaration | card above absent | card below absent |
|---|---|---|
| 3-run `1x` | ✅ | ✅ |
| 4-run `2x` | ✅ | ✅ |
| 5-run `3x` | ✅ | ❌ — could be the top of a longer run |
| four of a kind `4x` | — | — |
| bella `5x` | — | — |

Four of a kind and bella imply nothing: they are not runs, and holding a
neighbouring rank would not have changed what was declared.

Worth the care, because a wrong exclusion is a **false void** — it silently
removes a real candidate from every belief and every sampled world, and
nothing downstream can tell it from a true one.

### Scoring rules **[CONFIRMED by cross-checking `c` against declarations]**

- **Only the best sequence scores.** In one hand a Q-high and a 9-high tercă
  (40 points together) were both cancelled by an opponent's K-high tercă:
  `c(0,20)`. In another, an A-high tercă beat everything and **both** tercăs
  scored: `c(40,20)`.
- **Bella scores independently** of the sequence contest.
- Combination points never touch `p`.

### Two four-of-a-kinds that are not about points

- **Four 7s** (`FOUR_OF_SEVEN`) scores 0 and **cancels the deal**, exactly like
  `LESS_THAN_14`.
- **Four 8s** (`FOUR_OF_EIGHT`) scores 0 and **silences every combination except
  bella** — including the declarer's own.

Both are declarations a naive "announce everything offered" policy will fire by
accident, with effects nothing like the points it expected.

### Availability

Declarations exist **only during play, after all eight cards are dealt**
(phase ≥ 9). **[CONFIRMED]** Anything read from `combinationsCanShow` earlier
should be treated as stale — it may belong to the previous hand.

The server echoes **your own** declarations back as `SHOW_COMBINATION {who: you}`.

Opponents' declarations are **information**: they prove which cards that player
holds. Four 8s is worth decoding even by a client that would never declare it.

---

## 8. The seven-swap

A player holding the 7 of trump may trade it for the face-up card during phase 8.
The server sends **no "you may swap" event** — every precondition is deduced
locally:

1. phase 8 (⟹ a round-1 accept, ⟹ trump == face-up suit);
2. the player holds `trump * 8` — and it must be among the **first five cards**,
   since the rest arrive at phase 9, after the window closes;
3. the face-up card is not itself that 7.

**Server broadcast:** `SWAP_SEVEN {who, swappedCard, topCard}`. Afterwards `who`
holds the face-up card, and the natural recipient of the face-up card — the
declarer in a round-1 accept — holds the 7. Both are certainties.

**Two traps, both confirmed live:**

- `topCard` is **rewritten** from the flipped card to the 7 the declarer received
  (`"w"` → `"E"` in one capture). Everything downstream needs the *original*, so
  the face-up value must be frozen from phase 8 onward.
- `swapSeven` is the **seat index** of the swapper (`-1` = none), and it is
  **stale through the next deal**. Honouring it early re-arms a phantom swap
  immediately after the reset, freezing the face-up card on the previous hand's
  value for the whole hand.

The window waits on a timeout rather than closing as soon as a swap happens, so
a client has the full ~3s to decide.

**The outgoing payload is `{}`, confirmed by execution.** A live swap sent
`SWAP_SEVEN {}` and the server broadcast `{who: <our seat>, swappedCard: "E",
topCard: "v"}` back, after which the 7 had left our hand and the face-up card
was in it. The window closed in 1.5s rather than timing out at ~3s, which is
itself a tell that the swap was accepted.

---

## 9. Abnormal hand endings

| Event | Path | Scores? | Notes |
|---|---|---|---|
| `LESS_THAN_14` | `10 -> 1 -> 2` | **no** | deal cancelled, duplicate `scoreTable` row |
| `FOUR_OF_SEVEN` (four 7s) | same | **no** | identical effect |
| `WIN_ALL_HANDS` | `10 -> 12 -> 13` | yes | claimer takes every remaining trick |
| `SURRENDER_BT` | early → 13 | yes | claimer concedes and takes a bolt |
| `FOUR_OF_EIGHT` (four 8s) | none | n/a | hand continues; combinations except bella stop scoring |

A deal can be cancelled **at any point** — during bidding, during the swap
window, mid-play. Two subtleties:

- On cancellation the server appends a **duplicate cumulative row**, so the round
  delta is `(0,0)` and `roundTotals` still describes the previous *scored* hand.
  Cross-checking the delta against `b` there is meaningless.
- `round` incremented on every observed cancellation, but nothing guarantees it.
  A reused round number must not suppress a client's hand reset — key the abort
  on the phase transition (any live phase → a deal phase), not on `round`.
  Otherwise the aborted hand's history leaks into the fresh deal.

A `FOUR_OF_EIGHT` declaration is still a true statement about card holdings, so
information derived from it survives even though it scores nothing.

---

## 10. Timeouts and the platform bot

**The most consequential platform behaviour, and the least documented.**

When a turn times out, belot.md hands the seat to its own bot **for the rest of
the session**. From that moment the server silently ignores every message the
original client sends.

The signature in a log is unmistakable once you know it: the client appears to
play the *same card* in several consecutive tricks. The explanation is that the
hand still shrinks — the platform bot is playing it — but it happens never to
play that one card, so it stays in hand, stays legal, and a deterministic client
keeps selecting it. The tricks resolve normally around it.

Detection: `players[me].bot == true` in the state, and more directly, **a
dispatched card that never leaves your hand**. The card leaving your `cards`
string is the server's only confirmation that a play was accepted; anything else
is a rejection.

`BOT_ACTIVATION: "deactivate"` must be sent on join, but it does not protect
against this — it only stops the bot playing your seat *before* a timeout.

**The best defence is not triggering a timeout at all.** A rejected action that
is simply re-sent will be rejected again until the turn expires, so a retry has
to block the refused action and choose differently.

---

## 11. Still unverified

1. **`trumpWasPlayed` semantics.** It resets per hand and flips mid-hand, but it
   does not track "the declarer has played a trump" on the timing a client needs,
   so it is safer to derive that locally.
3. **Is `combinationsCanShow` stale across hands?** Never proven either way. It
   is gated to phase ≥ 9 defensively.
4. **Offer window width.** `combinationsCanShow` has been observed persisting for
   a run of frames, but it is not a latched field — a client that misses the run
   loses the declaration silently, and declarations are worth 20–200 points each.
