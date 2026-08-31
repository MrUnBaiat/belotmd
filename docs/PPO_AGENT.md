# The reference agent: observation contract

This describes **one** agent — `belotmd.agents.ppo`, a recurrent MAPPO policy
trained by self-play in a separate repository. It is not a requirement of the
bot. A replacement agent sees a reconstructed `BelotState` and returns an action
index; how it encodes that state is entirely its own business, and it should
bring its own encoder rather than edit this one.

Everything below is **fixed by the trained weights**. Change an offset and the
checkpoint becomes meaningless.

---

## 1. Local vector (513) — the actor's input

| Offset | Size | Feature |
|---|---|---|
| 0 | 32 | our hand |
| 32 | 32 | face-up card (**BIDDING only**, zeros otherwise) |
| 64 | 5 | trump: `[none, D, H, C, S]` |
| 69 | 5 | declarer, relative: `[none, me, left, partner, right]` |
| 74 | 3 | phase: `[bid r1, bid r2, playing]` |
| 77 | 108 | current trick — **Left, Partner, Right** (36 each) |
| 185 | 6 | match us/them ÷101, raw us/them ÷162, bolts us/them ÷2 |
| 191 | 4 | dealer, relative |
| 195 | 144 | last trick — **Me, Left, Partner, Right** (36 each) |
| 339 | 96 | belief matrix, 3 × 32 (Left, Partner, Right) |
| 435 | 8 | trick number |
| 443 | 38 | legal-action mask |
| 481 | 32 | graveyard |

Each 36-slot trick block is `card one-hot (32) + sequence position one-hot (4)`.
A player who has not played leaves the **entire block zero**. The current-trick
block excludes "Me" (108 wide) because we act before playing; the last-trick
block includes Me (144 wide).

The mask at offset 443 is an *observation feature*, and it is always the full
legal mask. It must not be narrowed by a retry — the network was trained on the
unnarrowed one. Narrowing applies only to the logits at selection time.

## 2. Action space (38)

Shared with every agent, defined in `belotmd.game.actions`:

```
 0-31  play that card
 32    pass
 33    accept the face-up card as trump   (bidding round 1)
 34-37 name a suit as trump               (bidding round 2), suit = a - 34
```

## 3. Global vector (332) — critic only

```
hands 0-127 (Me, L, P, R) · graveyard 128 · face-up 160 · trump 192 ·
declarer 197 · dealer 202 · phase 206 · current trick 209 · stats 317 ·
trick number 323 · declarer_has_played_trump 331
```

**This vector is fictional in live play.** Opponents' hands are unknowable, so
they are filled with placeholder ids of the correct *length*. That is harmless
only because the agent discards the value head at inference — it is built solely
to keep the model's forward signature intact. **Never reuse this vector for
anything else.**

## 4. Belief matrix semantics

Built from `unseen` (not in our hand, not in the graveyard, not on the table),
minus `impossible_cards` (suit and trump voids inferred during play), minus
`known_cards` (certainties), then column-normalised once and row-rescaled to
each opponent's hand size.

- Cards already played read **0.0**.
- `known_cards` reads **1.0** for its holder.
- It is a **single normalisation pass**: rows are exact (each sums to that
  opponent's hand size), columns are approximate. A card only one opponent can
  hold reads ~0.90, not 1.00. That is intended — do not "fix" it.

Critically, `unseen` excludes cards known to be in *someone's* hand:

```python
unseen[belot.known_cards.any(axis=0)] = False
```

Without that line, a pinned card leaks ~0.49 of phantom mass onto each *other*
opponent — the column sums to ~1.97 instead of 1.0 — and their genuine
candidates are diluted to compensate. It matters far more live than in training,
because live play pins 3–5 cards at once from declared combinations rather than
just the face-up card. `verify_observation_contract()` asserts it at startup.

## 5. Where the belief information comes from

The agent does not infer anything itself; the synchronizer hands it a state that
already carries:

- **voids** — a player who failed to follow suit cannot hold that suit
  (`impossible_cards`);
- **declared combinations** — decoded into the exact cards their holder must
  have, and pinned (`known_cards`);
- **the seven-swap** — both sides of the trade are certainties and both get
  pinned;
- **the face-up card** — its recipient provably holds it.

This is the main reason combinations matter to the policy even though there is
no "declare" action in the 38-slot space: declarations reach it as *belief*, not
as a decision. Declaring is handled by rule in `bot.py`, and combination points
land in a separate scoring column that no observation feature reads, so
declaring can never perturb the vectors above.

## 6. Invariants the auditor asserts

- `phase == "BIDDING"` ⟹ `trump is None` and `declarer is None`.
- `len(graveyard) == 4 * tricks_played`.
- `our hand + Σ numCards + graveyard + current trick == 32`.
- No belief mass on any card in the graveyard.
- Belief row mass ≤ that opponent's hand size; no row collapsed to zero.
- No `known_cards` bit on a card already in the graveyard.
- Declared cards never intersect **our** hand — if they do, the decoder is wrong
  and the declaration is discarded rather than believed.

## 7. Known limitation

The trainer's simulator does not implement the seven-swap or the less-than-14
deal cancellation, both of which the live bridge handles. The policy has
therefore never seen either mechanic during training; the bot applies them by
rule. Closing that gap means changing the simulator and retraining, which
belongs in the training repository, not here.
