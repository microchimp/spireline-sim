# The fight solver: design (owner decision D35)

2026-10-26, branch `rebuild/fight-solver-design` (from main 87b2c81b). Design round only: nothing here is built yet.
Sources: Spireline D35 (with its efficiency addendum and the owner's correction on perfect play), D4 (the combat score
L0), and the orchestrator's solver notes of 2026-10-26 07:30Z, 08:00Z, 08:45Z and 09:35Z. Game code is named by class
and method only and described in words.

**Perfect play, the owner's correction, which every section below obeys.** Perfect play is the action that is best in
expectation over the draws and rolls *at decision time*, not the line that wins once the draws are known. The solver
therefore has two outputs, kept apart in the API as separate calls with separate result types:

- the **expected value** (`SolveExpected`): the exact layer's value over the draw pile as a multiset and the enemies'
  rolls at the game's odds, or the deeper sampled search where the exact layer does not reach. Regret, policy labels and
  value labels come from this output only;
- the **hindsight bound** (`SolveHindsight`): the same fight with the record's actual draw order and rolls fixed. It
  answers "a winning line existed given these draws". It is an upper bound on avoidable losses, read by lost-fight
  attribution only. It never grades a decision as a mistake and never produces a training label.

Section 4a lists the consumers allowed to read each output and how that is enforced.

## 1. The exact layer

- **What it computes.** Expectimax over the emulator's own combat rules: max nodes where the player decides (a card play
  with its target, a potion with its target, end turn, an answer to a screen), chance nodes where the rules roll (a draw
  from the unknown part of the draw pile, an enemy's next-move roll, a random target, a random generated card), terminal
  nodes at the end of the fight scored by a terminal utility (section 10), and horizon leaves scored by the network when
  the solve is cut at a turn horizon. A solve whose every leaf is terminal is **exact**; its result says so
  (`exact_mass` = 1, the probability mass that reached only terminal leaves).
- **Where it runs.** C#, in a new namespace `Sts2Emulator.Solver` (`src/Sts2Emulator/Solver/`), inside the library the
  agent already loads (`out/Sts2Emulator.dll`, NativeAOT). It calls `CombatEngine.Step` (old engine) or the kernel's
  gate (`KernelGate.Step`, NATIVE_KERNEL.md section 5) directly on `CombatState` copies: no Python and no marshalling per
  node. Python calls it once per decision (or per in-tree node, section 10) and gets a result struct back.
- **The engine option.** The solver runs on the engine of the handle it is given (`old` or `core2:F08,F10`; a handle
  keeps the engine it was made with, NATIVE_KERNEL.md section 3). It adds no engine of its own and never mixes engines in
  one table: the transposition table is per solver handle and the handle is bound to one engine and one rules build. The
  result records `Sts2_GetEngine`'s canonical name and the solver version for provenance. Under core2 a position whose
  kernel queue holds a paused hook action is solved from the screen (section 3). A screen inside a card play or a potion
  pauses that player action in the queue (gathering the choice; executing in the stopped step's start): the key writes the
  queued player actions as plain data (section 2); a queued hook action is refused with a reason, never solved on old
  rules (D31's "no silent old").
- **Native exports** (new file `Interop/SolverExports.cs`, `NATIVE_API_VERSION` 38 to 39; the run API is unchanged at 30,
  because a run's fight is solved through `Sts2Run_ForkCombat` and a combat handle):
  - `Sts2Solver_Create(combatHandle, const SolveOptions*)` returns a solver handle that owns a transposition table bound
    to the fight (subtree reuse across decisions, section 5); `Sts2Solver_Destroy`, `Sts2Solver_Reset`.
  - `Sts2_SolveExpected(solver, combatHandle, const SolveOptions*, ExpectedResult*, float* qBuf, int maxActions)`:
    status 0 exact, 1 budget reached (result holds the last completed horizon), -1 error, -3 refused (reason in
    `Sts2_GetLastError`). `qBuf` receives one value per action id of `Sts2_ValidActions` (targets as separate entries,
    section 3).
  - `Sts2_SolveHindsight(solver, combatHandle, const HindsightOptions*, HindsightResult*)`: a separate call with a
    separate result type; refused unless the handle carries true streams and the solver was created with
    `SolveOptions.allowHindsight` (section 9).
  - `Sts2_SolvePotionCurve(solver, combatHandle, const SolveOptions*, CurveEntry* buf, int maxSubsets)` (section 8).
  - `Sts2Solver_PendingLeaves(solver, int* obsBuf, int max)` / `Sts2Solver_SetLeafValues(solver, const float* pdf, int n)`:
    the two-phase leaf protocol for the network at the horizon (section 10), the same shape as the search's batched
    prior requests (`combat_mcts.drive_batched`).
  - `Sts2_SolverInfo(buf, cap)`: solver version, pruning names and their bit positions, utility names.
  - `SolveOptions`: utility (`L0`, `L0Priced` with a price per potion id, `SearchHp` with its potion value and death
    value), node budget, time budget (ms), maximum horizon (turns), pruning bitmask (each lossless pruning can be switched
    off for the efficiency report), chance width cap (section 4), allowed-potion mask, `allowHindsight` (default off).
- **EmuDump command** `solve` (new `tools/emudump/Engines/Old/Solve.cs`): `EmuDump solve <ascension> <seed> --at <step>` or
  `EmuDump solve --record <file> --decision <i>` (re-anchored through the existing `reanchor` path), with
  `--engine old|core2`, `--budget-nodes`, `--budget-ms`, `--horizon`, `--prune <list>`, `--mode expected|hindsight`,
  `--curve`. One JSON object per decision: the expected result under `expected` or the hindsight result under
  `hindsight`, never both in one object, plus counters (nodes, engine steps, table hits, time, per pruning). It is the
  front end for the harness, the coverage curve and attribution, and the parity reference for the native exports (one
  code path: both call `Sts2Emulator.Solver`).

## 2. The state key

The key decides which positions are merged, so a missing field silently merges positions that play differently. The
rule is therefore **whole state by default**: the key encodes every field that `CombatState.CopyTo` copies (the one field
list `Clone` and the in-place restore share), and a field leaves the key only through an exclusion with a written reason.
Over-inclusion costs merges, never correctness.

**Encoding.** A generated canonical encoder (`Solver/StateKey.g.cs`, written by `tools/solver/gen_key.py` from
`tools/solver/key_fields.json`), no reflection on the hot path. Each field is classed `plain`, `multiset`, `regions`,
`card_ref` or `excluded(reason)`. The encoded bytes are hashed to 128 bits for the table; a debug mode stores the full
bytes and compares them on every hit. Mode (expected or hindsight), utility and the allowed-potion mask are part of the
table key, not of the state key (section 5).

- **The player:** `PlayerHp`, `PlayerMaxHp`, `PlayerBlock`, `Energy`, `MaxEnergy`, `Stars`, `OstyHp`/`OstyMaxHp`,
  `PlayerBuffs` (ordered: list order is hook order), `PlayerDebuffsAtRoundStart`, `Orbs` (ordered: slot order matters),
  `OrbCapacity`, `BaseOrbSlots`, stance-like and per-character fields as they appear in `CombatState`.
- **The piles:**
  - `Hand`: a multiset of `CardInstance` values. `CardInstance` is a value record whose uid slot equals every other slot
    (CARD_UID.md), so its value already carries every per-copy property (upgrade, cost bump, cost for combat, bonus
    damage, enchantment and its state, retain, free this turn, Tinker type). Kept ordered instead when the combat holds a
    hand-position reader (the order-reader registry below).
  - `DrawPile`: three regions, using the fields the engine already keeps: the known top (`KnownTopCount` cards, ordered),
    the unknown middle (a multiset), the known bottom (`KnownBottomCount` cards, ordered). In hindsight mode the whole
    pile is ordered.
  - `DiscardPile`, `ExhaustPile`: multisets.
  - The queues `ReturnToHandBeforeDraw`, `CopiesToHandBeforeDraw`, `AutoPlayQueue`, `StolenCards`, `StolenBackCards`:
    ordered (safe default).
- **Each enemy** (the list stays ordered: slot order is acting order and default-target order): `DefId`, `Hp`, `MaxHp`,
  `Block`, `Buffs` (ordered), `CurrentIntent`, `SecondaryIntent`, the pattern position (`MoveIndex`, `MoveHistory`,
  `LastMoveRepeats`, `OnceOnlyMoveUsed`, `LastBranch`, `RepeatStreak`, `BranchCooldown`, `StartsOnBranch`,
  `RollsNextMove`, `StarterMove`), the per-enemy flags (`Escaped`, `CurlUpArmed`, `NemesisIntangibleOn`, `OffBalance`,
  and the rest of `EnemyState`), `Slot`, and `StolenCard`. `CurlUpCardUid` is a `card_ref` (below).
- **Everything else that can change an outcome:** `Turn`, `PlayerTurn`, every "this turn" and "this combat" counter
  (`CardsPlayedThisTurn`, `AttackCardsPlayedThisTurn`, `EnergySpentThisTurn`, `CardsDrawnThisCombat`, and the others),
  `Relics` (ordered, with their counters and spent flags), the relic flags on `CombatState` (`PenNibArmed`,
  `VambraceSpent`, `BurningSticksUsed`, `MusicBoxUsedThisTurn`, ...), `PotionSlots` and `MaxPotionSlots` (a multiset
  plus the count of empty slots, ordered when a slot-position reader is present), `PendingSelection` (section 3),
  `EndTurnAwaitingSelection`, `TurnStartHeldPhase`/`TurnStartHeldRelic`, the escape and departed-enemy fields, gold
  fields, and the `Kernel`'s combat progress.
- **`card_ref` fields.** A field that holds a card uid (`EnemyState.CurlUpCardUid`, and any kernel `CardHandle` that
  outlives a step) is encoded as the referenced card's pile, value and rank among equal values in that pile, never as the
  uid number. A referenced copy is split out of its multiset as a marked card, so two equal values stay distinct only
  while something points at one of them. `gen_key.py` fails on an `int` field whose name ends in `Uid` that is not
  classed.
- **RNG positions.** In expected mode no stream position is in the key: chance is fresh at every chance node (section 4),
  the player cannot see seeds, and keying on a position would let the solver read hidden information. In hindsight mode
  every stream is in the key as (seed, position), as the full-state hash writes it.

**Which streams are chance nodes and which are fixed.** The engine has nine named streams (`StateCloning.StreamNames`:
`ShuffleRng`, `TargetRng`, `CardSelectionRng`, `EnergyCostRng`, `CardGenerationRng`, `PotionGenerationRng`,
`OrbGenerationRng`, `AiRng`, `NicheHpRng`) plus the step's own stream argument. In expected mode all of them are chance;
none is fixed. `ShuffleRng` is not enumerated call by call: a shuffle makes the pile's middle unknown and the draw is the
chance node (section 4). In hindsight mode all of them are fixed to the record's streams and the solve has no chance
nodes.

**Not in the key, and why that is safe** (each is a row in `key_fields.json` with its reason):

| Field | Reason |
|---|---|
| `CardInstance`'s uid slot, `LastCardUid` | identity only (`[IdentityOnly]`, CARD_UID.md); outcomes read values; references are `card_ref` |
| the ten streams (expected mode) | chance is fresh at each node; the player cannot see seeds |
| the middle region's order (expected mode) | the player cannot see it; draws from it are chance nodes |
| `LastIntentRoll` | a record for a resampled fork to roll the pass again; the solver's chance nodes replace it |
| `AiStateAssumed` | provenance of a re-anchor; carried in the result (`ai_state_assumed`) instead |
| the kernel's hook-action log, command log and action counter; the queue's ids and id counter | sequencing (the comparer's `kernel-sequencing` surplus); the queued actions themselves are keyed (`KeyWriter.KernelQueue`: per queue its flags, per action its kind, type, owner, state, step and answer), and a queued hook action or an early resumption is refused (`kernel-queue-busy: ...`) |

**The checklist.** Two lists and two tests make the key's coverage checkable:

- the full-state walk (`tools/emudump/Engines/Old/OldStateWalk.cs`, the field list `LifecycleTests` already checks
  against `CopyTo`) is the field census: `KeyFieldCoverageTests` fails on a walked field that `key_fields.json` does not
  class, so a field added to `CombatState`, `EnemyState`, `CardInstance`, `RelicInstance`, `BuffState`, `OrbState` or
  `PendingCardSelection` breaks the build of the key until it is classed;
- the struct_compare census (COMPARER.md: 33 schema paths per corpus fight, 110 per traced fight) is the list of what the
  game itself is compared on: each census path maps to a key field, and an unmapped path fails the same test;
- `KeyBisimulationTests`: on fuzzer positions (both engines), whenever two different states get equal keys, they must
  have the same legal actions, the same terminal utility, and for each action and each chance answer successors with
  equal keys. The debug build runs the same check on a sample of live table hits.

**The order-reader registry.** Hand order, potion-slot order and enemy interchangeability (section 5) are only canonical
when no rule in the combat reads the order. `tools/solver/order_readers.json` lists the cards, relics, powers, potions
and enemy moves that read hand position, slot position or enemy order (first living enemy, adjacency, ally effects,
reactive damage that depends on who acts first), found by a scan of the engine's effect code and reviewed. A combat
containing none gets the canonical forms; otherwise the ordered forms. The equal-value harness (section 6) checks the
canonical forms on positions with and without readers.

## 3. Legal actions

- **The engine's encoding** (`CombatEngine.ValidActions`, the comment at the top of `CombatEngine`): hand index for a
  card play, `Hand.Count` for end turn, `Hand.Count + 1 + slot` for a potion; a target is the `targetEnemyIndex`
  argument. The solver expands an aimed card or potion into one child per living, targetable enemy, as the live search
  does (`live_search._legal_moves`, using the same aimed-card and aimed-potion tables, moved into the library).
- **Deduplication** (section 5): one child per distinct (card value, target); one per distinct (potion id, target).
- **Potions** are actions like any other, limited by the allowed-potion mask (section 8). Potion use is subject to the
  engine's own refusals (`PotionEffects.CanUseInCombat`).
- **End turn** is always legal. Its step runs the enemy turn and the next turn's start; its children are chance outcomes
  (section 4).
- **Screens inside an action** (`PendingSelection` open: choose from hand, from the draw pile, discover, a hook choice
  under the kernel): a **decision node** over the candidates plus the decline when the screen is skippable, answered
  through the engine's replay (`CombatEngine.Step`'s stopped-step path, keyed answers under core2). The node's key is the
  step path: the key of the state before the step, the action and target, and the answers so far. For a pick-k screen
  whose result does not depend on pick order (a flag per `CardSelectionKind`), answers are canonical (non-decreasing
  candidate order only), so k picks are one subset, not k! sequences.
- **Random content before a choice** (a discover's offered cards, a random card generated into the hand): a **chance
  node** for the generation, then a decision node for the pick. A choice screen whose candidates come from a pile the
  player can see is a decision node only; one that picks at random is a chance node only.

## 4. Chance

**The mechanism: a chance hook on the streams.** `CountingRandom` (sealed; every engine stream is one) gains one
nullable field, an `IChanceSource`. Unset (the default everywhere outside the solver), it is a null check per call and
the bytes do not change; a byte-identity gate on the corpus proves it on both engines. Set in a solver copy, each call is
answered by the source:

- `Next(n)` and `Next(min, max)`: a chance node with uniform branches, exact for the rules as written (the game's draw is
  uniform over the range);
- `NextDouble()`: allowed only through the weighted helper below; a raw call in solver mode refuses the solve
  (`unlisted-roll`, with the call site), so no roll is ever given made-up odds. There are 12 such sites today (10 in
  `EnemyAI`, 2 in `PotionEffects`);
- a call whose answer is not yet known **stops the step** with a `ChancePending` (the twin of `HookChoicePending`), and
  each outcome is a child that replays the step from its start with the chance answers so far, through the replay
  machinery screens already use. Screen answers and chance answers share one answer list in the step path.

**The draw pile is a multiset, and its order is sampled when a draw happens.** In expected mode:

- a shuffle (`CardEffects.ShuffleDiscardIntoDraw`, `ShufflePile`) does not roll: the shuffled cards join the unknown
  middle region (`ForgetDrawOrder`);
- a draw from the top takes the known top if there is one, else is a chance node over the distinct values in the middle,
  each with probability count / size;
- a placement at a random position (the `placementRng` sites in `CardEffects`) inserts into the middle without a roll;
  a placement on top or bottom grows the known region;
- a random pick from the draw pile (by `CardSelectionRng`) is a chance node over distinct values with count weights;
- these draw-pile sites are listed in `tools/solver/draw_sites.json` (file, method, anchor text, as the kernel's
  `sites.json` lists dispatch sites); a `ShuffleRng` call that reaches the chance source from an unlisted site refuses
  the solve;
- **macro draw**: a draw of k cards with no order observer active is one chance node over the distinct sub-multisets of
  size k (multivariate hypergeometric probabilities) instead of k sequential nodes, with the hand-size cap applied. An
  order observer is anything that reacts per drawn card (an after-card-drawn listener in the kernel's listener tables,
  a card with an on-draw effect in the middle region, a draw-until rule); with one present the draws are sequential. This
  is the pruning that keeps a turn start at hundreds of children instead of thousands of sequences (section 5).

**Enemy moves come from the rules.** The shown intent (`EnemyState.CurrentIntent`, already rolled when the decision is
asked) is state, not chance: this turn's enemy move is fixed. The next move is rolled in the enemy turn
(`EnemyAI.ChooseIntents`, then `SelectIntent` per enemy, on `AiRng`), and each roll is a chance node at the game's odds.

**How the odds come from the game's move selection.** In the game an enemy's moves are a state machine
(`MonsterMoveStateMachine`): `RollMove` walks from the current state through `FindNextMoveState` until it reaches a move.
A `MoveState` names its follow-up state (no roll). A `ConditionalBranchState` takes the first branch whose condition on
the combat holds (no roll). A `RandomBranchState.GetNextState` is the only roll: it computes each branch's effective
weight (`GetStateWeight`), draws one float uniformly below their sum, and walks the branches in order subtracting each
weight until the remainder is not positive. The effective weight is the branch's base weight (or its weight function of
the owner) times an eligibility factor read from the machine's `StateLog`: zero for a use-only-once move already in the
log; zero for a cannot-repeat move (or a move with a maximum run length) when the last entries of the log are all that
move up to the limit; zero when a branch with a cooldown has that move among the last cooldown moves. So the odds of
branch i are its effective weight divided by the sum of effective weights, a function of the enemy's move history and
condition inputs, all of which are in the key.

In the emulator `EnemyAI.SelectIntent` is hand-written per enemy, with the branch tables written into the code
(`PickWeightedBranch` and inline draws). Piece S2 routes every enemy roll through one helper in the game's shape,
`Roll.Branch(rng, weights)` and `Roll.Uniform(rng, n)` (new `Core/Rng/Roll.cs`), which consumes exactly the draws the
inline code consumed (byte identity on the corpus) and, under the chance hook, hands the solver the weights: the odds are
weight / sum, exact as rational numbers (the game's float draw differs from them by at most one float step per branch,
the only rounding the layer accepts, recorded in the doc of `Roll`). `EnemyOddsTests` check each enemy's helper weights
against the game's branch tables as read from the state machines (eligibility rules included). The long-term form is the
owner's game-shaped one: when enemy AI migrates onto the game's state machine, the odds come straight from the branch
weights and the helper becomes the machine's own roll.

**Other random effects.**

- Random targets (`TargetRng`): uniform over the eligible enemies, via `Next(n)`.
- Random cost (`EnergyCostRng`), random card from hand or pile (`CardSelectionRng`): uniform via `Next(n)`; a pick from
  the hand or discard merges by value through the key's multisets.
- Card, potion and orb generation (`CardGenerationRng`, `PotionGenerationRng`, `OrbGenerationRng`): a chance node per
  roll; a pool pick with rarity weights goes through `Roll.Branch`. A generation node wider than the chance width cap
  (default 32 outcomes) is not enumerated: the subtree below it is a horizon leaf, so the result stays honest
  (`exact_mass` < 1) instead of exact by sampling.
- Enemy HP rolls of enemies spawned in the fight (`NicheHpRng`): `Next(min, max)`, uniform.
- Information rule: a chance outcome branches with later decisions conditioned on it only when the player sees it (a
  drawn card, a new intent, a generated card shown). An outcome the player does not see stays a distribution in the
  state (the draw pile's middle); a site that would need a hidden branch is refused. This is what keeps the exact layer
  free of the strategy fusion a determinized search has, and keeps it inside the search's firewall (section 10).

### 4a. The two outputs and who may read them

| Output | Call and result | May read it | Must not read it |
|---|---|---|---|
| Expected value | `Sts2_SolveExpected` / `ExpectedResult` (Python `ExpectedSolve`) | the oracle in the agent's search; value and policy labels; the regret benchmark; the potion curve and use-or-save; attribution's "exact play's win chance at each decision"; the coverage curve; the A/B's settled share | nothing excluded |
| Hindsight bound | `Sts2_SolveHindsight` / `HindsightResult` (Python `HindsightBound`) | lost-fight attribution reports only | the agent's search, any label writer, the regret benchmark, grading of a decision, promotion, the A/B |

Enforced in code: the two result types share no field; the label writers (`gen_loop.CombatSamples`, the prior loop's
writers) and the regret benchmark accept `ExpectedSolve` only and refuse a record carrying a hindsight field; the live
search's binding does not import `Sts2_SolveHindsight`, and a test (in the manner of `test_native_entry_pins.py`) runs
each search entry point and fails if the export is bound; the native call refuses unless the solver handle was created
with `allowHindsight`, which only the attribution tool sets.

## 5. Lossless prunings, first version

All are on from the first version. Each has a bit in `SolveOptions` so it can be switched off for the equal-value harness
and the efficiency report (engine steps per solve with and without it).

- **Transposition merging.** Expectimax values are memoised by key in the table, so commuting plays (Strike then Defend,
  Defend then Strike) reach one node and the tree becomes a graph. Exact because a node's expectimax value is a function
  of the state alone when the key is a bisimulation (section 2): equal keys have the same actions, the same successor
  distribution and the same utility, so they have the same value. A key seen while it is still being expanded (a cycle)
  refuses the solve; the turn counters make true cycles impossible in practice.
- **Multiset hands (identical cards are one).** Cards with equal `CardInstance` values are one card: a hand of three
  Strikes gives one play child per target, not three, and drawn hands are counted as multisets, not as C(n, k) labelled
  sets. Exact because the uid is identity only (CARD_UID.md) and every per-copy property is in the value, so the
  successors of playing either copy have equal keys; a copy pointed at by a `card_ref` is marked and stays distinct.
  The same holds for identical potions.
- **The draw pile as a multiset.** Draws from the unknown middle branch over distinct values with count weights, and the
  macro draw takes k cards as one node. Exact because, to the player, the shuffled order is uniform over the orderings of
  the multiset, so drawing value v has probability count(v) / size whatever the order representation; the macro draw is
  the product of the sequential draw probabilities summed over orderings, which is the hypergeometric probability, and it
  is used only when nothing observes the order between draws.
- **Interchangeable enemies.** Two enemies with equal state except `Slot` are interchangeable until their state diverges:
  aiming at either gives one child. Exact only when no rule in the combat reads enemy order in a way that tells them apart
  (the order-reader registry, section 2): with acting order or a first-living-enemy default in play, hitting the first
  copy and hitting the second are different positions. When a reader is present the pruning is off for that combat;
  the harness tests it on positions of both kinds.
- **The fixed shown intent.** This turn's enemy moves are state; only rolls that happen after the decision branch.
  Exact by the rules: the intent was rolled before the decision and the move performed is the shown one.
- **Subtree reuse across decisions.** The table lives in the solver handle for the whole fight; after the real step and
  the real draw, the new position's key is looked up first, and a solved node is answered at once. Exact because a value
  is a function of the key, the key does not depend on how the position was reached, and the table is bound to one
  engine, one rules build, one mode and one utility. Eviction (a node budget on the table, oldest generation first) only
  costs recomputation.
- **Shared expansion across solves.** Successor lists with their probabilities depend on the state alone, not on the
  utility or the potion mask; they are cached separately from values, so the potion-subset solves (section 8) and the
  priced solve share every expansion.

## 6. Dominance and bounds, behind the equal-value harness

**The equal-value harness** (`tools/solver/harness.py` over `EmuDump solve`, and `EqualValueHarnessTests`):

- a reference solver (`Solver/Reference.cs`): plain tree expectimax over full labelled states, no table, no pruning,
  sequential draws, enemies and copies distinct;
- a test set of random small positions on both engines: the fuzzer's and the scenario tool's fights cut down to one or
  two turns from their end (low enemy HP, small piles), plus positions chosen to contain each order reader, each screen
  kind and each chance kind; seeded and listed in `tests/fixtures/solver/positions.jsonl`;
- the rule: a pruning, dominance rule or bound is enabled only when, on the whole set, the root value and every root
  action's value equal the reference's to 1e-9, and the best action set is the same. A rule that fails stays off.
- **The chance-model gate** checks the reference itself against the engine's real streams: on 200 small positions, the
  exact value of the solver's policy equals the mean outcome of playing that policy on 10,000 copies with real seeded
  streams (real shuffle, real enemy rolls) within three standard errors. It catches wrong odds, a wrong draw model and a
  key that merges what the real game does not.

**Dominance (lossless with care, second version).** Drop an end-of-turn state dominated by a sibling under the same
parent: equal piles, equal enemy patterns and intents, and the sibling at least as good on player HP, block, energy and
each potion, and at most as good for the enemies on every enemy's HP and powers. Exact only if every rule is monotone in
those quantities, which is not true for every card (low-HP triggers, block-loss triggers); the registry of non-monotone
readers turns it off where needed, and the harness decides.

**Branch and bound (second version).** Optimistic and pessimistic bounds on expectimax (the Star1/Star2 family): a move
whose upper bound is below the best lower bound found is cut. Bounds: the maximum damage this turn cannot kill (the fight
does not end this turn); block covering every shown intent (no HP loss this enemy turn); the utility's own range. Exact
when the bounds are true bounds; the harness checks them.

**Heuristics last** (never called exact): policy-prior move ordering so bounds cut early; lethal-first termination;
a horizon with the network at the leaf (section 7). These change node counts or values and are reported, not gated as
lossless.

## 7. The budget

- **Per decision**: a node budget and a time budget, whichever is reached first. Iterative deepening over a turn horizon
  h = 0 (the rest of this turn, horizon leaves at this turn's end), 1, 2, ..., up to the end of the fight. Each completed
  iteration leaves root action values; the budget ending inside an iteration returns the last completed one with its
  `exact_mass` and horizon. A cheap size estimate (the week-3 sizing estimator, section 13) skips a solve that cannot
  finish within the budget before it spends anything.
- **Handover to the sampled search** (the live search of `rl/live_search.py`, configs in `rl/search_configs.py`):
  - the solve is **exact** (`exact_mass` = 1): its best action is played and its action values are the record's policy
    label; the sampled search does not run, and the time it would have spent is saved (reported in the A/B);
  - the solve is **horizon-limited**: the sampled search runs as today with its `sims` from the config, and the oracle
    answers inside it at nodes small enough to solve exactly (section 10). In the first version the horizon-limited root
    values are not used; a later arm may mix them into the root prior;
  - the solve is **refused** (a busy kernel queue, an unlisted roll, a hidden branch): the sampled search as today, the
    reason counted.
- **Default budget**: set from the coverage curve (section 11) and the week-3 numbers, not guessed here. The search at
  a3-v1 spends 30 simulations on a plain play and 120 on a target or choice screen, each a playout to the fight's end; the
  exact layer's first budget is of the same order in engine steps as that search, so turning it on does not change the
  decision's cost much.

## 8. A value curve over potion subsets

- For the potions held H (at most 3 today, 8 subsets; the solver accepts up to `PotionSlotCapacity`), solve the root
  once per subset S of H with only the potions in S usable (`SolveOptions` allowed mask). S empty is the base.
- Each entry `CurveEntry(S)` reports: expected L0, win chance, expected end HP when winning, and for each potion in S the
  chance it is still held at the end. L0 counts potions still held (D4: 10 HP each), including those outside S.
- When no rule in the combat reads potions held or empty slots (the order-reader registry has a potion-count class),
  the potions outside S are removed from the solve's copy, so a subtree after a potion is used in one subset's solve is
  the same node as in another's; otherwise the subsets share expansions only (section 5).
- **What the run layer uses it for.**
  - Use or save: the fight gain of potion p is V(S + p) - V(S); the run value head's saved value of p prices keeping it
    (today's flat 10 HP in L0 is the fallback). With a price vector the exact use-or-save is one more solve with the
    `L0Priced` utility, which shares every expansion with the curve.
  - Labels for the value head's potion pricing by situation: the curve, per fight kind, act and HP, is the target for
    what a potion is worth in that fight.
  - Later, through the value head trained on those labels and not by calling the solver: which potion to drop on a full
    belt, and buying potions before an elite or boss.

## 9. A hindsight mode

- **Input**: a record with the actual draw order and rolls. A self-play or lab emulator record carries seeds and steps,
  so the true streams are re-created exactly (a combat handle forked without a resample seed, or EmuDump's re-anchor
  with true streams). A real-game record re-anchored by the lockstep carries only what was observed: the cards drawn and
  the intents shown on the turns played.
- **Mode**: every stream fixed, no chance nodes; the key includes the stream positions and the whole draw-pile order. The
  question is an OR over lines ("does a winning line exist given these draws"), solved depth-first with lethal-first
  ordering, plus the best utility reachable.
- **Result** (`HindsightResult`, its own type): `win_line_exists` in {yes, no, unknown}, `best_utility`, the line, nodes.
  For a real-game record, a line that needs a draw or roll the record does not hold is unknown, never assumed; the result
  is "unknown" when every winning line needs one.
- **Use**: lost-fight attribution only. For a lost fight: the expected exact value at each decision (from
  `SolveExpected`) says whether the loss was likely from the start (decided by deck, route, rests or potions) or where
  the agent's move cost win chance (an in-fight error, by expected regret); the hindsight bound adds "a winning line
  existed given these draws", an upper bound on the losses that better play could have avoided. It never grades a
  decision and never becomes a label (section 4a).

## 10. The oracle in the search

- **Where it is called.**
  - At the root of every combat decision, before the sampled search, with the decision's exact budget (section 7). This
    is what removes missed lethal and missed forced survival wherever the rest of the turn (or the rest of the fight) is
    within the budget.
  - Inside the sampled search, at a node the size estimate says is solvable within a small per-node budget (typically the
    last turn or two of a fight): the solve replaces the playout from that node. An exact answer makes the node solved
    (its value is final and it gets no more simulations); a non-exact answer is ignored and the playout runs.
- **The firewall.** The search's worlds are determinized (`resample=True`); the expected-mode solve reads only public
  state (the draw pile's middle as a multiset, no stream positions), so every determinization of the same public
  position gets the same answer from the same table entry. `SolverLeakTests` (the solver's twin of `ResampleLeakTests`)
  check that two handles that differ only in hidden order and streams give identical results.
- **Caching.** One solver handle per fight per process (one emulator per process, as today), its table shared by the
  root call, the in-tree calls of every simulation and the following decisions of the same fight. Lanes of
  `ParallelSearcher` are separate processes with separate tables.
- **The value scale.** Labels, regret and the curve use D4's combat score L0 (0 if dead, else
  0.5 + 0.5 * min(1, (HP_end + 10 * potions) / maxHP)), maximised in expectation, so the policy is the one best for L0.
  Inside the live search the oracle answers in the search's own currency (`SearchHp`: end HP plus `potion_value` per
  potion held, -100 for a death, as `live_search._outcome`), so exact values and playout returns back up together with no
  rescaling and `c` keeps its meaning. Both utilities are terminal functions, so the same solve machinery serves both;
  the table key includes the utility.
- **The network at the horizon.** The combat net's value head is a distribution over end-HP buckets (bucket 0 is death,
  `train_combat`). A horizon leaf's value under any utility is the sum over buckets of the bucket's probability times
  the utility at that HP with the leaf's potions held. Leaves are batched to Python through the two-phase protocol and
  scored with the same model the prior uses.
- **How the values mix.** Inside a solve, by probability: each terminal leaf contributes its exact utility, each horizon
  leaf the network's expectation, weighted by the probability of reaching it; `exact_mass` is the share that came from
  terminals. In the search, a node with an exact solve takes that value in place of playouts; every other node keeps the
  search's own estimate. Where both an exact value and the network's value exist for a position, their difference is a
  value-head label and a measured error, not a blend.

## 11. The coverage-curve measurement

- **Input**: real decisions from the lab's records (the nightly real records and the S3 records), re-anchored through
  `EmuDump solve --record` on the engine each record was scored on, all decisions of a sample of fights (stratified by
  act, fight kind, won or lost).
- **Measure**: for each decision, solve with node budgets 10^3 to 10^7 (one per decade) and time budgets 0.01, 0.1, 1
  and 10 s, all lossless prunings on; record exact or not, nodes, engine steps, time, `exact_mass`, refusal reasons.
- **Curves**: the share of decisions solved exactly as a function of the budget, by act, deck size, turn of the fight,
  fight kind (hallway, elite, boss) and outcome (the last turns of lost and boss fights separately); the share of fights
  solved fully within 1 s and 10 s (every decision from the first exact), and from which turn a fight becomes exactly
  solvable; positions solved per second; engine steps per solve with each pruning switched off in turn (on a
  subsample).
- **Decisive positions**: a decision where the exact best and worst root actions differ by more than 0.01 L0, or the
  last two turns of a lost or boss fight. Their solved share within the per-decision budget is the A/B's coverage metric.
- `tools/solver/coverage.py`, results under the lab's results directory (not committed), no host names or paths in them.

## 12. The A/B

- **Arms**: the same generation, the same engine, the same seeds. A: config `a3-v1` (today's agent). B: `a3-v1-oracle`,
  a new named config in `rl/search_configs.py` adding the oracle (root and in-tree, exact answers only). Optional C, if
  the 30-against-120 block shows extra simulations still pay: A with simulations raised to B's mean decision time, to
  separate exactness from effort.
- **Method**: the lab's paired method on a fresh seed block (seeds never used by training, promotion or the week-3
  block): paired runs per seed, `gen_loop.paired_binary` (exact McNemar) for wins and Act-1 boss kills, the paired floor
  with its lower bound as in D24's promotion rule, read through the promotion guard (`tools/lab/promotion_guard`,
  docs/PROMOTION_GUARD.md). Size from the 30-against-120 block's paired variance (section 13); 500 paired seeds as that
  block uses, unless the power calculation says more.
- **Reported**: win rate, boss kills, act reached (the owner's prediction: highly important for the win rate; the
  orchestrator's: more for reaching Act 3 than Act 1), HP lost per fight by fight kind, the share of decisions the exact
  layer settles, the share of decisive positions solved within budget, positions solved per second, engine steps per
  solve with each pruning on and off, and time per decision for both arms.
- **Nothing on the defaults.** `SELFPLAY` and `SCORE_REAL` stay `a3-v1`; `jobs.json` is not touched; the A/B is run in
  the orchestrator's slot with `--config`. Promotion of the oracle into self-play or scoring is a later decision on the
  A/B's result, by the owner.

## 13. The pieces

Units: a reviewed round costs 1.0 with Sonnet and 1.7 with Opus.

| # | Piece | Files | Rounds, model, units | Gate | Tests |
|---|---|---|---|---|---|
| S1 | State key | `Solver/StateKey.g.cs`, `tools/solver/gen_key.py`, `tools/solver/key_fields.json`, `tools/solver/order_readers.json` | 1 Opus, 1.7 | every walked field and census path classed; 0 bisimulation failures on fuzzer positions, both engines | `KeyFieldCoverageTests`, `KeyBisimulationTests`, permuted-hand and uid-renumbered states give equal keys, streams change no expected-mode key |
| S2 | Chance | `Core/CountingRandom.cs` (hook), `Solver/ChanceSource.cs`, `Core/Rng/Roll.cs`, the 12 `NextDouble` sites, `Effects/CardEffects.cs` draw sites, `tools/solver/draw_sites.json`, `CombatEngine.Step` (`ChancePending`) | 2 Opus, 3.4 | hook off: 0 state-hash differences on the corpus, both engines; odds equal the game's branch tables; chance-model gate passes | `ChanceHookIdentityTests`, `EnemyOddsTests`, `DrawModelTests` (macro vs sequential), `ChanceModelTests` |
| S3 | Expectimax, table, lossless prunings, reference, harness | `Solver/Expectimax.cs`, `TranspositionTable.cs`, `Actions.cs`, `Utility.cs`, `Budget.cs`, `Reference.cs`, `tools/solver/harness.py`, `tests/fixtures/solver/positions.jsonl` | 2 Opus, 3.4 | equal-value harness: every pruning equal to the reference on the full set, both engines; per-pruning counters | `EqualValueHarnessTests`, screen and pick-k cases, refusal reasons, budget and iterative deepening |
| S4 | Native exports, EmuDump `solve`, Python binding, hindsight, potion curve | `Interop/SolverExports.cs`, `NativeExports.cs` (version 39), `tools/emudump/Engines/Old/Solve.cs`, `src/sts2_gym/native.py`, `Solver/Hindsight.cs`, `Solver/PotionCurve.cs` | 1 Opus, 1.7 | native equals EmuDump on the harness set; on every emulator record tried, the hindsight best utility is at least the outcome of the expected-best policy played on the same draws, and its mean over records at least the mean expected value | `SolverLeakTests`, `HindsightApartTests` (types, refusal without the flag), binding tests, the entry-pin test for hindsight |
| S5 | Oracle in the search | `rl/solver_oracle.py`, `rl/live_search.py` (option), `rl/search_configs.py` (`a3-v1-oracle`), label writers | 1 Opus, 1.7 | on the A3 dev set, where the solve is exact the search plays its best action (or a tie); no decision over its time cap; labels refuse hindsight | `test_solver_oracle.py`, `test_solver_labels_refuse_hindsight.py`, `test_search_configs.py` (defaults unchanged) |
| S6 | Coverage curve tool and run (built in the baseline round, section 16) | `tools/solver/coverage.py` | 1 Sonnet, 1.0 | reproducible curve on a seeded sample; a direct solve at a budget gives the exactness read from the deepening run | `test_coverage_curve.py` |
| S7 | The A/B run and report | orchestrator slot, no code beyond the config | 1 Sonnet, 1.0 | per section 12 | none (a measurement) |
| R | Review fixes reserve | | 1 Opus, 1.7 | | |
| S8 | Dominance and bounds (second version, optional) | `Solver/Dominance.cs`, `Solver/Bounds.cs` | 1 Opus, 1.7 | equal-value harness | harness cases with non-monotone readers |

**Total**: S1-S7 are 9 rounds, 13.9 units; with the reserve 15.6; with S8 17.3. That is more than D35's "about four
implementation rounds". A four-round cut exists: S1 with the draw model, S3, S4 without the curve, S5, with S2's enemy
odds deferred, so that a solve refuses any enemy roll (it then reaches only to the end of the current turn and to fights
that end before the next roll: lethal, forced survival, last turns). It delivers the "no missed lethal or forced
survival" half of the oracle and none of the multi-turn reach; the owner chooses.

**Order and parallelism.** S1 and S2 in parallel (the key reads the state, the chance piece changes the streams and draw
sites; S1 lands first). S3 after both (it can start on S1 with every roll refused). S4 after S3. S5 and S6 in parallel
after S4. S7 after S5, S6 and the week-3 numbers. S8 after S6, only if the curve says the prunings are the limit.

**Risks.**

- **State explosion.** A turn start draws 5 from a pile of 15 to 20 cards with 6 to 10 distinct values (one to a few
  hundred hands after the multiset), times 1 to 3 roll outcomes per enemy, times tens to hundreds of in-turn nodes: about
  10^4 to 10^5 nodes per remaining turn, so exact solves to the end reach the last one or two turns of most fights, and
  horizon solves everywhere else. The coverage curve replaces this arithmetic.
- **Keys missing state.** A silent wrong merge gives confident wrong values. Mitigations: whole state by default,
  exclusions with reasons, the field-coverage test, the bisimulation test and debug hit checks, the reference solver and
  the chance-model gate against real streams.
- **Speed.** Each `Step` clones the whole state for its replay record (`StepInProgress.Before`), the solver clones per
  child, the key encodes per node, and core2 adds the queue and dispatch per step. S3 lets the solver hand its own parent
  copy to the step as the replay record (one clone per child, not two), pools copies, and reports per-node costs from
  `Prof`. A chance-heavy step (an enemy turn with several rolls) replays its prefix once per chance point.
- **The switched-on families.** The solver must run on the handle's engine: core2's F08+F10 change who is targetable and
  when a fight is over, which changes values. The chance hook sits on the streams, so kernel commands are covered; the
  key writes the queued player actions (a screen inside a play or a potion) and solves screens through keyed answers; S1 to S4 gate on both engines. Each
  later family that switches on changes rules: tables never persist across processes, and the result records the engine.
- **Hidden information.** A branch on an outcome the player cannot see would make the solver clairvoyant. The
  information rule (section 4), the leak tests and S4's sanity check (the hindsight bound never below what the
  expected-best policy reaches on the same draws) guard it.

**What the week-3 session should feed in.**

- From the **sizing measurement** (state sizes of real fight decisions, share under 10^4 to 10^7 states, by act, last
  turns of lost and boss fights): the default per-decision node budget; the size threshold for in-tree oracle calls (from
  which turn the exact layer reaches); whether S8 or the four-round cut is the better spend; the strata and expected
  effect for the A/B; and a calibration of its size estimator against S6's real node counts, so the estimator can skip
  hopeless solves.
- From the **30-against-120 block** (gen 5, 500 paired seeds): the effect of four times the simulations on wins, boss
  kills and HP lost, the yardstick for what the oracle must beat; its paired variance, for the A/B's seed count; whether
  the sampled search is still budget-limited (if 120 beats 30, the time an exact solve saves can be re-spent on
  simulations and arm C is needed; if not, the oracle's gain must come from exactness); and the time per decision at 30
  and 120, which sets the exact layer's time budget.

## 14. S3 as built

- **Values are exact rationals** (`Solver/Utility.cs`: `Rational`, L0). Every odds the rules give and L0 itself are
  rational, so the reference and the pruned solver agree to the last bit or not at all; the harness compares them with
  `==`, not to 1e-9. `ValueDouble` is for reporting.
- **Draws in the engine's own pile** (`Solver/Successors.cs`, `DrawSteer`). The engine draws off the top of one list, so
  each step runs on a copy whose unknown middle is laid out by the solver: the cards drawn so far in this branch, then
  the rest in a canonical order (by key value, then uid). A step that takes the rest's front card has taken the next
  position of the unknown middle: it is run again once per distinct value that position can hold at count / size (the
  reference: once per card at 1 / size); a uniform pick further down the pile needs no branch. Uniform picks over
  the draw pile go through `StableShuffle`, which sorts by (id, upgraded) first, so the value picked is the same in
  every layout unless two cards tie there and differ in key value (a grown Rampage beside a fresh one): then the
  tie-break is the pile's order, the canonical layout fixes it (Uproar over the two came out 1/4 and 3/4, not 1/2
  each, on the reference too, which shares `Successors.Expand`), and the order left behind is not uniform; the solve
  is refused (`draw-tie`). The check runs at every
  roll, so a draw branches before any roll that follows it. A discard shuffle inside a step is laid out the same way
  by answering its Fisher-Yates calls. A screen's answer lays out the screen copy's pile (a draw-pile choice keeps its
  chosen card at its index), which the replay puts back at the screen. Refused: a card that peeks at the pile's order
  (I Am Invincible, Perfect Fit), a random placement into a known run (`hidden-branch`). Refusals left: `draw-peek`, `draw-tie`, `hidden-branch`, `draw-order`, `card-uid`, `unlisted-roll`, `empty-range`, `cycle`, `table-engine`, `engine-error`, `macro-draw-mismatch` (section 17), a queued hook action or an early resumption (`kernel-queue-busy: ...`, the key's refusal) and `budget` (no horizon completed). A roll wider than the width cap (default 32) is not refused since the baseline round (section 16): its branch is a horizon leaf. Since the macro-draw round (section 17) a run of plain draws is one node over the sub-multisets drawn where no draw observer is present; otherwise draws are sequential, merged by value
  and, for their successors, by key.
- **Exact means proved.** A node carries a sound upper bound (a horizon leaf's is L0's maximum, 1); a max node is exact
  when a move of greatest value is exact and no move's bound is above it, a chance node when every outcome is. An
  exact max node carries its exact move's mass, and a solve is `Exact` only with the full mass. A cut
  solve is never exact; a budget cut returns the last completed horizon as `Budget`, or `Refused` when none completed.
- **Pruning bits** (`Pruning`): `Transposition` (values by key and turns left, equal outcomes of a chance node merged),
  `MultisetHand`, `InterchangeableEnemies` (a no-op while every combat holds an enemy-order reader, S1's
  `EnemyOrderAlways`), `SubtreeReuse` (the table outlives a solve), `DrawByValue`. Pick-k screens are not made
  canonical by pick order: the table merges their equal results.
- **Budget** (`Solver/Budget.cs`): nodes, engine steps and time per decision; iterative deepening over turns, each
  horizon skipped before it starts when the week-3 estimate (W4-S1's formula) is over the node budget by more than the
  slack. At the cut L0 stands in for the network (S5).
- **Harness**: `tests/fixtures/solver/positions.jsonl` (400 recipes, `tools/solver/harness.py`) and
  `EqualValueHarnessTests`: every pruning set against the reference, root value, every root move's value, exactness and
  the best set. The horizon is drawn apart from the other fields; horizon-1 positions are held to a minimum per engine
  and source, root screens per kind (draw-pile choices included, which run the keep path), and skips are reported by
  class. Recipes past the 400 mixed ones stand on a screen card's screen, or on the owner's order-dependence fights
  (Hunter Killer with Tender up, the corpus's Toadpoles with Thorns up), which must be solved. A second test keeps one
  solver's table from a parent's solve into its children's and holds them to the reference (subtree reuse).
- **Chance model** (`ChanceModelTests`, section 6): the reference shares `Successors.Expand` with the pruned solver,
  so the harness cannot see a wrong chance model. Each harness position's first root moves are also stepped on plain
  copies with real seeded streams and the unknown middle in a seeded uniform order, and Expand's odds by key must
  match the count (every counted outcome in the model, every frequency within five standard errors).

## 15. S4 as built

- **One code path** (`Solver/SolverService.cs`): the native exports and EmuDump's `solve` and `hindsight` commands call
  the same service and print or write the same fields. Native API version 39 (`NativeExports.cs`); the run API is unchanged (32 on this base).
- **Exports** (`Interop/SolverExports.cs`): `Sts2Solver_Create/Destroy/Reset`, `Sts2_SolveExpected` (status, value,
  exactness, mass, best move, counters; the q buffer is `maxActions` x 7 floats, column 0 for an untargeted move and
  1 + t for target t, a merged move taking the value of the move it was merged into, NaN for none), `Sts2_SolveHindsight`
  (`HindsightResultNative`: outcome, yes/no/unknown, best utility, completeness, the line's first 64 moves),
  `Sts2_SolvePotionCurve`, `Sts2_SolverInfo`. The two result types share no field, in C#, across the boundary and in
  Python. Only the L0 utility is built; `SearchHp`, `L0Priced` and the horizon leaf protocol come with S5. A native
  handle leaves some streams unset (`Sts2_Create` shuffles on the step stream); the expected solve reads a copy with a
  stream of its own in each unset slot (`SolverService.ExpectedView`: no value changes, and the shuffle reaches the draw
  model on its own stream); hindsight refuses such a handle (`hindsight-unset-streams`).
- **Hindsight** (`Solver/Hindsight.cs`): depth first over every legal move (each hand index and target), lethal first,
  real steps on copies with the position's own streams, merged by the hindsight key plus the step stream; a line past the
  turn cap or the budget is unknown, never a loss. Refused unless `HindsightOptions.AllowHindsight` (natively: the solver
  made with `allowHindsight`) and unless the handle carries its own streams (`NativeCombat.TrueStreams`: false for a
  resampled fork and for a loaded document), and unless every stream is set (an unset one falls back to the step's
  stream, which the line itself advances, so the draws after a reshuffle would depend on the line: neither the game's
  world nor section 9's). EmuDump refuses it on a re-anchored record, whose streams are the emulator's, and on the seed
  form's fight (unset streams); its one accepted form is `--run`, a run's fight forked as it stands. A cut search's best
  line is a lower bound on the bound, so no export carries it as the bound: `best_utility` is NaN natively, null in
  EmuDump and None in Python unless the outcome is searched. In C# the result's `UpperBound` is the bound when searched and
  L0's maximum when cut (a line not finished may still win at full score), the bound any policy's outcome is held to.
- **Potion curve** (`Solver/PotionCurve.cs`): one expectimax per subset with the other held potions masked out
  (`SolveOptions.AllowedPotions`, a slot empty at the root stays usable), carrying the policy's win chance, win HP and
  per-potion held chance beside the value; the potions outside the subset stay in the copy (no table shared yet). One
  horizon per curve: every subset is solved at horizon 0, 1, ... in turn, and the curve is the deepest horizon every
  subset finished within its budget, or, when none did, refused whole (`budget:`). Monotonicity (more potions never lower
  the value) is a claim between entries of one curve, which share that horizon; across horizons it does not hold. The win
  chance, win HP and held chances follow the first move of greatest value, so among moves of equal value they depend on
  the tie-break (the move order of `Actions.Legal`); the value does not. Two bit conventions: an entry's `subset` is a bit
  set over the held potions in slot order (bit j: the j-th held potion), while `allowed_potions` (in the options, the
  expected result and each curve entry) is a slot mask (bit i: slot i usable; -1 all). The native options carry the
  complement, `ForbiddenPotions`, so a zeroed struct masks no potion, and the result echoes the mask the solve used.
  `CurveSolver` repeats `Expectimax`'s recursion with the statistics beside it: a change to the leaf protocol (S5) or to
  the search (S8) must be made in both, and `PotionCurveTests` holds their values equal.
- **Refusals carry no number**: a refused expected solve's value and mass, and a refused curve entry's value, chances and
  held chances, are NaN natively, null in EmuDump's JSON and None in Python; the C# Rational stays beside the status.
- **Python** (`sts2_gym/native.py`): `solver_create(..., allow_hindsight=False)`, `solve_expected` -> `ExpectedSolve`,
  `solve_potion_curve`, `solver_info`; `solve_hindsight` -> `HindsightBound` binds `Sts2_SolveHindsight` on its first
  call (which adds nothing: ctypes resolves an export on attribute access; the gate is the library's flag), and
  `open_solver` destroys its handle on exit. `test_solver_hindsight_pins.py` runs every lab and search entry point and
  fails if one reads a hindsight name or loads code naming one, the flag, or EmuDump's hindsight command, and scans rl/,
  referee/, src/sts2_gym/ and tools/ (but tools/emudump) for the same. Solver handles are claimed atomically; one handle
  serves one thread at a time, and the last error is one buffer for the process.
- **Gates**: `NativeSolverTests` (native equals EmuDump's path on the harness set), `SolverLeakTests`,
  `HindsightApartTests`, `HindsightBoundTests` (the bound is never below the expected-best policy played on the same draws,
  and its mean is at least the mean expected value), `PotionCurveTests`, EmuDump's `SolveModeTests`.
  `HindsightBoundTests` runs on 48 small fights at low HP (none won at full score) with a cheap policy (the per-record
  claim holds for any policy), and requires searched records with a bound above the outcome and a complete loss; the
  mean claim takes the records whose expected solve is exact and whose search was complete. `SolverLeakTests` has a
  positive control (the hindsight search tells the determinized worlds apart) and a floor on worlds whose draw order
  differs. `test_solver_aot_sites.py` runs in every `tools/build_and_test.sh` (`STS2_REQUIRE_AOT=1`: it fails, never
  skips, when the library or the Release EmuDump is missing) on the documents in `tests/fixtures/solver/aot_sites.jsonl`,
  the reshuffle, the draw tie and a sample of the harness positions. `SolverSiteGoldenTests` pins the engine's bytes
  under seeded play at df1db6a6, the commit before the site markers.

## 16. The baseline round as built (optimisation review changes 2 and 9, S6)

- **Counters, always on** (`SolveCounters`, on `ExpectedResult`; EmuDump's `stats` object; natively
  `Sts2_SolverStats(solver, buf, cap)`, the last expected solve's stats as JSON, size-and-retry, since the result struct
  has no room): nodes, engine steps, moves expanded and engine runs (step calls) by move kind (play, end turn, screen,
  potion), chance levels summed over runs (the answered rolls and drawn positions a run replays from) and the deepest,
  outcomes of end-turn expansions before merging, table hits split into exact entries and cut entries at the same turns
  left, misses, wide-roll leaves, refusals by reason class, wall time, and the deepening's horizons (for each: how it
  ended, `done`, `budget`, `estimate` or `refused`, the cumulative nodes, steps and seconds there, exactness, the size
  estimate). The native API version stays 39: two exports are added, none changed (`sts2_gym.native.solver_stats` and
  `solver_debug` bind them on first use).
- **Phase timers** (`Solver/SolverStats.cs`, `SolverProf`): a static switch, off by default (`SolverDebug.Flags` bit 0;
  natively `Sts2_SolverDebug(flags)`; EmuDump `--phase-timers`), compiled into the native library and the JIT builds.
  Exclusive time on a per-thread stack for the engine step (less its clones and the chance hook's calls), whole-state
  clones (`StateCloning.Clone`: the solver's copy and the engine's pre-step record), key encode, key SHA-256, `Rational`
  arithmetic and comparison wherever it runs, table lookups and inserts, chance expansion (`Successors.Expand`'s own work,
  `DrawSteer`'s layout, the odds, the chance hook's site, tie and width checks) and bound checks (`NodeValue.Max` and the
  budget check); the remainder is the solve's own frame (recursion, move lists, dictionaries), measured, not inferred,
  so the phases' sum is checked against the solve's stopwatch. Off, each site costs one static read.
- **Merge audit** (`MergeAudit`, a debug option: bit 1, EmuDump `--merge-audit`): at each expanded end-turn move, the
  pre-step key, the key with every candidate dead field masked (block, energy, the per-turn counters, the per-combat
  counters with no reader in any pile: Gold Axe, Murder, Voltaic, Banshee's Cry and Pull From Below, Supermassive), and the
  same keeping block. Distinct masked against distinct full keys per solve is what review change 6 could merge (block
  masked: the upper bound; block kept: the floor). A reader generated later in the fight is not seen.
- **Wide rolls are horizon leaves** (review change 9): a roll wider than `ChanceWidthCap` stops its run, and the branch
  (its probability as reached) is a horizon leaf at the move's position: L0, bound L0's maximum (1), no mass, never exact.
  `Successors.Expand` returns it as an `Outcome` with `WideRoll` set, so the expectimax, the reference and the potion
  curve value it alike and the equal-value harness compares those positions instead of counting them refused alike.
- **The coverage tool** (`tools/solver/coverage.py`, `test_coverage_curve.py`): the real game's nightly block keeps no
  fight positions (its rows and divergence records hold seeds, outcomes, and hand and energy at diverged decisions), so
  the decisions are its emulator arm's, replayed exactly (`gen0.play_emu_run`, same agent seed and run index, CPU prior
  path) on the night's seed block: the lockstep real arm plays those decisions to within its divergence rate. Whole fights
  are sampled stratified by act x kind x won with a fixed seed; each decision is solved once (fresh table, all lossless
  prunings, node cap and wall cap, engine steps uncapped) and the coverage at each node and time budget is read from the
  horizons of that run, with the size estimate's skip rule applied at the budget read; `check` holds a direct solve at a
  budget to the reading. Results go under the lab's results directory, never committed.

## 17. The macro draw and the streamed expansion as built (optimisation review change 3)

- **The macro draw** (`Pruning.MacroDraw`, bit 32, in `All`; `Solver/MacroDraw.cs`, `Successors.cs`). A run of plain draws
  from the unknown middle is one chance node over the distinct sub-multisets drawn, at the multivariate hypergeometric
  odds prod C(c_v, x_v) / C(n, k) in exact `Rational`, instead of k sequential nodes. It is wired as a k-prefix layout in
  `DrawSteer`: the run is stepped once with the rest laid out canonically; the chance source watches every take off the
  draw pile (`CombatEngine.TakeWatch`) and numbers every draw call (`CombatEngine.DrawWatch`), and the draw loop marks its
  own takes with its call's number (`CardEffects.PlainTake`; thread-statics the engine reads nowhere, null or 0 outside a
  solver run, so a step is unchanged: the corpus byte proof holds it); at the next check (a
  chance call or the step's end) `DrawPending` reports how many cards of the rest, from the front in layout order, one
  draw call's loop took in a row (`Take`). With two or more, each sub-multiset of that size is laid out at the front in
  canonical order (by value, then uid) and the step is run again. The reshuffle epoch is kept: a draw that runs the old
  rest out takes all of it as one outcome of probability 1, the reshuffle is steered as before, and the new epoch's
  draws are a node of their own, so Exact keeps the full mass. A take that is not the draw loop's (a top-card play, a
  pick, a screen's choice) or a chance call between draws (Confused, Slither) ends the run: those positions branch one
  at a time, as before. Hindsight keeps its sequential draws (its piles are ordered); the reference draws by card.
- **Off** where the hand is kept in order (a hand-position reader in the combat, `OrderedPiles` holding `Hand`) or a draw
  observer can act (`DrawObservers.All`): Hellraiser, Chains of Binding (binds the first cards drawn each turn),
  Speedster when ten of its hits can kill a living enemy, Iteration, Pagestorm, Stratagem (its pick inside the
  reshuffle), Pillage (draws until a non-Attack), Scrape (discards what it drew in draw order) and any kernel listener on
  `AfterCardDrawn` or `AfterCardDrawnEarly`. Checked at the expansion's position and again on the step's copy at each
  pending run. Off too for an open screen's answer (`screen-answer`): the answer replays the step from its start and the
  pile after the screen is the screen's own copy, so the takes there need not follow the layout (the harness's Burning
  Pact screens took the laid-out cards in another order and were refused, `macro-draw-mismatch`, before this rule). The list is a scanned registry: `tools/solver/scan_draw_observers.py` writes `draw_observers.json` from the
  draw loop's per-card calls, the callers that read a draw's result list, the engine's `X.AfterCardDrawn` hook names, the
  Core2 models registering on either hook and the kernel's hook ids, each classed in `draw_observer_classes.json`
  (observer, with the name of its test, or safe, with why); `--check` fails on an unclassed entry, a stale class, a scan
  older than the sources, or an observer the C# table does not hold, and `DrawObserverTests` holds the table to the file.
- **A replay that draws otherwise is refused** (`macro-draw-mismatch`): each macro block must be taken again by one draw
  call's loop, card for card, between the same two checks, and the loop must not take the next card of the rest right
  after it there. It fires only if the registry misses an observer whose count drawn depends on the cards, or a reader of
  the layout before the draw (below).
- **A take of the rest before the draw turns it off for the run** (`rest-taken`, `SolverChance.PlainRun`, the engine
  round). A card of the unknown middle taken outside a draw loop earlier in the same run read the layout the run gave the
  middle (a roll's index into the pile, the first card of a kind, the top), and a block laid out at the front would change
  what it took. The case: Thieving Hopper's steal (`EnemyAI.StealDrawOrDiscardCard`, `NextItem` over the draw and discard
  cards of the best rarity tier, in pile order) at the enemy turn, before the next turn's five-card draw. The run count
  was matched against a layout still holding the stolen card, so the block stopped at its slot (three of the five), and
  the replay with that block at the front drew past it in one loop (g7-emu N8ST89LTSW k0106 to k0112, six refusals).
  The count drawn never depended on the cards' values. `scan_draw_observers.py` lists every take off the draw pile
  (`take:` entries), classed `rest-take` (with the run rule's name, which `--check` finds in `Successors.cs`) or `safe`
  with why; an unclassed take fails the check. Harness: `off:rest-taken` (the Hopper's fight at horizon 1, both engines:
  macro on at the root, off in the run, equal to the reference).
- **Counters** (`SolveCounters`, the stats JSON): `macro_nodes`, `macro_draws_avoided` (the cards each macro node took,
  less one, summed), `draw_nodes_single` (draw nodes with the macro draw on that took one card), `macro_off` (draw nodes
  taken one card at a time because it was off, by reason) and `macro_root` (`on`, `hand-ordered`, `observer:Name`, or
  `pruning-off`).
- **The streamed expansion** (the memory fix). `Successors.Stream` yields one outcome at a time; `Expectimax.MoveValueOf`
  and the potion curve solve an outcome's child before the next outcome's copy is made, keeping each child's value by key
  for the later outcomes that merge into it. An expansion then holds one live copy (plus its pending answer lists), not
  every outcome's; the sums are exact rationals, so values, bounds, mass and table entries are unchanged.
  `Successors.Expand` (the reference, the tests) collects the stream.
- **Tests.** Harness recipes past the order-dependence ones (`macro` names them, both engines): a five-card draw over six
  values mid-turn (Expertise) and at the turn start (horizon 1, cards too dear to play twice), a draw across the
  reshuffle, the ordered hand and one per observer; the harness checks that the observers and the ordered hand switch it
  off at the root and the five-card draws take it, and adds the `no-macro-draw` configuration. `ChanceModelTests` holds
  the macro model equal to the sequential model and to the independent count on every harness move, and on draws past the
  reference (five of eight over six values at a turn start, five of seven over seven, a turn start across the
  reshuffle).
