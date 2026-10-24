# One comparer: struct_compare

Plan of record: referee/design/rebuild-plan.md section 1, "One comparer" and "Field census". This is round R7's
Tier 1 part: combat only (corpus fights and traced fights). Whole runs (`run_compare2`) are the run layer and out of
scope here; they get the same treatment in a later round.

## The idea

Every engine writes its state in the game's own shapes, with the game's names. One structural comparer judges every
engine against the game's record. Nothing on the comparer's side renames or projects: no alias table, no per-engine
code. Where an engine cannot say something the way the game does, that is a reviewed row in `allowed_diff.json`, with
its reason and the legacy rule it replaces. The comparer is calibrated on the old engine: it must flag every fight the
legacy comparers flag, at the same step or earlier for a listed reason.

## The shapes

A record holds two shapes of the game's state per step:

- **snap**: the referee's per-step snapshot (referee/driver/Combat.cs Snapshot): `in_combat`, `round`, `player`
  (hp, max_hp, block, energy, powers), `hand`, `draw` (in order), `discard`, `exhaust` as `[ModelId, upgraded]`,
  `enemies` (name, hp, max_hp, block, powers, `intents` as `[type]` or `[type, damage per hit, hits]`, `move`), `rng`
  (`{stream: {counter, seed}}`), and in setup fights `potions`; once a won fight is over, what the run holds.
- **net**: the game's NetFullCombatState after each checksum (`step["net"]`, traced fights only): creatures, players
  with piles of card objects, potions, relics, orbs, RNG counters, sequencing ids.

`IEngine.GameView()` (tools/emudump/Engine/IEngine.cs) writes the engine's current step as
`{snapshot, net, private}`. `EmuDump fight --game-view` adds it to each step as `step["game_view"]`; without the flag
the output does not move by a byte. `private` is the engine's own block for what the game's schema has no field for
(old: split buffs such as INFERNO_SELF_DAMAGE, relic counters); no comparer reads it.

The old engine's binding is `tools/emudump/Engines/Old/OldGameView.cs`. The translation fight_compare.py applied to
old's native dump moved there, each rule cited: `power()` (UPPER_SNAKE, `_POWER`), `EMU_RENAME` (power_aliases.json),
`NOT_POWERS` (to the private block; STUNNED becomes the Stun intent), `emu_intent` (a damage-carrying move leads with its
attack), `emu_view`'s living roster, and card, enemy, potion and relic ids through data/id_map.json. The net shape is
OldNetState's (what `--netstate` writes) with the same power rules and the relic counters moved to `private`.

## Structural rules

These are facts of the game's schema, the same for every engine:

- dicts by key; lists by index plus `path#len`; a list of scalars is also one leaf; an empty list is its `#len`;
- `powers` (`[[id, amount], ...]`) keyed by id: `snap.player.powers{STRENGTH_POWER}`;
- `keywords` is a set; a net card object omits default fields (`upgrades` 0, `keywords` [], no `cost`, `enchantment`,
  `affliction`), so absent and default compare equal and absent and set differ;
- a path one side lacks: an entry of a collection the other side has (an index past its end, a power it lacks) is a
  difference; a field the record has and the engine does not write is a difference unless a row lists it as
  unmodelled; a field only the engine writes is unjudged (the record does not say).

## Field census and its gate

Every comparison records the schema paths it compared (indices `[*]`, keys `{*}`) per fight, plus the unmodelled, unjudged and
waived ones. A path an allowed row waives (the `ignore` row: it stands for the comparison, so the path is waived whether the
values agree or not; or a difference a row excused) is not compared: it goes to the fight's `waived` set and not to its census.
`struct_compare.census_gate(A, B)` lists, per fight, the paths engine A was compared on and engine B was not, over the `compared`
sets only; the gate fails when the list is not empty. `struct_compare.waived_total` and `struct_calibrate.py ... --gate-against
<A's output>` report the waived counts next to it. On the old engine the census is 33 schema paths per corpus fight (median;
40 at most) and 110 per traced fight (117 at most), with 3 waived paths per traced fight (median).

## Allowed rows (tools/fidelity/allowed_diff.json)

16 rows. Each lists the `engines` it applies to: `old` for rows that exist only because of the old engine's limits, `*` for the rows
that state the record's own representation or integrity. `struct_compare --engine E` (default `old`; `compare_fight(engine=)`)
applies only the rows of E and of the engine whose state E runs over (`struct_compare.ENGINE_BASE`: core2 is old's state with the
kernel attached, so it has old's limits and is judged by old's rows; an engine with a state of its own is judged without them),
so a new state is judged without the old engine's rows.

| kind | rows | engines | what it does |
|---|---|---|---|
| unmodelled | enemy-move-id, rng-seeds, net-sequencing, saved-props-floor, relic-counters, after-fight-turn-state (6) | old | a record-only field old does not write is not compared (listed in the census as unmodelled): enemy move id, RNG seeds, net sequencing ids and phase, saved props and floors, relic counters, energy and round after a fight |
| drop_dead_enemies | dead-enemies | * | the record's enemies at 0 HP are not the roster |
| lost_outcome_only | fight-lost | old | a lost fight compares in_combat and the player's HP only (old plays the enemy phase out) |
| power_fold | power-folds | old | TemporaryStrength/Dexterity/Focus subclasses and SelfFormingClay summed into the base power on both sides; counted as fired only when a folded subclass is present |
| intent_coarse | intent-vocabulary | old | an intents diff is allowed when the first intent's class and the first attack's [damage, hits] agree. A kind outside the class map keeps its own name (Summon against Escape is a difference); Unknown, the kind old writes for what its five kinds cannot say, is the one wildcard for an unmapped kind of the record, never for a mapped one |
| ignore | cost-outside-hand | old | card cost outside the hand is waived (old prices the hand only); waived, not compared |
| same_first | affliction-amount | old | affliction id compared, not its amount (old's Bound is a flag) |
| both_positive | upgrade-count | old | upgrade count vs old's flag |
| settle | after-last-checksum | * | a net path the record's own snapshot shows changed after the step's last checksum is judged by the snapshot |
| surplus | kernel-sequencing (1) | core2 | the engine-written paths `net.last_action` and `net.last_hook` (the kernel's own queue counters, in the game's names) are taken out of both sides, listed in the census as `surplus`, never compared: the record's ids come from the live game's synchronizer and count actions the emulator does not run. Where the engine writes none the record's side stays under net-sequencing |
| late_steps | late-steps | * | record integrity: the net shape is not compared from a step recorded before the game finished it. The limit applies only when the row is present, and each skipped step is counted under the row |

Each row's `legacy` field names the fight_compare / netstate_diff rule it replaces.

## Calibration on the old engine (2026-10-24, allowed_diff sha 4b395b0af9192ff5)

One replay per fight (`fight --game-view`, plus `--netstate` when traced), judged by all three comparers
(`tools/fidelity/struct_calibrate.py`). A comparer crash has its own cell (struct_crash); there were none.

| | both, same step | both, struct earlier | both, struct later | legacy only | struct only | neither | struct crash |
|---|---|---|---|---|---|---|---|
| corpus (4135) vs fight_compare | 984 | 3 | 0 | 0 | 0 | 3148 | 0 |
| traced (77) vs fight_compare | 70 | 7 | 0 | 0 | 0 | 0 | 0 |
| traced (77) vs netstate_diff | 51 | 1 | 23 | 0 | 2 | 0 | 0 |

The traced set is the 77 first fights (with an ascension and net states) of the six newest `traced.out.json` files of the lab; they
are re-runs of fights that diverged, so every one of them differs somewhere. Another 77 (the oldest seven files) gave legacy-only 0
against both comparers too. The 5 legacy-only fights of the first calibration (below) came from a different selection that is not
recorded; they cannot be re-derived, and the explanation is kept.

Explained:

- **Struct earlier, corpus (3)**: the record lists the relics in combat (a setup fight with PEN_NIB twice; old holds
  one). fight_compare compares relics only after a won fight, struct_compare at every step.
- **Struct earlier, traced vs fight_compare (7)**: net-shape fields fight_compare has no shape for (a hand card's cost, the
  Ethereal keyword of a drawn card, an affliction, an enchantment, an orb's amounts) and the relics in combat; netstate_diff flags
  the net-shape ones at step 0.
- **Struct later than netstate_diff (23; nd at step 0 in 23, struct_compare at fight_compare's step in 20)**: first
  netstate_diff paths `piles.Hand.detail`. This is the legacy settle artifact described next: at step 0 the game played on after its last
  checksum, netstate_diff settles the pile's cards from the snapshot but compares the stale card details of the last checkpoint;
  struct_compare settles the whole pile and flags the fight where the snapshot comparison does.
- **Legacy only, netstate_diff (5 in the first calibration; two seeds, recorded 2 and 3 times)**: the same artifact on fights
  that never diverge afterwards. Not a missed divergence.
- **Struct only vs netstate_diff (2)**: s/traced.out.json#10 step 0: 7 steps identical on every modelled field ['snap.enemies[0].intents[0]']; s/traced.out.json#11 step 7: 10 steps identical on every modelled field ['snap.enemies[1].intents[0]'].

## Self check: the engine against its own game view

`struct_calibrate.py corpus|traced --self` takes the engine's own game view as the record (its snapshot as each step's state, its net
as the step's net checkpoint, no checksums so no step is skipped) and judges it against the same output: every fight must land in
"neither" with no difference, or the command exits 1. It checks the comparer's own mechanics (flatten, rows, census) and the game view's
shape. Result on old: traced 77 of 77 identical, corpus 300 of 300 identical (the first 300 fights), no crash.

Census gate on the traced set against itself: 0 of 77 fail; against a census with one more field: 77 of 77 fail (the ids were
not unique across traced files in the first version: the gate compared 24 of them; fixed).

## Core2 (K5, 2026-10-24, allowed_diff sha 11083dbf8259a2b4)

`IEngine.GameView()` of core2 is old's view plus the kernel's state when a switch is on (TIER1_KERNEL.md section 6): the game's
`lastExecutedActionId` and `lastExecutedHookId` as `net.last_action` and `net.last_hook`, the rest under `private.kernel`.
`struct_compare` carries a **surplus** list next to the census: the schema paths of the engine's `private.kernel` block and
the paths a `surplus` row names (kernel-sequencing). They are listed (`surplus` in a fight's result and in `struct_calibrate`'s
output), never compared against the record, and never counted into the census, so the census gate (core2 never compared on fewer
paths than old on the same item) does not move with them. `struct_calibrate corpus|traced --engine core2` runs EmuDump with
`--engine core2` and judges with core2's rows; `--gate-against <old output>` is the census gate.

With no switch on (the default `migrated_families.json`), first 300 corpus fights, old against core2: the table is identical
(both same step 37, neither 263, struct crash 0), the allowed rows fired are identical, every fight's first difference, paths,
census, waived set and leaves are identical (0 of 300 differ), the census gate reports 0 of 300 fights compared on fewer fields,
surplus empty. byte_proof old against `--b-args "--engine core2"`, all 11 modes (`game-view` and `game-view-netstate`
added), 300 fights: 3300 fight-mode pairs identical, 0 differ.

## Left for the next round

- The run layer (`run_compare2`) on the same shapes (rows, `--engine`, census, self check).
- Core2 on the traced set (`struct_calibrate traced --engine core2`), and the census gate there.
- Fields old could write natively and does not yet: RNG seeds, the enemy's game move id, the game's relic props. Each one removes
  an `old`-only unmodelled row.
- The legacy comparers keep running on old only, as the calibration reference and the history line; the corpus check
  (`corpus.py check`) still judges through fight_compare until its gate path moves to struct_compare.
