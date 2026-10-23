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

Every comparison records the schema paths it compared (indices `[*]`, keys `{*}`) per fight, plus the unmodelled and
unjudged ones. `struct_compare.census_gate(A, B)` lists, per fight, the paths engine A was compared on and engine B was
not; the gate fails when the list is not empty. `struct_calibrate.py ... --gate-against <A's output>` runs it over a
corpus or traced run. On the old engine the census is 33 schema paths per corpus fight (median; 40 at most) and 102
per traced fight (110 at most).

## Allowed rows (tools/fidelity/allowed_diff.json)

15 rows, by kind:

| kind | rows | what it does |
|---|---|---|
| unmodelled | 6 | a record-only field old does not write is not compared (listed in the census as unmodelled): RNG seeds, net sequencing ids and phase, saved props and floors, enemy move id, relic counters, energy and round after a fight |
| drop_dead_enemies | 1 | the record's enemies at 0 HP are not the roster |
| lost_outcome_only | 1 | a lost fight compares in_combat and the player's HP only |
| power_fold | 1 | TemporaryStrength/Dexterity/Focus subclasses and SelfFormingClay summed into the base power on both sides |
| intent_coarse | 1 | an intents diff is allowed when the first intent's class and the first attack's [damage, hits] agree |
| ignore | 1 | card cost outside the hand (old prices the hand only) |
| same_first | 1 | affliction id compared, not its amount (old's Bound is a flag) |
| both_positive | 1 | upgrade count vs old's flag |
| settle | 1 | a net path the record's own snapshot shows changed after the step's last checksum is judged by the snapshot |
| late_steps | 1 | record integrity: the net shape is not compared from a step recorded before the game finished it |

Each row's `legacy` field names the fight_compare / netstate_diff rule it replaces.

## Calibration on the old engine (2026-10-24, allowed_diff sha fb08098af794d38b)

One replay per fight (`fight --game-view`, plus `--netstate` when traced), judged by all three comparers
(`tools/fidelity/struct_calibrate.py`).

| | both, same step | both, struct earlier | both, struct later | legacy only | struct only | neither |
|---|---|---|---|---|---|---|
| corpus (4135) vs fight_compare | 984 | 3 | 0 | 0 | 0 | 3148 |
| traced (77) vs fight_compare | 19 | 4 | 0 | 0 | 0 | 54 |
| traced (77) vs netstate_diff | 9 | 0 | 0 | 5 | 14 | 49 |

Explained:

- **Struct earlier, corpus (3)**: the record lists the relics in combat (a setup fight with PEN_NIB twice; old holds
  one). fight_compare compares relics only after a won fight, struct_compare at every step.
- **Struct earlier, traced vs fight_compare (4)**: net-shape fields fight_compare has no shape for (a hand card's
  cost, an exhausted card's Sly keyword); netstate_diff flags each at the same step.
- **Legacy only, netstate_diff (5; two seeds, recorded 2 and 3 times)**: a legacy settle artifact. At step 0 the game
  played on after its last checksum; netstate_diff settles the pile's cards from the snapshot but compares the stale
  card details of the last checkpoint (`piles.Hand.detail`). struct_compare settles the whole pile, and the snapshot
  comparison of the same step (fight_compare too) finds the step identical. Not a missed divergence.
- **Struct only vs netstate_diff (14)**: fights where netstate_diff stops at a step the driver recorded too early
  (verdict "driver: ..."); the snapshot is still compared there, and fight_compare flags all 14 at the same step.

Census gate on the traced set against itself: 0 of 77 fail; against a census with one more field: 77 of 77 fail.

## Left for the next round

- The run layer (`run_compare2`) on the same shapes.
- Core2's GameView and the census gate against old's census on every corpus and traced fight.
- Fields old could write natively and does not yet: RNG seeds, the enemy's game move id, the game's relic props.
- The legacy comparers keep running on old only, as the calibration reference and the history line; the corpus check
  (`corpus.py check`) still judges through fight_compare until the gate path moves to struct_compare.
