# F08 (damage / HP) diagnosis, block rank-s2-m12

Date 2026-10-24. Input: `agent/gen0/rank-s2-m12` (ranked.json, by_family.json) and the lockstep records in
`agent/gen0/real-s2-m12/divergences.jsonl` / `runs.jsonl` (500 runs, 53,725 decisions). The block ran on
`fork/wt-d24` (integrate/m1m2 at d225d9f5, the build fork main points at; fork main's own `out/` is from
2026-10-15 and is stale). Groups: the four that hold F08 weight (Gb25e996655, G2891c25e0d's F08 part,
Geea146c773, G03884f363d) plus all of G2891c25e0d. Record weights recomputed with `tools/lab/rank.weigh`; each
record's share of its group is applied to the group's ranked weight. The four groups hold 4110.5 ranked weight
(58.4% of the block's 7043.8); F08 proper is 1939.7.

Diff convention (`referee/lockstep.py` `structural_diff`): `[path, referee value, emulator value]`. Since fork main,
`emu_enemy_hp` lists every enemy the emulator's roster holds, dead ones at 0, so a corpse on one side only is a diff
(the R04 text in by_family.md still describes the old comparer).

No game process was run: `referee/QUIET.lock` (taken 16:39Z by another lane) was held for the whole session. Every
mechanism below is therefore read from both sides' code plus the record pattern ("inferred"), not traced. The
trace targets for the first follow-up are listed at the end.

## Per-mechanism weight table

Share = of the four groups' 4110.5. "roots" = records with no earlier divergence in the run (clean state).

| mechanism | ranked weight | share | records | roots (weight) | how sure | remedy |
|---|---|---|---|---|---|---|
| A: death timing, the emulator answers a kill after the whole card, the game inside `CreatureCmd.Damage` | 2136.6 | 52.0% | 64 | 46 (1614) | inferred: code on both sides, 64/68 records one shape | F08 migration (Kill after the hit loop, inside Damage) |
| B: Slow applied outside the decimal fold, after a floor, in float | 401.8 | 9.8% | 15 | 6 (138) | inferred: code, all 15 Bygone Effigy records game 1-2 lower | F08 migration (Lane B interim possible) |
| F: multi-enemy fights, a whole hit more or less on one enemy | 579.5 | 14.1% | 21 | 10 (300) | open | trace first; candidates A (sweep moves target indices), hit counts |
| G: other one-enemy fights (Hemokinesis +3, Rampage +2-3, Whirlwind, Fossil Stalker, Haunted Ship...) | 394.2 | 9.6% | 16 | 3 (77) | open | trace first; Hemokinesis looks like listener order (strength from the HP loss before the hit) |
| E: Shrinker Beetle fights, emulator deals more (a whole hit in 3 of 5) | 173.9 | 4.2% | 5 | 2 (83) | open | trace; Shrink rule candidate (Lane B) |
| C: Soul Fysh, game deals more (1, 2, 6, 18) | 154.0 | 3.7% | 5 | 3 (97) | open | trace; Intangible / Vulnerable rule candidate (Lane B) |
| D: Lagavulin Matriarch, the emulator's block soaks a whole hit the game takes (2, 8, 9, 10) | 130.9 | 3.2% | 4 | 0 | inferred direction only | Lane B single site (Plating / Asleep) |
| H: player hp groups (Corpse Slug 86.3, others 44.9) | 131.2 | 3.2% | 9 | 4 (56) | open | Corpse Slug: Frail (F09 block) or Ravenous (A); trace |
| Z: Waterfall Giant hp stand-in (999999993 vs 32768) | 8.4 | 0.2% | 1 | 1 | measured (record) | representation artefact (R05) |

Of all agent weight in the block: A 30.3%, B 5.7%.

## G2891c25e0d (2277.0): mechanism A, 93.9% of the group

Shape: at a `select_cards_combat` decision the emulator lists one enemy more, at 0 HP; the other enemies agree.
The acted card is HEADBUTT in 50 records (70.8% of the group) and NEOWS_FURY in 14 (23.1%); 46 of the 64 are roots
(clean state, often the first fight, e.g. 2R788UE399 decision 7, ZZHYSB375C decisions 41 and 56). The other 4
records (6.1%, the group's F08:R06 part) are damage amounts and are counted under B, C, D and G above.

Game: `Headbutt.OnPlay` and `NeowsFury.OnPlay` run `DamageCmd.Attack(...)` and then `CardSelectCmd.FromCombatPile`.
`CreatureCmd.Damage` ends with `Kill(killedCreatures)` after its hit loop; `KillWithoutCheckingWinCondition` runs
`Hook.BeforeDeath`, `Hook.ShouldCreatureBeRemovedFromCombatAfterDeath`, `Hook.AfterDeath` and
`CombatState.RemoveCreature` (`CombatManager.RemoveCreature`). So the corpse has left the roster, and the death
listeners have fired, before the selection screen opens.

Emulator: `CombatEngine.ResolvePlay` answers deaths once, after the card's effects
(`AnswerDeathsSince(state, afterEffects, rng, sweep)`), and `AnswerDeaths` sweeps corpses with
`CombatState.RemoveDepartedEnemies` (the `CorpseSweep` block). A selection opened in the middle of an effect
(`PendingSelection`) is shown before that point, so the corpse is still in the roster at the screen, and every
`AfterDeath` listener (Ravenous, Gremlin Horn, Shrink's applier removal, Sapping-style splits...) runs late.

Remedy: the F08 migration's "Kill after the hit loop" inside the game-shaped `CreatureCmd.Damage` (kill, death hooks
and removal per Damage call, not per card). Do not patch the lockstep view: the corpse is really in the emulator's
roster at that point, and the late death hooks are a real order difference. Expected side effects: any multi-enemy
target index that depends on when a corpse is swept (part of F), and Corpse Slug's Ravenous timing (part of H).

## Gb25e996655 (1702.3): enemy hp at a card play

63 records, 23 roots. By mechanism: B 25.4% (Bygone Effigy), F 33.9%, G 23.2%, E 10.2%, C 6.2%, D 5.7%, Z 0.4%
(shares of the recomputed group weight).

### B: Slow (Bygone Effigy)

Every Bygone Effigy record has the game 1 or 2 HP lower (107/108, 108/110, 67/68, 114/115, 116/117, 81/82, 43/44,
44/46, 40/41, 32/33, 42/43, 39/40, 80/81, 30/31; G2891c25e0d's D6DQB5L2RF 44/45), whatever card was played.

Game: `SlowPower.ModifyDamageMultiplicative` (1 + 0.1 x SlowAmount for a powered attack on its owner) is one
multiplicative listener in `Hook.ModifyDamage`, which folds the additive, multiplicative and cap passes in decimal
and floors once, at the end. The counter rises in `SlowPower.AfterCardPlayed` and resets in `AfterSideTurnStart`.

Emulator: `CardEffects.DealDamageToEnemy` takes the already floored and capped result of
`BuffSystem.IncomingDamage` and then applies `(int)(damage * (1f + 0.1f * slowCount))`. Any fractional amount before
Slow (Vulnerable on an odd base: 9 x 1.5 = 13.5; Weak 0.75) is floored first: 13.5 x 1.2 = 16.2 -> 16 in the game,
13 x 1.2 = 15.6 -> 15 in the emulator. The counter timing matches the game (`CombatEngine` after-play increment,
reset at turn start).

Remedy: F08 migration (decimal `Hook.ModifyDamage` with Slow as a multiplicative listener). Lane B interim if the
migration is not next: move Slow into `IncomingDamage`'s multiplicative passes before its floor and cap.

### D: Lagavulin Matriarch (Lane B single site, inferred)

Four records (two in Gb25, one in G2891, all after an earlier divergence): the emulator's Matriarch loses a whole hit
less (Bash 8, Strike 9, Headbutt 10, Pommel Strike 2), so the emulator's block soaks a hit the game's does not.
Game: `PlatingPower` (`BeforeSideTurnStart` round-1 block, `BeforeSideTurnEndEarly` block, `AfterSideTurnStart`
decrement), `AsleepPower`, `LagavulinMatriarch.AfterDamageReceived` (wakes at half HP). Emulator: `EnemyAI` enemy
Plating block and the Matriarch's Asleep branch (the Plating section of the enemy turn-end). Needs one trace to say
which rule.

### C, E, F, G (open)

- C Soul Fysh: the game deals more (1, 2, 6, 18). Candidates: `IntangiblePower` on the enemy
  (`ModifyDamageCap`, `ModifyHpLostAfterOsty`, `AfterSideTurnEnd` decrement) or Vulnerable on the player.
- E Shrinker Beetle fights: the emulator deals more, a whole hit in 3 of 5. `ShrinkPower.ModifyDamageMultiplicative`
  is (100 - DamageDecrease)/100 and `ShrinkPower.AfterDeath` removes it when its applier dies; the emulator's
  `IncomingDamage` applies a flat 0.70 while any Shrink is held. A whole-hit difference is not the multiplier alone.
- F multi-enemy (slimes, rats, gardeners, Corpse Slugs, Phrog Parasite, cultists): one enemy differs by a whole hit
  (6, 10, 15, 18) or 2. Fits a target index that points at different creatures because a corpse was swept at a
  different time (A), or a hit count. E7ZDPNAXHG 74 (emulator keeps a rat at 8 that the game killed) is one.
- G: Hemokinesis gives the game +3 in two runs (7JQBAD5A6M, GQSYMF996Y): fits a strength gain from the card's own HP
  loss landing before the hit in the game and after it in the emulator (listener order). Rampage +2-3 four times in
  one run (JG3KEPAY59). Whirlwind 2-10.

## Geea146c773 (82.4) and G03884f363d (48.8): player hp (H)

Geea146c773: player hp only, 6 records (Corpse Slug x2, Mawler, Cubex Construct, Lagavulin Matriarch). G03884f363d:
enemy hp and player hp together, all three in Corpse Slug fights (3 HP to the player, 3-4 to an enemy, after a Strike,
a Defend or an end turn). Corpse Slug applies `FrailPower` (block pipeline, F09) and `RavenousPower` (strength and a
stun to the survivors on an ally's death, an `AfterDeath` listener, so A's timing). Open; trace first.

## What the F08 migration addresses

A + B = 2538.4 ranked weight, 61.8% of these four groups and 36.0% of the block's agent weight. Within F08 proper
(1939.7) the migration has a measured target only in B (about 21%); A sits in the unassigned G2891c25e0d but is the
migration's Kill step. F, G and H (26.9% of the four groups) may add to it once traced; C, D and E (11.1%) look like
single-model rules for Lane B.

## Trace targets (when QUIET.lock allows; game-trace copy, one fight per process)

- A: 2R788UE399 decision 7, floor 2 (first fight, Headbutt kill then screen).
- F: 8526CXYUBK decision 12, floor 2 (first fight, root) and 8V84HBYD5R decision 27, floor 3.
- B: QJCEN53FNG decision 71 (root), to confirm Vulnerable or Weak at the hit.
- E: BQNPC294Q6 decision 60 (root). C: FTV3HZZPFE decision 135 (root). D: 9U8H6MK3R2 decision 119.
