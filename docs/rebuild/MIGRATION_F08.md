# F08 migration (damage / HP with the kill sequence): plan

2026-10-24, rebuild lane round M0, branch `rebuild/m0-f08` (from fork main a1e4d7ea, the Tier 1 kernel K1-K4), game
v0.107.1. Input: `TIER1_KERNEL.md`, the F08 diagnosis (`D:/spireline/referee/rebuild/f08-diagnosis.md`: mechanism A,
kill timing, 52% of F08's groups and 93.9% of G2891c25e0d; mechanism B, Slow outside the decimal fold), the site
inventory (`python -m tools.kernel.sites groups`), the port manifest (`D:/spireline/referee/rebuild/manifest.md`) and
the decompile (read only). Game code is named by class and method only. This round changes no engine behaviour.

## 1. Confirming trace (traced, 2026-10-24)

Two setup fights on `referee/game-trace`, one fight per process, BelowNormal, `trace` + `trace_listeners`, no network
connection, real profile untouched. Ironclad A10, Corpse Slugs (weak, two slugs, each with Ravenous: an AfterDeath
listener that gives the survivor Strength), the player at Strength 40. Jobs and outputs in
`D:/spireline/referee/jobs-trace/rb-m0/` (`hb_*`: Defend, Defend, Headbutt on slug 0; `ts_*`: Twin Strike on slug 0).

**Headbutt kill, then its discard-pile screen (step 3).** The game's sequence inside the one PlayCardAction:
BeforeAttack, ModifyAttackHitCount, ModifyDamage (3 passes), AfterModifyingDamageAmount, BeforeDamageReceived,
ModifyHpLost (before Osty), ModifyUnblockedDamageTarget, ModifyHpLost (after Osty), the DamageReceived entry (28),
AfterCurrentHpChanged, AfterDamageGiven, then the kill: BeforeDeath, ShouldDie, ShouldCreatureBeRemovedFromCombatAfterDeath,
AfterDeath (Ravenous runs here: the survivor's PowerReceived Strength 5, with its own power passes nested), then
ShouldPowerBeRemovedOnDeath; then AfterAttack, whose listener list already lacks the dead slug (the survivor is now
enemy slot 0); then the selection (the chosen Defend's AfterCardChangedPiles); then the `played` mark. So the game has
answered the death, run its AfterDeath listeners and removed the corpse before the attack ends and before the screen
opens. During BeforeDeath, ShouldDie, ShouldCreatureBeRemoved and AfterDeath the dying slug's monster and powers are
still listed (the `MarkDying` window, TIER1 section 3).

**Emulator, same fight.** `EmuDump reanchor --trace` and `fight --trace` agree with the game on every mark and on the
step's result (`trace_diff`: "result differs in: nothing"; the old engine's marks are only start/play/played, so the
order inside the step is not visible there). Replayed without the game's answer (`hb.noanswer.*`), the emulator stops at
the screen (`DiscardToDrawPileTop`) with the corpse still in the roster at 0 HP with its Ravenous, and the survivor
without the Strength the game gave it before the screen: exactly the G2891c25e0d shape, now traced.

**Twin Strike killing one of two (step 1).** Game: one hit only. Hit 1 runs the whole damage sequence and the kill
(AfterDeath with Ravenous, removal) inside its CreatureCmd.Damage; at hit 2 AttackCommand finds no living target of
its own and stops, then AfterAttack (corpse gone). The emulator also lands one hit (result identical); the order
difference (the death answered after the card, not inside the hit) is the same as Headbutt's and is visible to any
later hit of the same card on another target (Whirlwind, Sword Boomerang) and to any listener between.

## 2. The switch and its bridge splits

The kernel has one switch group today (28 bridge sites). F08's damage command runs F10's kill sequence inside it
(BeforeDeath ... AfterDeath, removal), and two sites serve both (`EnemyAI.ShouldRemoveOnDeath`, F08+F10;
`RelicEffects.ApplyAfterPlayerHpChanged`, F08+F10), so **F08 and F10 switch together**. Bridges joining them to other
families, read for this plan:

| bridge | families | reading | record |
|---|---|---|---|
| `CombatEngine.DoomThePlayerAtTurnEnd` | F05+F10 | Doom's turn-end body (F05 moment) that then kills the player (F10). Two statements. | `split`: F05 `int doom = BuffSystem.Get(state.PlayerBuffs, BuffId.Doom);`, F10 `state.PlayerHp = 0;` + the next line `Effects.RelicEffects.ApplyAfterPlayerHpChanged(state);` |
| `RelicEffects.BiteForFurCoat` | F01+F08 | Fur Coat's AfterCreatureAddedToCombat body (F01) reads the hittable predicate (F08, the old `Hook.ShouldAllowHitting`). | `split`: F01 `public static void BiteForFurCoat(CombatState state, EnemyState enemy)`, F08 `state.FurCoatBites && EnemyAI.IsTargetable(enemy)` |
| `RunRewardGenerator.GenerateCombatRewards` | F10+F14+F15 | F10's part is AfterCombatVictory(Early) bodies, interleaved with F14's potion forcing and F15's reward modifiers. | none needed under option B below |

Anchors checked unique and in order in the current files; each record keeps the file's hash when M1 writes it.

**Option B (recommended): re-cut F10.** Move AfterCombatEnd, AfterCombatVictory and AfterCombatVictoryEarly from F10
to a new family F17 "combat end / victory" in `tools/fidelity/hook_families.json`. They run after the combat's result
(RunEngine.FinishCombat, the rewards, run counters), not inside the kill. Computed in memory with
`sites.build` and the two splits above: groups become **F08+F10** and the other fifteen (F01-F07, F09, F11-F17). The
switch then needs 94 handlers instead of 135 (41 victory-only models: 40 relics and one card; 26 of their 40
AfterCombatEnd overrides are not modelled). Cost: the family table, `Switches.g.cs` and the manifest are rebuilt;
by-family tables carry the new row.

Option A (F10 whole): also split `GenerateCombatRewards` (three parts, anchors to be read) and port the 41 victory
handlers in this migration.

## 3. Commands to write in Core2

Folder `src/Sts2Emulator/Core2/Commands/`. Each is the game's command in its shape, cites `game: <Class>.<Method>`,
reads and writes HP, block and buffs through the old state (`CombatState`, `EnemyState`, `BuffSystem`), and never
re-implements a model's effect (that is a handler calling the old body). Dispatch is only through `Hooks.*`.

1. **`AttackCmd.Execute`** (game: AttackCommand's execute): attacker dead: nothing. BeforeAttack; hit count through
   ModifyAttackHitCount (no old site today: the old cards pass a count); per hit: stop if the attacker is dead; the
   living possible targets; none left in a live combat: stop; one `CreatureCmd.Damage` per hit, single target or all
   valid targets in one call; then AfterAttack. Random targeting with or without duplicates as the game's flag.
2. **`CreatureCmd.Damage(targets, amount, props, dealer, cardSource)`** (game: CreatureCmd.Damage, the list overload
   all others reach): dealer dead: empty results. Per target in list order, skipping the dead:
   `Hooks.ModifyDamage` (all passes, decimal) then `AfterModifyingDamageAmount` with its changed set;
   `BeforeDamageReceived`; block absorption on the target or its pet owner (decimal, the old block field); `ModifyHpLost`
   before Osty on max(0, amount - blocked), `AfterModifyingHpLostBeforeOsty`; `ModifyUnblockedDamageTarget`;
   `ModifyHpLost` after Osty on that target, `AfterModifyingHpLostAfterOsty`; the HP loss with **the one cast**
   (game: Creature.LoseHpInternal truncates the decimal to int, capped at 999999999; overkill kept); when Osty took
   the hit, the overkill goes back to the original target through a second after-Osty fold; history entry
   (DamageReceived). Then, over all results in order: AfterBlockBroken, AfterCurrentHpChanged (unblocked > 0), the
   player's damage-dealt counter, AfterDamageGiven, AfterDamageReceived for a target not killed, else the target joins
   the killed list; then **`Kill(killed)` once, after the loop over targets** (so per hit, per Damage call: never per
   card).
3. **The decimal fold**: `Hooks.ModifyDamage` with its named steps (below) and passes add (changed: nonzero), mul
   (not-one), cap (lowers); the value stays decimal through block and HP loss and is cast once at the HP change. Slow
   becomes a `fold-mul` handler (game: SlowPower.ModifyDamageMultiplicative), which removes mechanism B with no
   special case. Callers that need an int today (intent numbers, the old `IncomingDamage` callers that are not
   migrated) keep the old path until their family moves; the announced intent uses the same fold with
   `PreviewHittable` when the game previews.
4. **`CreatureCmd.Kill(list, force)`** (game: CreatureCmd.Kill): each creature through `KillWithoutCheckingWinCondition`;
   then, all players dead: the loss (old `IsAboutToLose` / pending loss) and stop; else, during the player's side, a dead
   player ends their turn. No win check here.
5. **`KillWithoutCheckingWinCondition(creature, force, recursion)`**: not in a live combat: nothing. HP above 0: lose
   it (unblockable, unpowered) and AfterCurrentHpChanged. `MarkDying(creature)` from here to the removal.
   BeforeDeath; `ShouldDie` (both passes; preventer) unless forced or max HP <= 0. Dies:
   `ShouldCreatureBeRemovedFromCombatAfterDeath`; `AfterDeath(wasRemovalPrevented: false)` (per-listener: a listener's
   screen is a hook action that runs after the current action, K4); then, if removal is due and the creature is an
   enemy still in the roster: the removal, **skipped while its monster `IsPerformingMove`** (the move's end removes
   it); the creature's powers removed with `ShouldPowerBeRemovedOnDeath` (OwnerInCombat) and each removed power's
   AfterRemoved; a primary enemy whose living teammates are all secondary: `Kill(teammates)`; a player: orb queue
   cleared, Osty killed, `DeactivatePlayerHooks` from here, player death handling. Prevented: `AfterDeath(true)`,
   `AfterPreventingDeath(preventer)`, and again while still dead (at most 10, else the game's error).
   `KeepPlayerHooks` covers the player's own BeforeDeath/ShouldDie/AfterDeath.
6. **Removal in the old state** (game: CombatManager.RemoveCreature + CombatState.RemoveCreature): one function that
   does what `CombatState.RemoveDepartedEnemies` does for one enemy today (roster, departed ids, targeting indices),
   called from the kill; the old per-card sweep (`AnswerDeathsSince`, `HandleEnemyDeaths`, `AnswerDeaths`) does nothing
   under the switch for a death the kill answered.
7. **Win check at the game's points** (game: CombatManager.CheckWinCondition: pending loss first, then IsEnding with
   `ShouldStopCombatFromEnding`): K4's executor boundary (after every action but end turn and ready-to-begin-enemy-turn)
   plus StartTurn, EndEnemyTurn, ExecuteEnemyTurn's per-enemy check and EndPlayerTurnPhaseOneInternal; `CombatEnded`
   at the end. The old in-step checks (`CombatEngine.CheckWinCondition` callers) are gated off under the switch.

**Named steps** (`Core2/Steps.cs`, from the descriptor):
- `ClampMin0` (ModifyDamage post): the folded value is max(0, value).
- `EnchantAdd` (pre, when the Additive flag is set): with an enchanted card source, value += the enchantment's
  EnchantDamageAdditive(value, props).
- `EnchantMul` (pre, when Multiplicative): value *= its EnchantDamageMultiplicative(value, props).
- `PreviewHittable` (pre): only for a preview with no target in multi-creature targeting of an all-enemies card in
  hand or play: fold once per hittable enemy; if every result agrees after an int cast, that value and the distinct
  changed set, else the plain fold with no changed set. Never reached by a real hit (preview mode none).
- `NetIdGuard` (AfterDeath, AfterDiedToDoom pre): the dispatch's net-id guard (K3), tested at the fast path first.
- `OwnerInCombat` (ShouldPowerBeRemovedOnDeath pre): the power's owner must still be in the combat.
- `NegativeCostPassthrough` (ModifyEnergyCostInCombat, ModifyStarCost): F07, not needed by this switch.

## 4. Old sites replaced, and their game counterparts

All F08/F10 sites from `sites.json` (file:line at a1e4d7ea). Each gets `if (Kernel.If(Switch.F08, ...) is { } k)
<Core2 command> else <old>`.

| old site | what it does now | game counterpart |
|---|---|---|
| `CardEffects.DealDamageToEnemy` (3 overloads, 42 callers), `LandDamageOnEnemy` (5245) | one hit on an enemy: skip dead, fold via `IncomingDamage`, Slow after the floor (5113), Thorns, block, Intangible cap, HP, inline after-damage bodies (Curl Up, Slippery, Shriek ...), `AttackOutcome.Finish` | CreatureCmd.Damage per target |
| `CardEffects.DealDamage`, `FirstEnemy` (9344), `DealDamageMultiHit` (4186) | target choice, multi-hit loop | AttackCommand (targets, hit loop) |
| `CardEffects.DealDamageToPlayer` (3968), `LosePlayerHp` (3996), `AfterPlayerDamageReceived` (4054), `LoseHp` (5832, 33 callers) | damage and HP loss on the player with Buffer, Tungsten Rod, after-damage bodies | CreatureCmd.Damage on the player; LoseHp is Damage with unblockable props |
| `EnemyAI.DealAttackDamage` (4416, 19 callers) | an enemy's attack on the player | AttackCommand from the monster's move |
| `BuffSystem.IncomingDamage` (327), `CapIncomingDamage` (460), `CapHpLoss` (474) | int fold of Strength, Vigor, Weak, Vulnerable, Tank, Shrink ...; Intangible caps | Hook.ModifyDamage / ModifyDamageInternal; ModifyHpLost after Osty |
| `CardEffects.DmgFrom` (9240), `ShivDamage` (2803), `OstyAttackDamage` (9781); `RelicEffects.CardDamageBonus` (1622), `CardDamageMultiplier` (1661), `EnchantedCardDamageBonus` (2922), `IncomingDamageMultiplierFromDoom` (1695), `Vulnerable/WeakMultiplierDelta*` (2187, 2175) | pieces of the fold by hand | listeners of the three ModifyDamage passes, EnchantAdd/EnchantMul |
| `RelicEffects.ModifyHpLost` (2544), `BootDamageFloor` (1932), `CardEffects.DamageOsty` (9722), `DealOstyDamage` (9791) | HP-loss folds, Osty | ModifyHpLost before/after Osty, ModifyUnblockedDamageTarget |
| `RelicEffects.ApplyAfterBlockBroken` (1913), `ApplyAfterUnblockedDamageReceived` (1313), `CombatEngine.RefillHardenedShells` (4653), `EnemyAI.WakeMatriarchIfStruck` (4836) | after-damage bodies | AfterBlockBroken, AfterDamageReceived listeners |
| `CombatEngine.FirstHittableEnemy` (6495), `IsPrimaryEnemy` (4856), `TurnPlayerTowardTarget` (5187), `EnemyAI.IsTargetable` | hittable filter | Hook.ShouldAllowHitting / CombatState.HittableEnemies |
| `CombatEngine.HandleEnemyDeaths` (4358, from `ResolvePlayRepetition`), `AnswerDeathsSince` (4821, 7 callers: play, potion, selection answer, turn end/start), `AnswerDeaths` (4402, with the corpse sweep at 4590), `AnswerDeathsHoldingTheFightOpen` (4686) | deaths answered after the card or stretch, then the sweep | CreatureCmd.Kill inside Damage |
| `CombatEngine.EnrageQueenIfWaitingToBurnBright`, `TryEnrageCrabPartner`, `TryReattachSegment`, `TryRespawnAxebot`, `TryRespawnTestSubject`, `TryReviveIllusion`, `TurnPlayerToFaceSurvivor`, `SpawnGremlinMercReinforcements`, `SpawnPhrogParasiteWrigglers` (5014-5409) | per-monster death bodies | those monsters' BeforeDeath/ShouldDie/AfterDeath (handlers calling these bodies) |
| `EnemyAI.ShouldRemoveOnDeath` (5344), `CardEffects.RemoveDeathBoundPowers` (5483) | removal veto, powers removed on death | ShouldCreatureBeRemovedFromCombatAfterDeath, ShouldPowerBeRemovedOnDeath |
| `RelicEffects.ApplyAfterPlayerHpChanged` (1514), `LizardTailUnspent` (1508), `CombatEngine.PlayerIsDead` (3212), `IsAboutToLose` (4997), `CardEffects.KillDoomedEnemies` (10137), `RelicEffects.ApplyAfterDiedToDoom` (2332) | player death and its prevention, Doom kills | Kill/ShouldDie(Late)/AfterPreventingDeath, AfterDiedToDoom |
| `CombatEngine.CheckWinCondition` (4891), `NoPrimaryEnemyLeft` (4877), `HoldsTheFightOpen` (4623), `DiedHoldingTheFightOpen` (4638), `IsOverOrEnding` (4983) | win/loss at the old safe points | CombatManager.CheckWinCondition, IsEnding, ShouldStopCombatFromEnding |

Under option B the victory sites (`RunEngine.FinishCombat`, `RunNonCombatEffects.ApplyWarHammerOnVictory`,
`CountEliteVictoryForSwordOfStone`, `TriggerFishingRod`, `RelicEffects.CountCombatFor*`, `CarryRunCountersBack`,
`GenerateCombatRewards`) leave this switch.

## 5. Handlers

`python -m tools.kernel.port_list --families F08 F10 [--size 15] [--md ...] [--json ...]` (this branch, test
`tests/python/test_kernel_port_list.py`) lists every model reached by records that overrides an F08 or F10 hook, with
each hook's old status and location (from `hook_table.classify`: the dispatch site and line where the old engine
implements that override, else the methods naming it, or the not-modelled reason) and its batch. Full output for this
plan: `D:/spireline/referee/rebuild/port-f08-f10.md`.

- Option A: 135 models (power 60, relic 58, monster 14, card 2, potion 1); 174 (model, hook) pairs.
- **Option B: 94 models (power 60, relic 18, monster 14, card 1, potion 1); 128 pairs: 101 registered (an old
  location to call), 22 open (the old engine names the model elsewhere or not at all: each is read in its batch), 5
  not modelled (no state change or multiplayer only: no handler).**

Batches of 15 by reach (option B):
- power-1: Strength, Vulnerable, Weak, Thorns, Ravenous, Shrink, Inferno, Vigor, Rupture, Tank, Knockdown, Colossus,
  Shriek, Flame Barrier, Curl Up
- power-2: Slippery, Buffer, The Gambit, Imbalanced, Intangible, Surprise, Possess Speed, Possess Strength, Paper Cuts,
  Tainted, Illusion, Swipe, Gigantification, Slow, Flutter
- power-3: Reattach, Slumber, Hard to Kill, Hex, Personal Hive, Dampen, Stock, Asleep, Plow, Crab Rage, Surrounded,
  Sandpit, Soar, Adaptable, Steam Eruption
- power-4: Infested, Hardened Shell, Constrict, Burrowed, Heist, Painful Stabs, Calcify, Hang, Accuracy, Sic 'Em,
  Conqueror, Phantom Blades, Lethality, Tracking, Double Damage
- relic-1: Self-Forming Clay, Tungsten Rod, Strike Dummy, Red Skull, Centennial Puzzle, Pen Nib, Lizard Tail, Gremlin
  Horn, Miniature Cannon, Beating Remnant, Mystic Lighter, Meat on the Bone, Lava Lamp, Demon Tongue, Fake Strike Dummy
- relic-2 + card + potion: Hand Drill, The Boot, Vitruvian Minion, Melancholy, Fairy in a Bottle
- monster-1: Decimillipede segments (back, front, middle), Queen, Vantom, Aeonglass, Lagavulin Matriarch, Crusher,
  Rocket, The Insatiable, Soul Fysh, Test Subject, Waterfall Giant, Kin Priest

Porting pattern: the handler calls the model's body where the old engine has it; a body that sits inline in an old
site (Ravenous inside `AnswerDeaths`, Curl Up inside `LandDamageOnEnemy`) is first extracted into a named function that
the old site calls too, so the switch-off path stays byte-identical.

## 6. Acceptance

Per round:
- rule tests from the decompile for what the round adds (commands: the Headbutt and Twin Strike sequences of section 1
  as tests; the decimal fold, e.g. Vulnerable on 9 with Slow 2: 13.5 x 1.2 = 16.2 -> 16; ModifyHpLost's truncation
  criterion; removal deferred while the monster performs its move; a Whirlwind hit's Kill after all its targets);
  handlers: one test per override from its game body;
- the fuzzer (K6) over the ported set, core2 against old: every difference is a listed fix (A, B, or a `hook_open.txt`
  row the switch closes) or goes to a reading the same day; with the switch off, none;
- `struct_compare` on the corpus fights with `--engine core2` (test switch) against old and against the game: no fight
  worse against the game; improvements listed;
- the gates: C# and EmuDump tests, corpus 0 regressed on old, core2 with no switch byte-identical, `sites check`,
  `construct shape`, and (from P0) `construct commands`: every F08/F10 site gated.

Final (switch on in `migrated_families.json`): the by-family table re-measured on a lockstep block with core2;
target G2891c25e0d's kill part and mechanism B at 0, F08 proper down by at least its A+B share; any group that grows is
a stop.

## 7. Rounds and estimate

| round | model | content |
|---|---|---|
| M1 | Opus | F10 re-cut (option B), the two SPLITS records, gates scaffolding with the switch for tests only; `CreatureCmd.Damage` and `AttackCmd` with rule tests |
| M2 | Opus | `Kill` / `KillWithoutCheckingWinCondition`, the removal, the marks, the win check moved to the host; the old answers gated off |
| M3 | Opus | the old entry points routed (the 42 + 33 + 19 + 7 callers through adapters returning the old ints), hybrid audit, fuzzer and struct_compare harness on the empty handler set |
| M4-M10 | Sonnet | seven handler batches (section 5), 13-15 models each |
| M11 | Opus | switch on, corpus and lockstep re-measure, by-family table |

About 11 rounds: 5 Opus, 7 Sonnet, ~15.5 reviewed-round units at Opus 1.7x (range 13-22 with amends). Depends on K5
(writer) and K6 (fuzzer) landing first for M3's harness.

## 8. Risks

- **Hybrid re-entry.** A handler that calls an old command which hand-dispatches (Thorns' retaliation through
  `DealDamageToPlayer`, Inferno through `LoseHp`, a death body that deals damage) would run F08/F10 hooks by hand
  inside a kernel dispatch. Rule: under the switch every old damage/HP/kill entry is gated into Core2, so a handler's
  call re-enters the kernel; M3's audit lists every call from a handler into old code and the `commands` check fails on
  an ungated F08/F10 site.
- **Inline bodies.** Old sites hold several models' bodies inline; extraction must keep the old path identical (the
  byte-identity gate catches a slip).
- **Other families' sites calling damage.** Turn-end poison, Bomb, Combust and similar F05/F02 bodies call the old
  damage functions; under the switch these reach Core2's Damage through the gated entry, with their own family still
  old: the order around them stays the old one until that family moves.
- **Roster indices.** Removal inside the hit changes enemy indices mid-card; old code that holds an index across a
  card (multi-target loops, `ResolvePlayRepetition`'s position pairing) must read creatures, not positions (diagnosis
  F: part of the multi-enemy weight may move here).
- **IsPerformingMove.** An enemy killed during its own move (Thorns) stays in the combat state until its move ends; the
  old engine has no such flag: the enemy turn site sets it.
- **Option B changes family numbers**; the by-family tables must be rebuilt before the comparison of section 6.
