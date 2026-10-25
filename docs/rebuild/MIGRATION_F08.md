# F08 migration (damage / HP with the kill sequence): plan

2026-10-24, rebuild lane round M0, branch `rebuild/m0-f08` (from fork main a1e4d7ea, the Tier 1 kernel K1-K4), game
v0.107.1. Input: `TIER1_KERNEL.md`, the F08 diagnosis (`D:/spireline/referee/rebuild/f08-diagnosis.md`: mechanism A,
kill timing, 52% of F08's groups and 93.9% of G2891c25e0d; mechanism B, Slow outside the decimal fold), the site
inventory (`python -m tools.kernel.sites groups`), the port manifest (`D:/spireline/referee/rebuild/manifest.md`) and
the decompile (read only). Game code is named by class and method only. This round changes no engine behaviour.
Amended after review (same day): the F06 bridge, the kill's step sequence, the pseudo-hooks, the unlisted HP writes and
Escape, the open-pair gate, the F17 hand-off and the round split.

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

Anchors checked unique and in order in the current files. R1 wrote both records in `tools/kernel/sites.py` (`SPLITS`, kind
`split`) with the hash of each file; any later edit of `CombatEngine.cs` or `RelicEffects.cs` (R4 reroutes sites in both) makes
`sites check` fail until the two parts are read again and the new hash recorded.

**Option B (taken, landed in R1): re-cut F10.** Moved AfterCombatEnd, AfterCombatVictory and AfterCombatVictoryEarly from F10
to a new family F17 "combat end" in `tools/fidelity/hook_families.json`. They run after the combat's result
(RunEngine.FinishCombat, the rewards, run counters), not inside the kill. With
the two splits above `python -m tools.kernel.sites groups` shows **F08+F10** as one group; every other family (F01-F07, F09,
F11-F17) stays together in the second group only through bridges that belong to later migrations (the turn-start flow
F02+F03+F13 and the F01/F02/F03/F05/F06/F07/F09/F11/F12/F14/F15/F16 sites; with every one of them split the groups would be
F01, F02+F03+F13, F04-F07, F09, F11, F12, F14-F17). The
switch then needs 94 hook-overriding models instead of 135, 106 with the pseudo-hooks of section 5 (41 victory-only
models left: 40 relics and one card; 26 of their 40
AfterCombatEnd overrides are not modelled). Cost: the family table, `Switches.g.cs` and the manifest are rebuilt;
by-family tables carry the new row.

Option A (F10 whole): also split `GenerateCombatRewards` (three parts, anchors to be read) and port the 41 victory
handlers in this migration.

**The F17 hand-off under option B (amend, 2026-10-24).** Once the old in-step win checks are gated off (section 3
item 7), nothing in the old step calls `RunEngine.FinishCombat` any more. Under the switch the call moves to the host,
at the game's point: Core2's win check (CombatManager.CheckWinCondition) finds the combat ending, sets the old state's
`IsInProgress` false (the game's EndCombatInternal start: no action runs after it, the executor drains nothing more),
and then calls the old `RunEngine.FinishCombat` once, from the same place the game reaches EndCombatInternal, with the
same arguments the old in-step caller passed. The loss path is the same with the pending loss. F17's hooks
(AfterCombatEnd, AfterCombatVictoryEarly, AfterCombatVictory) keep their old dispatch inside `FinishCombat`,
`GenerateCombatRewards` and the run counters until F17 migrates; F17's switch then replaces those inside the same call.
Test (round 3): a fight won by an AfterDeath kill (Ravenous' survivor killed by Thorns) calls `FinishCombat` exactly
once, after the last action, with `IsInProgress` false and the same rewards as the switch-off path.

## 3. Commands to write in Core2

Folder `src/Sts2Emulator/Core2/Commands/`. Each is the game's command in its shape, cites `game: <Class>.<Method>`,
reads and writes HP, block and buffs through the old state (`CombatState`, `EnemyState`, `BuffSystem`), and never
re-implements a model's effect (that is a handler calling the old body). Dispatch is only through `Hooks.*`.

1. **`AttackCmd.Execute`** (game: AttackCommand's execute): the combat over or ending (`IsOverOrEnding`): return at
   once; attacker dead: nothing. BeforeAttack; hit count through ModifyAttackHitCount; per hit: stop if the attacker
   is dead; the living possible targets; none left in a live combat: stop; one `CreatureCmd.Damage` per hit, single
   target or all valid targets in one call; then AfterAttack. Random targeting with or without duplicates as the
   game's flag.
   **The F06 bridge.** BeforeAttack, ModifyAttackHitCount and AfterAttack are F06 in `hook_families.json`, and their
   old sites are F06-only (`CombatEngine.ResolvePlayRepetition[effect]`/`[after]`, `QueueAttackPlayLifecycleEffects`,
   `EnemyAI.DealAttack`, `RelicEffects.ApplyAfterOstyAttack`; ModifyAttackHitCount has none: the old card passes its
   count). Until F06 migrates, AttackCmd dispatches none of the three: with `Switch.F06` off it takes the count it is
   given and the old F06 points keep firing BeforeAttack/AfterAttack where they do today; with F06 on, AttackCmd
   dispatches them through `Hooks.*` and the F06 switch gates the old points off. Either way AfterAttack fires exactly
   once per attack. Test (round 2): F08 on and F06 off, then both on: a Twin Strike and a Whirlwind with a counting
   AfterAttack listener see one BeforeAttack and one AfterAttack per card play, and an enemy's multi-hit move one per
   move. The game's AfterAttack-before-the-card's-next-effect order (section 1: before Headbutt's screen) comes with
   F06, not here; the fuzzer lists it as an F06 difference.
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
5. **`KillWithoutCheckingWinCondition(creature, force, recursion)`**, in the game's sequence:
   1. not in a live combat: nothing.
   2. HP above 0: lose it (unblockable, unpowered) and AfterCurrentHpChanged.
   3. `MarkDying(creature)` from here to the removal (step 9).
   4. BeforeDeath.
   5. `ShouldDie` (both passes; preventer) unless forced or max HP <= 0. Prevented: go to step 12.
   6. The creature's `Died` event (pseudo-hook, section 5: SlumberingBeetle and SoulNexus subscribe), raised
      **before** the removal query.
   7. `ShouldCreatureBeRemovedFromCombatAfterDeath` (removal due unless a listener says no).
   8. `AfterDeath(wasRemovalPrevented: false)` (per-listener: a listener's screen is a hook action that runs after the
      current action, K4).
   9. Removal due and the creature still in the combat: **`CombatManager.RemoveCreature` always** (item 6: the
      monster's BeforeRemovedFromRoom pseudo-hook, its state machine reset, its hooks unsubscribed), **even while its
      monster `IsPerformingMove`**; only `CombatState.RemoveCreature` (the roster) is skipped during the move.
   10. Whether the creature is a primary enemy is read here, before its powers go (Minion decides it). Then **powers on
       every death**, removal or not (game: Creature.RemoveAllPowersAfterDeath): each power whose
       `ShouldPowerBeRemovedOnDeath` (OwnerInCombat step) and its own `ShouldPowerBeRemovedAfterOwnerDeath`
       (pseudo-hook, overridden by Adaptable, Die For You, Minion, Painful Stabs, Reattach, Steam Eruption) allow it is
       removed; then `AfterRemoved(oldOwner)` (pseudo-hook) per removed power, in removal order.
   11. A primary enemy (step 10's reading) whose living teammates are all secondary: `Kill(teammates)`; a player: orb
       queue cleared, Osty killed, `DeactivatePlayerHooks` from here, player death handling.
   12. Prevented: `AfterDeath(true)`, `AfterPreventingDeath(preventer)`, and again while still dead (at most 10, else
       the game's error).
   `KeepPlayerHooks` covers the player's own BeforeDeath/ShouldDie/AfterDeath.
   **The move's end** (game: the monster's PerformMove after its move's last action): if the monster died during its
   move, ask `ShouldCreatureBeRemovedFromCombatAfterDeath` again and, removal due, `CombatState.RemoveCreature` now.
   The old engine has no `IsPerformingMove`: round 3 adds it to the enemy-turn site that runs a move.
6. **Removal in the old state**, two functions as the game's two: `RemoveCreature` (CombatManager's: BeforeRemovedFromRoom,
   the monster's move state reset in the old `EnemyState` (intent, move index), its listener entries dropped from the
   kernel's tables) and `RemoveFromRoster` (CombatState's: what `CombatState.RemoveDepartedEnemies` does for one enemy
   today: roster, departed ids, targeting indices). The kill calls both (the second skipped during the move), Escape
   (item 8) calls both, the move's end calls the second. The old per-card sweep (`AnswerDeathsSince`,
   `HandleEnemyDeaths`, `AnswerDeaths`) does nothing under the switch for a death the kill answered.
7. **Win check at the game's points** (game: CombatManager.CheckWinCondition: pending loss first, then IsEnding with
   `ShouldStopCombatFromEnding`): K4's executor boundary (after every action but end turn and ready-to-begin-enemy-turn)
   plus StartTurn, EndEnemyTurn, ExecuteEnemyTurn's per-enemy check and EndPlayerTurnPhaseOneInternal; `CombatEnded`
   at the end. The old in-step checks (`CombatEngine.CheckWinCondition` callers) are gated off under the switch; the
   old `RunEngine.FinishCombat` is then called from here (section 2, the F17 hand-off).
8. **`CreatureCmd.Escape(creature)`** (game: CreatureCmd.Escape): dead: nothing. The creature's powers removed but the
   one the game keeps (RemoveAllPowersInternalExcept), `RemoveCreature` and `RemoveFromRoster` (item 6), the escaped
   list (CombatState.CreatureEscaped). **No death hooks**: no BeforeDeath, ShouldDie, Died, AfterDeath, no power
   AfterRemoved; the win check follows at the next check point (item 7). Callers: ThievingHopper's and FatGremlin's
   flee moves, Battleworn Dummy's time limit (section 4).
9. **`CreatureCmd.Heal`, `SetCurrentHp`, `SetMaxHp`, `GainMaxHp`, `LoseMaxHp`** (game: the CreatureCmd methods of
   those names): thin commands over the old HP fields with the game's hook calls (AfterCurrentHpChanged where the game
   dispatches it; SetMaxHp does not). They exist so the HP writes of section 4's second table have a Commands home;
   each caller moves when its own family does, through these commands.

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

**Settled in R2 (branch `rebuild/f08-r2`, 2026-10-24).** Items 1-3 and the named steps are in code, gated by the
F08+F10 switch (a `KernelState` exists only with a switch on) and called from no old site yet (R4 reroutes callers).
- Files: `Core2/Commands/CreatureCmd.cs` (`Damage`, the `Kill` hook point), `Core2/Commands/AttackCmd.cs` (`AttackCmd`,
  `AttackBridge`), `Core2/Commands/DamageResult.cs` (`DamageResult`, `CommandLog`, `ValueProps` with the game's bits,
  `PreviewModes`), and the adapter at the end of `Core2/OldState.cs`: `OldCreature` (player, Osty or an enemy by object)
  and `OldCreatures` (HP, block, pet owner, opponents, hittable enemies, the source card's enchantment, and
  `Ref`/`At` between a creature and the `CreatureRef` a dispatcher carries: the player 0, Osty 1 when present, then the
  enemies in the combat, as `OldState.Combat` numbers listeners). The commands touch HP and block only there.
- `CreatureCmd.Damage(k, targets, amount, props, dealer, cardSource, ctx)` returns the results in the game's order;
  each result gets a `DamageResultRef` in `k.Commands` (`Result(ref)` for handlers; copied with the state, enemies
  re-pointed by roster index). A creature the call kills is marked dying (`OldWorld.MarkDying`) from its HP loss until
  `Kill` returns, since the game keeps it in the combat until the kill removes it. An event dispatch that pauses where
  the command cannot resume goes through `KernelState.Settle`. The old state's damage history is owned too (see "Fixes after R2's
  review" below). Not in the old state, so not done: the run statistics (the player's damage-dealt counter, the map
  point's damage taken) and visuals.
- The enchantment steps call the old bodies: `CardEffects.EnchantDamageAdditive` and `EnchantDamageMultiplicative`,
  extracted from `DmgFrom` (which calls them, byte-identical), applied only on a powered attack (every damage
  enchantment of the game checks `IsPoweredAttack`). `NetIdGuard` returns true (one local player). `OwnerInCombat` is
  true while the power's owner position names a creature in the combat. `PreviewHittable` as above, over
  `OldCreatures.HittableEnemies` (the old `EnemyAI.IsTargetable`). The ModifyBlock enchantment steps (F09) and
  `NegativeCostPassthrough` (F07) still throw.
- **The F06 bridge.** `AttackCmd.Execute(k, ctx)` dispatches BeforeAttack, ModifyAttackHitCount and AfterAttack only
  with F06 on. With F06 off it takes the count given and runs the `AttackBridge` it carries once at each of the two
  places: R4's rerouted caller passes its old point there instead of running it (a monster's attack: the post-loop
  bodies of `EnemyAI.DealAttack`, `TriggerSuck` with the hits that landed and Painful Stabs; Osty's attack:
  `RelicEffects.ApplyAfterOstyAttack`). A card's attack carries no bridge: its old points
  (`ResolvePlayRepetition` [effect]/[after], `QueueAttackPlayLifecycleEffects`) stay at the play site, once per play.
- **What R3a must provide (the Kill contract).** `CreatureCmd.Kill(KernelState k, IReadOnlyList<OldCreature> killed)`
  is called exactly once per `Damage` call that got past the dead-dealer and empty-target returns, after every result's
  hooks, with the receivers killed by this call in result order (empty when none died; game: `await
  Kill(killedCreatures)`), while each killed enemy is still marked dying. The stub counts calls
  (`CommandLog.KillCalls`), runs the test seam `CommandLog.KillStandIn` when set, and otherwise throws
  `NotImplementedException` for a non-empty list. R3a replaces the stub's body with item 4 (keeping the count, or
  updating `DamageCommandTests`, which use the seam), and must leave a removed enemy out of `OldCreatures.Enemies`, so
  AttackCmd's per-hit target check sees it gone.
- Card uids in tests: a listing pass renumbers live cards whose uid is above `LastCardUid` (`CardUidInvariant.Normalise`),
  so a test card needs `CombatState.Fresh`.
- Tests: `src/Sts2Emulator.Tests/Core2/DamageCommandTests.cs` (16).
- Acceptance (R2): `dotnet test src/Sts2Emulator.Tests` 6765 passed, 7 skipped, 0 failed (6772); `tools/emudump.Tests`
  246 passed; byte_proof in the 9 modes (fight, netstate, state-hash, rng-log, reanchor, -scramble, -stale,
  -perturb-moves, -trace) over 4304 corpus fights: old against `--engine core2` 38736 pairs identical, 0 differ;
  `rebuild/f08-r1` (5657cd54) against this tree, both `--engine old`, 38736 identical, 0 differ; `construct shape` 0
  differences; `sites check` 0 problems (sites.json rebuilt for the moved CardEffects lines).

**Settled in R3a (branch `rebuild/f08-r3a`, 2026-10-24).** Items 4-6 and 8 are in code, gated as R2's (a `KernelState`
exists only with a switch on) and called from no old site yet (R4 reroutes). Item 9 (the HP commands) and the win check
(item 7) are not here.
- Files: `Core2/Commands/CreatureCmd.Kill.cs` (`Kill`, `KillWithoutCheckingWinCondition`, `Escape`, `PerformMove`,
  `RemoveCreature`, the pseudo-hook argument records), `Core2/OldState.cs` (`OldRoster`, the `Moving` mark, creature
  helpers), the four pseudo-hook ids in `HookId` (generated: `gen_hooks.py` appends `port_list.PSEUDO_HOOKS` after the
  dispatchers, so no existing id moves; `sites.py` gives them F10 in `SwitchGroups.FamilyOf`).
- `CreatureCmd.Kill(k, killed, force = false)` (game: CreatureCmd.Kill): each creature through
  `KillWithoutCheckingWinCondition` in list order; a list that held the player, the player dead: `LoseCombat`, **R3b's one
  call point** (today it only counts `CommandLog.LossCalls`). `KillCalls` counts every call (chained kills too) and the
  `KillStandIn` seam stays for R2's damage tests.
- The sequence as the game's (the decompile's order, which differs from item 5's list in two places: the living
  teammates are read after AfterDeath, and the primary-enemy reading comes after the removal, before the powers): not
  attached to the combat (an enemy out of the roster): nothing; the creature marked dying from the start (its HP loss
  included) to the end, and the player's hooks kept (`KeepPlayerHooks`) over its own passes; HP above 0 lost with
  AfterCurrentHpChanged; BeforeDeath; ShouldDie unless forced or max HP <= 0. Death: `Died` (pseudo-hook, the monster's
  own model), ShouldCreatureBeRemovedFromCombatAfterDeath (asked for every creature, used for enemies), AfterDeath(false),
  teammates; removal due and in the roster: `RemoveCreature` (BeforeRemovedFromRoom to the monster's own model, then
  `OldRoster.ResetMoveState`: move index and intents) always, `OldRoster.RemoveFromRoster` unless `w.IsPerformingMove`;
  primary read (`OldCreatures.IsPrimaryEnemy`: no power whose game class is MinionPower, the game's only
  OwnerIsSecondaryEnemy override); the powers; a primary enemy whose living teammates are all secondary: `Kill(teammates)`;
  a player: orbs cleared, a living Osty `Kill([osty], force)`, then `w.DeactivatePlayerHooks()` left set for the rest of
  the combat (HandlePlayerDeath is multiplayer only). Prevented: AfterDeath(true), AfterPreventingDeath(preventer), again
  while dead, the 11th round throws as the game.
- **Powers** (game: Creature.RemoveAllPowersAfterDeath): every power the listeners list (one per game power instance,
  `OldCreatures.Powers`) is judged first: its own ShouldPowerBeRemovedAfterOwnerDeath (pseudo-hook, default true), then,
  the owner still in the combat, `Hooks.ShouldPowerBeRemovedOnDeath` (an owner removed from the roster: true with no
  listener asked, as the game's `Owner.CombatState == null`); then the rest removed in order (every buff entry of that game
  power) and AfterRemoved per removed power, in removal order. Engine-only, folded and unclassified buffs are not listed powers, so this pass does not remove them; a corpse
  that stays loses the ones its revive would strip ("Fixes after R3a's review").
- **Pseudo-hooks** dispatch to the one model through `HandlerTable.Find` (no listener pass): BeforeRemovedFromRoom and
  Died take `HookEvent<BeforeRemovedFromRoomArgs/DiedArgs>` on the monster, ShouldPowerBeRemovedAfterOwnerDeath
  `HookPredicate<PowerDeathArgs>` and AfterRemoved `HookEvent<AfterRemovedArgs>` (owner, first BuffId, the entries
  removed) on the power. They carry `OldCreature`, since the owner may have left the combat; a pseudo-hook handler that
  pauses throws (the game calls them directly).
- **The old roster's removal** (item 6): `OldRoster.RemoveFromRoster(w, enemy)` is `CombatState.RemoveDepartedEnemies`
  for that one enemy (roster out, kind noted in `DepartedEnemyDefIds`, the old sweep's own helper), so the enemy drops out
  of `OldCreatures.Enemies`, the listener lists and the `CreatureRef` numbering. `OldRoster.Escaped` sets the old
  `Escaped` flag and `EscapedEnemies` (the escaped list) and removes; `FatGremlinEscaped` stays the caller's (R4).
- **IsPerformingMove**: `CreatureCmd.PerformMove(k, monster, move)` (game: MonsterModel.PerformMove) sets the
  `OldWorld.Moving` mark (copied with the state by roster index) around the move; a monster dead in the roster under the
  mark is in the combat (`OldState.EnemyInCombat`); after the move, dead and still in the roster:
  ShouldCreatureBeRemovedFromCombatAfterDeath again and, due, `RemoveFromRoster`. R4 wraps `EnemyAI.ExecuteIntent` in it.
- `CreatureCmd.Escape(k, c)` (game: CreatureCmd.Escape): dead or not attached: nothing; every old buff cleared (no
  AfterRemoved), `RemoveCreature`, `OldRoster.Escaped`. No death hook.
- **Damage changes**: the dealer's ref is read at each use and is null once the dealer has left the combat
  (`OldCreatures.TryRef`; a dealer a nested Damage killed, R2's review); a player the call killed keeps its hooks from the
  results loop to the kill (`KeepPlayerHooks`), so those passes are not reported `PlayerHooksUndeclared`. A receiver a nested kill removed mid-loop is skipped ("Fixes after R3a's review").
- Not modelled: a creature's death with removal vetoed stays listed after the kill only while the old
  `EnemyAI.ShouldRemoveOnDeath` agrees (the old state keeps no veto mark); Osty's own powers (none in the old state).
- What R3b must provide: `LoseCombat`'s body (the pending loss and its processing at CheckWinCondition, game:
  CombatManager.LoseCombat / ProcessPendingLoss), the win check at the game's points (item 7) including the executor
  boundary after a kill, and the F17 hand-off (`RunEngine.FinishCombat` from the host, once). The dead-player EndTurn of
  Kill's other branch needs a second player and stays out.
- Tests: `src/Sts2Emulator.Tests/Core2/KillCommandTests.cs` (11): the full sequence order; a revive; the 10-round limit;
  powers kept and removed by both predicates with AfterRemoved; removal during the monster's move then at its end;
  Twin Strike and Whirlwind with the real kill; Headbutt's corpse gone before the next effect (G2891c25e0d); Escape with no
  death hook; the player's death (Osty, deactivation after AfterDeath, the loss point); secondary teammates; a removed
  dealer. `DamageCommandTests`' stub test now checks the kill runs.
- Acceptance (R3a): `dotnet test src/Sts2Emulator.Tests` 6776 passed, 7 skipped, 0 failed (6783); `tools/emudump.Tests`
  246 passed; the kernel's Python tests (sites, shape, port_list) 49 passed; byte_proof in all 11 modes (fight, netstate,
  game-view, game-view-netstate, state-hash, rng-log, reanchor, -scramble, -stale, -perturb-moves, -trace) over 4304
  corpus fights: old against `--engine core2` 47344 pairs identical, 0 differ; `rebuild/f08-r2` (b78ff33d) against this
  tree, both `--engine old`, 47344 identical, 0 differ; `construct shape` 0 differences; `sites check` 0 problems (279
  sites; sites.json unchanged); `gen_hooks --check` clean. No old-engine file changed.

**Fixes after R2's review (branch `rebuild/f08-r3a`, 2026-10-24).** R2 said the damage history is not in the old state.
It is: the old damage sites update counters that old code reads later. Core2 owns each, through `OldCreatures`
(`Core2/OldState.Damage.cs`), at the game's point; R4 deletes the old line when it reroutes the site, and a ported model's
handler takes its line over.
- **Counters owned.** At `History.DamageReceived` (per result, in the first loop, after both results' flags are set;
  unguarded, since the old engine has no IsEnding flag inside a Damage call): `PlayerHpLostThisTurn` (Spite: player
  receiver, unblocked > 0, on the player's side turn, the entry's round and side), `UnblockedDamageHitCount` (Tear Asunder:
  every entry with unblocked > 0), `PlayerHitTurn`/`PlayerHitTurnBefore` (Emotion Chip: every entry not fully blocked),
  `EnemyState.PoweredHitsThisTurn` (Beat Into Shape: enemy receiver, the player itself as dealer, powered attack, the
  player's turn). At AfterDamageReceived, for a receiver not killed, after the hook dispatch: `UnblockedDamageThisTurn`
  (Beating Remnant, held or not: the old field is global), `TookUnblockedDamage` (Lava Lamp: unblocked > 0 and not
  Unblockable), `HardenedShellDamageThisTurn` (the power's owner, not fully blocked), and the old player-side body
  `CardEffects.AfterPlayerDamageReceived` (now `internal`; Rupture, Inferno, Centennial Puzzle, Self-Forming Clay, Demon
  Tongue, Red Skull) with Rupture's `fromPlayedCard` derived as the game reads it: the damage's card source is the card in
  flight (`OldWorld.Flying`) on the player's turn. In `AttackCmd`, at `History.CreatureAttacked` (once per command, after
  its last hit, before AfterAttack): `OstyAttacksThisTurn` (Flatten, Rattle; Osty as attacker, the player's turn).
- **Where this reads the game, not the old engine** (R4's fuzzer will list these as differences to read, not regressions):
  `UnblockedDamageHitCount` now counts every route (old: monster attacks only); `PlayerHitTurn` stamps a result that is not
  fully blocked even with no HP lost (old: unblocked > 0); Osty attacks count per command (old: per hit, Rattle patched it
  back); `PlayerHpLostThisTurn` is not counted on the enemy's turn (unobservable: it resets at the player's turn start).
- **Not counters, left to the handler batches:** the other old writes at the damage sites are model bodies (Slippery,
  Curl Up's armed flag, Flutter, Slumber, Plow, Shriek, Envenom, Personal Hive, Reaper Form, Monarch's Gaze, Paper Cuts,
  Imbalanced, The Gambit, Thorns, Skittish through `AttackOutcome`). A killed player gets no old
  `AfterPlayerDamageReceived`: that body's HP-changed call (Red Skull, Lizard Tail) would revive it before `Kill`, which
  would kill it again; Lizard Tail's revival belongs to the F10 `ShouldDie` handler.
- **`AttackCmd.BeforeDamage`** (game: AttackCommand.BeforeDamage): an `Action` run per hit after the target pick and before
  Calculate and Damage. The game's seven users are Dagger Spray, Fiend Fire, Hyperbeam, Sovereign Blade (both builders),
  Sweeping Beam, Calcified Cultist and Damp Cultist; every body plays effects or sounds only, none changes simulation
  state, so **the plan's handler batches wire none of them**. The seam exists for a model that does.
- **Random targeting, pinned.** `AttackCmd.RandomlyTargeted` draws the attacker's player's CombatTargets stream (the pet
  owner's for Osty; the old `CombatState.TargetRng`) **once per hit, after the duplicates filter**, as
  `Next(valid.Count)` over the living opponents, drawn even over one candidate (the game's Rng.NextItem). An enemy
  attacker throws (it has no such stream). Duplicates disallowed: the filter drops creatures already in the attack's
  results, none left throws. `RandomPick` stays only as a test seam. R4's callers (Sword Boomerang, the random-target
  cards and relics) set `RandomlyTargeted` and pass no stream.
- Tests: `src/Sts2Emulator.Tests/Core2/DamageCounterTests.cs` (15): one per counter, Rupture banked against immediate,
  random targeting (stream and draw count, over one candidate), duplicates disallowed (filter then draw, throw), an enemy
  attacker, BeforeDamage's place, Calculate per hit (single target, null, a lone opponent), 0 damage into block fully
  blocked with AfterDamageReceived firing.
- Acceptance (R2 fix): `dotnet test src/Sts2Emulator.Tests` 6791 passed, 7 skipped, 0 failed (6798); Core2 tests 127 (15 new in `DamageCounterTests`); byte_proof old against `--b-args "--engine core2"`, modes fight, state-hash, rng-log, `--limit 500`: byte_proof: rng-log: 500 fights, 500 identical, 0 differ, 0 run problems (0 with an error text compared to its first line); byte_proof: ok: 1500 fight-mode pairs identical, 0 differ; `construct shape` 0 differences; `sites check` 0 problems. `tools/emudump.Tests` not rerun (no EmuDump file changed). `CardEffects.cs` changes by one keyword (`private` to `internal`), no line moves.

**Fixes after R3a's review (branch `rebuild/f08-r3a`, 2026-10-24).**
- **The player's hooks stay deactivated.** `OldWorld.DeactivatePlayerHooks()` sets `PlayerHooks.Deactivated` and nothing
  restores an earlier state: the game's `Player.DeactivateHooks` is never scoped, and only a heal of the dead player
  (`Creature.HealInternal`'s `ActivateHooks`, multiplayer revives) turns the hooks on again, which item 9's HP commands own.
  `KeepPlayerHooks()` does nothing once the hooks are deactivated, and a Keep scope that ends after a deactivation inside
  it leaves it standing (it restores its earlier value only while the state is not Deactivated). So a second kill of a dead
  player (BeforeDeath, ShouldDie), a prevented death of a dead player, and an outer `Damage`'s Keep around a nested player
  kill no longer re-enable the hooks. `DeactivatePlayerHooks()` still returns a disposable, a no-op. The old scoped-form
  tests in `KernelTests` and `ListenerOrderTests` now assert the absorbing form (the "undeclared" case uses a fresh world).
- **A receiver or target removed mid-loop is skipped.** In `CreatureCmd.Damage`: the first loop skips a target that is dead
  (the game's `IsDead` check over its copy of the target list) or no longer in the combat; the results loop re-checks
  `OldCreatures.InCombat` before each pass (AfterBlockBroken, AfterCurrentHpChanged, AfterDamageGiven, AfterDamageReceived)
  and skips the receiver's remaining passes once a nested kill or escape has removed it, instead of throwing in `Ref`. The
  basis is the game's per-visit membership test in `CombatState.IterateHookListeners` (a model is visited only while
  `Contains` holds); the game itself still dispatches the others' passes with the removed Creature object, which a
  `CreatureRef` cannot carry, so those passes are not modelled (a known difference: relics see no AfterDamageGiven for a
  receiver a hook already removed). The result stays in the returned list and in `k.Commands`. `AttackCmd`'s per-hit target
  filter adds the same `InCombat` test to its `IsAlive` filter, so a single target that left the combat ends the hits. The
  dealer was already read through `TryRef`.
- **A corpse that stays loses what its revive would strip.** After the powers, a dead enemy still in the roster has the
  buffs the game's power removal does not reach (folded TEMPORARY_*, engine-only such as Stunned, unclassified) stripped by
  the predicate of the old revive path that claims it (`OldRoster.StripCorpse`): `CombatEngine.IllusionDeathStrips` (debuffs;
  `TryReviveIllusion`), `ReattachDeathStrips` (everything but Reattach and Reviving; `TryReattachSegment`) and
  `AxebotDeathStrips` (everything but Stock and Reviving; `TryRespawnAxebot`, which clears and re-applies the Stock it read,
  so the kernel keeps Stock for it). The old paths call the same predicates (byte-identical), so the two cannot drift. A corpse
  none of the paths claims keeps its unlisted buffs. Listed game powers are still the hooks' business (a model's
  ShouldPowerBeRemovedAfterOwnerDeath handler), so only unlisted buffs are touched here.
- Tests: `KillCommandTests` 21 (10 new: second kill of a dead player, prevented death of a dead player, nested player kill
  inside an outer Damage's Keep, a hook killing a later target in the first loop and in the results loop, a primary enemy
  dying in a hook with its minion later in the loop, an attack's single target that escaped mid-attack, a revived segment, a
  respawned Axebot, an Illusion corpse and an unclaimed corpse); eight of the ten fail on the R3a code. `KernelTests` and
  `ListenerOrderTests` adjusted for the absorbing mark.
- Acceptance: `dotnet test src/Sts2Emulator.Tests` 6801 passed, 7 skipped, 0 failed (6808); Core2 tests 137; byte_proof old
  against `--b-args "--engine core2"`, modes fight, state-hash, reanchor, `--limit 500`: 1500 fight-mode pairs identical, 0
  differ, 0 run problems (no switch is on in those runs); `construct shape` 0 differences; `sites check` 0 problems (279 sites;
  `sites.json` rebuilt for the moved `CombatEngine.cs` lines and the `DoomThePlayerAtTurnEnd` split's file hash re-recorded: the
  two anchors it names are unchanged and the method untouched). `CombatEngine.cs` changed by three predicates and the two old
  paths that call them.

**Settled in R3b (branch `rebuild/f08-r3b`, 2026-10-24).** Item 7 (the win check), `LoseCombat`'s body and the F17
hand-off are in code, gated by F10 (a `KernelState` exists only with a switch on); the old in-step checks and the turn
end's points stay old until R4 reroutes them.
- Files: `Core2/Commands/CombatEnd.cs` (`CombatEnd`, `CombatProgress`, `ICombatEndHost`), `KernelState.Progress` (copied
  with the state), `CreatureCmd.Kill`'s loss branch, `RunEngine` (the hand-off), `tools/kernel/commands.py` and
  `construct commands`.
- **The check** (game: CombatManager.CheckWinCondition): a pending loss first (`ProcessPendingLoss`: the loss cleared,
  `IsInProgress` false), else `IsEnding` (in progress; a pending loss; no living primary enemy in the roster, primary as
  `OldCreatures.IsPrimaryEnemy`; `Hooks.ShouldStopCombatFromEnding` holds no listener) and `EndCombatInternal`
  (`IsInProgress` false first). Both then hand off once (`ICombatEndHost.EndCombat(k, playerWon)`; a second hand-off
  throws) and call `ActionQueueSet.CombatEnded` (the game's NotInCombat at EndCombatInternal's end; for a loss the run's
  reset), which cancels every combat action not executing. `IsInProgress`, `IsAboutToLose`, `IsEnding`, `IsOverOrEnding`
  read `CombatProgress`.
- **The game's points**: `CombatEnd.AfterAction` (K4's executor boundary: `IActionHost.CheckWin`, F10 on, after every
  action but EndTurn and ReadyToBeginEnemyTurn, only while `InProgress`), `StartPlayerTurn` (both StartTurn player
  points, for `ITurnStartFlow.CheckWin`), `StartEnemyTurn`, `AfterEnemyTakesTurn` (ExecuteEnemyTurn per enemy; go on while
  `IsInProgress`), `EndEnemyTurn` (switch sides on `!IsEnding`, as the game), `AfterBeforeTurnEnd` and
  `EndPlayerTurnPhaseOne` (EndPlayerTurnPhaseOneInternal's two). K4's `KernelState.WinCheck` still calls `CombatEnded` after a
  flow's true answer; the second call cancels nothing new.
- **LoseCombat** (game: CreatureCmd.Kill's loss branch, CombatManager.LoseCombat): every player dead and the combat in
  progress: the loss pending, once. Nothing else runs: no hook, `IsInProgress` stays true, `IsEnding`/`IsOverOrEnding`
  turn true; HandlePlayerDeath (multiplayer) and the dead player's EndTurn are not modelled. `LossCalls` counts every reach.
- **The F17 hand-off**: `RunEngine` implements `ICombatEndHost` (explicitly): `LastPlayerWon` set, then the old
  `FinishCombat(playerWon)` once (victory heal, rewards, run counters, `SyncAfterCombat`; a loss as the old engine finishes
  it, although the game runs no AfterCombatEnd on a loss: F17 decides); the step returns what FinishCombat returned
  (`TakeHandOff`). `QueueHost.InProgress` is Core2's with F10 on; `QueueHost.CheckWin` runs `AfterAction` with the run
  engine as the end's host; a combat stepped on its own (`StepCombat`) ends as the step's terminal result.
- **R4's bridge** (`CombatEnd.OldStepEnded`): with F10 on, `StepInner` no longer calls FinishCombat on an old terminal
  result; the old in-step check's point stands for the game's: a loss is the old death answered (`LoseCombat`), then
  `CheckWinCondition` there. R4 deletes it when it gates the old checks and routes the player's death through `Kill`.
- Known differences under F10 until the handler batches: the five old holders (`HoldsTheFightOpen`: Steam Eruption,
  Adaptable, Surprise, Infested, Stock) have no Core2 ShouldStopCombatFromEnding handler yet, so a boundary check can end
  a fight the old engine holds open; their handlers are in the batches (Surprise power-2, Stock power-3, Steam Eruption,
  Adaptable and Infested power-4) and the handler gate keeps the switch off until they land. A living Illusion enemy is
  primary in Core2 (the game's MinionPower-only rule) and not in the old `IsPrimaryEnemy`: no handler closes that, so it
  is its own switch-on check (section 6).
- **`construct commands`**: per command the game method's `Hook.X` calls and named callees in order with branch labels
  (enclosing blocks; `<kind>-cond` inside a header, `cond` after `&&`, `||`, `?:`; `IsEnding` inlined) against the
  `// construct: game X` block in the Core2 source, both ways, and the block's modelled names against the Core2 method's
  `Hooks.X` calls (not the Resume continuations) and mapped callees, both ways. Six commands: CreatureCmd.Damage (15
  calls), AttackCommand.Execute (5), Kill (3, the multiplayer EndTurn marked not modelled), KillWithoutCheckingWinCondition
  (18, HandlePlayerDeath not modelled), Escape (3), CombatManager.CheckWinCondition (3): 0 differences. The write scan:
  0 Core2 writes outside `Core2/Commands` and `OldState*.cs`; **86 old HP/block writes run with the switch on (R4's list,
  `construct commands --list-writes`): 50 F08 (47 HP and the damage's 3 block absorptions, which move with the HP
  write: `CardEffects.DealDamageToPlayer`, `LandDamageOnEnemy`, `EnemyAI.DealAttackDamage`; R3b's review) and 36 F09
  (block) in `EnemyAI.cs` 35, `CombatEngine.cs` 25, `CardEffects.cs` 21, `RelicEffects.cs` 5** (the setup allow-list: CombatFactory, StateCloning, FightAnchor*, Run*,
  CreateEnemy; the spawned hatchling's two lines in `ApplyBuffIntent` are setup R4 allow-lists, F09's stay until F09).
- Tests: `src/Sts2Emulator.Tests/Core2/CombatEndTests.cs` (8): victory at the boundary after the action's end, a minion
  left, ShouldStopCombatFromEnding holding then releasing, the pending loss beating a victory at the same point,
  CombatEnded cancelling an unstarted hook action (and a later enqueue), the turn-flow points' answers, the progress
  copied with the state, and a whole run (PRB00058 A8) with F08+F10 on beside the old engine: same masks, results, HP,
  gold and phase at every step, `FinishCombatCalls` equal at every step, each combat handed off exactly once
  (`HandOffs` 1, `IsInProgress` false, `PlayerWon` the old `LastPlayerWon`), at least one victory, a loss only for the
  death. `tests/python/test_kernel_commands.py` (13): labels, inlining, strings and comments; each broken case caught
  (a call missing from the declaration, a label, the order, a call missing from Core2, an undeclared Core2 call, a
  not-modelled call Core2 makes, no declaration, bad blocks); the write scan's gated else and else-if branches, the
  allow-list, Core2's stray write; the real decompile (0 differences, and a broken CombatEnd caught).
- Acceptance (R3b): `dotnet test src/Sts2Emulator.Tests` 6809 passed, 7 skipped, 0 failed (6816); Core2 tests 145 (8 new);
  `tools/emudump.Tests` 246 passed; the kernel's Python tests (sites, shape, port_list, commands) 62 passed (13 new);
  byte_proof in all 11 modes over 4304 corpus fights: old against `--b-args "--engine core2"` 47344 pairs identical, 0
  differ; `rebuild/f08-r3a` (76ed5a1a, built from a clean worktree) against this tree, both `--engine old`, 47344
  identical, 0 differ; `construct shape` 0 differences; `construct commands` 0 differences for the six commands, 0 Core2
  stray writes, 86 old write sites listed; `sites check` 0 problems (279 sites; `sites.json` rebuilt for the moved
  `RunEngine.cs` lines); `gen_hooks --check` clean.

**Settled in R4 (branch `rebuild/f08-r4`, 2026-10-24).** The old entry points are rerouted behind
`if (Core2.Kernel.If(Core2.Switch.F08, state) is { } core2)` (F08 and F10 are one group; with no switch a combat holds no
kernel state and every site runs its old code).
- Core2: `Core2/Commands/OldEntry.cs` (HitEnemy, HitPlayer, HitOsty, EnemyAttack, EnemyHit, Kill, Escape: the old call's
  arguments in, the old int out; powered is Move, else Unpowered; HP loss is Unblockable and Unpowered; the card in flight
  is the source), `CreatureCmd.Hp.cs` (item 9: Heal, SetCurrentHp, SetMaxHp, SetMaxAndCurrentHp, GainMaxHp, LoseMaxHp),
  `OldCreatures.SetMaxHp`, `KernelState.EndHost` (the end's host for a check inside the old step, set by RunEngine and the
  bare queue host, else `CombatEnd.NoHost`). R3b's `CombatEnd.OldStepEnded` is deleted.
- Sites rerouted: `CardEffects.cs` 13 (DealDamageToEnemy, LandDamageOnEnemy, DealDamageToPlayer, LosePlayerHp, LoseHp,
  HealPlayer, GainMaxHp, LoseMaxHp, Spur's Osty heal, SummonOsty, DamageOsty, KillOsty, KillDoomedEnemies); `EnemyAI.cs` 12
  (DealAttack as AttackCmd with Suck and Painful Stabs as the F06 bridge's AfterAttack, DealAttackDamage as a one-hit
  AttackCmd, the GasBomb and IsExploding kills, the Thieving Hopper and Fat Gremlin escapes, the Knowledge Demon and
  heal-intent heals, The Gambit's body as Kill, Paper Cuts' as LoseMaxHp, TriggerAboutToBlow as SetMaxAndCurrentHp; the
  hatchling's HP moved to `SetUpHatchling`, setup); `RelicEffects.cs` 2 (Fur Coat, Lizard Tail: SetCurrentHp);
  `CombatEngine.cs` 21 (Regen, Reattach, Sandpit as Kill force, Battleworn Dummy as Escape, Fairy in a Bottle, Doom as Kill,
  the Axebot and Test Subject respawns; the sweeps HandleEnemyDeaths, AnswerDeaths, AnswerDeathsSince,
  AnswerDeathsHoldingTheFightOpen and the GasBomb roster sweep do nothing; CheckWinCondition, PlayerIsDead, IsOverOrEnding
  and IsAboutToLose read Core2; ExecuteIntent inside PerformMove with ExecuteEnemyTurn's per-enemy check; EndEnemyTurn's
  check; the InFlight mark over the card's effects).
- The write scan: 0 F08 writes run with the switch on (36 F09 stay for F09). Allow-listed with reasons: `SetUpHatchling`,
  the run-level heals `ApplyAfterRoomEntered` and `ApplyBeforeBossCombat` (RunState, F01's), Baseline's copy of the HP. A
  gated `if` whose block ends in return gates the rest of its block.
- No handler added (H0 changes the handler delegates). Dropped under the switch until their handler batch lands: the
  damage fold (Strength, Vigor, Vulnerable, Weak, Intangible, Slow, Hard to Kill, the relic bonuses), Thorns, Buffer,
  Tungsten Rod, Osty's redirect, every after-damage and death body (Curl Up, Ravenous, revives, respawns, Fairy in a Bottle
  and Lizard Tail), Necro Mastery, the heal modifiers. `DealUnpoweredDamageToEnemy` gates at its top, so its old Intangible
  and Hard to Kill caps and Thorns never run under the switch (they are ModifyDamage's and BeforeDamageReceived's hooks).
  Skittish is not dropped: it is the card attack's AfterAttack (F06's), and with F06 off it runs once per card attack as
  the AttackCmd's F06 bridge (`Bridge.AfterAttack`: `AttackOutcome.Finish` over the command's results); with F06 on the
  hooks dispatch and the bridge does not run.
- Card attacks are one AttackCmd each (`CardEffects.CardAttack`, review fix): the one-hit attack, `DealDamageMultiHit`
  (Twin Strike), `DealDamageAsSeparateAttacks` (one command per attack), `DealDamageToAll` (Cleave),
  `DealDamageToAllMultiHit` (Whirlwind's X hits), `DealDamageToRandomEnemiesMultiHit` (Sword Boomerang) and Ricochet. The
  player is the attacker and the card in flight the source; the hit count; single target, all opponents or random; one
  Damage call per hit over every live target and one Kill after it, so a death resolves after the whole hit (no longer
  between Cleave's targets); Calculate per hit when the caller passes one; the history's CreatureAttacked entry; the F06
  points through the bridge once. Random targets draw the CombatTargets stream (`TargetRng`) once per hit over the live
  enemies, as the old card did; with no stream, the card's rng as the old fallback.
- Left old: Osty's swings are one Damage call per hit, each hit noted so their `Finish` runs Skittish once after the
  last (Omnislice, Echoing Slash and Seeking Edge's sweep take the game's shape since the second review, below);
  the intent preview keeps the int fold; Doom kills with one Kill call per creature, as the game's `DoomPower.DoomKill`
  does (`OldEntry`'s unused list overload is removed); the turn start's two points are reached only through K4's flow.
- R7 checks (recorded, not run now): the intent preview equals the AttackCmd hit it previews (same fold, same amount per
  hit). Settled in R4c (below): relic and power damage fired inside `CardEffects.Apply` carries the in-flight card as
  its source where the game's call passes one, and none where the game passes none.
- Second review (one Damage over a list where the game makes one). `OldEntry.AttackContext` is the game's AttackContext
  (BeforeAttack when opened, AddHit, AfterAttack when disposed; no CreatureAttacked entry and no hit-count hook, as the
  game's context; with F06 off the card attack's bridge, Skittish). `OldEntry.HitEnemies` is one CreatureCmd.Damage over
  CombatState.HittableEnemies. Under the switch: Omnislice is its OnPlay (the aimed hit with Move; the splash ONE Damage
  over the other hittable enemies for TotalDamage + OverkillDamage with Unpowered | Move and the card as source, so
  ThornsPower, which answers a powered attack or an Omnislice, sees the game's props and every splashed enemy is hit
  before any death resolves); Echoing Slash is one context whose volleys are one Damage each over the hittable enemies,
  adding a volley per result with WasTargetKilled; Sovereign Blade is one AttackCmd of the base amount, at all opponents
  under Seeking Edge, else the target (Conqueror's doubling is its ModifyDamageMultiplicative hook's, not the amount's).
  The cards that read their attack read the AttackCmd's results as the game: Feed, Hand of Greed, The Hunt, Sunder and
  Knockout Blow a result with WasTargetKilled (not the target's HP after), Fisticuffs the results' TotalDamage +
  OverkillDamage, Beat Into Shape the CalculatedForge after the attack (the history's powered hits on the target this
  turn) less the attack's Results.Count; Blight Strike already summed TotalDamage; Tear Asunder's count is 1 + the
  history's unblocked hits on the player, which Core2's Damage writes, and its hits are one AttackCmd (ModifyAttackHitCount
  inside it with F06 on; the game has no listener of it).
- The sweep (a script over `Core/`: every for/foreach/while loop that names the enemies and calls a per-target damage
  helper; 17 loops). Fixed: 9 loops at 8 sites, `CardEffects` Omnislice (`DealOmnislice`), Echoing Slash (its volley loop
  and `DealDamageToAllCountingKills`), Seeking Edge's sweep (`case "SovereignBlade"`), `DamageAllForSpeedster`, Black Hole
  (in `GainStars`), `TriggerInfernoAfterPlayerDamage`, `CountPoisonForOutbreak` and `DealUnpoweredDamageToAll`, the root
  helper of 21 callers (CombatEngine 4: Hailstorm, Rolling Boulder, Panache, The Bomb; CardEffects 6: Smokestack, the Glass orb
  twice and three card bodies; PotionEffects 2: Explosive Ampoule, Foul Potion's enemy part; RelicEffects 9: Metronome,
  Festive Popper, Mercury Hourglass, Mr. Struggles and five more after-card, exhaust and end-of-turn ones): each is now one
  Damage over the hittable enemies. Already one command under the switch (old branch only): 4 (`DealDamageToAll`,
  `DealDamageToAllMultiHit`'s two, Ricochet). Per target in the game too: 3 (`CombatEngine.EndTurn`'s enemy loop and
  Poison ticks, `ApplyAfterCardPlayedPowers`' Strangle, each power its own Damage). Listed: 1 (Necro Mastery, below).
- R7 checks from the sweep (listed, not fixed): Osty's swings wait for Necrobinder, each hit one Damage where the game's
  Osty attack is one AttackCommand whose CreatureAttacked entry Flatten and Rattle count: `CardEffects.cs:7319, 7328,
  7331, 7343` (Rattle's loop at 7340), `7356, 7381, 7388, 10268` (`DealOstyDamage`); Necro Mastery's per-enemy loop at
  `CardEffects.cs:10195` sits behind `DamageOsty`'s gate (under the switch it is NecroMasteryPower's handler: one Damage
  over the hittable enemies, Unblockable | Unpowered); Foul Potion (`PotionEffects.cs:287`) hits the player then the
  enemies, where the game's FoulPotion makes one Damage over every non-pet creature; the list sites' props are
  Unpowered where the game passes a DamageVar's props (Letter Opener, Festive Popper, Mercury Hourglass, Metronome,
  Screaming Flagon, Stone Calendar, The Bomb, Charon's Ashes, Lost Wisp, Explosive Ampoule): read each at its batch.
- Acceptance (R4, after the second review): `dotnet test src/Sts2Emulator.Tests` 6852 passed, 7 skipped, 0 failed (6859;
  four new RerouteTests: Omnislice's splash as one Damage call with both Thorns answers before the death, Seeking Edge's
  sweep as one attack at the base amount, Echoing Slash's kill-counted volleys, Tear Asunder's hits from the history);
  `tools/emudump.Tests` 281 passed; the Python suite ran 2894 tests, 14 failures and errors, all in the reviewed baseline
  (0 new); `sites check` 0 problems (280 sites); `construct commands` 0 differences; the build's corpus check 4298
  compared, 0 regressed. With no switch: byte_proof over 500 fights in the modes fight, state-hash, rng-log and reanchor,
  main's build against this tree, 2000 pairs identical, 0 differ, and this tree's old engine against `--engine core2`,
  2000 identical, 0 differ. The fuzzer with F08+F10 on (`--fights 300 --procs 3`): 298 of 300 fights differ, 0 errors
  (no handler is registered yet).
- For the batches: the extracted bodies that write HP are gated, so a handler calling them under the switch reaches the HP
  commands; TriggerAboutToBlow, TryRespawnAxebot and RespawnTestSubject take the combat as an optional last argument (pass
  it); DealAttackDamage's callers get one AttackCmd per hit, so when F06 migrates each multi-hit caller becomes one
  EnemyAttack; Rupture's fromPlayedCard comes from the InFlight mark.
- Tests: `RerouteTests` (13; the review added six: Cleave as one Damage call with its death after both hits, Twin Strike's
  dead target eating the second hit, Whirlwind's X hits, Sword Boomerang's draws equal to the old engine's over 20 seeds,
  Skittish once per command, the unpowered path's caps gone under the switch).
  `ActionQueueTests`' lockstep test now uses F05, and `CombatEndTests`' run is played with F08+F10
  alone (each combat handed off once): neither expects F08+F10 to play as the old engine any more.
- Acceptance (R4, after the review): `dotnet test src/Sts2Emulator.Tests` 6848 passed, 7 skipped, 0 failed (6855);
  `tools/emudump.Tests` 281 passed; the Python suite ran 2894 tests, 14 failures and errors, all in the reviewed
  baseline (0 new); `sites check` 0 problems (280 sites); `construct commands` 0 differences; the build's corpus check
  4298 compared, 0 regressed. With no switch: corpus identity, main's Release EmuDump against this tree, 4304 fights,
  4304 identical, 0 differ; byte_proof over 500 fights in the modes fight, state-hash, rng-log and reanchor, main's
  build against this tree, 2000 pairs identical, 0 differ, and this tree's old engine against `--engine core2`, 2000
  identical, 0 differ. The fuzzer with F08+F10 on (`--fights 300 --procs 3`): 298 of 300 fights differ, 0 errors (no
  handler is registered, so the fold and the bodies listed above are missing until their batches).
- R4c (Conqueror as a LOCAL model; the list and unpowered hits take their card source from the caller). The source
  sweep: every call that reaches an `OldEntry` hit while a card is in flight, read against the game's call, 60 calls
  checked: 42 fixed (a power, relic, orb or potion Damage passes no card source in the game, where the `InFlight` mark
  gave it the card), 6 card sources kept (the card's own hits: the game passes the card).
  `ConquerorPower` (ModifyDamageMultiplicative) gives Sovereign Blade's one AttackCmd its doubling.
- The dealer sweep beside it (R4c's review): `OldEntry.HitEnemy` and `HitEnemies` take the dealer from the caller, as
  the source. The game's PoisonPower, StranglePower, HauntPower and DemisePower Damage calls pass no dealer and no
  source, so under the switch their ticks fire no AfterDamageGiven with the player as dealer and take no dealer-dead
  skip (`CardEffects.DealUnblockableUnpoweredDamageToEnemy`'s `noDealer` at the four sites); every other caller keeps
  the player, and Flame Barrier, Reflect and Thorns keep their owner as dealer (the game passes the owner and no
  source). Test: `RerouteTests.PoisonTickHasNoDealerSoNoAfterDamageGivenForThePlayerAndNoDeadDealerSkip`.
- Acceptance (R4c): with no switch, byte_proof main's build against this tree 2000 pairs identical, 0 differ, and this
  tree's old engine against `--engine core2` 2000 identical, 0 differ; `dotnet test src/Sts2Emulator.Tests` 6894
  tests; the fuzzer with F08+F10 on (`--fights 300`): 298 of 300 fights differ.

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

**HP writes the site table does not list (amend).** Kills and escapes done by writing HP in place, all routed in round 3
(each becomes `if (Kernel.If(Switch.F08, ...)) <command> else <old>`):

| old write | what it does | remedy |
|---|---|---|
| `EnemyAI.ExecuteIntent` (357) GasBomb | its explode move zeroes its own HP | `CreatureCmd.Kill([self])` from the move (IsPerformingMove: roster removal at the move's end) |
| `EnemyAI.ExecuteIntent` (365) `IsExploding` | Waterfall Giant's explosion zeroes its own HP | `Kill([self])`, same |
| `EnemyAI.ExecuteIntent` (791, 803) ThievingHopper, FatGremlin | the flee move zeroes HP and sets `Escaped` | `CreatureCmd.Escape` (section 3 item 8) |
| `CombatEngine.EndTurn` (2064) Sandpit | force-kills the player (no prevention) | `Kill([player], force: true)` |
| `CombatEngine.EndTurn` (2227) Battleworn Dummy | the time limit zeroes HP with the escape flag | `Escape` (BattlewornDummyTimeLimitPower) |
| `EnemyAI.DealAttackDamage` (4491-4492) The Gambit | removes the power and zeroes the player's HP inside the hit | The Gambit's AfterDamageReceived handler (batch power-2) calling `Kill([player])` after the hit's results |
| `CardEffects.KillOsty` (9758) | zeroes Osty's HP | `Kill([osty])` (the player's death step 11; Osty's own deaths come through Damage) |

Further HP and block writes from the grep of the old engine (every write of `PlayerHp`, `PlayerMaxHp`, `PlayerBlock`,
`OstyHp`, an enemy's `Hp`, `MaxHp`, `Block` in `Core/`, outside setup: `CombatFactory`, `CreateEnemy`, `StateCloning`,
`FightAnchor`, `Run*`):

| old write | remedy |
|---|---|
| `CombatEngine.EndTurn` (1643) Regen heal | `Heal` (item 9), with F05's Regen body |
| `CombatEngine.EndTurn` (1995) Reattach's HP set at turn | `SetCurrentHp` inside Reattach's handler (power-3) |
| `EnemyAI.ExecuteIntent` (466) Knowledge Demon heal; `ApplyBuffIntent` (4061) heal intent | `Heal` |
| `EnemyAI.ApplyBuffIntent` (4095-4096) a spawned hatchling's HP | creature setup (the game's Add), not an HP change: allow-listed |
| `EnemyAI.DealAttackDamage` (4527-4528) Paper Cuts | `SetMaxHp`/`LoseMaxHp` in Paper Cuts' handler (power-2; no AfterCurrentHpChanged, as the game) |
| `EnemyAI.TriggerAboutToBlow` (4924-4925) Waterfall Giant | `SetMaxAndCurrentHp` |
| `CardEffects.HealPlayer` (5778), `GainMaxHp` (5792), `LoseMaxHp` (5821-5822) | `Heal`, `GainMaxHp`, `LoseMaxHp` |
| `CardEffects.ApplyNecrobinderCard` (6920), `SummonOsty` (9704, 9709) | `Heal` on Osty; the summon stays with AfterSummon (F10 handlers) |
| `RelicEffects.BiteForFurCoat` (863) | `SetCurrentHp` (its F08 split, section 2) |
| `CombatEngine.RespawnTestSubject` (5364-5376), `TryRespawnAxebot` (5298-5300) | `SetMaxAndCurrentHp` inside the respawn bodies (Test Subject, Stock handlers) |
| `RelicEffects.ApplyAfterRoomEntered` (2850, 2855), `ApplyBeforeBossCombat` (3122) | `Heal`/`SetCurrentHp`, with F01 |
| `CardEffects.ApplyCore` (1562) a card strips the first enemy's block | `LoseBlock` (F09) |
| `CardEffects.Finish` (5216) Skittish block inside the hit's outcome; `CombatEngine.PayCurlUp` (5967) | `GainBlock` from Skittish's and Curl Up's AfterDamageReceived handlers |
| enemy block gains in `ExecuteIntent`/`ApplyBuffIntent`/`ApplyDebuffIntent`, `GrantRampartBlock`, the turn-start block clears in `EndTurn` (1939, 2328, 2332), `BreakBurrowIfBlockGone` (5323), the revival bodies' block resets | F09's (GainBlock, LoseBlock, the turn-start clear): stay old until F09 moves |

`construct commands` (round 3) checks writes, not only hook sites: every write of those fields in `Core/` outside
`Core2/Commands`, outside the else branch of a gated site and outside the setup allow-list (each entry with its reason)
fails, and an F09 write is reported under F09.

## 5. Handlers

`python -m tools.kernel.port_list --families F08 F10 --size 15 [--md ...] [--json ...] [--gate]` (this branch, test
`tests/python/test_kernel_port_list.py`, 14 tests) lists every model reached by records that overrides an F08 or F10
hook or takes part in an F10 pseudo-hook, with each hook's old status and location (from `hook_table.classify`: the
dispatch site and line where the old engine implements that override, else the methods naming it, or the not-modelled
reason) and its batch. R1 re-cut F17 in `hook_families.json`, so no exclusion is needed (`--exclude-hooks` stays as a general option). Output: `D:/spireline/referee/rebuild/port-f08-f10.md`.

**Pseudo-hooks** (`PSEUDO_HOOKS`, F10): the kill's model calls that are not AbstractModel hooks, found in the
decompile by override (inherited ones included) or event subscription: `BeforeRemovedFromRoom` (MonsterModel; 11
monsters: Cubex Construct, Knowledge Demon, Ovicopter, Owl Magistrate, Queen, Rocket, Snapping Jaxfruit, Soul Nexus,
Test Subject, Thieving Hopper, Waterfall Giant), `ShouldPowerBeRemovedAfterOwnerDeath` (PowerModel; 6 powers:
Adaptable, Die For You, Minion, Painful Stabs, Reattach, Steam Eruption), `AfterRemoved(oldOwner)` (PowerModel, not
the relic virtual of the same name; 10 powers: Burrowed, Dampen, Hex, Ringing, Sandpit, Shrink, Slumber, Sword Sage,
Tangled, Vital Spark), the `Died` event (SlumberingBeetle, SoulNexus). The old engine has no dispatch point for them, so
each pair is open with the methods that name the model. Core2 dispatches each to the one model it belongs to (the dying
monster, the removed power), not over the listener list; round 3 adds their ids to `HookId` with F10.

**Locations that apply the model.** A value hook (Modify*, Should*) only reads its model, so a location whose statement
applies or removes the model's buff is moved to elsewhere: `EnemyAI.TriggerSuck` (Suck applying Strength) left
Strength's ModifyDamageAdditive; the same check moved three more: Buffer's ModifyHpLostAfterOstyLate
(`CardEffects.LosePlayerHp`'s decrement, which is AfterModifyingHpLostAfterOsty's), Slippery's ModifyHpLostAfterOsty
(`LandDamageOnEnemy`'s decrement, likewise) and Tracking's ModifyDamageMultiplicative (`CardEffects.ApplyCore` applying
Tracking). Each keeps another location, so all four stay registered. An After/Before hook's body may write its own
buff (Slippery's decrement, Asleep's removal) and is kept.

- Option A: 135 models (power 60, relic 58, monster 14, card 2, potion 1); 174 (model, hook) pairs (hooks only).
- **Option B with the pseudo-hooks: 106 models (power 64, relic 18, monster 22, card 1, potion 1); 155 pairs, 27 of
  them pseudo-hook (BeforeRemovedFromRoom 11, AfterRemoved 9, ShouldPowerBeRemovedAfterOwnerDeath 5, Died 2; Die For
  You and Sword Sage are not reached): 101 registered (an old location to call), 49 open (22 hook pairs the old engine
  names elsewhere, 27 pseudo-hook pairs; each is read in its batch), 5 not modelled (no state change or multiplayer
  only: no handler).** The pseudo-hooks add 12 models (8 monsters, 4 powers).

**Handler gate (fixed in R1).** `--gate` exits 1 while any listed (model, hook) pair has no Core2 handler (one
`EmuDump kernel-handlers` lists: the HandlerTable with every switch on, built from `Core2/Models/Registry.cs`; a handler marked
`[StubHandler]` does not count, nor does a comment) and is not
`not_modelled`. It used to count only open, missing or unmatched pairs, so the 101 registered pairs (an old location to
call) passed with no handler at all. Today: 150 (101 registered, 49 open); the 5 not-modelled pairs need no handler. Each
handler batch closes its own pairs; switch-on (section 6) requires 0. The 22 open hook pairs: Slow and Hang
(ModifyDamageMultiplicative), Hard to Kill (ModifyDamageCap), Imbalanced (AfterDamageGiven), Shrink (AfterDeath), Fairy
in a Bottle (AfterPreventingDeath), Lava Lamp (AfterDamageReceived), Meat on the Bone (AfterCurrentHpChanged), Crusher
and Rocket (AfterCurrentHpChanged, BeforeDeath), and AfterDeath of the three Decimillipede segments, Aeonglass, Kin
Priest, Lagavulin Matriarch, Soul Fysh, The Insatiable, Vantom and Waterfall Giant.

Batches of 15 by reach (option B with the pseudo-hooks; the four small ones form one batch):
- power-1: Strength, Vulnerable, Weak, Thorns, Ravenous, Shrink, Inferno, Vigor, Rupture, Tank, Knockdown, Colossus,
  Shriek, Minion, Flame Barrier
- power-2: Curl Up, Slippery, Buffer, The Gambit, Imbalanced, Intangible, Surprise, Possess Speed, Possess Strength,
  Paper Cuts, Vital Spark, Tainted, Illusion, Swipe, Gigantification
- power-3: Slow, Flutter, Reattach, Slumber, Hard to Kill, Hex, Personal Hive, Dampen, Stock, Asleep, Plow, Crab Rage,
  Surrounded, Sandpit, Soar
- power-4: Adaptable, Steam Eruption, Infested, Hardened Shell, Constrict, Burrowed, Heist, Tangled, Ringing, Painful
  Stabs, Calcify, Hang, Accuracy, Sic 'Em, Conqueror
- relic-1: Self-Forming Clay, Tungsten Rod, Strike Dummy, Red Skull, Centennial Puzzle, Pen Nib, Lizard Tail, Gremlin
  Horn, Miniature Cannon, Beating Remnant, Mystic Lighter, Meat on the Bone, Lava Lamp, Demon Tongue, Fake Strike Dummy
- monster-1: Ovicopter, Soul Nexus, Thieving Hopper, Cubex Construct, Decimillipede segments (back, front, middle),
  Slumbering Beetle, Queen, Vantom, Aeonglass, Lagavulin Matriarch, Crusher, Rocket, Owl Magistrate
- monster-2: The Insatiable, Snapping Jaxfruit, Knowledge Demon, Soul Fysh, Test Subject, Waterfall Giant, Kin Priest
- mixed (power-5, relic-2, card-1, potion-1): Phantom Blades, Lethality, Tracking, Double Damage, Hand Drill, The Boot,
  Vitruvian Minion, Melancholy, Fairy in a Bottle

Porting pattern: the handler calls the model's body where the old engine has it; a body that sits inline in an old
site (Ravenous inside `AnswerDeaths`, Curl Up inside `LandDamageOnEnemy`) is first extracted into a named function that
the old site calls too, so the switch-off path stays byte-identical.

**R5 result (`rebuild/f08-r5`).** Every inline body of a located (model, hook) pair in a shared dispatch site is now an
`internal static <Model>_<Hook>(...)` function in the same class, with `game: <Class>.<Method>` in its doc comment, called
from the old place under the old condition (the call site keeps the `if`, the function holds what was inside it; a value
hook's function returns its number or factor and the fold calls it in the same order). The port list's location column
reads the new names from the source (`hook_table.classify` follows a dispatch method's bare-name callees), so no mapping
file is needed and the statuses did not move. Pairs whose old code was already its own function keep that function
(`TriggerShriekIfWounded`, `TryRespawnAxebot`, `BootDamageFloor`, ...). Left: the pairs with no old location (the 27
pseudo-hook pairs and the open hook pairs: their bodies sit in EndTurn, ExecuteIntent, the factories; each is read in its
handler batch), `CombatEngine.IsPrimaryEnemy` (Creature.IsPrimaryEnemy, not a hook body), Fairy in a Bottle's
`ShouldDie` (the slot lookup `PlayerIsDead` and `IsAboutToLose` share) and the Tungsten Rod line of
`RunNonCombatEffects.LoseHp` (run-level HP loss, outside the combat code).

**Integration (landing 6, `rebuild/integrate-f08a`).** R3b's tip `d9a0a6f0`, R1's review fixes (`622271a6`), R5 (`50af5b24`) and
`rebuild/tier1-kernel` (fork main `7727a1b7` plus the scenario format) merged in that order. Conflicts: the Doom split record
in `sites.json` (R1's corrected reason kept), `CardEffects.cs` (doc comments only: R2's `EnchantDamageAdditive` and
`EnchantDamageMultiplicative` kept, R5's `Gigantification_ModifyDamageMultiplicative` placed ahead of `DmgFrom`), `sites.py`
and `sites.json` (the Doom and Fur Coat parts re-read, anchors unchanged and unique; the `CombatEngine.cs` hash re-recorded,
`sites.json` rebuilt). Acceptance: `dotnet test src/Sts2Emulator.Tests` 6809 passed, 7 skipped, 0 failed (6816);
`tools/emudump.Tests` 281 passed; byte_proof against a Release EmuDump of fork main `7727a1b7`, all 11 modes over 4304
corpus fights, `--engine old` and `--b-args "--engine core2"`: 47344 pairs identical, 0 differ, each; `corpus.py identity`
4304 identical in every field at every step; `fuzz_core2.py --fights 300` 0 differences; `construct shape` 0, `construct
commands` 0 (6 commands, 0 stray writes), `sites check` 0 problems (279 sites); `EmuDump scenario tests/scenarios` 14 pass,
9 xfail on old and on core2. `fuzz_core2.py --switches F08,F10 --fights 300` finds 8 differing fights (all untriaged: the
combat-end fields, Phase, rewards and HP); the R3b tip `d9a0a6f0` alone gives the same 8 fights with byte-identical diff
files, so the merge adds none: they are the open F08/F10 handlers and the F17 hand-off, for R4 and the batches.

### How to write a handler

The template is `src/Sts2Emulator.Tests/Core2/HandlerContextTests.cs`: test-only handlers for Strength
(`ModifyDamageAdditive`, its own amount) and Vulnerable (`ModifyDamageMultiplicative`, the old engine's extracted body),
registered under the F08+F10 switch and run by `Hooks.ModifyDamage` over an old `CombatState`. A handler batch copies it:

1. Take the pair from the port list and read the game method (`game: <Class>.<Method>`). The handler is
   `internal static` in `Core2/Models/<Kind>/<GameClass>.cs`, with that citation in its doc comment.
2. The signature is the dispatcher's op (TIER1_KERNEL.md section 4): the context first, then `ListenerRef self`, then
   `in <Dispatcher>Args a`, then the op's extra: a fold-add or fold-mul `decimal Method(HookCtx c, ListenerRef self, in
   ModifyDamageArgs a, decimal current)` (the addend, or the factor), a cap `decimal Method(c, self, in a)`, a predicate
   `bool`, an event `HookStep Method(c, self, in a, ChoiceCtx? ctx)`.
3. Read the combat only through `c`. Engine-neutral (a Tier 2 context answers them too): `c.Power(self)` for the power's
   owner and amount (null when it is gone from its owner: answer the identity or name the gap); `c.Owner(self)` for any
   other model's creature; `c.Card(self)` for a card model's own copy (`CardView`: model id, pile, index; by the K2b uid)
   and `c.Card(handle)` for a card an argument names (the card source, a played card); `c.Relic(self)` for a relic's
   counter (`RelicView`; a relic's other saved properties are fields of the old combat state its old body reads).
   Tier 1 only (each use goes on `TIER2_REWRITE_LIST.md`): `c.OldCombat` (the `CombatState`: the first argument of most
   old bodies, `CardEffects.PhantomBlades_ModifyDamageAdditive(c.OldCombat)`, `RelicEffects.BootDamageFloor(c.OldCombat,
   hpLost)`, and the state a Lethality or Melancholy body reads), `c.OldBuffs(creature)` (a body the old engine already
   has as `BuffSystem.<Model>_<Hook>(...)`, R5), `c.Enemy(creature)` (the old `EnemyState` of a body that takes one, as
   `RelicEffects.ApplyAfterBlockBroken(state, enemy)` does for Hand Drill) and `c.World`. The handler keeps the game's
   condition (the dealer, the target, `props.IsPoweredAttack()`) and the old function holds the number; an input the old
   fold takes from the state is passed the way the old site passes it (the template passes
   `RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(c.OldCombat)`, not 0). State changes go through the commands,
   which take `c.K`, never by writing the old state (TIER1_KERNEL.md section 4: in Tier 1 the commands take the
   `KernelState`; Tier 2 maps each `c.K` use).
   **One hook call per copy.** The game runs a card's hook once per card model in the combat, so a handler for a card
   acts on its own copy (`c.Card(self)`: its pile and index), never on every copy. Melancholy is the case in the pilot: the
   old `CardEffects.CheapenMelancholyForDeath(state)` cheapens every copy in every pile in one call, so called from a
   per-copy handler it would cheapen N copies N times. The handler must act on its own card only, or the batch first
   splits the old function into a per-card part (`CheapenMelancholy(CardInstance)`, returned with its `CostBump` lowered) that
   both the old caller loop and the handler call, keeping the switch-off path byte-identical.
4. Where the power does not apply, return the identity: 0 for an addend, 1 for a factor, the cap's `decimal.MaxValue`, the
   current value for a set. The changed set is the dispatcher's (by the pass's `changed` criterion): a handler never
   tracks it.
5. Register in the model's `Register(HandlerTable t)`: `t.Model(ListenerKind.Power, "StrengthPower").On(HookId.
   ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)StrengthPower.ModifyDamageAdditive)`, and list the model in
   `Core2/Models/Registry.cs`. It registers only while its hook's family switch is on.
6. Test it the way the example does: build an old `CombatState`, bind a `Kernel` with the family's switches and the model
   registered, call the generated dispatcher (`Hooks.ModifyDamage(k, args, out var changed)`), and assert the value the game
   returns and the changed set; add the cases where the power does not apply.
7. `Power(self).Amount` is the old buff's `Magnitude`; where the engine keeps the game's amount somewhere else (a buff's
   `Data`, a counter on the state) the model's handler reads that and says so in its comment.

### Batch monster-1, first half (R6.m1, branch `rebuild/f08-m1`)

Nine game models, 13 (model, hook) pairs, the first eleven port-list monsters by corpus reach (the three Decimillipede
segments are one game base class): Ovicopter, Thieving Hopper, Cubex Construct (BeforeRemovedFromRoom), Soul Nexus (Died,
BeforeRemovedFromRoom), Slumbering Beetle (Died), Queen (AfterDeath, BeforeRemovedFromRoom), Vantom, Aeonglass and
DecimillipedeSegment for Back, Front and Middle (AfterDeath). Each is `Core2/Models/Monsters/<GameClass>.cs` with its test
in `src/Sts2Emulator.Tests/Core2/Models/Monsters/`, listed in `Core2/Models/Registry.cs`.
- Read from the game: every pair but Queen's AfterDeath has no combat-state effect (sound loops, animation tracks, music
  parameters, a sleeping effect, a phobia-mode texture, unsubscribing a visual handler), so its handler is the game's call at
  the game's point and changes nothing. Queen's AfterDeath is the one body with state: the old
  `EnrageQueenIfWaitingToBurnBright` looped over every Queen, so it is split into a per-Queen
  `CombatEngine.Queen_AfterDeath(EnemyState)` that the old loop and the handler both call (one hook call per copy); the
  old path is unchanged.
- Accessor gaps closed in `HookCtx` (engine-neutral): `IsAlive(CreatureRef)` (game `Creature.IsAlive`) and
  `IsMonster(CreatureRef, gameClass)` (game `creature.Monster is T`). Queen still reads her old `EnemyState` through
  `c.Enemy` for the old body (listed in `TIER2_REWRITE_LIST.md`).
- Registration gap closed in `HandlerTable`: the old state has one enemy id for the three Decimillipede segment models, and
  two game names may not share an old id, so `ModelRegistration.InheritedBy(...)` declares game models that inherit the
  registered model's overrides; `Registered()` (what `EmuDump kernel-handlers` and the port-list gate read) names each.
- The test harness (`MonsterHarness`) registers a model as it registers itself, wraps each handler to log its call, and
  kills through `CreatureCmd.Damage`: the log gives the pass and order (Died, then AfterDeath on every listener, then
  BeforeRemovedFromRoom to the leaving monster), and a deleted handler leaves its pair unregistered and unlogged.
- Acceptance (R6.m1): the monster tests 24 pass; `dotnet test src/Sts2Emulator.Tests` 6895 passed, 7 skipped, 0 failed
  (6902); `tools/emudump.Tests` 281 passed; `port_list --gate --families F08 F10`: 137 pairs without a handler (150 before,
  this batch's 13 closed); byte_proof of the Release EmuDump against itself, `--engine old` and `--b-args "--engine
  core2"`, no switch, `--modes fight,state-hash --limit 300`: 600 fight-mode pairs identical, 0 differ;
  `fuzz_core2.py --switches F08,F10 --fights 300 --procs 3`: 299 differing (R4: 299), all untriaged, top fields the
  generic ones (ActiveCombat.Enemies 186, PlayerHp 84, UnblockedDamageHitCount 59); 29 of them are encounters of this
  batch's monsters, with the same generic signatures (the damage fold and the death bodies the power batches carry).

### Batch monster-1, second half, and monster-2 (R6.m2, branch `rebuild/f08-m2`)

Eleven game models, 17 (model, hook) pairs: the rest of the port list's monster rows. Lagavulin Matriarch
(AfterDamageReceived, AfterDeath), Crusher (AfterCurrentHpChanged, BeforeDeath), Rocket (the same two and
BeforeRemovedFromRoom), Owl Magistrate, Snapping Jaxfruit and Knowledge Demon (BeforeRemovedFromRoom), The Insatiable, Soul
Fysh and Kin Priest (AfterDeath), Test Subject and Waterfall Giant (AfterDeath, BeforeRemovedFromRoom). Same layout and
harness as m1.
- Read from the game: all 17 bodies are presentation only (the Kaiser Crab background's arm and body animations, sound
  loops, music parameters, the Matriarch's sleeping effect and shell-eye animation with the `IsShellAwake` flag only that
  animation reads, the Kin Priest's all-followers-dead talk line, the Test Subject's colour and its unsubscribing of the
  PowerApplied and PowerRemoved handlers that set it). So each handler is the game's call at the game's point, changes
  nothing and reads nothing; no open pair needed a body.
- Two monster pairs are `registered` in the port list on an old location that is a power's body, and the power's row
  lists the same location: Lagavulin Matriarch's AfterDamageReceived on `EnemyAI.WakeMatriarchIfStruck`
  (AsleepPower.AfterDamageReceived, batch power-3) and Test Subject's AfterDeath on `CombatEngine.TryRespawnTestSubject`
  (AdaptablePower.AfterDeath, power-4). Waterfall Giant's stand-up, `EnemyAI.TriggerAboutToBlow`, is
  SteamEruptionPower.AfterDeath (power-4), and the Axebot's `TryRespawnAxebot` is StockPower's. The power handlers call
  them, passing the combat as R4's optional last argument (`RespawnTestSubject` too, from the respawn move); the monster
  handlers do not, so no body runs twice once both rows are ported. The tests pin that: with the monster registered alone
  the Matriarch stays asleep, the Test Subject gets no Reviving and the giant does not stand up.
- `HookCtx` gaps: none (no handler reads the combat). `MonsterHarness` now also wraps BeforeDeath, AfterCurrentHpChanged
  (logging the delta) and AfterDamageReceived. A `CreatureRef` is the roster slot, so a death in a slot an earlier death
  emptied logs the same index (the Kin Priest's test).
- Acceptance (R6.m2): the monster tests 47 pass (23 new); `dotnet test src/Sts2Emulator.Tests` 6918 passed, 7 skipped, 0
  failed (6925), ModelSourceScanTests included; `tools/emudump.Tests` 281 passed; `port_list --gate --families F08 F10`: 120
  pairs without a handler (137 before, this batch's 17 closed; every monster row now has its handlers); byte_proof of the
  Release EmuDump against itself, `--engine old` and `--b-args "--engine core2"`, no switch, `--modes fight,state-hash
  --limit 300`: 600 fight-mode pairs identical, 0 differ; `fuzz_core2.py --switches F08,F10 --fights 300 --procs 3`: 299
  differing (m1: 299), all untriaged, top fields the generic ones (ActiveCombat.Enemies 186, PlayerHp 84,
  UnblockedDamageHitCount 59). As expected for handlers that change nothing: the differences are the power and relic
  batches' (the damage fold and the death bodies).


### Batch L1: item 9's HP commands and the first LOCAL non-monster models (branch `rebuild/f08-l1`)

**Item 9, the spec per method (read from the decompile; `Core2/Commands/CreatureCmd.Hp.cs`, each with its `construct`
declaration).** `construct commands` checks all six against the game (12 commands with R3a's six).
- `Heal(creature, amount)`: a non-player heals nothing while the combat IsEnding; Creature.HealInternal (HP = min(HP +
  amount, max), truncated; a dead creature brought back has its player's hooks activated again: `OldWorld.ActivatePlayerHooks`,
  R3a's left item); then AfterCurrentHpChanged with the **asked-for amount** whenever it is positive, even at full HP (the old
  engine announced only a change, with the change).
- `SetCurrentHp(creature, amount)`: Creature.SetCurrentHpInternal (min(amount, max), no floor); AfterCurrentHpChanged with
  asked-for minus old HP when they differ; then Kill when the creature is dead. No hook activation (only Heal's
  HealInternal has it). An enemy set to 0 HP is marked dying over the hook, as the game keeps it in the combat until the kill.
- `SetMaxHp(creature, amount)`: Creature.SetMaxHpInternal with the amount floored at 0 (max = min(amount, 999999999), HP
  lowered to it) with **no hook**; Kill at a max of 0; returns the change of the max.
- `SetMaxAndCurrentHp`: SetMaxHp, then SetCurrentHp.
- `GainMaxHp(creature, amount)`: negative refused (the game throws); SetMaxHp(max + amount), then Heal by the change it
  returns (so the hook and the ending gate are the heal's).
- `LoseMaxHp(ctx, creature, amount, isFromCard)`: negative refused; when the lowered max is under the HP, the excess is real
  damage first (CreatureCmd.Damage, Unblockable and Unpowered, plus Move when from a card; no dealer, no card source: so
  Rupture applies Strength at once), then SetMaxHp(max(1, lowered max)). Callers: Brightest Flame (`CardEffects.LoseMaxHp`,
  from a card: the InFlight mark) and Paper Cuts (not from a card).
- Old sites: R4 had already routed every F08 heal and max-HP write (the write scan: 0 F08 HP writes with the switch on); L1
  passes `isFromCard` and moves the two death-prevention heals of the old bodies from SetCurrentHp to the game's Heal
  (`RelicEffects.LizardTail_AfterPreventingDeath`, `CombatEngine.FairyInABottle_AfterPreventingDeath`; switch off unchanged).
  Not modelled: the heal's sound, animations, numbers, the revive animation, the map point's history and the scaled wait.

**Models (11 game models, 13 pairs; `Core2/Models/Powers`, `Relics`, `Potions`; tests `Core2/Models/L1ModelTests.cs`,
`Core2/HpCommandTests.cs`).**
- Paper Cuts AfterDamageGiven (registered; the switch skips its inline old body): dealer the owner, target the player, a powered
  attack with unblocked damage: LoseMaxHp by the amount, not from a card.
- The Gambit AfterDamageReceived (registered): target the owner, powered, unblocked > 0: the old body (the buff removed,
  then Core2's Kill).
- Hard to Kill ModifyDamageCap (open): the amount when the target is the owner, else no cap; any props.
- Slow ModifyDamageMultiplicative (open, mechanism B): 1 + 0.1 x SlowAmount for a powered attack on the owner, a decimal factor
  inside the fold; SlowAmount is the old `BuffId.SlowCount` buff (its per-turn card counter, kept by the old engine).
- Minion, Reattach, Adaptable, Steam Eruption, Painful Stabs ShouldPowerBeRemovedAfterOwnerDeath (open pseudo): false (each
  game override); their other pairs stay with their batches and go in the same class later.
- Lizard Tail ShouldDieLate and AfterPreventingDeath (registered): the owner with the relic unused is not let die; then the
  relic is spent (its counter) and the owner heals max(1, max HP x 50%) through Heal.
- Fairy in a Bottle ShouldDie and AfterPreventingDeath (open; the potion's slot is the listener): the owner is not let die
  while it is held; then the slot is emptied and the owner heals max(max HP x 0.3, 1) through Heal. Not modelled: the
  potion-use hooks of OnUseWrapper (another family).
- `HookCtx` (engine-neutral): `IsPlayer(CreatureRef)` (game Creature.IsPlayer) and `Damage(DamageResultRef)` (the result a
  damage hook is handed). Tier 1 uses: Slow's `c.OldBuffs`, The Gambit's, Lizard Tail's and Fairy's old bodies through
  `c.OldCombat`.
- Left LOCAL: Inferno, Rupture, Self-Forming Clay, Centennial Puzzle, Demon Tongue, Red Skull (one change: Core2's call of
  the player-side `CardEffects.AfterPlayerDamageReceived` in `OldState.Damage.cs` goes, six handlers take it over), Beating
  Remnant and Hardened Shell (their Core2-owned tallies), Curl Up (needs F09's GainBlock), Shrink (AfterDeath reads the
  applier, which the old buff does not keep; AfterRemoved is a node scale only), Illusion, Imbalanced, Vital Spark, Hang,
  Meat on the Bone, Lava Lamp, the AfterRemoved pseudo pairs (Slumber, Hex, Dampen, Sandpit, Burrowed, Tangled, Ringing),
  and the registered pairs of Minion, Reattach, Adaptable, Steam Eruption and Painful Stabs.
- Acceptance (L1): the new tests 21 pass (HpCommandTests 10, L1ModelTests 11); `dotnet test src/Sts2Emulator.Tests` 6939 passed, 7 skipped, 0 failed (6946), ModelSourceScanTests included; `construct commands` ok: 12 commands, 0 differences, 0 Core2 writes outside Commands, 0 F08 HP writes with the switch on; `sites check` 0 problems (279 sites; the CombatEngine.cs and RelicEffects.cs hashes re-recorded, the Doom and Fur Coat parts unchanged); `port_list --gate --families F08 F10`: 107 pairs without a handler (120 before, this batch's 13 closed); byte_proof of the Release EmuDump against itself, `--engine old` and `--b-args "--engine core2"`, no switch, `--modes fight,state-hash --limit 300`: 600 fight-mode pairs identical, 0 differ; `fuzz_core2.py --switches F08,F10 --fights 300 --procs 3`: 299 differing (m2: 299), all untriaged. `tools/emudump.Tests` not rerun (no EmuDump file changed).

### Batch L2: the LOCAL models L1 left (branch `rebuild/f08-l2`)

47 (model, hook) pairs over 27 game models (`Core2/Models/Powers`, `Relics`, the classes in `KeptAfterOwnerDeath.cs`; tests
`Core2/Models/L2PlayerDamageTests.cs`, `L2TallyTests.cs`, `L2DeathTests.cs`, `L2DeathExtraTests.cs`).

**The player-side body goes (one change, item 1).** Core2's damage command called the old
`CardEffects.AfterPlayerDamageReceived` after the AfterDamageReceived pass for a player it did not kill: Rupture, Inferno,
Self-Forming Clay, Centennial Puzzle, Demon Tongue, then `ApplyAfterPlayerHpChanged` (Red Skull, and Lizard Tail, which is
L1's handler) and Emotion Chip's `RecordPlayerHit` (the same entry `NoteDamageReceived` writes in the first loop). The
call is gone and six handlers take it over at the game's hook and pass, in the game's listener order (the creature's
powers before the player's relics) rather than the old body's fixed order:
- Inferno (AfterDamageReceived): the game's one `CreatureCmd.Damage` over the hittable enemies (the old body hit each enemy
  with its own call, so each had its own AfterDamageReceived passes and Kill); the owner hit, unblocked > 0, its own turn.
- Rupture (AfterDamageReceived): the old body with the fromPlayedCard Core2 computed before (the card source is the
  in-flight card on the player's turn: the bank, `RuptureOwed`, paid by the old `PayRupture`).
- Self-Forming Clay, Centennial Puzzle, Demon Tongue (AfterDamageReceived): the old extracted bodies under the game's
  conditions (IsInProgress; Centennial Puzzle's used flag and Demon Tongue's turn flag are the old relic counters).
- Red Skull (AfterCurrentHpChanged): any creature's HP change while the combat is in progress re-reads the owner (the old
  body is idempotent). With the switch on the old heal returns into the Heal command and no longer re-read Red Skull; the
  handler answers the Heal, SetCurrentHp and kill passes as well as the damage one.

**The tallies (item 2) and Lava Lamp.** `OldCreatures.AfterDamageReceived` is gone with its call; each line is its
listener's handler: Beating Remnant (`UnblockedDamageThisTurn`, old body `CardEffects.BeatingRemnant_AfterDamageReceived`),
Hardened Shell (`HardenedShellDamageThisTurn`, old body `CardEffects.HardenedShell_AfterDamageReceived`, handed the HP
now plus the unblocked damage) and Lava Lamp (`TookUnblockedDamage`, through the new
`RelicEffects.LavaLamp_AfterDamageReceived`; open pair). As in the game, only a holder keeps them: with core2 the two
global old fields stay 0 for a player without Beating Remnant or Lava Lamp. Nothing else reads them (the cap and the
reward upgrade check the relic first), so the difference against old is a field, not a behaviour. The two models' caps
are theirs too (the whole class is LOCAL): Beating Remnant's ModifyHpLostAfterOsty (min(amount, 20 - tally)) and Hardened
Shell's ModifyHpLostBeforeOstyLate (min(amount, Amount - tally), amount 0 left alone), in the decimal fold with the old
tallies, and their AfterModifying flashes (presentation).

**Curl Up (item 3).** The game's AfterDamageReceived only remembers the card (powered attack, a card source, the first
card); the block is CurlUpPower.AfterCardPlayed's. The handler arms the old flag (`CardEffects.CurlUp_AfterDamageReceived`).
Hybrid: the block is the old `CombatEngine.PayCurlUp` through the old block code once the card has finished
(AfterCardPlayed and F09's GainBlock are not migrated), so the handler calls no block-gain command.

**Item 4.**
- Shrink AfterDeath (open): the old buff keeps no applier, so the applier is read as the old engine reads it: a Shrink on
  the player came from the Shrinker Beetle (the one monster that applies it), whose death runs
  `CardEffects.RemoveDeathBoundPowers`. Not modelled: an enemy's Shrink (the player applied it, by a potion) goes with the
  player's death, which ends the combat; two Shrinker Beetles in one combat, which no encounter has. AfterRemoved: the node
  scale, nothing. ModifyDamageMultiplicative: the owner dealing a powered attack, the old factor
  (`BuffSystem.Shrink_ModifyDamageMultiplicative`, 0.70).
- Illusion: AfterDeath calls `CombatEngine.TryReviveIllusion` (the Reviving marker, the intent, the block; it also strips
  the debuffs); ShouldAllowHitting reads the Reviving marker; ShouldCreatureBeRemovedFromCombatAfterDeath keeps the owner;
  ShouldPowerBeRemovedOnDeath answers by the power's type (`HookCtx.IsDebuff`, from the game's power data): a debuff goes, a
  buff stays, for any dying creature, as the game asks every listener. The R3b primary-enemy rule is unchanged.
- Imbalanced AfterDamageGiven (open): a fully blocked attack of the owner puts the Bowlbug Rock off balance
  (`EnemyAI.Imbalanced_AfterDamageGiven`); another owner's Stun is not modelled (the old engine gives the power to the
  Rock only).
- Vital Spark AfterRemoved: nothing to clear (the old engine reads the Tainted stamp off the living enemies' Vital Spark).
- Meat on the Bone AfterCurrentHpChanged (open): the relic's status only, so nothing; its heal is F17's.
- Lava Lamp: above. **Hang is not ported**: the old Hang card multiplies its damage by the stacks before the damage call
  (`CardEffects.ApplyNecrobinderCard`, a card-attack path the R4 fix round owns), so a ModifyDamageMultiplicative handler
  would multiply a second time with the switch on. The handler lands with that site's switch gate.

**Item 5.**
- AfterRemoved (pseudo) of Shrink, Slumber, Hex, Dampen, Sandpit, Burrowed, Tangled, Ringing and Vital Spark (Sword Sage is
  not reached): Dampen gives the upgrades back (`CardEffects.RemoveDampen`); Hex, Tangled and Ringing have no affliction on
  the old cards (the old engine reads the buff); Sandpit's old owner is dead at every AfterRemoved Core2 dispatches (the
  death removal), so the game returns; Burrowed's dead owner leaves the combat, so its block is read by nothing; Slumber's
  override is empty. A removal outside a death (PowerCmd.Remove) is not F08/F10.
- Read again at the F08 integration (landing 8), against the game's overrides: Hex, Dampen, Slumber and Burrowed agree
  with the game's conditions (Slumber on unblocked damage not 0, Burrowed on its owner's block broken, Hex on the
  applier's death, Dampen on the last caster's). Dampen's `RemoveDampen` runs once under the switch: its one old caller
  (`CombatEngine.Dampen_AfterDeath`, from `AnswerDeaths`) returns at the gate on its top; a second Magi Knight's
  AfterDeath finds the buff gone (`HookCtx.Power` re-reads it); AfterRemoved is dispatched only from the owner's death
  removal; and the body is idempotent (the flags it clears). One timing note, not a count: two Knights killed by one
  Damage lift it at the first one's AfterDeath (the living Knights are read by HP), where the game lifts it at the
  second's (its caster set).
- The other pairs of the AfterRemoved models (their class files are this batch's): Slumber AfterDamageReceived
  (`CardEffects.Slumber_AfterDamageReceived`), Hex AfterDeath (the applier read as the Spectral Knight,
  `CombatEngine.Hex_AfterDeath`), Dampen AfterDeath (the casters read as the living Magi Knights,
  `CombatEngine.Dampen_AfterDeath`), Burrowed AfterBlockBroken (`EnemyAI.BreakBurrowIfBlockGone`).
- The registered pairs of the L1 pseudo-pair powers, calling the power-owned old bodies the monster batches left: Reattach
  AfterDeath (`CombatEngine.TryReattachSegment`), ShouldAllowHitting, ShouldCreatureBeRemovedFromCombatAfterDeath; Adaptable
  AfterDeath (`CombatEngine.TryRespawnTestSubject`; the respawn move already passes the combat), ShouldAllowHitting,
  ShouldStopCombatFromEnding, ShouldCreatureBeRemovedFromCombatAfterDeath; Steam Eruption AfterDeath
  (`EnemyAI.TriggerAboutToBlow` with the combat), ShouldStopCombatFromEnding, ShouldCreatureBeRemovedFromCombatAfterDeath;
  Painful Stabs ShouldCreatureBeRemovedFromCombatAfterDeath. Minion has no other F08/F10 pair. Asleep AfterDamageReceived
  (`EnemyAI.WakeMatriarchIfStruck`) is here as the batch was asked; it is also in the CLOUD brief C3, which drops it.
  `TryReviveIllusion`, `TryReattachSegment` and `TryRespawnTestSubject` went from private to internal (no other change).
- **Stock is not ported**: the game's AfterDeath adds a new Axebot (CreatureCmd.Add) and the dead one leaves; the old
  `TryRespawnAxebot` revives the same `EnemyState` in place, which Core2's Kill then takes out of the roster (the removal
  is decided before AfterDeath and nothing vetoes it for an Axebot), so the respawn would be lost. It needs a
  CreatureCmd.Add over the old roster (a new enemy in the slot) first. Stock is C3's CLOUD model; C3 should skip its
  AfterDeath for the same reason.

`HookCtx` (engine-neutral): `CombatInProgress` (CombatManager.IsInProgress), `HittableEnemies()`
(CombatState.HittableEnemies), `IsDebuff(PowerRef)` (PowerModel.Type). Tier 1 uses (on the rewrite list): `c.World.InFlight`
(Rupture), `c.K` (Inferno's Damage), `c.OldCombat` and `c.Enemy` (the old bodies), `c.OldBuffs` (the Reviving marker).
Hybrid calls (an old body run from a handler; each re-enters Core2 only through the switch-gated old entries): Demon
Tongue's heal (`CardEffects.HealPlayer`, gated to Heal), Steam Eruption's HP write (gated to SetMaxAndCurrentHp), Rupture's
and Red Skull's Strength (`RelicEffects.GainPlayerStrength`, the old buff write: F-power, not migrated), Centennial
Puzzle's draws (`CardEffects.DrawCards`), Curl Up's block (above), and the old buff writes of Shrink, Hex, Dampen, Slumber,
Burrowed, Asleep, Illusion, Reattach and Adaptable (PowerCmd is not migrated).

- Acceptance (L2): the new tests 36 pass (L2PlayerDamageTests 6, L2TallyTests 7, L2DeathTests 18 with the 9 AfterRemoved cases, L2DeathExtraTests 5); a run with every L2 registration deleted failed every L2 test but the one that registers no model (32 of 33, before the cap pairs were added); `dotnet test src/Sts2Emulator.Tests` 6975 passed, 7 skipped, 0 failed (6982), ModelSourceScanTests included, `TIER2_REWRITE_LIST.md` regenerated; `construct commands` ok: 12 commands, 0 differences, 0 Core2 writes outside Commands, 0 F08 HP writes with the switch on (R4's informational list: 36 F09 block writes, 4 of them in the old bodies L2 calls: TryReviveIllusion, TryReattachSegment, TryRespawnTestSubject, BreakBurrowIfBlockGone); `sites check` 0 problems (279 sites; the CombatEngine.cs and RelicEffects.cs hashes re-recorded, the Doom and Fur Coat parts unchanged); `port_list --gate --families F08 F10`: 60 pairs without a handler (107 before, this batch's 47 closed; of the LOCAL models only Hang and Stock are left); byte_proof of the Release EmuDump against itself, `--engine old` and `--b-args "--engine core2"`, no switch, `--modes fight,state-hash --limit 300`: 600 fight-mode pairs identical, 0 differ; `fuzz_core2.py --switches F08,F10 --fights 300 --procs 3`: 299 differing (L1: 299), all untriaged; top fields ActiveCombat.TookUnblockedDamage and the run's TookUnblockedDamageThisCombat 151 each and UnblockedDamageThisTurn 48 (new: the old global tallies of players without the relic, which only a holder keeps now), ActiveCombat.Enemies 125 (m1: 186), PlayerHp 68 (84), UnblockedDamageHitCount 54 (59). `tools/emudump.Tests` not rerun (no EmuDump file changed).

### Batch L3b: the LOCAL models that needed a kernel addition (branch `rebuild/f08-l3b`)

7 (model, hook) pairs over 4 game models (`Core2/Models/Powers/SurprisePower.cs`, `InfestedPower.cs`, `StockPower.cs`,
`HangPower.cs`; Registry block `L3b`; tests `Core2/Models/L3bTests.cs`).

- **The stream: `HookCtx.Niche` (engine-neutral).** The old death answer (`CombatEngine.AnswerDeaths`) hands its bodies the
  `Random` the combat step threads: in a run `RunState.ActiveCombatRng`, the old combat stream (seeded from the run's Niche
  seed, restarted per combat, restreamed with Niche; `combat` in the RNG log). `Surprise_AfterDeath` and `Infested_AfterDeath`
  pass it to `CreateEnemy`, whose HP roll (`EnemyAI.RollSummonedHp`) reads the combat's `NicheHpRng` first (the run's Niche
  stream at its position; `Niche` in the RNG log) and the handed stream only when the combat has none; `TryRespawnAxebot`
  draws the handed stream itself. The game draws all three through `CombatState.CreateCreature`
  (`Creature.SetUniqueMonsterHpValue` on `RunState.Rng.Niche`), whatever kills the owner. The accessor is named by the
  game's stream and answers with the combat's own copy of it, `CombatState.NicheHpRng` (taken at the run's Niche position
  when the fight starts, restreamed with Niche, handed back to the run when the fight ends, cloned with the combat); a
  hand-made combat with none answers with its Shuffle stream, and one with neither throws. (Review round: the first cut
  answered with `RunState.ActiveCombatRng` through the kernel's run, which a fork shared with the live run; see "A fork's
  run view" in the R7 checks below.)
- **A copy's run.** `KernelState.CopyTo` kept the source's run when a copy was given none, so a combat's clone (a fork of
  `RunEngine.ForkActiveCombat`, a snapshot, a probe) stood in the live run: a forked Stock, Surprise or Infested death drew,
  and advanced, the live run's combat stream. The old engine's fork carries the fight and its own streams only (the combat
  clone and a clone of `ActiveCombatRng`). A copy now binds to the run it is given: none for a combat's clone (its run
  deck is then a DeckWithoutRun gap), and a restored holder keeps the run it stands in; a run's clone rebinds its fight
  as before. The other `World.Run` reads in Core2 (OldState's deck, relic and potion fallbacks, `HookCtx.Card` for a deck
  card) read the same binding, so none reaches the live run from a fork. Test: `ResampleLeakTests`, a forked step with a
  Stock death.
- **The command: `CreatureCmd.Add`** (`Core2/Commands/CreatureCmd.Add.cs`). game: CreatureCmd.Add(Creature): refused outside
  a combat in progress and for a creature the combat already holds; CombatState.AddCreature puts it at the end of its side's
  list; CombatManager.AfterCreatureAdded's parts (AfterAddedToRoom, the move roll) are the caller's old body in Tier 1.
- Surprise and Infested: AfterDeath under the game's condition (the dead creature is the owner, its removal not prevented),
  the old bodies on `c.Niche`; ShouldStopCombatFromEnding, the old functions.
- Stock: AfterDeath builds a NEW Axebot from the old respawn's parts, split out of `TryRespawnAxebot` with its behaviour
  unchanged (`NewRespawnedAxebot`, the state the in-place respawn leaves), rolls its HP as CreateCreature does
  (`RollAddedAxebotHp`: unique against the living side and the dying bot, on `c.Niche`; the old respawn keeps
  `RollAxebotRespawnHp`, flat on its site's stream), and lists it with `CreatureCmd.Add`, which fires
  Hook.AfterCreatureAddedToCombat behind F01's switch. The dead bot leaves through the Kill's removal after AfterDeath, so the new one stands at the
  roster's end, as in the game. The old condition (an Axebot) is kept; R3a's corpse rule (`AxebotDeathStrips`) is unchanged.
  ShouldStopCombatFromEnding: the old function.
- Hang: ModifyDamageMultiplicative, read against HangPower (the target is the owner and the card source is Hang: the
  amount; else 1). The old Hang body multiplied its base by `Math.Max(1, stacks)` before `DealDamage`; under the F08 switch
  that factor is 1 (`Kernel.If(Switch.F08, state) is null`), so the fold applies it once. No switch: unchanged.
- Tier 1 uses: `c.OldCombat`, `c.Enemy`, `c.K` (Stock's Add). Hybrid: the spawn bodies write the old roster themselves
  (`state.Enemies.Add` and `Insert`, `RelicEffects.Spawned`) and remove their own buff.

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
  `construct shape`, and (from round 3) `construct commands`: every F08/F10 site gated and no ungated HP or block
  write outside `Core2/Commands` (section 4);
- the AfterAttack-once test of the F06 bridge (section 3 item 1) and the single `FinishCombat` test of the F17
  hand-off (section 2) stay in the suite from the round that adds them.

Final (switch on in `migrated_families.json`): **the handler gate first**: `python -m tools.kernel.port_list
--families F08 F10 --gate` exits 0, that is every (model, hook) pair of the reached set, pseudo-hooks included, has a Core2
handler or is not modelled (today 150 pairs have neither: section 5); a pair is closed only by a Core2 handler or a
not-modelled entry with its reason. **The Illusion primary rule** (no handler closes it, R3b): Core2 reads a living
Illusion enemy as primary (the game's MinionPower-only `IsPrimaryEnemy`), the old engine as not; before switch-on the
corpus fights with an Illusion enemy are read against the game for the win point (the fight ends when only Illusions
are left, or not), and the old rule's users are listed and either follow the game or carry the difference.

### R7 proof rerun after R7f (2026-10-25): not passed, nothing switched on

R7f: the struct_compare census applies the unmodelled rows in the missing-entry branch (a record-only field no row lists is
still a difference); Diamond Diadem's and Undying Sigil's damage folds (F08) had no handler because the port list skipped models
no record in its manifest reaches (`564797d1d47ddfb9` fixed); `port_list --unreached` lists such models. Run with F08+F10 on:

| check | after R7e | after R7f | gate |
|---|---|---|---|
| O1: fail only switched on | 0 | 0 | pass |
| traced (77): better / same / worse; fewer fields | 8 / 69 / 0; 3 | 8 / 69 / 0; 1 | fail (fewer 1) |
| corpus (4,392): better / same / worse; fewer fields | 63 / 4,326 / 3; 8 | 63 / 4,327 / 2; 0 | fail (worse 2) |
| step test, strict | 4 of 4 | 4 of 4 | pass |
| fuzz (300): differing / ending disagreements | 291 / 0 | 291 / 0 | report |
| port gate (manifest-reached) / `--unreached` pairs without a handler | 0 / - | 0 / 11 | fail (11) |

Open, before the switch:
- `2b49779387ccd8b9`: Shrink on enemies[0] where the game has it on enemies[2] at step 4 (the reanchored step is right, the full
  replay is not; an aim left on a world copy was suspected, a potion-aim change did not move it and was reverted).
- `7171bcd6dd0b0bd6`: Gremlin Horn's AfterDeath pays an extra draw and energy for an egg Parrying Shield kills at turn end.
- The traced fight on fewer fields (one left after the census fix).
- 11 F08/F10 pairs of models no record reaches have no handler (`port_list --unreached`: Die For You's other pair, Envenom,
  Monarch's Gaze, Necro Mastery, Reaper Form, Reflect, Sword Sage, Book Repair Knife, Emotion Chip, ...): under the switch the
  old fold returns early for them, so they would silently drop, as Diamond Diadem did. Port them; the gate counts them.

### R7 proof rerun after R7d and R7e (2026-10-25): not passed, nothing switched on

R7d added the game's two turn-start win checks (CombatManager.StartTurn; under F10 the old turn start never checked, so a
kill or a loss there was dropped and the fight went on where the game's had ended: the triage above read it backwards) and
walks a copy in the Demise loop (a crash); R7e makes `CreatureCmd.Damage` skip a target the hooks removed (a crash: an
Inferno burst kills a Toadpole inside the Toadpole's Thorns answer). Run with F08+F10 on, gated against the old engine:

| check | on `a42c41a3` | after R7d | after R7e | gate |
|---|---|---|---|---|
| O1 (291 old-suite tests): fail only switched on | 0 | 0 | 0 | pass |
| traced (77) against the game: better / same / worse | 8 / 68 / 1 | 8 / 69 / 0 | 8 / 69 / 0 | pass |
| traced: fights on fewer fields | 3 | 3 | 3 | fail (see below) |
| corpus (4,392) against the game: better / same / worse | 67 / 4,203 / 122 | 63 / 4,324 / 5 | 63 / 4,326 / 3 | fail |
| corpus: fights on fewer fields | 20 | 9 | 8 | fail (see below) |
| step test, strict | 2 of 4 | 4 of 4 | 4 of 4 | pass (`RerouteLanded` set) |
| fuzz F08,F10 (300): differing / ending disagreements | 291 / 0 | 291 / 0 | 291 / 0 | report |

Open, before the switch (next round):
- Three worse corpus fights, not yet read: `2b49779387ccd8b9` (Shrink on enemies[0] where the game has it on enemies[2] at
  step 4: a roster index after a corpse removal, a comparer alignment question, or real targeting), `564797d1d47ddfb9` (the
  player's HP 51 where the game has 58 at step 1), `7171bcd6dd0b0bd6` (the draw pile's order at step 10).
- A target removed inside its own hit (R7e): the kernel keeps the game's zero result in the results but does not note it in
  `History.DamageReceived` as the game does (the removed creature has no position to name); check a recorded Thorns-Inferno fight.
- The fewer-fields gate: R7e reads it as a census fault (neither view writes `enemies[*].move` or the net piles' floor; the old
  run's census holds them only from missing-entry diffs, and `net` from old steps with no net view). Settle it in the comparer
  (in the missing-entry branch `compared.add(p)` runs before the unmodelled-row test: apply the unmodelled rows there, keeping
  the documented rule that a record-only field is a difference), then rerun.
- Then rerun this table; the switch goes on in `migrated_families.json` only with every gate passing.

### R7 proof on `a42c41a3` (2026-10-24): not passed, nothing switched on

Port gate 0 (model, hook) pairs without a handler. Run with F08+F10 on (`struct_calibrate --switches F08,F10 --gate-against` an
old-engine run of the same tree, `suites_switch.py`, `StepApiKillTests` strict, `fuzz_core2.py`):

| check | result | gate |
|---|---|---|
| O1, the old behaviour suites (291 tests) | 215 pass both, 76 not covered, 0 fail only switched on | pass |
| traced set against the game (77 fights) | struct_compare 8 better, 68 same, 1 worse (on `ad9c8909`: 5 / 59 / 13); 3 fights on fewer fields | fail |
| corpus against the game (4,392 fights) | struct_compare 67 better, 4,203 same, 122 worse; 20 fights on fewer fields; 2,027 exposed by the migrated families' rows | fail |
| step test, strict (traced Headbutt and Twin Strike) | switch-off 2 pass; switched-on 2 fail: the death-event order lacks `AfterAttack` after `AfterDeath` | fail |
| fuzz F08,F10 (300 fights) | 291 differ (untriaged), ending disagreements 0 | report |

Triage of the 122 worse corpus fights (first differing path): 66 `snap.discard`, 36 `snap.draw`, 11 `snap.in_combat`, 7 a step
count one short, 1 Shrink, 1 player HP. The 113 pile and in-combat cases are one mechanism: under the switch the combat has ended
(the old `FinishCombat`, F17's, runs at the kill: in_combat false, the piles cleared) at a step where the game is still in combat
with its piles intact (the game ends the combat after its action queue drains). The traced set's one worse fight is the same
(`snap.draw`). Next, before any switch-on:
- **Combat end after the queue drains (blocker).** Under F10, hold the old `FinishCombat` until the kernel's queue has drained
  at the step boundary the game ends on (CombatManager's end of combat after the card's actions), or migrate F17's AfterCombatEnd
  and AfterCombatVictory with it. Gate: the 113 corpus fights and the traced fight same or better, the 7 step-count errors gone.
- **AfterAttack after a killing attack (blocker).** The step test's switched-on order is BeforeDeath, ShouldDie, ShouldRemove,
  AfterDeath and stops: the attack's AfterAttack (F06, through the bridge) is not seen after the kill. Read AttackCommand's end
  in the game (AfterAttack fires whether or not the target died) and the F06 bridge's recording.
- **Fewer fields (20 corpus, 3 traced).** Read which fields go missing under the switch (the census gate) and close each.
- Then rerun this table; the switch goes on in `migrated_families.json` only with every gate passing.

### R7d: the blockers (branch `rebuild/f08-r7d`, on `dd48f22e`)

**The triage above had the direction reversed.** In all 112 pile cases the game's piles are empty and ours are not, and
in the 11 `snap.in_combat` cases the game reads false and we read true: the game's combat is over at the recorded step and
ours goes on. Every one is an `end_turn` step. With F10 on, the combat's end is Core2's `CheckWinCondition` at the game's
points only, and `RunEngine` drops the old step's terminal reading (Settle) unless a hand-off ran; the old turn start had no
such point, so a kill or a loss in the turn start (Inferno, a power's damage, a pending loss from the turn start's own
Kill) left the combat in progress with nothing standing (the missing victory heal is the HP difference). Fixed at the
game's two `CombatManager.StartTurn` checks: the player side after the auto-pre-play phase (and at the turn start's two
player-death returns), the enemy side before ExecuteEnemyTurn (`CombatEngine.StartTurnWinCheck`). The 7 step-count
errors were a crash: under the switch a Demise kill removes its enemy inside the end-of-enemy-turn pass
(`EnemyAI.ApplyEndOfEnemySideTurn` now walks the roster as the pass began). Tests: `R7dTests`.
F17 is not migrated: the game runs AfterCombatEnd and AfterCombatVictory inside `EndCombatInternal`, inside
`CheckWinCondition`, and the hand-off already runs the old `FinishCombat` there; nothing in the corpus needed it.

**AfterAttack after a killing attack.** AfterAttack is F06's. With F06 off the attack's end runs the bridge (the old
bodies) where the game runs Hook.AfterAttack, after the kill, outside the kernel's dispatch, so no handler could log it.
`Kernel.BridgePoint` (a test seam) is called at the bridge's two points; the step test logs AfterAttack there. The strict
run passes and `RerouteLanded` is set.

R7 proof rerun on `rebuild/f08-r7d` (same commands):

| check | result |
|---|---|
| O1 (291 tests) | 215 pass both, 76 not covered, 0 fail only switched on |
| traced (77) | 8 better, 69 same, 0 worse; 3 fights on fewer fields (b00#2, b03#13, b03#14, unchanged: not the early end) |
| corpus (4,392) | 63 better, 4,324 same, 5 worse; 9 fights on fewer fields (`snap.enemies[*].move`; was 20) |
| step test, strict | 4 pass |
| fuzz F08,F10 (300) | 291 differ (untriaged), ending disagreements 0 |

Left before switch-on: the 5 worse corpus fights (26562e018fdc5e6a, 2b49779387ccd8b9, 41193a3af1dede7e,
564797d1d47ddfb9, 7171bcd6dd0b0bd6; read so far: the enemies after a Beetle Juice, 7 HP more lost on the first end turn, the draw pile),
the 9 corpus and 3 traced fights on fewer fields, the fuzz triage.

**R7 checks before switch-on** (each closed, with a test, before the switch goes on in `migrated_families.json`):
- **The enemy side's turn start order (R7e).** The game's `CombatManager.StartTurn` runs `Hook.AfterSideTurnStart`
  (PoisonPower's tick) before the enemy side's CheckWinCondition; the old engine ticks poison inside the per-enemy loop,
  after it (`CombatEngine.StartTurnWinCheck`'s enemy-side call), so an enemy can act before a later enemy's poison ticks,
  and a poison kill of the last enemy is caught by a later check. The enemy-side check has no reproducing test in the old
  order: nothing between the turn end's last check and it deals damage (`R7dTests.ADoomKillAtThePlayersTurnEndEndsTheCombatBeforeTheEnemiesAct`
  passes with it off; the turn end's check ends that fight). Test before closing: an enemy-side poison kill ends the combat
  before any enemy acts.
- **The loss path (R7e).** On a loss the game runs neither AfterCombatEnd nor AfterCombatVictory, but the hand-off runs the
  old `FinishCombat` for a lost combat too (as the old engine does). Whatever `FinishCombat` does on a loss must be the
  game's loss path, or not run. `R7dTests.ALossAtThePlayersTurnStartEndsTheCombatThere` pins one hand-off and one
  `FinishCombat` call on a loss.
- **Inferno's turn-start burst in a test setup (found in R7e, unread).** Run PRB00058 at A8, its first combat, every
  enemy at 1 HP, Inferno 999 and InfernoSelfDamage 1 set with BuffSystem.Apply, end the turn: switch off the enemies die,
  switch on they stay at 1 HP (the player loses the 1). In the corpus the switched-on turn-start burst lands (26562e018fdc5e6a,
  its end turn at step 8), and so does a Thorns-set-off burst in `R7eTests`, so the setup may be at fault. Read before
  switch-on; the R7d turn-start kill test uses Rolling Boulder until then.
- **A fork's run view (blocker, batch L3b review).** Since the L3b fix a combat's clone (a fork of the search, a snapshot,
  a probe) binds to no run, so under the switch a run+combat pass in a fork leaves out the run deck's cards and enchantments
  (`DeckWithoutRun`, reported by `Listeners.Report` only) and `HookCtx.Card` answers null for a deck card. The search forks
  combats, so before switch-on a fork gets a read-only view of its run's deck, or the gap refuses (throws) instead of
  answering empty. Test: a forked step with a run-deck listener equals the same step unforked.
  **Closed in R7a** (branch `rebuild/f08-r7a`): `RunEngine.ForkActiveCombat` keeps the fight and its streams, and the old
  combat step never reads the run (its deck changes are made by the run as the combat ends), so a copy bound to no run
  keeps the deck its source read, frozen at the copy (`KernelState.CopyTo`: the same card values in a list nothing
  writes; a fork of a fork shares it), and its world reads it as `OldWorld.Deck` (the run pass, the deck member check and
  probe, `HookCtx.Card`). No live list or stream is reached. Test:
  `ResampleLeakTests.AForkedStepWithARunDeckListenerEqualsTheSameStepUnforkedAndLeavesTheLiveRunAlone`.
- **The pet redirect at the enemy-attack site (R7a).** Under the switch the monster's hit is AttackCmd's
  `CreatureCmd.Damage`, whose `ModifyUnblockedDamageTarget` fold had no holder (the old state lists no power on Osty), so
  the player took the hit (O1's three OstyDamageSinkTests). `DieForYouPower` (Registry block `R7a`) ports the game's
  override; OldState lists the power on the Osty a combat holds (the game applies it once, at OstyCmd.Summon's first
  add, and never removes it), and `HookCtx.PetOwner` names the player for Osty. Tests: `R7aTests`.
  **Die For You on revive (R7b2)**: the game applies the power at OstyCmd.Summon's first add only; the revive branch re-adds the same creature, which kept it (ShouldPowerBeRemovedAfterOwnerDeath is false), so a revived Osty holds it. OldState lists it whenever the combat holds an Osty (`OstyMaxHp` above 0, dead or alive under the switch), so the code already acts after a revive; the comments now say so. Test: `R7b2Tests.ARevivedOstyStillTakesThePlayersPoweredHit` (passes before and after).
- **Dealers: the player (R7a).** The game's ConstrictPower and DisintegrationPower damage their owner with the owner as
  dealer; `CardEffects.DealDamageToPlayer` and `OldEntry.HitPlayer` take an `OldCreature` dealer, and those two sites
  pass `OldCreature.Player` (read only under the switch). Test: `R7aDealerTests`.
- The index holders (the independent review, docs/reviews/2026-10-24-f08-migration.md, recommendation 3, its 1.3 A and B):
  `HandleEnemyDeaths` gated off whole under the switch; Oblivion paid by reference, not by roster position; the play's
  target resolved by reference (`FirstEnemy` reads `PlayTarget`, null for a dead or removed target).
  **Done (R6.9)**, tests in `Core2/IndexHolderTests`, three enemies each: the positional pairing was already off whole
  under the switch (R4: `HandleEnemyDeaths` and `AnswerDeathsSince` return first), now pinned
  (`APositionalDeathCheckDoesNotRunUnderTheSwitch`); Oblivion's snapshot keeps its holders by reference under the switch
  (`OblivionIsPaidToTheCreatureNotTheSlotAfterAMidCardRemoval`); the play opens an aim mark (`OldWorld.Aiming`, the
  play's `PlayTarget` creature, nested plays set their own), and `FirstEnemy` reads it, null for a dead, untargetable or
  removed target, the index fallback only with no target given
  (`ACardsSecondTargetedEffectFollowsTheCreatureNotTheIndex`). Switch off: unchanged.
- The in-move death (recommendation 4, 1.3 C): no post-attack bookkeeping (`MoveIndex++`, Ritual, Vigor,
  `RestoreTemporaryEnemyStrength`) on a creature that died in its own move, or the forced move re-applied at
  `PerformMove`'s end; `WaterfallGiantTests` stay true under the switch.
  **Done (R6.9)**: the kill notes a death of the monster performing its move (`OldWorld.MoverDied`, reset by each
  `PerformingMove` scope), and `EnemyAI.ExecuteIntent` returns before its post-attack bookkeeping when it is set
  (`AMoverKilledInItsOwnMoveGetsNoPostAttackBookkeeping`, a stand-in Thorns handler until L3a's lands);
  `WaterfallGiantTests.KilledByThornsOnItsOwnTurnItBlowsARoundLater` still waits for L3a's Thorns handler.
- One ending (recommendation 7): Fogmog's primary rule read against the game and closed; the two "ending" predicates one
  under F10, with an agreement report (the old predicate against Core2's at every step) in `EmuDump fuzz`.
  **Done (R6.9), the agreement part reported, not asserted**: the game's `IllusionPower.AfterApplied` gives its owner MinionPower (decompile read), so Core2's
  `OldCreatures.IsSecondaryEnemy` counts an Illusion holder and the Fogmog's death is the last primary's
  (`AnIllusionHolderIsNotAPrimaryEnemy`); F08 and F10 switch together, so `CombatEngine.IsOverOrEnding` is Core2's
  `CombatEnd.IsOverOrEnding` whenever F10 is on, and the old reading is `CombatEngine.OldIsOverOrEnding`; `EmuDump fuzz`
  compares the two after every step with F10 on and lists each disagreement under the output's `ending_disagree`
  (absent when they agree, so the old engine's output is unchanged). The key is only a list: nothing reads it yet, so
  a disagreement fails no test (see the R7 acceptance item below).
- Curl Up: the old flag pays on the next card played (`CombatEngine.PayCurlUp`), where the game's
  CurlUpPower.AfterCardPlayed pays only on the card that armed it.
  **Done (R6.9)**: under the switch the Core2 handler remembers the card (`EnemyState.CurlUpCardUid`, IdentityOnly,
  and a second card's hit is not remembered, as the game's playedCard test), and `PayCurlUp` pays only at that card's
  AfterCardPlayed (`CurlUpPaysOnlyAtTheCardThatArmedIt`).
- Ravenous: `CombatEngine.Ravenous_AfterDeath` split per owner (the review's 3.3), so N holders give N owners' effects,
  not N squared. Done in batch L3a: the loop calls `Ravenous_AfterDeathOf`, the handler (`RavenousPower`) calls it for its
  own owner; `L3aModelTests.RavenousActsForEachSurvivorOnceAmongFourHolders`.

**R7 acceptance items** (the oracle rounds O1 and O2 are building them): recommendation 1, the step-API acceptance test
from the traced Headbutt and Twin Strike fights; recommendation 2, the old behaviour suites run under F08+F10 through a
test-fixture switch, each failure classified (R4 to deliver, handler batch, or old deviation updated with a trace);
recommendation 5, the traced set with net states (`struct_calibrate traced --engine core2`), no fight worse against the
game. Then the by-family table re-measured on a lockstep block with core2;
target G2891c25e0d's kill part and mechanism B at 0, F08 proper down by at least its A+B share; any group that grows is
a stop.

R7 item, the ending agreement made an assertion: `tools/fidelity/fuzz_core2.py` fails on any step listed under a fuzz
output's `ending_disagree` that is outside an expected list (held-open corpses without a handler), and that list is empty at switch-on.
**Done (R7b1)**: `EXPECTED_ENDING_DISAGREE` (power id, readings), empty; the summary counts `ending_disagree` and
`ending_unexpected`, and an unexpected one exits 1 (`test_an_ending_disagreement_outside_the_expected_list_exits_1`).

**R7 checks: old-site divergences the handlers inherit or settle.** Each is a place where a handler equals the old function and the
game does something else, found while porting; R7's corpus re-measure decides each, and none moves a gate by itself.

- **Phantom Blades: name against tag.** The game's override tests the Shiv tag on the card source; the old site
  (`CardEffects.ShivDamage`, and `PhantomBlades_ModifyDamageAdditive`'s caller) tests the card's name, and so does the handler
  (`PhantomBladesPower.IsShiv`), so that it equals the old function. The old card data carries the tag as `CardDef.ShivTag`.
  Check: list the defs with `ShivTag` whose name is not Shiv; a recorded Phantom Blades fight that plays one under the power
  differs from the game, and the handler then takes the tag test.
  **Closed in R7b2, no behaviour change** (branch `rebuild/f08-r7b2`): the one card the game tags Shiv is Shiv (CanonicalTags, every card read), so the name test is the game's tag test. `CardDef.ShivTag` is not the tag: `scripts/extract_data.py` sets it on any card whose text names `CardTag.Shiv`, so Knife Trap (a Skill that counts the exhaust pile's Shivs) carries it, and taking the flag would add a card the game does not count. The handler stays on the name, its comment says why. The flag's other reader is old: `RelicEffects` HelicalDart pays on a Knife Trap play, which the game's tag test does not: a Lane B item (the extractor's text match, `RelicEffects` HelicalDart, and `UnmodelledRelicHooksTests.TheTagIsOnShivAndKnifeTrap`, which pins the bug). Test: `R7b2Tests.PhantomBladesAnswersTheGamesShivTagWhichOnlyShivCarries` (passes before and after: a guard against the flag).
- **Tracking and Double Damage: the pet case.** The game accepts a dealer that is the owner's pet (Osty); the handlers do too
  (`HookCtx.IsOwnerOrPet`), while the old fold (`BuffSystem.IncomingDamage`) reads the attacker's own buff list. Check: a recorded
  fight where Osty strikes with a card source under either power; the handler is the game's answer.
- **Sic 'Em on the last enemy's killing hit.** Under the switch Osty's max HP grows and its current HP stays (GainMaxHp's Heal
  skips a non-player while the combat is ending); the old engine raised both. Check: a recorded Necrobinder fight that ends on
  Osty's hit on a Sic 'Em holder.
- **Intangible: the ending guard.** The game's `IntangiblePower.ModifyHpLostAfterOsty` returns the amount unchanged while the
  combat is not in progress; HookCtx has no accessor for it and the handler (`IntangiblePower`, batch C3) does not model it.
  **Closed in R7b2**: `IntangiblePower.ModifyHpLostAfterOsty` returns the amount while `HookCtx.CombatInProgress` is false (the accessor existed: `CombatEnd.IsInProgress`). Test: `R7b2Tests.IntangibleLeavesAnHpLossAloneOnceTheCombatIsNotInProgress`.
- **Buffer: any owner.** The game's Buffer acts for any owner, an enemy's too; the old functions read the player's buff list alone
  and the old enemy site has no Buffer, so the handler acts for the player's Buffer only (`BufferPower`). Check: a recorded fight
  where an enemy holds Buffer and takes an HP loss; the game spends a stack and the loss becomes 0, the handler leaves it, and the
  handler then takes the any-owner test.
  **Closed in R7b2**: `BufferPower` acts for any owner; an enemy's stack is spent on its own list (the old function's step), the player's by the old function. Tests: `R7b2Tests.AnEnemysBufferTakesItsHpLossAndSpendsAStack`, `PowerTwoTests.BufferActsForItsOwnersLossOnEitherSide` (was `BufferLeavesAnEnemysLossAlone`).
- **Personal Hive: the `dealer != null` test.** The game's override tests the target, a non-null dealer and the powered props; the
  old site (`CardEffects.LandDamageOnEnemy`) has no dealer test and the handler (`PersonalHivePower`, batch C4, whose comment omits
  it) keeps the old site's tests, the owner's HP above 0 included. Check: a recorded powered hit on a Personal Hive owner with no
  dealer; the handler shuffles Dazed, the game does not, and the handler then takes the dealer test.
  **Closed in R7b2**: the handler tests a non-null dealer; the old site's HP test is kept. Test: `R7b2Tests.PersonalHiveShufflesNoDazedForAPoweredHitWithNoDealer`.
- **Constrict: the Applier test.** The game's override removes the power when the dead creature is the power's Applier; the old
  site and the handler (`ConstrictPower`) test that the fallen is a Slithering Strangler by def id, whichever Strangler applied
  it. Check: a recorded fight with two Stranglers where the one that did not apply Constrict dies; the handler removes the
  player's Constrict, the game does not, and the handler then takes the Applier test.
  **Closed in R7b2, with a gated old-engine edit**: the old state had no Applier, so the Strangler's Constrict move (`EnemyAI`, the debuff intent) records it under the F08 switch only, when its Apply creates the player's power (`EnemyState.AppliedPlayersConstrict`, IdentityOnly, cloned; a later Apply stacks and keeps it, as PowerCmd.Apply's ModifyAmount); the handler removes the power at its Applier's death, and falls back to the old Strangler test when no enemy is recorded (a loaded document, a test). Switch off: nothing recorded (byte proof 0 differ). Tests: `R7b2ConstrictTests.ConstrictStaysWhenTheStranglerThatDidNotApplyItDiesAndGoesWithItsApplier`, `TheStranglersConstrictMoveRecordsTheApplierOnlyUnderTheSwitchAndOnlyWhenItCreatesThePower`.
- **Strike Dummy, Fake Strike Dummy, Miniature Cannon, Mystic Lighter, Pen Nib: props and owner.** Miniature Cannon and Mystic Lighter also keep the old site's Attack-def test. The game's overrides answer the no-bonus value off a powered attack (`ValueProp.IsPoweredAttack`) and for a dealer or card that is not the owner's (`HookCtx.IsOwnerOrPet` for Pen Nib); the old sites (`RelicEffects.CardDamageBonus`, `CardDamageMultiplier`, `CardEffects.DmgFrom`) read neither, so the handlers take the game's tests. Check: a recorded fight with an unpowered hit carrying a card source (an upgraded Attack's self-damage) under one of the five; the old engine gave the bonus, the handler gives none. The old state has one card owner, so the card-owner tests are always met until a second owner exists.
- **Thorns, an enemy's: the target.** The game's `ThornsPower.BeforeDamageReceived` damages the dealer; the old body
  (`CardEffects.TriggerEnemyThorns`) and the handler (`ThornsPower`, batch L3a) damage the player whoever dealt, so Osty's
  powered hit on a Thorns holder costs the player, not Osty. The retaliation's dealer is the owner under the switch
  (`DealDamageToPlayer`'s dealer, L3a). Check: a recorded Necrobinder fight where Osty strikes a Thorns holder; the handler
  then takes the dealer as its target.
  **Closed in R7b2**: the enemy-owned retaliation goes to the dealer: the old body when the player dealt, else CreatureCmd.Damage on the dealer (Osty, an enemy), unpowered, the owner as dealer. Test: `R7b2Tests.AnEnemysThornsHitsOstyWhenOstyStrikes`.
- **Crab Rage: whose death.** The game's `CrabRagePower.AfterDeath` answers any ally's death; the old body and the handler
  (`CrabRagePower`) answer only the death of a Crab Rage holder. With the Kaiser Crab's two halves the only enemies, no
  reachable fight differs until another ally joins them.
  **Closed in R7b2**: the handler answers any death on its owner's side but its own (the handler needs an enemy owner; a player-side Crab Rage is not reached), with the old body's step for its own owner (Strength, unpowered Block, the power removed). Tests: `R7b2Tests.CrabRageAnswersTheDeathOfAnAllyThatHoldsNoCrabRage`, `L3aModelTests.CrabRageTakesTheGamesTestOfTheFallen` (was `CrabRageKeepsTheOldTestOfTheFallen`).
- **Surrounded: the props.** The game's `SurroundedPower.ModifyDamageMultiplicative` has no props test; the old fold
  (`BuffSystem.IncomingDamage`) is the powered one and the handler (`SurroundedPower`) asks for a powered attack. Check: an
  unpowered hit on the player from the half at the back; the game multiplies it by 1.5. Its AfterDeath takes the game's
  condition (a real death on the other side), which the old site reached only after a Crab Rage.
  **Closed in R7b2**: the fold has no props test. Tests: `R7b2Tests.SurroundedMultipliesAnUnpoweredHitFromTheBack`, `L3aModelTests.SurroundedMultipliesAPoweredHitFromBehindAsTheOldBodyDoes` (its unpowered line now 1.5x).
- **Surprise, Infested, Stock: the HP stream by site (batch L3b).** The old engine draws by site: the in-blow death
  answer of a card's kill (`CardEffects.OnEnemyDeath`, `AnswerDeathsHoldingTheFightOpen` on `state.ShuffleRng`) hands the
  Shuffle stream, every other death answer (`AnswerDeaths`, `AnswerDeathsSince`) the step's combat stream
  (`RunState.ActiveCombatRng` in a run). Surprise's and Infested's spawns read `NicheHpRng` first and the handed stream only
  without one; Stock's respawn always draws the handed stream, flat over the band. The game draws `RunState.Rng.Niche`
  (SetUniqueMonsterHpValue, unique against the side) whatever the site. Under the switch the handlers run from Kill and draw
  `HookCtx.Niche` (the combat's `NicheHpRng`); Stock's HP is the unique roll (`RollAddedAxebotHp`). So an Axebot respawn
  under the switch moves the Niche stream where the old engine moved the Shuffle or the combat stream. Check: a recorded
  Axebot respawn's HP against the Niche stream's next unique draw, and the Shuffle stream's next shuffle after a card's kill.
  GremlinHorn's AfterDeath draws `ShuffleRng` for its card draw's reshuffle (the game's Shuffle stream), not a spawn: kept.
- **Stock: the unique-HP side (batch L3b).** The game's set excludes every creature still listed on the side, the dying
  one included (CreatureCmd.Kill removes it after AfterDeath); `RollAddedAxebotHp` excludes the living enemies and the dying
  bot, as the old roster keeps corpses the game has removed. `EnemyAI.RollSummonedHp` (Surprise, Infested, every old summon)
  excludes the living only. Check: a recorded respawn whose dead bot's max HP is in the band.
- **CreatureCmd.Add: AfterCreatureAddedToCombat (batch L3b).** The game's CreatureCmd.Add ends with
  Hook.AfterCreatureAddedToCombat (PhilosophersStone, FurCoat, Sandpit, Murderous). Its family is F01 (the other switch
  group): Add fires it behind F01's switch, and with F01 off nothing runs for the new Axebot, as the old in-place respawn
  never called `RelicEffects.Spawned` (the old body: PhilosophersStone's Strength, FurCoat's bite). Core2 registers no F01
  handler for it yet. Check: a recorded Axebot respawn under PhilosophersStone or FurCoat; the game gives the new bot
  Strength 1 or bites it, the old engine does neither.
- **Stock and Infested: the roster position (batch L3b).** The game appends every added creature (CombatState.AddCreature)
  and the dead one leaves after AfterDeath; Stock's handler does so (the old engine revived the bot at its index), while
  Infested's old body inserts the Wrigglers right after the Phrog Parasite and stops at six enemies. Check: a recorded fight
  with an enemy listed after the Phrog Parasite or the Axebot; enemy turn order and first-enemy targeting.
- **Hang: the multiply's place (batch L3b).** The old Hang body multiplied the card's base before the fold's additive
  terms (Strength, Vigor); the game multiplies after them. Under the switch the fold gives the game's order. Check: a
  recorded Hang played with Strength.
- **Surprise, Infested, Stock: a prevented removal (batch L3b).** The game's AfterDeath does nothing when the removal was
  prevented; the old site has no such test; the handlers follow the game.
- **A HookCtx accessor for the card's live copy.** `MiniatureCannon.Copy` and `MysticLighter.Copy` are the same private helper (the card source's copy read back off `OldCombat` and `World`, for the upgrade flag and the enchantment, which `CardView` does not carry); HookCtx has no near fit, so R7 should add one accessor (the copy for a handle) and both models, and any later one, take it.

### The old suites under the switch (O1; the review's recommendation 2)

The old engine's behaviour suites pin game behaviour the old engine was made to match; run with F08+F10 on they are a
switched-on regression suite. `STS2_TEST_SWITCHES=F08,F10` (src/Sts2Emulator.Tests/Support/TestSwitches.cs) makes one
kernel with those switches and `Registry.All`. A combat a suite steps itself (the `Fight` helper's steps,
`TestSwitches.Step`) is bound at its first step, after all of the test's setup (factory, fluent `Enemy` / `PlayerBuff` /
`Seed`, direct writes), with `Attach(combat, run)`, the run being the state of the run engine (one per combat) whose
`RunEngine.StepCombat`, the kernel's entry gate, takes that and every later step: the order `RunEngine.StartCombat` uses
(setup, pre-play phase, `Attach(combat, State)`, then the first step). An engine call a test makes on a combat before
its first step runs unbound. A suite's own `RunEngine` gets the kernel before its `Reset` (`TestSwitches.Run`) and binds
the combats it starts; its steps go through `TestSwitches.Step(engine, ...)`. With `STS2_TEST_SWITCHES_COVERAGE` set to
a file, every kernel-bound step appends the test method that made it (found on the stack; a theory's cases share it).
Unset, empty or `none`: no kernel, nothing bound or written, every step `CombatEngine.Step` / `RunEngine.Step` (the
whole suite and the byte proof unchanged). The runner runs the suites once off and once on (with the coverage file) and
writes each test's verdict (pass both, not covered: passed both ways with no kernel-bound step, fail on only, fail off
only, fail both, skipped, missing off / on) to `DIR/suites_switch.json` and `DIR/suites_switch.md`; it exits 1 when a
run crashed, wrote no trx or lost tests:

    python -m tools.fidelity.suites_switch --out DIR [--switches F08,F10] [--no-build] [--suites A,B]

The suites are `NAMED` (the ten of review 2.3 item 3; DeathCheckTests.cs is six classes) and `ADDED` (the other damage,
death, HP and combat-end suites) in tools/fidelity/suites_switch.py. By hand: `STS2_TEST_SWITCHES=F08,F10 dotnet test
src/Sts2Emulator.Tests --no-build --filter "FullyQualifiedName~.DeathCheckTests.|..."`. Each round that switches more
on reruns it; a "fail on only" is classed in MIGRATION_F08_O1.md (R4 to deliver, handler batch, old deviation).

### R7e: the R7d proof's worse fights (branch `rebuild/f08-r7d`)

- 26562e018fdc5e6a, 41193a3af1dede7e (fewer steps): a crash, `Enemy(93) is not in the combat`. A card's hit on a Toadpole
  with Thorns: Thorns hits the player in BeforeDamageReceived, the player's Inferno burst kills the Toadpole and the kernel
  removes it; `CreatureCmd.Damage` then named the removed target. The game's loop goes on with the dead target, which has no
  HP to lose; the kernel now skips a target the hooks took out of the combat, as its results loop does. Test: `R7eTests`.
- The census's fewer fields (`snap.enemies[*].move`, `snap.exhaust[*]`, `net`, `net...piles.*[*].floor`): no view drops a
  field. Neither view writes `move` or `floor`; the old engine's census holds them only from the comparer's missing-entry
  diffs (the game lists an enemy or a card the old engine has not), and `net` from steps where the old view had no net
  state. Core2 has those entries, so the paths become game-only (unmodelled) and leave its census. A comparer question:
  the census should count a path the engine wrote, not one compared only because the engine lacked the entry.

### Damage terms applied twice (round R7c, branch `rebuild/f08-r7c`)

Found by R7b1: once the fixture gave every hand-made card a uid at the bind, 8 cases failed with F08+F10 on (7
CalculatedDamageTests Sharp / Corrupted cases, VitruvianMinionDoublesBothHalves). With a card source that resolves (a real
fight's cards always have one) Core2's `Hook.ModifyDamage` folded the enchantment and Vitruvian Minion's factor onto an
amount the old card pre-calculation had already folded them into. The class Hang had (batch L3b). Rule: each Calculate
term runs once, and under the switch it is the kernel's.

Method: `python -m tools.kernel.fold_precalc` lists, for every model of `Registry.All` with a damage fold, the old
engine's calls of its extracted body and whether an F08 test comes first in that method; each method with none was
followed to its callers by hand (below). The enchantment is the dispatcher's EnchantAdd / EnchantMul steps, not a pair,
and is listed with them. 36 pairs and 2 steps checked; 11 applied twice, all 11 fixed at 2 gates.

| model | hook | old site | applied twice |
|---|---|---|---|
| enchantment (step) | EnchantAdd, EnchantMul | `CardEffects.DmgFrom` (`EnchantDamageAdditive`, `EnchantDamageMultiplicative`) | yes: fixed (gate 1) |
| GigantificationPower | ModifyDamageMultiplicative | `DmgFrom` | yes: fixed (gate 1) |
| MiniatureCannon, StrikeDummy, FakeStrikeDummy | ModifyDamageAdditive | `RelicEffects.CardDamageBonus` from `DmgFrom` | yes (3): fixed (gate 1) |
| MysticLighter | ModifyDamageAdditive | `RelicEffects.EnchantedCardDamageBonus` from `DmgFrom` | yes: fixed (gate 1) |
| VitruvianMinion, PenNib | ModifyDamageMultiplicative | `RelicEffects.CardDamageMultiplier` from `DmgFrom` | yes (2): fixed (gate 1) |
| AccuracyPower, PhantomBladesPower | ModifyDamageAdditive | `CardEffects.ShivDamage` (after `Dmg`) | yes (2): fixed (gate 2) |
| HangPower | ModifyDamageMultiplicative | the Hang case's factor | no: gated in L3b |
| ConquerorPower | ModifyDamageMultiplicative | `ConqueredDamage`, Sovereign Blade | no: only on the switch-off branch |
| CalcifyPower | ModifyDamageAdditive | `OstyAttackDamage` | no, but once by the OLD site: under the switch Osty's hits name the player as dealer, so the handler (dealer the owner's pet) answers 0. Open: Osty as dealer (the Strength gap of `OstyAttackDamage`'s remarks) |
| Strength, Vigor, Weak, Vulnerable, Shrink, Flutter, Tank, Surrounded, Tracking, DoubleDamage, Soar, Tainted | additive / multiplicative | `BuffSystem.IncomingDamage` | no: its damage callers (`DealDamageToEnemy`, `EnemyAI.DealAttackDamage`) are below their F08 return; `Enemy.AnnouncedDamage` is the intent, no damage |
| Lethality, Colossus, Slow, HardToKill | multiplicative / cap | `DealDamageToEnemy`, `EnemyAI.DealAttackDamage` | no: below the F08 return |
| Intangible | cap, ModifyHpLostAfterOsty | `BuffSystem.CapIncomingDamage` / the HP-loss caps | no: below the F08 returns |
| HardenedShell, Slippery | ModifyHpLostBeforeOstyLate, AfterOsty | `LandDamageOnEnemy` | no: below the F08 return |
| Buffer, TungstenRod, BeatingRemnant, TheBoot | ModifyHpLost passes | `LosePlayerHp`, `RelicEffects.ModifyHpLost`, `LandDamageOnEnemy` | no: below the F08 returns |

Gate 1, at the shared entry: under F08 `DmgFrom` returns the base (the card's DamageVar or CalculatedDamageVar value) and
the command folds everything, which is the game's shape (the attack hands the base to the damage command and
Hook.ModifyDamage folds enchantment, additive, multiplicative and cap per hit). Every play-path caller (`Dmg`, the
calculated cards, Tinker, Hang, Sovereign Blade, Osty's card attacks) goes through it. The old body is
`ModifiedCardDamage`, which Thrash keeps calling for the card it eats either way: that is a value the game reads through
Hook.ModifyDamage with the eaten card as source, not an amount the damage command folds. Gate 2: under F08 `ShivDamage`
adds neither Accuracy nor Phantom Blades. Switch off, both are byte-identical (the same computation in the same order).
The fixture keeps the uid at the bind (`TestSwitches`), so the O1 runner sees real card sources.

## 7. Rounds and estimate

| round | model | content |
|---|---|---|
| R1 (done, `rebuild/f08-r1`) | Sonnet | F17 re-cut in `hook_families.json` (AfterCombatEnd, AfterCombatVictory, AfterCombatVictoryEarly), `Switches.g.cs`, the manifest and the by-family tables rebuilt; the two SPLITS records; `port_list --exclude-hooks` no longer needed |
| R2 (done, `rebuild/f08-r2`) | Opus | `CreatureCmd.Damage`, the decimal fold and `AttackCmd` with the `IsOverOrEnding` return and the F06 bridge; rule tests (section 1's sequences, the fold, the truncation, AfterAttack once) |
| R3a | Opus | `Kill` / `KillWithoutCheckingWinCondition` in the sequence of section 3 item 5, the two removals, `IsPerformingMove` and the move's end, the pseudo-hook ids, `Escape`, the HP commands (item 9) |
| R3b | Opus | the win check at the game's points, the old answers and in-step checks gated off, the F17 hand-off (`FinishCombat` from the host), `construct commands` with the write check |
| R4 | Opus | after K5/K6 land: the old entry points rerouted (the 42 + 33 + 19 + 7 callers through adapters returning the old ints, and section 4's write sites), the hybrid audit, fuzzer and struct_compare harness on the empty handler set |
| R5 (done, `rebuild/f08-r5`) | Sonnet | the inline bodies extracted into named functions the old sites call too (Ravenous and the other AnswerDeaths bodies, Curl Up and the LandDamageOnEnemy bodies, The Gambit and Paper Cuts in DealAttackDamage, the EndTurn bodies), byte-identical old path; done before the batches that need them |
| R6.1-R6.8 | Sonnet; Opus for monster-1 and monster-2 | the eight handler batches of section 5; each closes its pairs on the open-pair gate |
| R7 | Opus | switch-on behind the open-pair gate (0 open pairs), corpus and lockstep re-measure, by-family table |

Estimate: 15 rounds, 7 Opus (R2, R3a, R3b, R4, two monster batches, R7) and 8 Sonnet (R1, R5, six batches), about 20
reviewed-round units at Opus 1.7x. Range 13-17: 13 when R3 fits one round and the mixed batch rides with relic-1, 16-17
with one or two amends (the reviewer's 13-16). R4 waits for K5 (writer) and K6 (fuzzer); R1-R3b do not.

## 8. Risks

- **Hybrid re-entry.** A handler that calls an old command which hand-dispatches (Thorns' retaliation through
  `DealDamageToPlayer`, Inferno through `LoseHp`, a death body that deals damage) would run F08/F10 hooks by hand
  inside a kernel dispatch. Rule: under the switch every old damage/HP/kill entry is gated into Core2, so a handler's
  call re-enters the kernel; R4's audit lists every call from a handler into old code and the `commands` check fails on
  an ungated F08/F10 site.
- **Inline bodies.** Old sites hold several models' bodies inline; extraction must keep the old path identical (the
  byte-identity gate catches a slip).
- **Other families' sites calling damage.** Turn-end poison, Bomb, Combust and similar F05/F02 bodies call the old
  damage functions; under the switch these reach Core2's Damage through the gated entry, with their own family still
  old: the order around them stays the old one until that family moves.
- **Roster indices.** Removal inside the hit changes enemy indices mid-card; old code that holds an index across a
  card (multi-target loops, `ResolvePlayRepetition`'s position pairing) must read creatures, not positions (diagnosis
  F: part of the multi-enemy weight may move here).
- **IsPerformingMove.** An enemy killed during its own move (Thorns) leaves the room at once (RemoveCreature:
  BeforeRemovedFromRoom, state reset, hooks dropped) but stays in the roster until its move ends; the old engine has no
  such flag: the enemy turn site sets it (R3a).
- **Option B changes family numbers**; the by-family tables must be rebuilt before the comparison of section 6.
