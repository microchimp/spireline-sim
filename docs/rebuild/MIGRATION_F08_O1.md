# F08 migration, round O1: the old suites under the switch (first measurement and triage)

The F08 review's recommendation 2 (docs/reviews/2026-10-24-f08-migration.md, 2.3 item 3): the old engine's behaviour
suites run once with the test switches off and once with `STS2_TEST_SWITCHES=F08,F10`, so their game-pinned cases become
switched-on regression tests. The fixture and the runner are in MIGRATION_F08.md, "The old suites under the switch".

Classes of a test that fails only with the switch on:

- **R4 to deliver**: the reroute of the damage and kill path (R4) has to make it pass.
- **handler batch (model)**: a Core2 handler of the named model is missing; its batch makes it pass.
- **old deviation (needs a trace)**: the test pins an old-engine behaviour the game does not have; update it from a trace.
- **queue gate (no model)**: none of the three: the kernel's queue or its entry gate (`RunEngine.StepCombat`), not a
  model and not R4.

The six below were first seen with the round's first fixture (bound at construction, no run state) and are unchanged
with the reviewed one (bound at the first step with `Attach(combat, run)`); the exhaust-answer probe was made with the
first.

## Suites

42 xunit classes, 287 tests. The ten the review names (15 classes: DeathCheckTests.cs is six): PotionKillTests,
TurnStartDeathTests, SelectionAnswerDeathTests, AfterPlayDeathTests, FightHeldOpenTests, AutoPlayDeathTests,
CombatEndingGateTests, CorpseRemovalTests, WaterfallGiantTests, IllusionAndMinionTests, PhrogParasiteTests,
MercPowerListTests, HeistGoldRewardTests, LethalHitHookTests, HeadbuttTests. Added (damage, death, HP, combat end):
CalculatedDamageTests, CombatEndRewardPowerTests, CombatEndsWithTheEnemyTurnTests, CorpseSlugsNormalTests,
DamageAndPowerModifierRelicTests, DamageReceivedRelicTests, DeadlyPoisonTests, DeathMarchTests, DeathbringerTests,
DeathsDoorTests, HunterKillerTests, LethalityTests, MinionDiveBombTests, MinionSacrificeTests, MinionStrikeTests,
OstyDamageSinkTests, PlayerHpLossTests, RunHpLossTests, PoisonedStabTests, TestSubjectTests, ThornsOrderTests,
TurnEndSelfDamageTests, UnpoweredDamageHookTests, TwinStrikeTests, WhirlwindTests, GremlinHornTests, FeedTests.

## Measurement 1: main (4245d9cb; H0, no reroute, `Registry.All` empty), 2026-10-24

Variable unset: the whole suite 6871 tests, 6864 passed, 7 skipped, per test identical to the tree before the fixture.
byte_proof main vs this tree (`fight,state-hash,rng-log,reanchor --limit 300`): 1200 pairs identical, 0 differ.

What was exercised: on this base `Registry.All` is empty (no model file under Core2/Models; the pilot's models reached
main after 4245d9cb), so the switched-on run exercised F10's queue and its boundary win check only, with no Core2
handler. Main as it is now (pilot merged) is to be measured with the integration tree.

Fixture as reviewed (rerun 15:45Z): a combat is bound at its first step with `Attach(combat, run)`, after the test's
setup, and every kernel-bound step is counted per test method. Variable unset: again 6871 tests, 6864 passed, 7 skipped,
0 failed (no test added or removed, so per test identical); nothing outside the test project changes, so the byte proof
above stands.

Off vs on: 287 tests, **209 pass both, 72 not covered, 6 fail on only**, 0 fail off only, 0 fail both, no crashed run.
"Not covered" passed both ways with no kernel-bound step, so it proves nothing about the switch. Every suite not listed
passes all its tests both ways with the kernel.

| suite | tests | pass both | not covered | fail on only |
|---|---|---|---|---|
| AutoPlayDeathTests | 3 | 2 | 0 | 1 |
| CombatEndRewardPowerTests | 5 | 0 | 5 | 0 |
| CombatEndingGateTests | 52 | 42 | 10 | 0 |
| CorpseRemovalTests | 10 | 1 | 9 | 0 |
| CorpseSlugsNormalTests | 8 | 0 | 8 | 0 |
| DamageAndPowerModifierRelicTests | 6 | 1 | 5 | 0 |
| DamageReceivedRelicTests | 7 | 6 | 1 | 0 |
| DeathMarchTests | 5 | 4 | 1 | 0 |
| FightHeldOpenTests | 3 | 2 | 0 | 1 |
| HeistGoldRewardTests | 7 | 2 | 5 | 0 |
| IllusionAndMinionTests | 10 | 7 | 3 | 0 |
| LethalHitHookTests | 5 | 4 | 1 | 0 |
| MercPowerListTests | 9 | 4 | 5 | 0 |
| MinionDiveBombTests | 3 | 2 | 1 | 0 |
| MinionSacrificeTests | 3 | 2 | 1 | 0 |
| PhrogParasiteTests | 5 | 3 | 2 | 0 |
| PlayerHpLossTests | 9 | 1 | 8 | 0 |
| PotionKillTests | 10 | 9 | 0 | 1 |
| RunHpLossTests | 3 | 0 | 3 | 0 |
| SelectionAnswerDeathTests | 2 | 1 | 0 | 1 |
| TestSubjectTests | 11 | 6 | 3 | 2 |
| UnpoweredDamageHookTests | 10 | 9 | 1 | 0 |

Uncovered classes: wholly CombatEndRewardPowerTests, CorpseSlugsNormalTests, RunHpLossTests; mostly CorpseRemovalTests
(9 of 10), PlayerHpLossTests (8 of 9), DamageAndPowerModifierRelicTests (5 of 6), HeistGoldRewardTests (5 of 7),
MercPowerListTests (5 of 9); partly CombatEndingGateTests (10 of 52), IllusionAndMinionTests (3 of 10), TestSubjectTests
(3 of 11), PhrogParasiteTests (2 of 5), DamageReceivedRelicTests, DeathMarchTests, LethalHitHookTests,
MinionDiveBombTests, MinionSacrificeTests, UnpoweredDamageHookTests (1 each). By the suites' shape (not read test by test), these are tests that call an engine
function on a state or an enemy (a predicate, a setup, a removal rule), read a run engine's setup, or step a run engine
outside combat: they never reach the kernel, and covering them needs a step (or a kernel-aware call) in the test, not
a wider fixture. Theory cases share their method's count, so a theory counts as covered if any case stepped.

| test (fails with F08+F10 on) | class | why |
|---|---|---|
| AutoPlayDeathTests.ATestSubjectKilledByAQueuedAutoPlayRespawns | handler batch (AdaptablePower) | the kernel's boundary win check (F10) finds no Core2 `ShouldStopCombatFromEnding` holder for the Test Subject between forms and ends the combat on the kernel's side, so the respawn turn does not run (max HP 111 where 212 is pinned); the step itself is still reported non-terminal. MIGRATION_F08.md lists Adaptable among the five old holders with no Core2 handler yet |
| FightHeldOpenTests.ATestSubjectKilledByAfterPlayDamageIsAnsweredBeforeTheStepReturns | handler batch (AdaptablePower) | as above (the respawn form's height or state) |
| PotionKillTests.APotionKilledTestSubjectRespawnsRatherThanStallingTheFight | handler batch (AdaptablePower) | as above (the respawn form's height or state) |
| TestSubjectTests.TheRespawnHeightsFollowAscension(ascension: 0, second: 200, third: 300) | handler batch (AdaptablePower) | as above (the respawn form's height or state) |
| TestSubjectTests.TheRespawnHeightsFollowAscension(ascension: 8, second: 212, third: 313) | handler batch (AdaptablePower) | as above (the respawn form's height or state) |
| SelectionAnswerDeathTests.AnExhaustAnswerThatKillsTheLastEnemyEndsTheFightAtOnce | queue gate (no model) | a probe: off, the play opens the exhaust screen and the answer is the winning step; on, the play's own `StepCombat` step comes back terminal with the enemy dead and no screen open, and the answer step is refused (`StepResult.Invalid`). Where the screen is answered inside the queue is not read yet: one probe in `RunQueue` / `ActionExecutor.Drain` (the run path, `QueueStep`, reads the host's `Terminal`; `StepCombat` returns the body's `StepResult`) |

The game side of the five: the game's AdaptablePower overrides `ShouldStopCombatFromEnding` (decompile read, not
quoted). The six show the fixture reaches the old sites' kernel and the queue: main has no reroute, so the only switched
behaviour is F10's queue and boundary win check, and the failures are exactly the held-open corpse (review 2.3 item 4's
expected disagreement) plus one queue-gate case.

## Measurement 2: the R4 scratch tree (rebuild/f08-l2 + rebuild/f08-r4): not made

`git merge --no-commit rebuild/f08-r4` into a detached rebuild/f08-l2 conflicts in
`src/Sts2Emulator/Core2/Models/Registry.cs` (both branches add registrations; git had a recorded resolution, not
applied). Per the round's rule the scratch part stopped: the merge was aborted and the scratch tree removed. Rerun on the
integration branch once R4 and the batches are merged:

    python -m tools.fidelity.suites_switch --out DIR --switches F08,F10

To watch there (the review's 1.3, passing on main because nothing is rerouted yet):
`WaterfallGiantTests.KilledByThornsOnItsOwnTurnItBlowsARoundLater` (1.3 C, the in-move death: R4 to deliver) and
`IllusionAndMinionTests.KillingTheLastPrimaryWinsWithAnIllusionStillAlive` (1.3 D, the Fogmog fight: handler batch, the
Fogmog / Illusion closer still to be read).

## Measurement 2: the merged tree (rebuild/land-oracles: main with R4, the batches C1-C6, m1/m2, L1/L2, O1 and O2), 2026-10-25

`suites_switch --switches F08,F10`: 42 classes, 291 tests: **183 pass both, 74 not covered, 34 fail on only**, 0 fail off only,
0 fail both, no crashed run. The five Test Subject / Adaptable tests of measurement 1 now pass (the AdaptablePower batch has
landed). Classes by reading each test and the game's override list for the model (not traced; "handler batch" names the model
that is not registered, or registered and not reading what the test seeds):

| class | tests |
|---|---|
| handler batch (power-1: Thorns; old bodies private, LOCAL) 8 | ThornsOrderTests (EveryHitOfAMultiHitCardFixesItsDamageBeforeItsOwnThorns, RuptureStrengthFromThornsDoesNotRaiseTheProvokingHit, ThornsStillRetaliatesOncePerPoweredHit); UnpoweredDamageHookTests (AnAoeSkipsAnEnemyThatAnEarlierTargetsInfernoBurstKilled, EchoingSlashDoesNotCountAnEnemyItsVolleySkippedAsDead, InfernoBurnsWhenAnEnemysThornsHitsThePlayer, OmnislicesSplashSkipsAnEnemyThatAnInfernoBurstKilled); WaterfallGiantTests.KilledByThornsOnItsOwnTurnItBlowsARoundLater (rerun after Thorns: the in-move death is then R4's) |
| handler batch (power-1: Flame Barrier; LOCAL) 4 | UnpoweredDamageHookTests.FlameBarrierDrawsOnAShellRefilledForTheEnemyTurn; MercPowerListTests (AnAttackerKilledMidAttackLandsNoFurtherHits for defId 1 and 37, AMercKilledByItsOwnAttackStealsNothingMore, which also needs Surprise) |
| handler batch (power-1: Ravenous; LOCAL) 1 | TurnStartDeathTests.ASlugKilledAtTurnStartLeavesTheOthersStunnedNotRerolled |
| handler batch (power-2: Surprise) 5 | AfterPlayDeathTests.StrangleFinishingALoneGremlinMercLeavesTheFightToItsReinforcements; FightHeldOpenTests.ALoneGremlinMercKilledAsTheTurnEndsSendsInItsReinforcements; MercPowerListTests (ANothingStolenMercLeavesNoHeist, KillingTheFatGremlinSpendsTheHeist, TheFatGremlinHoldsAHeistOfTheStolenGold) |
| handler batch (power-3: Stock) 1 | AfterPlayDeathTests.AnAfterPlayKillOfAnAxebotWithStockLeftBringsItBack |
| handler batch (power-4: Infested; LOCAL) 1 | AfterPlayDeathTests.PanacheFinishingALonePhrogParasiteLeavesTheFightToItsWrigglers |
| handler batch (mixed: The Bomb, Doom) 2 | CombatEndingGateTests.TheBombAheadOfLethalDoomWinsAndBehindItLoses (both cases) |
| handler batch (Heist, Swipe: registered; the kill returns 0 gold or no card for a fight the test seeds through the old `HeistGold` / `StolenCard` field only) 4 | CombatEndingGateTests.AFirstPhaseKillStillReturnsWhatTheEnemyStole (thief 28 and 92); HeistGoldRewardTests.KillingTheFatGremlinBanksItsGoldForTheRewardScreen (encounter 3 and 7) |
| R4 to deliver (the enemy-attack site's Osty redirect: the player takes the hit, Osty keeps its HP) 3 | HookListenerTests/OstyDamageSink: AHitOstyAteDoesNotCountAsOneThePlayerTook, DamageBeyondOstysHpStillReachesThePlayer, OstyTakesTheUnblockedDamageInsteadOfThePlayer |
| old deviation (needs a trace): a Minion enemy dies or stays with the last primary's death 4 | CombatEndingGateTests.ARandomTargetAttackStopsWithTheLeadersDeath (both cases), CombatEndingGateTests.NoExhaustHookRollsAfterTheKill, FeedTests.KillingAMinionFeedsNothing. Read R6.9: the game's `CreatureCmd.KillWithoutCheckingWinCondition` kills the living teammates when a primary dies and every one is secondary (MinionPower); Core2 does the same, the old engine leaves them standing (the review's 1.3 E). Not ours: the tests pin the old order and are updated from a trace with the switch-on suite |
| queue gate 1 | SelectionAnswerDeathTests.AnExhaustAnswerThatKillsTheLastEnemyEndsTheFightAtOnce (the measurement-1 probe, unchanged) |

The step test (`StepApiKillTests`, O2): on this tree the two switched-on tests failed as plain assertions in the whole-suite run
(R4 has landed, Ravenous's AfterDeath and Strength's fold have not, so outside `Pinned` the survivor reads differently too); they
are now whole-body strict expected failures, and `STS2_STEP_TEST=strict` runs them plain (Headbutt and Twin Strike both fail on the
survivor's Strength, 0 against 5). `RerouteLanded` stays false.

## Triage on f34f64db

The 15 failing cases (11 methods) of the old suites with `STS2_TEST_SWITCHES=F08,F10`, each run, read and probed on f34f64db
(temporary probes, not kept). 15 cases: old deviation 6, fixture gap 7, Core2 bug 1, missing handler 1. This supersedes the
class column of the 2026-10-25 table above for these tests (Stock, Surprise, Heist, Swipe, Thorns and Gremlin Horn are
registered now; no test of this list waits on a registered model except the Bomb and Doom case).

| test (case) | class | fix hint |
|---|---|---|
| AfterPlayDeathTests.AnAfterPlayKillOfAnAxebotWithStockLeftBringsItBack | old deviation: the test holds the dead Axebot; StockPower.AfterDeath adds a NEW Axebot (CreatureCmd.Add) and the corpse leaves | re-read `fight.State.Enemies[0]` after the play and assert its Hp and Stock 1 |
| CombatEndingGateTests.AFirstPhaseKillStillReturnsWhatTheEnemyStole (thief 28) | fixture gap: the enemy is seeded through the old `HeistGold` field with no Heist power, so HeistPower.BeforeDeath has no listener | seed BuffId.Heist next to the field (with it the probe banks 40) |
| CombatEndingGateTests.AFirstPhaseKillStillReturnsWhatTheEnemyStole (thief 92) | fixture gap: `StolenCard` seeded with no Swipe power, so SwipePower.BeforeDeath has no listener | seed BuffId.Swipe next to the field (with it the probe returns the card) |
| CombatEndingGateTests.ARandomTargetAttackStopsWithTheLeadersDeath (seed 0 rolls 3) | old deviation: the roll counts pass; the minion is dead with the last primary (CreatureCmd.Kill, the all-secondary teammates branch; MinionPower.OwnerIsSecondaryEnemy), the test pins it standing | assert the minion Hp is 0 (keep the CallCount assert) |
| CombatEndingGateTests.ARandomTargetAttackStopsWithTheLeadersDeath (seed 2 rolls 1) | old deviation: same | same |
| CombatEndingGateTests.NoExhaustHookRollsAfterTheKill | old deviation: same (CallCount 0 passes; minion Hp 0, test pins 20) | assert the minion Hp is 0 |
| CombatEndingGateTests.TheBombAheadOfLethalDoomWinsAndBehindItLoses (bombFirst true, won) | old deviation: the Bomb kills the lone enemy and the game removes the corpse from the roster (Kill removal; ShouldCreatureBeRemovedFromCombatAfterDeath), so the test reads Enemy0 of an empty list: ArgumentOutOfRangeException, stack top `List<T>.get_Item` from `Fight.get_Enemy0` (Support/Fight.cs:31) at CombatEndingGateTests.cs:584 | hold `var enemy = fight.State.Enemies[0]` before EndTurn and assert enemy.Hp 0 |
| CombatEndingGateTests.TheBombAheadOfLethalDoomWinsAndBehindItLoses (bombFirst false, lost) | missing handler (TheBombPower and DoomPower, BeforeSideTurnEnd): neither is registered, the old turn-end site orders them and the gated Damage drops the Bomb hit after the Doom kill (enemy stays 20, test pins 0) | register both handlers in the game order; read the game gate on Damage while the combat is ending (CreatureCmd.cs around lines 296 and 694) before deciding whether the test pins the old order |
| FeedTests.KillingAMinionFeedsNothing | fixture gap: a roster with no primary enemy is already ending (CombatManager.IsEnding), so the gated damage never lands (minion 1/1, fight won); not reachable in the game | add a primary enemy, Feed the minion (target 1), assert minion Hp 0 and max HP unchanged; do not assert an empty roster |
| FightHeldOpenTests.ALoneGremlinMercKilledAsTheTurnEndsSendsInItsReinforcements | old deviation: Hook.BeforeTurnEnd (the Bomb, so the Horn draw) runs before CombatManager.FlushPlayerHand, so the Horn card is flushed: hand 5, discard 1, draw 2 (test pins 6) | assert 5 and the Horn card in the discard pile, or update from a trace |
| HeistGoldRewardTests.KillingTheFatGremlinBanksItsGoldForTheRewardScreen (encounter 3) | fixture gap: hand-built Fat Gremlin with `HeistGold` 80 and `Buffs = []`, no Heist power | add BuffState(BuffId.Heist, 1) to the enemy |
| HeistGoldRewardTests.KillingTheFatGremlinBanksItsGoldForTheRewardScreen (encounter 7) | fixture gap: same | same |
| SelectionAnswerDeathTests.AnExhaustAnswerThatKillsTheLastEnemyEndsTheFightAtOnce | fixture gap: one exhaust candidate is answered inside the play step (CardSelectCmd.FromHand opens no screen when the candidates do not exceed the minimum), so the fight ends in the play and the answer step runs on a finished combat (old repeats the terminal result, Core2 StepCombat returns Invalid); with a second candidate the screen opens and the answer wins with the switch on and off. Replaces the earlier queue-gate reading | add a second card to the hand so the screen opens |
| UnpoweredDamageHookTests.OmnislicesSplashSkipsAnEnemyThatAnInfernoBurstKilled | fixture gap: the hand-made Omnislice has card uid 0; Thorns resolves the damage card source by uid (OldState.LocateCard rejects uid 0; ThornsPower.IsOmnislice sees source uid 0, no view), so Thorns never answers the splash and Inferno never bursts (player 60, struck 42, thorny 22) | stamp uids on hand-made cards in Fight.Hand or TestSwitches.Attach, as CombatFactory does |
| WaterfallGiantTests.KilledByThornsOnItsOwnTurnItBlowsARoundLater | Core2 bug (Core/EnemyAI.cs:603-614, mechanism: the STOMP riders read `enemy.MoveIndex` after the hit, but SteamEruption.AfterDeath inside the hit already forced ABOUT_TO_BLOW (Core2/Models/Powers/KeptAfterOwnerDeath.cs:119), so Weak and the 3 Steam are skipped: Weak 0, Steam 15, off gives 1 and 18) | decide inGiantRing and the Weak turn before the attack damage and apply the riders from that snapshot |

### Mechanism groups

- Seeded through an old field with no power (4 cases): both AFirstPhaseKill cases and both HeistGoldReward cases. One fixture change: the seeders add the power.
- A minion dies with the last primary (3 cases): both ARandomTarget cases and NoExhaustHookRollsAfterTheKill. One assertion change.
- A corpse or a replaced creature read through the old roster or reference (2 cases): the Axebot test and the Bomb-first crash. Hold the reference before the step, as the game removes or replaces the creature.
- A hand-made state the game does not reach (3 cases): FeedTests (no primary), SelectionAnswerDeathTests (one candidate), the Omnislice test (card uid 0). Three separate fixture edits; the uid one may belong in the fixture for every hand-made card.
- Turn-end order of BeforeSideTurnEnd against the hand flush and the Doom kill (2 cases): the Gremlin Merc test (old deviation) and the Bomb-behind-Doom case (missing handler). The Bomb and Doom handlers are the open item.
- Alone: WaterfallGiantTests (Core2 bug in the rider decision).
