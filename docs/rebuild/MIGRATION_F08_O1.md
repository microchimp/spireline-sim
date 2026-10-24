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

Off vs on: 287 tests, **281 pass both, 6 fail on only**, 0 fail off only, 0 fail both. Every suite not listed below
passes all its tests both ways.

| suite | tests | pass both | fail on only |
|---|---|---|---|
| AutoPlayDeathTests | 3 | 2 | 1 |
| FightHeldOpenTests | 3 | 2 | 1 |
| PotionKillTests | 10 | 9 | 1 |
| SelectionAnswerDeathTests | 2 | 1 | 1 |
| TestSubjectTests | 11 | 9 | 2 |

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
