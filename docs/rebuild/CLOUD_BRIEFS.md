# Cloud briefs: what the F08 migration hands to cloud sessions, and what stays local

2026-10-24, rebuild lane, fork main `2fbd17ad` (the Tier 1 kernel K1-K6, F08 rounds R1-R3b and R5 landed). Input:
`MIGRATION_F08.md` (section 3 commands, section 4 old sites, section 5 handlers and the R5 result, section 6 acceptance, section 7
rounds), `TIER1_KERNEL.md` (section 4 handlers, section 6 fuzzer), `SCENARIOS.md`, `port-f08-f10.md` (the port list, next to this file)
and `tools/kernel/port_list.py`. Plan item: `CLOUD_CREDIT_PLAN.md` item 4 (kernel handler batches the rebuild lane marks as cloud-able).

**What a cloud session has:** a clone of the fork (GitHub `microchimp/spireline-sim`), `dotnet test`, Python. **What it does not have:** the
game, its decompile, the corpus or any recording, `data/` beyond what the fork holds, the port manifest. So a CLOUD item is one whose
inputs are the migration doc, the port list and the fork's own sources, and whose acceptance is unit tests. A LOCAL item needs the
decompile, the corpus, a trace, the game, the PC's built EmuDump, or coordination with a round in flight (R4 edits the same old files).

**Verdict: separable.** 49 of the 106 reached models (60 of the 150 gate pairs) are CLOUD handler work, in six branches; the five listed repo-only
nits are CLOUD, in two branches; everything else is LOCAL. One blocking prerequisite (H0) is LOCAL and comes first.

## 0. H0 (LOCAL, blocks every handler batch): a handler cannot reach the old state yet

Found while writing these briefs; verify it first, and strike this section if a path exists. As of `2fbd17ad` every handler delegate
(`HookEvent`, `HookFold`, `HookCap`, `HookTry*`, `HookPredicate`, `HookMutate` in `Core2/HookRuntime.cs`) takes `(ListenerRef self, in TArgs a, ...)`
and nothing else: not the `KernelState`, not the `OldWorld`, not the old `CombatState`. `ListenerRef` is (kind, model id, instance), and
the args carry `CreatureRef`s (positions), not creatures or power amounts. `Register(HandlerTable)` runs once for the shared, immutable
`Kernel` (TIER1 section 4), so a static model class cannot capture a per-combat state; the existing tests close over `rt` or the combat they built
(`ActionQueueTests`, `DamageCommandTests`). Every old extracted function a handler must call takes old state (`CombatState`, `EnemyState`, a power
amount), so no real model handler can be written until the handler's access is decided. Candidate shapes, for the local session to choose
(the choice changes every delegate type and `tools/kernel/gen_hooks.py` output, so it is a design decision, with R4 touching the same dispatch):
(a) `HookRuntime rt` (or `KernelState k`) as the first delegate parameter, with `k.World`, `OldCreatures` and a power-amount accessor by
`ListenerRef`; (b) the args records carry a read-only old-state view; (c) a per-combat model instance. Once the shape is fixed and
landed on fork main, the handler items below start; the mechanical part (generator, regenerated `Hooks.g.cs`, existing tests moved to the
new shape) could itself be handed to cloud as a first item with the shape written into its brief.

## 1. Handler batches (MIGRATION_F08 section 5): the split

A model is CLOUD when **every one of its (model, hook) pairs is `registered`** (an old location to call) or `not_modelled` (no handler), and
the body it would call is not already run by a Core2 command and needs no Core2 command that is not built. A model with an open pair, a
Core2-owned body or a missing command is LOCAL as a whole (one class file, one owner). The registered pairs of mixed models may follow as a
second wave once the pilot has proven the pattern.

| batch | CLOUD models (branch) | LOCAL models, and why |
|---|---|---|
| pilot: power-5, relic-2, card-1 (8 models, 9 pairs) | Phantom Blades, Lethality, Tracking, Double Damage, Hand Drill, The Boot, Vitruvian Minion, Melancholy (`cloud/f08-pilot`) | none |
| power-1 (15) | Strength, Vulnerable, Weak, Thorns, Ravenous, Vigor, Tank, Colossus, Shriek, Flame Barrier (10 models, 10 pairs, `cloud/f08-power-1`) | Shrink (open: AfterDeath and AfterRemoved have no old location); Minion (open pseudo ShouldPowerBeRemovedAfterOwnerDeath); Inferno, Rupture (Core2's `CreatureCmd.Damage` already calls the old player-side body `CardEffects.AfterPlayerDamageReceived`; the handler takes the line over in step with R4); Knockdown (not modelled, nothing to port) |
| power-2 (15) | Slippery, Buffer, Intangible, Surprise, Possess Speed, Possess Strength, Tainted, Swipe, Gigantification (9 models, 14 pairs, `cloud/f08-power-2`) | Curl Up (its block gain is a `GainBlock` command in section 4's table, F09, not built); The Gambit (the handler must call `Kill([player])`, together with R4's gate at its old inline zeroing); Paper Cuts (needs item 9's `SetMaxHp`/`LoseMaxHp`, not built); Imbalanced, Vital Spark (open); Illusion (ShouldAllowHitting's old location is `CombatEngine.IsPrimaryEnemy`, which R5 left unextracted, and a living Illusion is primary in Core2 only: R3b's known difference) |
| power-3 (15) | Flutter, Personal Hive, Stock, Asleep, Plow, Crab Rage, Surrounded, Soar (8 models, 11 pairs, `cloud/f08-power-3`) | Slow (open: mechanism B, the fold-mul over SlowAmount, no old location); Hard to Kill (open ModifyDamageCap); Reattach, Slumber, Hex, Dampen, Sandpit (open pseudo AfterRemoved or ShouldPowerBeRemovedAfterOwnerDeath) |
| power-4 (15) | Infested, Constrict, Heist, Calcify, Accuracy, Sic 'Em, Conqueror (7 models, 8 pairs, `cloud/f08-power-4`) | Adaptable, Steam Eruption, Painful Stabs, Burrowed, Tangled, Ringing (open pseudo pairs); Hang (open); Hardened Shell (its counter `HardenedShellDamageThisTurn` is owned by Core2 in `OldState.Damage.cs`; the handler takes that line over, which edits Core2) |
| relic-1 (15) | Tungsten Rod, Strike Dummy, Pen Nib, Gremlin Horn, Miniature Cannon, Mystic Lighter, Fake Strike Dummy (7 models, 8 pairs, `cloud/f08-relic-1`) | Self-Forming Clay, Centennial Puzzle, Demon Tongue, Red Skull (same Core2 call of the player-side body as Inferno and Rupture); Beating Remnant (its AfterDamageReceived tally `UnblockedDamageThisTurn` is Core2-owned); Lizard Tail (player-death prevention, coupled to `Kill`, `PlayerIsDead` and R4); Meat on the Bone, Lava Lamp (open) |
| monster-1 (15), monster-2 (7) | none | All 22 monsters: every pair is open (no old location; 19 + 8 pairs, most pseudo-hooks `BeforeRemovedFromRoom`/`Died`, the rest AfterDeath, AfterCurrentHpChanged, BeforeDeath of the Decimillipede segments, Crusher, Rocket, Vantom, Aeonglass, Soul Fysh, Kin Priest, The Insatiable, Waterfall Giant, Lagavulin Matriarch, Test Subject, Queen); the bodies sit in `EndTurn`, `ExecuteIntent` and the factories and are read from the decompile; Opus |
| potion-1 | none | Fairy in a Bottle: AfterPreventingDeath open; ShouldDie's location is the slot lookup `PlayerIsDead` and `IsAboutToLose` share (R5 left it) |

Pair arithmetic against the gate (150 pairs without a handler): 60 CLOUD (pilot 9, power-1 10, power-2 14, power-3 11, power-4 8, relic-1 8),
41 registered pairs of LOCAL models, 49 open pairs (22 hook pairs, 27 pseudo-hook pairs; no old location, each is read in its batch). Five
not-modelled pairs need no handler.

## 2. Everything else (MIGRATION_F08 sections 3, 7; TIER1 section 7; PLAN P0)

| item | verdict | reason |
|---|---|---|
| H0, handler access to the old state | LOCAL | design decision on the delegate shape (section 0) |
| R4, rerouting the old sites (42 + 33 + 19 + 7 callers, the 86 HP/block write sites, the hybrid audit, fuzzer and struct_compare on the empty handler set) | LOCAL | runs locally now; edits `CombatEngine.cs`, `RelicEffects.cs`, `CardEffects.cs`, `EnemyAI.cs`, `BuffSystem.cs` and re-records `sites.json` hashes |
| R7, switch-on (`migrated_families.json`), lockstep re-measure, by-family table | LOCAL | needs the corpus, the lockstep records and the open-pair gate at 0 |
| Item 9, the HP commands (`Heal`, `SetCurrentHp`, `SetMaxHp`, `GainMaxHp`, `LoseMaxHp`) | LOCAL | which hooks each game method dispatches is read from the decompile; once the local session writes that spec per method, the commands are CLOUD-able |
| The 49 open pairs, the 22 monsters, Fairy in a Bottle, the Core2-owned and command-needing models of section 1 | LOCAL | no old location, a decompile reading, or a Core2 edit shared with R4 |
| `construct pauses` and `construct rng` beyond the kernel's own items; `port_check`'s game side | LOCAL | read each game override's awaits, RNG streams and enqueue-or-await use from the decompile |
| P0 local tables (`corpus.py identity` and `rng_diff` 0 differences after them); the L coverage set (D2 `game_order_key` against the game's `L` records) | LOCAL | corpus, recordings and a traced game |
| The 8 untriaged `fuzz_core2.py --switches F08,F10` fights (combat-end fields, Phase, rewards, HP) | LOCAL | triage needs EmuDump runs and decompile readings |
| `python -m tools.kernel.port_list --gate`, `EmuDump kernel-handlers`, byte proof, corpus, fuzz, `construct shape/commands`, the decompile review | LOCAL | manifest, decompile, built EmuDump, corpus; they are the gate of every CLOUD branch |
| Repo-only cleanups C7, C8 (section 4) | CLOUD | listed review nits; sources and unit tests only |
| The "caller keeps the condition" doc nit (C9) | LOCAL until R4 lands | the three functions sit in files R4 edits, two of them carry the `sites.json` split-record hashes |

## 3. Handler items (C1 pilot, C2-C6): shared rules

**Setup.** Start from fork main once H0 is on it. Branch `cloud/f08-<name>`, PR to `main`, one branch per batch row above, the pilot first (it
proves the pattern and is the only batch that restructures `Registry.cs`; the other five wait for its landing). Switch nothing on:
`migrated_families.json` and `Switches.g.cs` stay as they are, so the handlers are inert in every engine run and live only in unit tests
(R7 flips the switch). No game text, no decompiled text in code, comments or the PR: identifiers (model ids, hook names, our function names).

**Files.** New: `src/Sts2Emulator/Core2/Models/<Kind>/<GameClass>.cs` (`Power`, `Relic`, `Card`; one static class per model implementing
`IMigratedModel`, TIER1 section 4), and the batch's test file `src/Sts2Emulator.Tests/Core2/Models/<Batch>Tests.cs`. Edited: `Core2/Models/Registry.cs`
only (the pilot turns `All` into one block per batch; later branches append their block, and the local session resolves the textual
conflict by keeping every block). **Nothing else**: not `src/Sts2Emulator/Core/` (the old engine: any edit can move the lines `sites.json` and the port list
record, change the file hashes of the split records, and needs the byte proof), not `Core2/Commands/`, not generated files, not `tools/kernel/*`, not `hook_families.json`.
`git diff --stat main...HEAD` must show exactly those.

**Shape of a model class.** `Register(HandlerTable t)` does `t.Model(ListenerKind.<Kind>, "<GameClass>").On(HookId.<Hook>, <handler>)` per pair. The
game class name is the key in `data/id_map.json` for the row's model id (`power:STRENGTH_POWER` to its `powers` entry; add `power_aliases.json`
if `OldModelIds.Resolve` throws). One method per game override, named like the override, with a doc comment `game: <Class>.<Method>`.
The HookId is the row's hook, exactly: Early and Late passes are their own ids (`ModifyHpLostAfterOstyLate`, `ModifyHpLostBeforeOstyLate`,
`ShouldDieLate`); the delegate type is the one the pass op in the `Hooks.g.cs` doc comment of that hook implies (fold-add and fold-mul
`HookFold<ModifyDamageArgs, decimal>` returning the addend or factor, cap `HookCap`, events `HookEvent` returning `HookStep.Done`, `Should*`
`HookPredicate`). Values stay `decimal`; the cast is the damage command's. The handler calls the old function the row names, with
the **condition the old call site has**: R5 left the `if` at the call site and put what was inside it in the function, so read the call
site (its line is in the row) and put the same condition in the handler. Where the row names a function whose name differs from its hook
(Slippery's AfterDamageReceived row names `Slippery_AfterModifyingHpLostAfterOsty`) or two locations, register the row's hook, call what the
row names, and say so in the PR. Order among listeners of one pass is the dispatcher's (K2), not the handler's.

**Stop rules (report, do not fix).** If the named function is missing, `private`, still inline in a shared site, or its signature does not fit
the handler's arguments, skip that model and list it in the PR: extraction and visibility are old-engine edits and need the byte proof. If a
body deals damage, loses HP or kills (Thorns and Flame Barrier retaliate; read every other body for it),
call it as it is and list it as a **re-entry** in the PR (R4's hybrid audit gates the old entry; MIGRATION_F08 section 8); do not gate it.

**Tests (all must pass).** `dotnet test src/Sts2Emulator.Tests` green, the count the branch point's plus the new tests. Per model, in the batch's test
file, with the F08+F10 test switch (`new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => Ids, [Registry...])`, as `KillCommandTests` builds
it): (1) every pair of the model is in `HandlerTable.Registered()` for the row's hook and none is a `[StubHandler]`; (2) behaviour: dispatch the
hook through `Hooks.<Dispatcher>(k, args)` (or `CreatureCmd.Damage` for the damage and HP-loss folds) on a combat that holds the model, and the result and
the `changed` set equal what the old function returns for the same state when called directly; (3) delete-mutant: the same dispatch with
the model's registration left out gives the no-handler value and a different result, so deleting the handler fails the test; (4) with F08
off the model's pairs are absent from the table. The batch also has one table test: the (model, hook) set of `Registry.All` for the
batch equals the batch's pair list written out in the test (identifiers only), which stands in for `EmuDump kernel-handlers` and
`python -m tools.kernel.port_list --gate --families F08 F10` (these need the manifest, the decompile and a built EmuDump, so they are local; run
`EmuDump kernel-handlers out.json` only if EmuDump builds in the cloud). Also run `python -m tools.kernel.gen_hooks --check` and
`python -m tools.kernel.sites check` (0 problems: the sites' lines and the split-record hashes are intact, so no old file changed).

**What the local gate checks after, for every handler branch.** `git diff --stat` is only the files above; `python -m tools.kernel.port_list
--gate --families F08 F10` lists the batch's pairs as ported (the gate count falls by the batch's pair count) via `EmuDump kernel-handlers`; byte
proof in all 11 modes over the corpus, `old` against `--b-args "--engine core2"`, 0 differ (the handlers are inert with no switch), and
`corpus.py identity`; `sites check`, `construct shape`, `construct commands`, `gen_hooks --check`, the C#, EmuDump and Python suites; a decompile
review of each handler's shape (pass and delegate, early or late, the condition, `decimal` against int, the amount and where the game truncates) and
of the effect against the game's override, not against the old engine; `fuzz_core2.py --switches F08,F10 --fights 300` (informative only once
R4's reroutes land: until then Core2's commands are called from no old site, so the handlers run in unit tests alone and the decompile review is the
substantive gate). A branch is landed commit by commit, rejecting only the commit that breaks a gate. The PC runs nothing in the quiet window
2026-10-24 10:00Z-11:45Z.

### C1 pilot: `cloud/f08-pilot` (8 models, 9 pairs)

Inputs: the port list rows power-5, relic-2, card-1 and MIGRATION_F08 sections 3 item 3 (the decimal fold and its named steps) and 5. Old functions:
`CardEffects.PhantomBlades_ModifyDamageAdditive` (ModifyDamageAdditive), `CardEffects.Lethality_ModifyDamageMultiplicative`,
`BuffSystem.Tracking_ModifyDamageMultiplicative`, `BuffSystem.DoubleDamage_ModifyDamageMultiplicative`,
`RelicEffects.VitruvianMinion_ModifyDamageMultiplicative`, `RelicEffects.BootDamageFloor` (The Boot, two pairs:
AfterModifyingHpLostAfterOsty and ModifyHpLostAfterOstyLate), `CardEffects.CheapenMelancholyForDeath` (Melancholy, AfterDeath event), and Hand Drill's
AfterDamageGiven, whose old location is the shared `RelicEffects.ApplyAfterBlockBroken` (no `HandDrill_*` function exists: if the body is inline with
other relics' bodies, apply the stop rule and leave Hand Drill out). Files: the eight classes (`Core2/Models/Power/{PhantomBladesPower,
LethalityPower, TrackingPower, DoubleDamagePower}.cs` by the game class names in `id_map.json`, `Core2/Models/Relic/{HandDrill, TheBoot,
VitruvianMinion}.cs`, `Core2/Models/Card/Melancholy.cs`), `Registry.cs` restructured to one block per batch, `src/Sts2Emulator.Tests/Core2/Models/PilotTests.cs`. The pilot's first duty is the pattern: one folder layout, one test helper
(`ModelTestKit`: build a combat holding a power, relic or card, bind a kernel with the test switch, dispatch) that the five later batches reuse
unchanged. Tests and local gate: as the shared rules; the pilot's gate also reviews the test helper and the `Registry.cs` layout before the others start.

### C2: `cloud/f08-power-1` (10 models, 10 pairs)

Inputs: port-list rows power-1. Old functions: `BuffSystem.Strength_ModifyDamageAdditive`, `Vigor_ModifyDamageAdditive`,
`Weak_ModifyDamageMultiplicative`, `Vulnerable_ModifyDamageMultiplicative` (its second location, `EnemyAI.DealAttackDamage`, is the old enemy-attack
fold: the handler ports the one fold, not both), `Tank_ModifyDamageMultiplicative`; `EnemyAI.Colossus_ModifyDamageMultiplicative`;
`CombatEngine.Ravenous_AfterDeath` (AfterDeath event); Thorns' BeforeDamageReceived: `CardEffects.TriggerEnemyThorns` (enemy-owned) and
`EnemyAI.ApplyPlayerThorns` (player-owned), one handler choosing by the owner's side as the two call sites do (re-entry: retaliation damage);
`EnemyAI.TriggerShriekIfWounded` (Shriek, AfterDamageReceived); `EnemyAI.ApplyPlayerFlameBarrier` (Flame Barrier, AfterDamageReceived; re-entry). Files:
ten classes under `Core2/Models/Power/`, `Registry.cs` (one block), `src/Sts2Emulator.Tests/Core2/Models/PowerOneTests.cs`. Tests: shared rules, plus the
integer-versus-decimal check for the damage folds (a handler returns the addend or the factor as a `decimal`; the one cast to int stays in `CreatureCmd.Damage`). Local gate: the shared
list, and the review reads Thorns' and Flame Barrier's re-entry against R4's audit list. Excluded and why: section 1.

### C3: `cloud/f08-power-2` (9 models, 14 pairs)

Inputs: rows power-2. Old functions: `CardEffects.Slippery_ModifyHpLostAfterOsty`, `Slippery_AfterModifyingHpLostAfterOsty` (see the row-name rule above:
Slippery's registered hooks are AfterDamageReceived and ModifyHpLostAfterOsty), `Buffer_AfterModifyingHpLostAfterOsty` and `Buffer_ModifyHpLostAfterOstyLate`,
`BuffSystem.CapIncomingDamage` and `CapHpLoss` (Intangible: ModifyDamageCap, ModifyHpLostAfterOsty, AfterModifyingHpLostAfterOsty; its AfterModifyingDamageAmount is not
modelled), `CombatEngine.Surprise_AfterDeath` and `Surprise_ShouldStopCombatFromEnding`, `PossessSpeed_AfterDeath`, `PossessStrength_AfterDeath`,
`BuffSystem.Tainted_ModifyDamageAdditive`, `CombatEngine.Swipe_BeforeDeath`, `CardEffects.Gigantification_ModifyDamageMultiplicative`. Files: nine classes
under `Core2/Models/Power/`, `Registry.cs` block, `PowerTwoTests.cs`. Tests: shared rules; Intangible's cap handler returns a `HookCap` (the pass keeps the
running minimum); `Surprise_ShouldStopCombatFromEnding` is also exercised through `CombatEnd` (R3b's boundary check) to show the held fight stays open, which
closes one of R3b's known differences for Surprise. Local gate: the shared list, plus the decompile review of the Intangible cap and the HP-loss passes against R2's
fold tests.

### C4: `cloud/f08-power-3` (8 models, 11 pairs)

Inputs: rows power-3. Old functions: `BuffSystem.Flutter_ModifyDamageMultiplicative` and `CardEffects.Flutter_AfterDamageReceived`, `CardEffects.PersonalHive_AfterDamageReceived`,
`CombatEngine.TryRespawnAxebot` (Stock, AfterDeath) and `Stock_ShouldStopCombatFromEnding`, `EnemyAI.WakeMatriarchIfStruck` (Asleep; the same function serves the LOCAL Lagavulin
Matriarch, so the handler passes the Asleep case only), `EnemyAI.TriggerPlowIfWounded`, `CombatEngine.TryEnrageCrabPartner` (Crab Rage), `BuffSystem.Surrounded_ModifyDamageMultiplicative`
with `CombatEngine.FacePast` and `TurnPlayerToFaceSurvivor` (Surrounded, AfterDeath), `BuffSystem.Soar_ModifyDamageMultiplicative`. Files: eight classes under
`Core2/Models/Power/`, `Registry.cs` block, `PowerThreeTests.cs`. Tests: shared rules; Stock's AfterDeath keeps the corpse rule of R3a (`AxebotDeathStrips`: the kernel keeps Stock
for `TryRespawnAxebot`), so the test runs it through `CreatureCmd.Kill` with an Axebot-shaped combat as `KillCommandTests` does. Read the `Surrounded_ModifyDamageMultiplicative`
body, not its doc comment (the comment says the caller keeps the condition; C9). Local gate: the shared list; the review checks Stock's respawn HP writes against R4's list.

### C5: `cloud/f08-power-4` (7 models, 8 pairs)

Inputs: rows power-4. Old functions: `CombatEngine.Infested_AfterDeath` and `Infested_ShouldStopCombatFromEnding`, `Constrict_AfterDeath`, `Heist_BeforeDeath`
(with `SpawnGremlinMercReinforcements` at its second location), `CardEffects.Calcify_ModifyDamageAdditive`, `Accuracy_ModifyDamageAdditive`, `SicEm_AfterDamageGiven`,
`CardEffects.ConqueredDamage` (Conqueror, ModifyDamageMultiplicative; an own function, not a `<Model>_<Hook>` one). Files: seven classes under `Core2/Models/Power/`,
`Registry.cs` block, `PowerFourTests.cs`. Tests: shared rules; `Infested_ShouldStopCombatFromEnding` through `CombatEnd` as in C3. Local gate: the shared list; the review
reads Heist's BeforeDeath spawn against the kill sequence (step 4) and Constrict's AfterDeath against the dying-window mark.

### C6: `cloud/f08-relic-1` (7 models, 8 pairs)

Inputs: rows relic-1. Old functions: `RelicEffects.TungstenRod_ModifyHpLostAfterOsty` (ModifyHpLostAfterOsty and AfterModifyingHpLostAfterOsty; the second location,
`RunNonCombatEffects.LoseHp`, is run-level HP loss outside the combat code, which R5 left: the handler covers the combat location only), `StrikeDummy_ModifyDamageAdditive`,
`FakeStrikeDummy_ModifyDamageAdditive`, `PenNib_ModifyDamageMultiplicative`, `MiniatureCannon_ModifyDamageAdditive`, `RelicEffects.EnchantedCardDamageBonus` (Mystic Lighter,
ModifyDamageAdditive; compare the named step `EnchantAdd` in `Core2/Steps.cs` so the enchantment bonus is not applied twice), `CombatEngine.GremlinHorn_AfterDeath`. Files: seven
classes under `Core2/Models/Relic/`, `Registry.cs` block, `RelicOneTests.cs`. Tests: shared rules; Tungsten Rod's pair is exercised through `CreatureCmd.Damage` (the fold after
Osty, with the truncation at the HP loss). Local gate: the shared list; the review checks Mystic Lighter against `EnchantAdd` (shape) and the Tungsten Rod floor against the game.

## 4. Repo-only cleanups (CLOUD, one branch each, tests only)

### C7: `cloud/f08-tooling-nits`

The review nits that touch only tooling. Inputs: this list. (1) `tools/emudump/Engines/Old/ExtractedBody.cs` (`Is`: an underscore after an upper-case initial, an upper-case letter after
it, an old-engine namespace) and `tools/fidelity/hook_table.py` (`callees`, which follows `<Model>_<Hook>` helpers) must name the same set of functions: add one test in each
suite that pins the set of extracted names (read by reflection in `tools/emudump.Tests`, by a source scan in `tests/python/test_hook_table.py`) against a literal list, so a function the
two predicates would judge differently fails; if the predicates differ, make the Python one match the C# one. (2) `tools/emudump/Engine/IEngine.cs`, the doc of `StateFieldHashes`, names the old
engine's field prefix ("ActiveCombat."): reword it engine-neutrally (the combat's fields carry a prefix the engine chooses; callers treat the names as opaque). (3) the flag tuple
`("--rng-log", "--trace", "--trace-to")` is written twice, as `LABELLED_FLAGS` in `tools/fidelity/packet.py` and inline in `referee/trace_divergent.py`: define it once next to `NO_INLINING` in
`tools/fidelity/rng_diff.py`, import it in both, and add a test that the two modules hold the same object. (4) `tools/emudump/HookProbe.cs`: a missing or unreadable scenarios file (and an uninstrumented
build) must write the usage or reason to stderr and return 2, as the other bad calls do, not throw from `File.ReadAllText`; test in `tools/emudump.Tests`. Tests: `dotnet test` on both
test projects and the Python tests it touches (`tests/python/test_hook_table.py`, `test_packet.py`, `test_byte_proof.py`, `test_kernel_port_list.py`). Local gate: no old-engine file changed
(`sites check` 0 problems), byte proof unchanged (the tooling is not in the engine), and the nit list in `MIGRATION_F08.md` ticked.

### C8: `cloud/f08-scenario-setters`

`IEngine` setters for the player's block, the enemies' HP and block, so a scenario can express them (SCENARIOS.md: "Not expressible yet: block and enemy HP/block"). Inputs: `SCENARIOS.md`
(the format, the setup keys, the setup order relics, deck, potions, HP, the fight, then the combat items), `tools/emudump/Engine/IEngine.cs`, `tools/emudump/Setup.cs`, `tools/emudump/Scenario.cs`,
`tools/fidelity/scenarios.py`. Files: new setup members on `IEngine` implemented by `Engines/Old/OldEngine.cs` (and so by `Core2Engine`, which derives from it), new setup items `block` and
`enemy_hp`/`enemy_block` in `Setup.cs` run with the other combat items, the format keys in `Scenario.cs` and its format check, and `SCENARIOS.md`. Tests: `tools/emudump.Tests/ScenarioTests.cs`
(format, setup translation, an unknown enemy slot, a negative value, a value above max HP is refused or capped as the doc says), one scenario under `tests/scenarios/` that sets enemy HP and block
and passes on both engines, and `tests/python/test_scenario_runner.py`. No `src/Sts2Emulator/Core/` file may change (the setters write the state through the old fields from the tool side).
Local gate: `EmuDump scenario tests/scenarios` unchanged for the 23 existing scenarios (14 pass, 9 xfail, on old and on core2), byte proof untouched, and the new scenario's expectation read against
the game's rule, not the engine's.

### C9 (LOCAL until R4 lands): the "caller keeps the condition" doc nit

The doc comment "Extracted from <site>; the caller keeps the condition" is untrue for `BuffSystem.Surrounded_ModifyDamageMultiplicative`, `CombatEngine.Ravenous_AfterDeath` and
`RelicEffects.RedSkull_AfterCurrentHpChanged` (each function holds a condition itself). Reword the three comments in place, keeping the line count. It is LOCAL because `CombatEngine.cs` and
`RelicEffects.cs` carry the `sites.json` split-record file hashes (any edit makes `sites check` fail until the two parts are re-read and the hash recorded) and R4 is editing both files.
