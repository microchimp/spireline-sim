# The Tier 1 kernel in the native library (step N1, owner decision D31)

**Size: fits.** Five pieces in 5-6 reviewed rounds (section 12), inside the 4-6 of the plan (HANDOFF, 2026-10-25 06:00Z).
Piece 1 is built with this note; pieces 2-5 are later rounds.

The native library (`out/Sts2Emulator.dll`, NativeAOT; `src/sts2_gym/native.py` loads it) is what the agent's search,
the lab's self-play and the lockstep scorer run. Today it is the old engine only: `RunNativeExports` builds `RunEngine`
with no kernel, and the combat-handle API (`NativeExports`: `Sts2_Create`, `Sts2_Clone`, `Sts2_Step`,
`Sts2Run_ForkCombat`, `Sts2_LoadCombatSnapshot`) steps `CombatEngine` directly. EmuDump's `--engine core2`
(`tools/emudump/Engines/Core2/Core2Engine.cs`) is the old engine plus the kernel of TIER1_KERNEL.md, switched by
`tools/fidelity/migrated_families.json` (F08+F10 today). N1 makes the native library able to run the same kernel, as an
explicit option, without moving anything that runs old today.

## 1. Rules (D31)

- **Old by default.** A process that does not ask for core2 runs the old engine byte for byte. The lab's self-play, the
  promotion arms and the train job keep old; nothing in `jobs.json` changes.
- **Explicit option.** The lockstep scorer (`referee/envproto.py play`) and `rl/gen0.py score` gain `--engine core2`.
- **No silent old.** With core2 on, a path the kernel does not reach yet is refused with its reason, never run old.
- **One recipe, two front ends.** The native core2 and `EmuDump --engine core2` build the same kernel from the same
  files; piece 5 gates their identity over the corpus.

## 2. Pieces, files and order

| # | Piece | Files | Round |
|---|---|---|---|
| 1 | Engine option; kernel built in the library; run API's `Sts2Run_Step`/`Sts2Run_StartCombat` lifted; every other combat path refused under core2 | `Interop/NativeEngine.cs` (new), `Interop/RunNativeExports.cs`, `Interop/NativeExports.cs`, `Sts2Emulator.csproj` (embedded files), `src/sts2_gym/native.py`, `referee/envproto.py`, `rl/gen0.py`, tests | R1 (this round) |
| 2 | Combat handles through the kernel's entry gate; `Sts2_Clone`/`Sts2Run_Clone`/`Sts2Run_ForkCombat` carry the engine and the per-combat `KernelState` | `NativeExports.NativeCombat`, `RunEngine.Clone`/`ForkActiveCombat`, a standalone gate (section 5) | R2 (Opus) |
| 3 | Snapshot load binds a fresh kernel (refused while a hook action is paused); the native re-anchor binds the anchored combat | `NativeExports.LoadCombatSnapshot`, `CombatSnapshot` (the refusal), `RunNativeExports.ReanchorFight*` | R3 (Opus) |
| 4 | Screen answers on combat handles and loaded snapshots reach the kernel's paused queue (keyed answers) | `NativeExports.StepOrRefuse`, `HookChoiceReplay` keyed lookup through the handle | R4 (Opus) |
| 5 | Gates: native vs `EmuDump --engine core2` over the corpus; `ResampleLeakTests`/fork tests under the switch; the speed benchmark; the 200-seed smoke | `tools/fidelity/native_parity.py` (new), tests, a bench run, a lab smoke | R5 (+ R6 for the smoke and review fixes) |

Order: 1, 2, 3, 4, 5. Pieces 2-4 each lift exactly the refusals piece 1 leaves, so a round that lands a piece removes its
refusal and adds its test; the smoke (piece 5) needs 2-4, because the search forks runs and fights and the lockstep
re-anchors at every fight.

## 3. How the option reaches the library (piece 1)

- **Native:** `Sts2_SetEngine(spec, len)` (UTF-8): `old`, `core2` (the shipped family list) or `core2:F08,F10` (the
  families named; a group is still all or nothing, `KernelOptions`). Returns 0, or -1 with `Sts2_GetLastError` saying
  why (an unknown spec or family). It sets the engine of the handles made after it; a handle keeps the engine it was made
  with (a run holds its `Kernel`, a combat its `KernelState`), so nothing already live changes rules mid-way.
  `Sts2_GetEngine(buf, cap)` returns the canonical name (`old`, `core2:F08,F10`) for provenance. The library never reads
  the environment: a C# test or an embedding host sets it explicitly.
- **Python, per process:** `native.py` reads `STS2_ENGINE` once, when it loads the library, and calls `Sts2_SetEngine`
  with it before any handle exists (unset or `old`: no call). A refused spec raises at import: a block never starts on
  the wrong engine. `native.engine()` reads `Sts2_GetEngine`. The env var is inherited by every `spawn` worker and by
  `envproto play`'s subprocess, so one flag on the parent reaches every process that loads the library.
- **Front ends:** `envproto.py play --engine {old,core2}` sets `STS2_ENGINE` before its workers spawn and records the
  engine in `summary.json`; `gen0.py score --engine` passes it to envproto, sets it for the emulator arm it plays, records
  it, and pairs only with an `--emu-dir` arm played on the same engine; `gen0.py bench-emu --engine` and `rl/bench_emu.py --engine`
  (default old) do the same for the benchmark. Every other command keeps old, and they pin it themselves, because
  `STS2_ENGINE` is inherited: `rl/run_train/cli.py` (the lab's self-play and train jobs, pinned before its first import, which
  loads `native`), `rl/bench_emu.py`, and the hand-run self-play, promotion and training mains (`rl/train.py`,
  `train_combat.py`, `evaluate.py`, `vrun_sweep.py`, `run_train/{evaluate,smoke,probe_throughput,prior_check}.py`,
  `prior_loop/coach_check.py`, each when run as `__main__`) clear the variable before the import that loads `native`. A new
  entry point that loads `native` does the same (`sts2_gym.engine_select.select_engine`, which loads no library);
  `tests/python/test_native_entry_pins.py` runs each with the variable set and fails on one that loads core2.
- **Refusals:** a kernel-on `Sts2Run_Clone` returns -3 with the reason in `Sts2_GetLastError` (as `Sts2Run_ForkCombat`), so
  `native.run_clone` reports it as a refusal and not as an unknown handle or a full pool (-1).

## 4. The kernel build in the library (piece 1)

- `NativeEngine.Parse(spec)` builds a `Kernel(options, ids, Registry.All)` as `Core2Engine.Make` does: the options from
  `migrated_families.json` (`core2`) or from the spec (`core2:...`); the ids `OldModelIds(id_map.json, power_aliases.json)`;
  the handler registrations `Core2.Models.Registry.All`. Built once per spec and shared (the kernel is immutable).
- The three files are embedded in the library (`EmbeddedResource` of `tools/fidelity/migrated_families.json`,
  `data/id_map.json`, `tools/fidelity/power_aliases.json`), the same source files EmuDump copies next to itself, so the two
  front ends cannot read different lists, and a NativeAOT library hosted in python.exe needs no file beside it.
- `core2` with an empty family list is a kernel with no switch on: no refusal and old bytes (as EmuDump's core2).

## 5. Combat handles through the entry gate (piece 2)

- `NativeCombat` gains the engine's `Kernel` and steps through a standalone gate: `RunEngine.StepCombat`'s queue run
  (`RunQueue` with a bare-stream `QueueHost`) lifted into a static `KernelGate.Step(kernel, combat, rng, action, target)`
  that `RunEngine.StepCombat` and the handle both call, so the two are one code path. With no switch on it is
  `CombatEngine.Step` (old bytes).
- `Sts2_Create`/`Sts2_Reset*` bind the fresh combat before `CombatFactory.Reset` (as `StartCombatWithDeck` does) and
  resolve the pre-play phase through the kernel's turn start.
- `Sts2_Clone`: `CombatState.Clone` already copies `KernelState` (rebound to the copy); the handle copies the kernel
  reference. `RunEngine.Clone` today drops `_kernel` (the copy's state keeps a `KernelState`, its engine has no gate: a
  silent mix); it copies `_kernel`. `ForkActiveCombat`: the fork's `KernelState` binds to its own run copy (L3b) with
  R7a's frozen deck view; the handle takes the run's kernel.
- Tests: `ResampleLeakTests` and `RunForkCombatTests` with core2; fork vs run step parity under the switch; a handle's
  step goes through the queue (a counter on `ActionExecutor.Drain`).

## 6. Snapshots (piece 3)

- `CombatSnapshot` carries no kernel state. A loaded combat binds a fresh `KernelState` (the queue empty, no hook action).
- Refused while a hook action is paused: `Sts2_GetStateJson` refuses (-3, reason) a combat whose queue holds a
  `GatheringPlayerChoice` hook action or whose screen is a kernel screen (`PendingSelection.KernelAction > 0` or a
  `HookKey`); loading a document whose `stopped_step` carries keyed answers is refused the same way. Serializing the queue
  is a later option, not N1.

## 7. The lockstep re-anchor (piece 3)

- `ReanchorFight`/`ReanchorFightHidden` call `run.BindActiveCombat()` right after the anchor, as EmuDump's
  `OldReanchor` binds the combat it anchors, so the lockstep's re-anchored fight steps through `RunEngine.Step`'s gate.
  The `kernel_on` refusal goes. Test: a re-anchored fight under core2 carries a `KernelState` and steps through the queue.

## 8. Screen answers (piece 4)

- The run API already answers through `RunEngine.QueueStep` (kernel screens via `KernelState.Answer`, old screens by the
  gathering action's resume, keyed answers inside `CombatEngine.Step`'s replay), the same path EmuDump drives.
- The combat handles and loaded snapshots answer through the gate of section 5, so a kernel screen's answer resumes the
  hook action and a screen inside old code replays with the keyed answer, never matched by order. Tests: a hook choice at
  a mid-combat pause answered through `Sts2_Step` equals the same answer through `Sts2Run_Step` and EmuDump.

## 9. Parity: native vs `EmuDump --engine core2` (piece 5)

- `tools/fidelity/native_parity.py`: for every corpus fight, one process loads the library with `STS2_ENGINE=core2`,
  re-anchors the fight from the record (`Sts2Run_ReanchorFightHidden`), steps the recorded actions and writes the
  state hash per step; EmuDump `reanchor --engine core2` writes the same per-step hashes; 0 differ is the gate (a spawn
  pool, one library per process). The same harness with `old` against EmuDump old is its own control (must be 0 too).

## 10. Speed benchmark (piece 5)

- `rl/gen0.py bench-emu --engine old` and `--engine core2` (the gpu lane's harness) on the same seconds and procs. The flag is
  the only way in: `bench-emu` clears `STS2_ENGINE` like every command, and the default is old. `bench.json` records the
  engine. Under core2 the clone and sim metrics need piece 2 (`Sts2Run_Clone`), so until then `rl/bench_emu.py --engine core2
  --metrics steps` is the run that works: decisions/s, and the per-step dispatch cost from `Prof`. Reported, not gated, unless core2 is more than 10% slower,
  which sends it to the speed lane before any switch of the lab.

## 11. The 200-seed smoke (piece 5, R6)

- `rl/gen0.py score --engine core2 --runs 200` and the same block with `--engine old` (the scorer's lockstep, the
  orchestrator's slot, not `jobs.json`): no crash, no refusal (`reanchor_fail_reasons` has no `kernel_on`, no fork
  refusal), and both divergence rates side by side.

## 12. Risks

- A path that skips the kernel and runs old rules silently (R7f-R7i's census): every combat path is refused under core2
  until its piece lands, and piece 5's native-vs-EmuDump identity is the catch-all.
- Speed: the queue and the hook dispatch per step are unmeasured in the search; piece 5 measures before the lab moves.
- A throw across the native boundary aborts the host: the run API checks the gate's precondition (an unbound fight) and
  returns -1 instead of reaching `RunEngine.Step`'s guard.
- Debug exports (`Sts2Run_DebugResolveCombat`, `DebugSetPlayer`, `DebugSetStreams`) under core2 run the run engine's
  own code on a bound fight; the replay script is not a D31 consumer, and piece 5's parity covers the lockstep's uses.
- The lab's models were trained on old dynamics; core2 shifts them a little. Measured by the smoke, not by changing the lab.

## 13. Estimate

R1 (Sonnet-sized; piece 1), R2 (Opus; piece 2), R3 (Opus; piece 3), R4 (Opus; piece 4), R5 (parity, fork safety,
speed), R6 if needed (the smoke and review fixes): 5-6 reviewed rounds, about 7-9 units, as the plan said.
