# Byte proof on the old engine (R6)

2026-10-24, branch `rebuild/r6-proof` (from `2ecdf207`). `rebuild-plan.md` section 1, "Proof on old first": after the
oracle tools moved onto `IEngine`, the old engine's results must not move by a byte before Core2 plugs into the same
adapters. This file records the proof, surface by surface. The local tag `old-p0` (`97b03d1e`, the last main before the
moves) marks the baseline; it is not pushed.

## Builds compared

- A: fork main `ccb04363` (R2 `IEngine` + Fight and R3 Reanchor already in it). Release `EmuDump`, built in a scratch
  worktree at that commit (`dotnet build tools/emudump/EmuDump.csproj -c Release`, offline from the NuGet cache).
- B: `rebuild/tier1-kernel` at `2ecdf207`: A plus R4 Setup/NetState, R5 HookProbe/Trace/RngLog/StateWalk, K2b (the
  per-combat card uid on `CardInstance`) and the L patch. Release `EmuDump` and `out/` from a finished
  `tools/build_and_test.sh` gate (exit 0), clean checkout.
- Both are the JIT builds (`EmuDump.dll`). The judging code is one tree's for both sides: `tools/fidelity` of B, run
  from a worktree of `2ecdf207` with only this file added.

## Surfaces

| Surface | A vs B | Command | Counts | Result |
|---|---|---|---|---|
| Corpus fights: `fight`, `--netstate`, `--state-hash=fields`, `--rng-log`; `reanchor`, `--scramble`, `--stale all`, `--perturb-moves`, `--trace` | outputs as text, per fight | `python tools/fidelity/byte_proof.py --a <A dll> --b <B dll>` (earlier rounds, run by the rebuild lane) | 9 modes x 4,055 corpus fights | 0 differences |
| Refset v2 scores (whole set: realistic, coverage, regression, whole runs) | stored scores | `python -m tools.fidelity.refset_score --version v2 --subset all --emu <tree> --codex --procs 7 --label <label>`, then `refset_score compare <A score> <B score>` | 996 fights judged on each side; 676 identical on A, 676 on B | gate: ok: 0 worse, 0 unsound, 0 explained, 0 better; 0 fight verdicts differ; headline, per-cell table, coverage, regression, open bug classes, whole runs and exclusions equal: True |
| Codex walk, per run (inside the score; each side walked its own tree, none reused) | `per_run` of the two scores | same command | 20294 runs walked of 20381 in the set; 15092 complete, 19274 judged | 0 runs differ (status, floors walked, stop class and component); components equal: True, stops equal: True |
| Traced `netstate_diff` | `EmuDump fight --netstate` output per fight, then the `netstate_diff` text and exit code | per traced recording: `EmuDump fight <recording> --netstate` with each build, then `python -m tools.fidelity.netstate_diff <recording> <output>`; both builds run with `rng_diff.NO_INLINING` | 1120 traced recordings (12060 fights, 2322 MB) from the referee's trace jobs and the lab's packs | 12060 of 12060 fight outputs byte-identical (1120 of 1120 whole outputs); 1120 of 1120 `netstate_diff` texts and exit codes identical; 0 problems, 0 harness exceptions, 0 emulator exits |

The refset record is the current `refs-all` record of v2. Scores are stored as `all-rebuild-a-ccb0436` and
`all-rebuild-b-2ecdf20` (not committed). A score takes 19.0 (A) and 23.8 (B) minutes
with the Codex walk, 7 emulator processes at BelowNormal.

### Refset numbers (same on both sides)

Per part (specs, kept, excluded, identical, steps verified of transitions): {"realistic": {"specs": 600, "kept": 597, "excluded": 3, "identical": 368, "verified": 15840, "transitions": 17427}, "coverage": {"specs": 283, "kept": 280, "excluded": 3, "identical": 202, "verified": 5033, "transitions": 5575}, "regression": {"specs": 119, "kept": 119, "excluded": 0, "identical": 106, "verified": 1918, "transitions": 1937}}.
Regression classes passing: {"emulator": [45, 56], "driver": [1, 1]}. Coverage (clean, items, diverging, not reached) per kind:
{"card": [122, 276, 144, 10], "encounter": [68, 80, 12, 0], "relic": [15, 245, 91, 139], "potion": [34, 50, 15, 1]}.

### Codex walk

20294 runs of 20381 walked per side; 15092 complete; 19274 judged (complete, or stopped by an
emulator difference); 5202 runs stop before the end, each at the same floor with the same class and component on both sides
(the first-divergence floor per run is `per_run[index][1]` in the stored scores). Stops by class: {"EMULATOR": 4182, "WALKER": 652, "PROFILE-DEPENDENT": 249, "FIGHT-DEPENDENT": 119};
walker skips: {"other character": 272539, "other ascension": 65976, "not single-player": 244964, "not standard mode": 12441, "mod data in the record (modded)": 33195, "modded content id (modded)": 620, "modifiers": 3512, "modded act (modded)": 99, "unknown relic (modded)": 75, "unknown card (modded)": 12, "ascension outside the engine": 6}.

### Traced netstate

The set is every recording the referee's trace jobs and the lab's packs hold with the checksum tap's full state
(`net` on its steps): 1120 files, 12060 fights. `netstate_diff` over A's output and over B's output
printed the same text and exited with the same codes in 1120 of 1120 files; it judged
12060 fights, 9486 of them not identical or not comparable (these are recorded
divergences, the same ones on both sides). Exit codes (A, B): {"[1, 1]": 1076, "[0, 0]": 44}. Run time: done 1008 recordings in 11.6 min.
Error texts are compared up to their first line (stack traces name build paths); 0 fights had one cut.

## Deterministic JIT

The rng log and the trace name each draw's caller from the stack, and which engine frames tiered JIT has inlined by a given
draw depends on timing. With the default tiered JIT one build against itself differed in 5 of 4,043 reanchor `--trace`
fights and 14 `rng-log` fights. So `byte_proof.py` and the traced run above set `DOTNET_TieredCompilation=0`,
`DOTNET_JITMinOpts=1` and `DOTNET_TieredPGO=0` (`rng_diff.NO_INLINING`): the same IL then gives the same frames every run.
The refset scores and the Codex walk use the scorer's own launch (plain `fight`, `reanchor` and `codex`, which read no
stack labels) and need no such setting.

## Not covered here

`hookprobe` runs only on the instrumented scratch build `hook_order.py` makes of a whole tree; it is proved by comparing
`python -m tools.fidelity.hook_table order --json` between the trees, not by this file. A is not `old-p0`: R2 and R3 are
inside it, and their own proof against `old-p0` is the earlier rounds' byte-proof numbers above.
