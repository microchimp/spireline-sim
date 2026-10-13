# Referee runbook: find and fix emulator divergences

For an agent working unattended. Read `CLAUDE.md` and `FORK.md` first; their rules win. Goal: a
high-fidelity simulator with complete coverage. Ironclad A10 first, then every character.

## Paths (Windows PC, Git Bash)

| What | Where |
|---|---|
| Fork (our simulator) | `D:\spireline\fork\sts2`, pushes to private `microchimp/spireline-sim` |
| Game copy the referee runs | `D:\spireline\referee\game` (verified copy of `D:\Archive\sts2-0.107.1-23811903`; never touch the archive) |
| Decompiled game source (local only, never commit or quote bodies) | `D:\spireline\referee\decompiled` |
| Jobs and outputs | `D:\spireline\referee\jobs` |
| Python | `/d/spireline/src/sts2/.venv/Scripts/python.exe` (prefix commands with `PYTHONDONTWRITEBYTECODE=1`) |
| Reviewed upstream checkout (read-only, never edit) | `D:\spireline\src\sts2` |

## The loop

Shortcut for steps 1-4: `python referee/batch.py --name <name> --fights 100 --start <n>` (from the fork's root) does them all and prints a summary grouped by the first differing field. Use a new `--start` for new seeds.

1. **Make a job** (JSON). One fight per entry, all in one session:
   `{"out": "D:/spireline/referee/jobs/<name>.out.json", "runs": [{"kind": "combat", "seed": "<10 chars>", "character": "IRONCLAD", "ascension": 10, "neow": 0|1|2, "chooser_seed": <int>, "max_actions": 120}, ...]}`.
   Vary seeds, `neow` and `chooser_seed`. 50 to 200 fights per job.
2. **Run the game**: `cd referee && python run_referee.py --game 'D:\spireline\referee\game' --job '<job>' --timeout 3600`.
   It must print `"exit": 0`, `"network_connections": []`, `"real_profile_touched": false`.
   If not: stop, report, do not retry blindly.
3. **Replay in the emulator**: `dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll fight '<out.json>' > <name>.emu.json`
   (rebuild with `cd tools/emudump && dotnet build -c Release --source "$HOME/.nuget/packages"` after any engine change).
4. **Compare**: `python -m tools.fidelity.fight_compare '<out.json>' '<emu.json>'`. It prints the first diverging step per fight.
5. **Classify each difference** before touching code:
   - *Driver gap*: the game side errored or skipped a screen (e.g. `SelectLocalReward RelicReward did not finish`). Fix `referee/driver/*.cs`, never the engine.
   - *Comparer gap*: same game state, different representation (e.g. STUNNED is a buff in the emulator, a move in the game). Fix `tools/fidelity/fight_compare.py` and write the reason next to the rule.
   - *Emulator bug*: the game's own code (decompiled) shows the emulator does something else. Fix `src/Sts2Emulator`.
6. **Fixing an emulator bug** — root cause first, never a symptom patch:
   - Ask why the emulator differs: which mechanism is wrong (pool or list order, RNG stream, number or order of draws, hook timing, a shared helper)? A fix that only makes this one fight match is not done.
   - Find every other place that uses the same mechanism (grep), check each against the game, and fix them together.
   - In FINDINGS.md record the root cause and the sibling sites checked, not just the symptom.
   - Then think one level up: name the bug CLASS (hand-written orderings, a draw on the wrong RNG stream, an effect before rather than after a hook, a helper that ignores a modifier) and sweep the engine for other instances no fight has hit yet. Verify each against the decompiled game; aim the next referee batch at that class.
   - Read the game's class for that card, relic, power or monster in the decompiled tree, and the emulator's code for it.
   - Make the smallest change that matches the game; cite the game's class/method in a comment (never paste game code).
   - Add or adjust an xunit test in `src/Sts2Emulator.Tests` that fails before and passes after.
   - Re-run the failing fights; the fix must make them identical.
7. **Gate and push**: `BASELINE_FAILURES=/d/spireline/logs/baseline-python-failures.txt bash tools/build_and_test.sh /d/spireline/logs/fork-$(date +%Y%m%d-%H%M%S) /d/spireline/src/sts2/.venv/Scripts/python.exe`.
   Push only if C# shows `Failed: 0`, Python has no failures beyond the baseline (the deckfights stall test is a known 1-in-3 timing flake)
   and `corpus_check_exit` is `0`: no recorded fight regressed (section "Recorded-fight corpus" below; `5-corpus-check.log` lists any). Commit message: what diverged, the game class it was checked against, the fights that now match, ending with the attribution line the session gives. Then `git push origin main`.
8. Log every finding in `referee/FINDINGS.md` (one line: date, seed, step, cause, fix commit or "open").

## Recorded-fight corpus (regression gate)

Every fight ever recorded is test data. `tools/fidelity/corpus.py` keeps each one once (local only, `D:\spireline\referee\corpus`,
never committed) and replays all of them against the current emulator:

```
python tools/fidelity/corpus.py build                 # scan every *.out.json under jobs/ and jobs-*/ (also done by run/check/baseline)
python tools/fidelity/corpus.py add <out.json> [...]  # a recording kept anywhere else
python tools/fidelity/corpus.py check [--procs 16]    # replay all, compare with the baseline; exit 1 on a regression
python tools/fidelity/corpus.py baseline              # on main, after a fix is merged: accept the current results
python tools/fidelity/corpus.py status                # size, formats, flags, tags, excluded folders, baselines
```

- Sources: every `*.out.json` at any depth under `jobs/` and the sibling `jobs-*/` folders. Each folder has an entry in
  `corpus.POLICY` with its reason: included, included with a tag (`jobs/frames`: `settle-trial`, `jobs-nodeless`:
  `nodeless-trial`), or excluded (two files recorded with the frame-count settle and no frame sleep, which snapshots
  before the game settles). A new folder is read and tagged `unreviewed` until it gets an entry. A fight's tag is one
  all its sources share. Sources are kept by path.
- Dedup: by spec (seed, character, ascension, Neow pick, deck additions, room, and `enemy_hp_mult` or a console setup
  when a recording has them) plus each step's action and state and the choices, so a fight in a shard, a wave and the
  merged file is one entry with all its sources. Per-step instrumentation (cost, trace, checksums) is dropped before
  hashing and storing. Only the game side is stored.
- A fight the game broke off (idle_timeout and the like) is kept up to its last good step and flagged `truncated`:
  before the first step that changed nothing (the driver acted during the game's turn setup and the game parked it)
  and the end of turn before it, and without a trailing end of turn. A prefix of a complete fight is only its source.
- Tags: game build (sts2.dll sha256), recording format (1: choices not tied to a step; 2: choices carry their step;
  3: refused plays recorded, fork 732aacc). A format 1-2 fight with a RandomEnemy card play is flagged
  `driver-artifact-risk` (the driver aimed it and the game refused it unrecorded); it stays in the gate. `no-op-step`:
  a complete fight with a step that changed nothing. `intents`: the recording has the enemies' intents.
- A fight that needs `enemy_hp_mult` (speed/long-fights) or a console setup (feat/console-setup) is skipped by `run`
  until EmuDump's source applies it, and counted as skipped.
- `run` (and `check`, `baseline`) first picks up new or changed recordings, then runs `EmuDump fight` and `reanchor`
  split across processes (each call times out: 120 s + 0.1 s per step, `--timeout` overrides) and writes
  `corpus/runs/<time>-<commit>/results.json`: per fight, identical or the first divergence (step, field, card), any
  emulator exception, and one re-anchored verdict per step (`M` match, `D` diverge, `E` error, `U` unjudged,
  `-` not re-anchored) with every diverging field.
- `check` compares with `corpus/baselines/<commit>.json` of this commit or its nearest ancestor. REGRESSIONS: a fight whose
  plain replay diverges earlier than before (identical before included), a new emulator exception in either replay,
  or a re-anchored step that matched and no longer does, now throws, or now also diverges in another field.
  IMPROVEMENTS: the reverse. A full check (no `--only`/`--limit`) also FAILS when a baseline fight was not compared.
  New fights without a baseline are counted, never a failure. A note says when the judge (`tools/fidelity`,
  `data/id_map.json`) changed since the baseline.
- A regression is a bug in the change unless the old match was luck (two wrongs cancelling): then say so in the commit
  and store a new baseline after the merge. A baseline needs a committed emulator and judge, a fresh EmuDump build, a
  full run and no EmuDump failure.
- `tools/build_and_test.sh` builds EmuDump and runs `check` as its last step (`corpus_check_exit`). `CORPUS_CHECK=0`
  skips it (`corpus_check_exit` "skipped"), which is never a push gate: step 7 needs `0`.

## Rules

- Only the referee session touches the game copy; never launch the game any other way.
- No game files, decompiled code or game text in any commit. Short identifiers only.
- Never run anything on the E2 list in `FORK.md`. Never edit `D:\spireline\src`.
- Don't change engine code to make a test pass; change it to match the game.
- Keep output small: grep and head, don't print whole JSON files.

## Escalate (stop and write up, so a stronger model or the owner takes it)

- The same divergence survives two fix attempts.
- The fix would touch RNG plumbing (`Core/Rng`, `StateCloning`, `RunEngine` stream sync), cloning or resampling.
- A fix changes more than about 40 lines of engine code, or breaks other fights that matched before.
- The referee itself misbehaves: network connections, profile touched, hangs not explained by a missing screen.
- Anything needing a download, a credential, a setting change or publishing.

Write-up: seed, step, the two states (diff lines), what the game code does, what you tried.

## Current coverage of the driver

Handles: Neow choice, card-choice screens, rewards screens (card rewards answered first card),
map travel to the first row-1 monster room, one fight with seeded random legal play, Instant fast mode.
Also nested reward screens (Neow's Bones). Not yet: potions, events beyond Neow, shops, rest sites, fights after the first,
characters other than the Ironclad (the emulator's run engine starts every run as the Ironclad).

## Table check (content tables vs the game)

`{"out": "D:/spireline/referee/exports/v0.107.1.json", "runs": [{"kind": "export"}]}` run through `run_referee.py` writes the game's pool orders, rarities, costs, act encounter and event lists and Neow's options (local only, never commit). Then `dotnet build` in `tools/emudump` and `python -m tools.fidelity.table_check` lists every difference with the emulator's tables.

## Long-lived servers (`serve.py`)

For many fights at once: `python referee/serve.py --game <game copy> --specs <job.json> --servers 4 [--repeat K] [--out <file>]`
(from the fork's root). It boots N headless games, each on its own scratch profile, feeds the job's `runs` through one
inbox folder and merges the results into `--out` (same format as a runs-mode out file, in spec order). Each game keeps
running between fights (no 10 s boot per session), so cost per fight is the fight. Details and rules:

- Safety checks are `run_referee.py`'s (known `sts2.dll` hash, `override.cfg`, scratch `APPDATA`, real profile untouched);
  one `netstat` call per second covers every server. It exits 1 if any server opened a connection.
- A server that dies or hangs is replaced; the spec it held goes back to the inbox once and is recorded as `stalled` if it
  kills a second server. One hot spare is kept booted (`--spares 0` turns it off). Servers run at BelowNormal priority.
- `--verify-against <runs-mode out.json>` compares each result (ignoring timing) with the reference, spec by spec.
- `tools`: `python referee/linkcopy.py --src <game> --dst <game-name>` makes a private game copy out of hard links, so a
  session can install its own driver without waiting for the shared copy's `mods/*.dll` to unlock.
- Run files: `<game>/../serve/<ts>/{inbox,out/s*.jsonl,scratch/s*,results.json,results.jsonl,summary.json}`.
- Driver side (`driver/Server.cs`): job `{"out": "x.jsonl", "inbox": "<dir>", ...}` instead of `"runs"`; claim by renaming
  `<id>.json` to `<id>.json.taken.<pid>`, append one line per fight, `STOP` in the inbox ends it.

## Re-anchoring (every recorded step tested on its own)

`fight_compare` stops learning at the first divergence. Re-anchoring puts the emulator into the game's state before each step, applies the game's action and compares the result with the game's next state:

```
dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll reanchor '<out.json>' > <name>.reanchor.json
python -m tools.fidelity.reanchor_compare '<out.json>' <name>.reanchor.json [--after-enemy-desync plays] [-v]
    [--emu <name>.emu.json] [--coverage cov.json --commit <sha>]
```

- A matching step verifies its card; a divergence is blamed on its card only when the step before it matched ("primary"). Fights stop being re-anchored where the enemies' move state cannot be reproduced (the snapshot does not record intents); `--after-enemy-desync plays` goes on with card plays and leaves later ends of turn unjudged.
- What the snapshot lacks (relic counters, enemy AI and intent, per-card modifiers, combat counters) comes from the emulator's own previous result. The `--scramble` and `--stale enemy,cards,relics,counters|all` options of `reanchor` test the write-back and show which hidden state a step needs.
- `reanchor_compare.learn` / `coverage.update_reanchored` replace `coverage.update` for a re-anchored batch (do not use both).
