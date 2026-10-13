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
   Push only if C# shows `Failed: 0` and Python has no failures beyond the baseline (the deckfights stall test is a known 1-in-3 timing flake). Commit message: what diverged, the game class it was checked against, the fights that now match, ending with the attribution line the session gives. Then `git push origin main`.
8. Log every finding in `referee/FINDINGS.md` (one line: date, seed, step, cause, fix commit or "open").

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

Fight mode (`"kind": "combat"`): Neow choice, card-choice screens, rewards screens (card rewards
answered first card), nested reward screens (Neow's Bones), map travel to the first row-1 monster
room, one fight with seeded random legal play, Instant fast mode. Whole runs (`"kind": "run"`): see
"Whole runs" below. Characters other than the Ironclad: not yet (the emulator's run engine starts
every run as the Ironclad).

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

## Whole runs (`runs.py`)

One spec plays a whole A10 run in the game with seeded random legal choices on every screen,
through the game's own synchronizers and commands (`driver/RunPlay.cs`, spec `{"kind": "run", "seed",
"chooser_seed", "buff_max_hp", "avoid", "stop_after_act", "max_decisions", "potions"}`), and records the
run state (the game's own save, minus timestamps, drawings, unlocks and the map) after every decision,
with the offer on screen and the choice by id. Fights inside the run use the fight loop and are
recorded as fight steps.

```
python referee/runs.py --name <name> --runs 10 --start <n> [--procs 2] [--buff 2000] [--stop-after-act 3] [--potions]
python referee/runs.py --name <name> --no-game        # replay and compare again after an engine change
```

It runs `run_referee.py` sessions and stops on a session report that is missing, unreadable or lacks
`"network_connections": []` and `"real_profile_touched": false`, and on a non-zero exit (a crash or a
timeout: reported with its scratch folder; the merged results are written, `--no-game` compares them).
Then `EmuDump run` (tools/emudump/RunReplay.cs: each decision applied as the emulator's own actions,
looked up by id) and `python -m tools.fidelity.run_compare2 run`.

What the comparison does, per run:

- The first divergence, fights included: a fight step that differs (fight_compare's views), a replay
  that ends at another step than the game's fight, or a fight identical at every step whose OUTCOME
  differs. The outcome is the emulator's own run snapshot right after the fight, before any anchoring
  (`outcome_snap`), against the game's post-fight save, and its reward screen (`outcome_offer`)
  against the game's next rewards offer.
- Then every fight is anchored: the replay puts the emulator's holdings, odds, RNG counters, grab
  bags, relic counters, wax relics, card properties and reward screen where the game's are
  (`RunReplay.Anchor`, `ForceRewardScreen`), so the screens after it are still compared. The first
  divergence outside fights is printed too. A difference in the snapshot after anchoring is
  `combat: not anchored -> <field>`: something the anchoring does not reset.
- At every decision: replay errors, the offer, the room entered (its RoomType and its id), the event
  options enabled at every index and the first page's option count, a new act's map, and the run
  snapshot (`run_snapshot.ORDER`, including relic counters in the game's units: `RunSnap.RelicProps`).
- Never taken as equal: what is not compared is printed with every comparison. `run_snapshot.NOT_COMPARED`
  lists the save's fields left out and why (among them event option identity: the emulator has no
  option ids, and melted wax relics, which the emulator drops), and every saved relic or card property
  a snapshot carries that the emulator does not export is named per run (e.g. Spoils Map's
  `SpoilsActIndex`, Touch of Orobas' and Dusty Tome's pickup records, Byrdpip's skin).

To compare a new relic counter, add it to `RunSnap.RelicProps` (getter and setter in the game's units;
`Absent` for a `SaveIfNotTypeDefault` property, mirrored in `run_snapshot.SAVED_IF_NOT_DEFAULT`; a
counter the game reads modulo n goes in `run_snapshot.PROP_MOD`).

`EmuDump runstart` + `run_compare2 start` compare run starts (all three acts' room sets, grab bags,
odds, counters) without playing.

Driver notes: `?` points are entered in Fast mode (PunchOff's endless animation loop freezes the game
in Instant mode), the chest opens through `NTreasureRoom.OpenChest`, the Crystal Sphere is played with
`SetTool` and `CellClicked`, the Fake Merchant's stall is bought from, potions are drunk in fights with
spec `"potions": true` (never outside fights: the emulator has no action for that).
