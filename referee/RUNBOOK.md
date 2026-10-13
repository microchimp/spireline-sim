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
It streams: one pool of game servers (`--parallel`) plays the whole batch, finished fights are replayed in the emulator in chunks (`--chunk`, default 48) while the servers keep playing, and each chunk updates the coverage the next decks are drawn from. The canary (`--canary` fights per server) is the first chunk; if it fails the batch stops (exit 3). `--game`, `--jobs` and `--coverage` point it at a private game copy, job folder and coverage file. `<name>.json` holds the specs as played (`serve.py --specs <name>.json` replays them).

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

Handles: Neow choice, card-choice screens, rewards screens (card rewards answered first card),
map travel to the first row-1 monster room, one fight with seeded random legal play, Instant fast mode.
Also nested reward screens (Neow's Bones), and with a spec's setup keys (next section) any encounter of any act,
relics, potions and their use, powers and upgraded cards. Not yet: events beyond Neow, shops, rest sites, fights
after the first, characters other than the Ironclad (the emulator's run engine starts every run as the Ironclad).

## Setup fights (states random Ironclad decks never reach)

The developers' console commands set up a situation through the game's own commands; the driver
(`driver/Setup.cs`) calls those same commands, never one that changes or skips a rule (godmode, kill, win,
damage or heal shortcuts are refused). Spec keys, all optional; without them the output is byte-identical:

- `"setup": [{"cmd": "relic", "id": "VAJRA"}, {"cmd": "potion", "id": "FIRE_POTION"}, {"cmd": "card", "id": "BASH", "upgraded": true},
  {"cmd": "fight", "id": "BYRDONIS_ELITE"}, {"cmd": "power", "id": "VULNERABLE_POWER", "amount": 2, "target": "enemy:0"}, {"cmd": "energy", "amount": 1}]`.
  Items before `fight` run on the map after Neow, items after it once the room has settled (turn 1, hand drawn), before the
  first action. `fight` is FightConsoleCmd's room without the clock-seeded encounter Rng, at the row-1 coordinate. Each item is
  logged (`setup_log`: ok, message, RNG counters); a choice it opens is tagged step -(i+1).
- `"use_potions": true`: potion use joins the random actions (PotionModel.EnqueueManualUse, the popup's enable rules); snapshots
  then also record the belt (compared) and the relics (not compared).
- `"script": [<recorded actions>]`: replays actions by card id instead of random ones (the shrinker's reproduction).

EmuDump mirrors every item at the same moment (`tools/emudump/Setup.cs`); an item it cannot build is logged `unsupported`
(a mirror gap, not a divergence). `python -m tools.fidelity.setup_compare <out.json> <emu.json>` names the first item after which
the run's RNG counters or the item's outcome differ.

`python referee/scenarios.py --name <n> --mode coverage|dense --fights 60 --start <k> [--shrink 8 --shrink-budget 200]` (game copy
`D:\spireline\referee\game-setup`, 2 servers, outputs in `D:\spireline\referee\jobs-setup`) makes the specs from the game's tables,
plays, replays, judges and prints a summary. Coverage fights: one encounter of any act and kind plus 1-3 least-covered relics,
potions, powers or upgraded cards. Dense fights: 20-30 card decks (~35% upgraded), 10-20 relics, a full belt, powers on both sides.
Plausibility flags (`cross_act`, `debug_room`, `cross_character`, `duplicate_relic`) mark setups no real run has; packs rank them last.
`--shrink K` reduces K diverging fights to a minimal setup (`referee/shrink.py`: script-mode re-runs of item subsets, ddmin, all
cases in one game round at a time, 48 fights a case at most) and writes `<n>.packs/minimal/<seed>.json`, a spec that runs as is.

## Table check (content tables vs the game)

`{"out": "D:/spireline/referee/exports/v0.107.1.json", "runs": [{"kind": "export"}]}` run through `run_referee.py` writes the game's pool orders, rarities, costs, act encounter and event lists and Neow's options (local only, never commit). Then `dotnet build` in `tools/emudump` and `python -m tools.fidelity.table_check` lists every difference with the emulator's tables.

## Long-lived servers (`serve.py`)

For many fights at once: `python referee/serve.py --game <game copy> --specs <job.json> --servers 4 [--repeat K] [--out <file>]`
(from the fork's root). It boots N headless games, each on its own scratch profile, feeds the job's `runs` to them one
spec at a time through an inbox folder per server and merges the results into `--out` (same format as a runs-mode out
file, in spec order). Each game keeps running between fights (no 10 s boot per session), so cost per fight is the fight.
The dispatcher is `serve.Pool` (`submit(specs)`, `poll()` returns finished fights, `close()`); `batch.py` keeps one for a
whole batch. Details and rules:

- Safety checks are `run_referee.py`'s (known `sts2.dll` hash, `override.cfg`, scratch `APPDATA`, real profile untouched);
  one `netstat` call per second covers every server. It exits 1 if any server opened a connection.
- The driver's watchdog arms once a server is ready and counts frames as well as seconds (`driver/Watchdog.cs`): it exits after
  `stall_s` seconds and 600 frames without progress, or when no frame ends while the main thread is blocked or burns CPU for 2 x
  `stall_s` seconds, so a slow, starved server is not killed. serve.py kills on a 10 x `stall_s` old status file only as a backstop.
- A server that dies or hangs is replaced; the spec it held goes back to the front of the queue once and is recorded as
  `stalled` if it kills a second server. One hot spare is kept booted (`--spares 0` turns it off). Servers run at BelowNormal priority.
- One inbox per server, never a shared one: a claim is a rename, and on Windows two servers renaming the same file can both
  succeed (night04 lost 4 of 600 fights that way). The dispatcher writes a spec there only under its final name (tmp, then
  rename) and withdraws specs (an abandoned run) only once the server's status reads `hold`, or kills it after 60 s. A used
  `--run-dir` is refused.
- `--verify-against <runs-mode out.json>` compares each result (ignoring timing and the diagnostics below) with the
  reference, spec by spec.
- Driver job keys (SPEED_PLAN steps 3-4, `driver/Settle.cs`), set in the specs file or by serve.py's and batch.py's
  defaults, which are the same (`serve.job_options`, which batch.py's pool uses too): `settle` `"flags"` (default: the game's own turn flags, held on two checks one frame apart)
  or `"frames"` (the old six-stable-frame rule); `settle_audit` 0..1 (default 0.05: the share of fights, by seed hash,
  whose every flag settle is re-checked with the old rule at stock pacing; the result's `settle_audit` lists the steps,
  an object for each disagreement, and batch.py's summary counts them); `fast_tweens` (default true: awaited tweens
  finish at once); `sleep_usec` (default 0: no per-frame padding; only safe with the flag settle); diagnostics
  `trace_frames` (per-frame settle trace in each step's `frame_trace`; `trace` is trace mode's, below), `step_costs` (frames and main-thread megacycles in each step's
  `cost`) and `idle_probe` N (a quiet frame's cost after step 0). `--old-settle` (serve.py and batch.py) runs `"frames"`
  with stock tweens and pacing and no audit; `--settle-audit X` sets the share. A settle that does not hold within
  `idle_timeout_s` (default 10 s, and 600 frames; below `stall_s`) ends the fight with `idle_timeout`.
- Frame cost (speed/frame-cost up to `still_visuals`): `silent` (no sound is started) and `still_visuals` (visual-only nodes
  do no per-frame work) are on with `fast_patches` unless `--no-patches`; `--job-opt KEY=JSON` sets any job key for every
  server (e.g. `profile=true`, the per-fight frame-cost profile of `driver/Lean.cs`); the games start with
  `DOTNET_TieredPGO=0` unless `--no-dotnet-tuning`. `no_vfx` is not merged (it breaks Slumbering Beetle and Lagavulin
  Matriarch at combat start).
- `tools`: `python referee/linkcopy.py --src <game> --dst <game-name>` makes a private game copy out of hard links, so a
  session can install its own driver without waiting for the shared copy's `mods/*.dll` to unlock.
- Run files: `<game>/../serve/<ts>/{inbox/s*,out/s*.jsonl,scratch/s*,results.json,results.jsonl,summary.json}` (a batch:
  `serve/<name>`, with `chunks/` for the emulator's working files).
- Driver side (`driver/Server.cs`): job `{"out": "x.jsonl", "inbox": "<dir>", ...}` instead of `"runs"`; claim by renaming
  `<id>.json` to `<id>.json.taken.<pid>` in its own inbox (`inbox/s<NN>`, `driver/InboxClaim.cs`), append one line per
  fight, `STOP` in the inbox ends it; `hold.<pid>` makes it wait without claiming, and its status file then reads `hold`.

## Re-anchoring (every recorded step tested on its own)

`fight_compare` stops learning at the first divergence. Re-anchoring puts the emulator into the game's state before each step, applies the game's action and compares the result with the game's next state:

```
dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll reanchor '<out.json>' > <name>.reanchor.json
python -m tools.fidelity.reanchor_compare '<out.json>' <name>.reanchor.json [--after-enemy-desync plays] [-v]
    [--emu <name>.emu.json] [--coverage cov.json --commit <sha>]
```

- A matching step verifies its card; a divergence is blamed on its card only when the step before it matched ("primary").
- Enemy moves: the snapshot records each enemy's next move (`move`, `intents`), EmuDump dumps the emulator's (`intent`), and fight_compare compares them, so a wrong move shows at the step that chose it. Before each step `reanchor` puts the emulator's AI on the game's moves where its own state allows (a cycle's position, ranked by the game's later moves; a drawn choice redrawn after the step that chose it) and records `moves` = same / set / unknown; an end of turn whose moves are unknown is left unjudged. `--perturb-moves` tests this (every cycle moved one on, the AI on another stream). Records without intents fall back to the old guess: re-anchoring stops at the first end of turn an enemy move could explain, or with `--after-enemy-desync plays` leaves later ends of turn unjudged.
- What the snapshot lacks (relic counters, enemy AI and intent, per-card modifiers, combat counters) comes from the emulator's own previous result. The `--scramble` and `--stale enemy,cards,relics,counters|all` options of `reanchor` test the write-back and show which hidden state a step needs.
- `reanchor_compare.learn` / `coverage.update_reanchored` replace `coverage.update` for a re-anchored batch (do not use both).

## Trace mode (re-runs of fights that diverged)

A fight replays identically from its spec, so the costly recording is paid only on re-runs:

```
python referee/trace_divergent.py <batch out.json> --game <your own game copy> [--specs <job.json>] [--max 10] [--servers 1..3]
```

- It finds each diverging fight's step (first divergence of the plain replay, else the first primary re-anchored one), re-runs the spec through `serve.py` with `"trace": true` and `max_actions` = that step + 2, checks the re-run reproduces the batch up to the step, replays it with `EmuDump reanchor --trace-to`, and writes `<name>.packs/traces/<seed>.trace.txt` (+ `.json`, `index.md`, the traced records). `context_pack` folds each group's trace diff into its pack. `--reuse` redoes only the emulator side.
- Game side (`driver/Trace.cs`, off unless the spec asks): per step the game's CombatHistory entries, each GameAction run, each `Hook.Before*/After*` called, each RNG draw (stream, counter, two callers) and marks with the visible state (play/played, turn_end, enemy_turn, move/moved, turn_start), plus hidden state (card costs with modifiers, enchantments, afflictions, relic counters and fields, power amounts and fields).
- Emulator side (`EmuDump reanchor --trace`, `tools/emudump/Trace.cs`): the same marks from `Core/CombatTrace` probes, every draw of a combat stream with its caller, and the hidden state. The probes cost one null check in normal runs.
- `python -m tools.fidelity.trace_diff <traced out.json> <traced reanchor.json> [--seed S] [--step J]` aligns the marks, measures each side from its own start state, and prints the first segment whose changes differ (with the game's log, hooks and draws there), whether the re-anchored start already differed (a write-back gap), the first draw one side makes and the other does not (per stream), and hidden-state differences at the step start.

## The game's own checksums (per-action oracle)

The game hashes its full combat state (`NetFullCombatState`) after every finished game action and at
each end-of-turn phase, for multiplayer desync checks (`ChecksumTracker`); single player leaves it off.
`driver/ChecksumTap.cs` switches it on per fight:

- Spec (or runs-mode job) key `"checksums": true`: each step gets `checksums` = `[context, action type,
  checksum]` per checksum. Cost measured on r20v2: 0.48 ms per checksum, ~2.2 per step, ~22 ms per
  fight (under 1% of a fight's CPU); `checksum_cost` in each result. In serve mode put it on the specs.
- `"trace": true` (also trace mode's key, above: a traced re-run gets both): also `net` = the full state behind each checksum (creatures, piles with card
  keywords/cost/enchantment/saved props, relics with saved props, potions, orbs, every RNG counter,
  relic grab bag). About 6x the output size; for a handful of fights.
- An end of turn yields five checksums: `After player turn phase one end` (end-of-turn hooks,
  ethereal, turn-end-in-hand), `after player turn phase two end` (hand flush), `After enemy turn
  start`, `After enemy turn end` (every enemy's move), `After player turn start` (block clear, energy,
  draw, intents). A play yields one (`PlayCardAction`); the auto-play phase after `After player turn
  start` and commands inside a play have none.

```
dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll fight '<trace out.json>' --netstate > <name>.net.json
python -m tools.fidelity.netstate_diff '<trace out.json>' <name>.net.json [--seed S] [-v]
python -m tools.fidelity.netstate_diff '<out.json>' '<other out.json>'   # game against game, by checksum
python -m tools.fidelity.netstate_diff --check '<out.json>'              # driver integrity only
```

`netstate_diff` names the first step and the game sub-action where the emulator's value stops matching,
and lists the fields the emulator does not model (`tools/emudump/NetState.cs`). `--check` (also run
first by the other modes) flags a step whose action left no checksum: the driver recorded it before
the game finished (seen with Mayhem, Hellraiser and Rolling Boulder, whose turn-start auto-play
outlasts `Idle()`), so that fight is a driver artifact from the step before it, not an emulator bug.
