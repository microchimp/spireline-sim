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
   For many calls, `bash tools/emudump/aot.sh` also builds `tools/emudump/bin/aot/EmuDump.exe` (NativeAOT: no JIT, about a
   third of the CPU on a 12-fight call, half on 100). It runs only `fight <out.json> [--compact]` and `reanchor <out.json>`,
   byte-identical to the DLL on the corpus; rebuild it with the DLL. `corpus.py --emudump <EmuDump.exe>` takes it.
   `EMUDUMP_PROFILE=1` prints a call's time per phase on stderr.
4. **Compare**: `python -m tools.fidelity.fight_compare '<out.json>' '<emu.json>'`. It prints the first diverging step per fight;
   a replay that stops early differs at the first step it lacks. Exit 1 on a difference, an emulator error (also after the
   last step), a missing result or one of another seed (broken cases: `tests/python/test_fight_compare_broken.py`).
4b. **Diagnosis packet** (for a backlog item, or one failing fight): `python -m tools.fidelity.packet --item <ID>` / `--top 10` / `--fight <id>` writes
   `D:\spireline\referee\backlog\packets\<ID>.md` (section "Diagnosis packets" below) and prints its path. **Every fixer is given the path of its item's
   packet in its prompt ("Your diagnosis packet is <path>; read it first"), and so is the reviewer of its branch.** Record `packet=yes` in
   `inputs_given` of `backlog/fix-attempts.csv`. A packet is evidence, not a verdict: its candidate case is tied to the item by a cited seed or by
   model names, and the packet says which; confirm it is the same cause before using it.
5. **Classify each difference** before touching code:
   - *Driver gap*: the game side errored or skipped a screen (e.g. `SelectLocalReward RelicReward did not finish`). Fix `referee/driver/*.cs`, never the engine.
   - *Comparer gap*: same game state, different representation (e.g. STUNNED is a buff in the emulator, a move in the game). Fix `tools/fidelity/fight_compare.py` and write the reason next to the rule.
   - *Emulator bug*: the game's own code (decompiled) shows the emulator does something else. Fix `src/Sts2Emulator`.
6. **Fixing an emulator bug** — root cause first, never a symptom patch:
   - Ask why the emulator differs: which mechanism is wrong (pool or list order, RNG stream, number or order of draws, hook timing, a shared helper)? A fix that only makes this one fight match is not done.
   - Find every other place that uses the same mechanism, check each against the game, and fix them together: start from
     the sweep table (`python -m tools.fidelity.sweep_table <Cmd.Method>`, section "Sweep table"), fill its status
     column and attach it with `--check` passing.
   - In FINDINGS.md record the root cause and the sibling sites checked, not just the symptom.
   - Then think one level up: name the bug CLASS (hand-written orderings, a draw on the wrong RNG stream, an effect before rather than after a hook, a helper that ignores a modifier) and sweep the engine for other instances no fight has hit yet. Verify each against the decompiled game; aim the next referee batch at that class.
   - Read the game's class for that card, relic, power or monster in the decompiled tree, and the emulator's code for it.
   - Make the smallest change that matches the game; cite the game's class/method in a comment (never paste game code).
   - Add or adjust an xunit test in `src/Sts2Emulator.Tests` that fails before and passes after.
   - Re-run the failing fights; the fix must make them identical.
7. **Gate and push**: `BASELINE_FAILURES=/d/spireline/logs/baseline-python-failures.txt bash tools/build_and_test.sh /d/spireline/logs/fork-$(date +%Y%m%d-%H%M%S) /d/spireline/src/sts2/.venv/Scripts/python.exe`.
   Push only if C# shows `Failed: 0`, Python has no failures beyond the baseline (the deckfights stall test no longer races the process start: its workers are in a light module)
   Push only if it exits 0 (`gate` is `pass` in `results.json`: restore and publish passed, `src/Sts2Emulator.Tests` and
   `tools/emudump.Tests` each show `Failed: 0` in a summary; exit 1 when a step failed or its output is unreadable, 3 when
   the corpus check was skipped), Python has no failures beyond the baseline (the deckfights stall test is a known 1-in-3 timing flake)
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
  IMPROVEMENTS: the reverse. A full check (no `--only`/`--limit`) also FAILS when a baseline fight was not compared, and
  any check FAILS (`UNSOUND`) when it compared no baseline fight, ran an EmuDump older than the emulator's sources or
  lost fights to EmuDump failures (broken cases: `tests/python/test_corpus_check_broken.py`).
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

Fight mode (`"kind": "combat"`): Neow choice, card-choice screens, rewards screens (card rewards
answered first card), nested reward screens (Neow's Bones), map travel to the first row-1 monster
room, one fight with seeded random legal play, Instant fast mode; with a spec's setup keys (next section)
any encounter of any act, relics, potions and their use, powers and upgraded cards. Whole runs
(`"kind": "run"`): see "Whole runs" below. Characters other than the Ironclad: not yet (the emulator's
run engine starts every run as the Ironclad).

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
A relic whose pickup opens a deck selection is mirrored by the pick the game's selector logged for it (a `cards` choice tagged
with the item's step): the game lists the eligible deck cards in deck order, except a removal (Biiig Hug, Empty Cage, Precarious
Shears, Precise Scissors, Preserved Fog, Pael's Tooth), which lists its curses first and then deck order, and the selector takes
the first n. The mirror takes the cards at those list positions (followed through the removals and transforms earlier picks
made), and closes a zero-to-N screen (Kifuda, Gnarled Hammer, Claws) with DONE once the game's picks run out. When the
emulator's screen lists other cards than the game's (an earlier item already left its deck elsewhere), the pick is matched by
card id and the setup log's message says so (`note: the emulator's selection lists ...`), which is no gap; it is a gap when the
emulator holds copies of that id that differ (upgrade, enchantment), or the id is not on its screen, or the screen wants more
picks than the game's required. A screen the game never showed is closed with DONE and a note.

`python referee/scenarios.py --name <n> --mode coverage|dense --fights 60 --start <k> [--shrink 8 --shrink-budget 200]` (game copy
`D:\spireline\referee\game-setup`, 2 servers, outputs in `D:\spireline\referee\jobs-setup`) makes the specs from the game's tables,
plays, replays, judges and prints a summary. Coverage fights: one encounter of any act and kind plus 1-3 least-covered relics,
potions, powers or upgraded cards. Dense fights: 20-30 card decks (~35% upgraded), 10-20 relics, a full belt, powers on both sides.
Plausibility flags (`cross_act`, `debug_room`, `cross_character`, `duplicate_relic`) mark setups no real run has; packs rank them last.
`--shrink K` reduces K diverging fights to a minimal setup (`referee/shrink.py`: script-mode re-runs of item subsets, ddmin, all
cases in one game round at a time, 48 fights a case at most) and writes `<n>.packs/minimal/<seed>.json`, a spec that runs as is.

A `card` item after `fight` goes into a combat pile: `{"cmd": "card", "id": "HAVOC", "pile": "Hand"}` (the end of the hand),
`"pile": "Discard"`, or `"pile": "Draw"` at the bottom or with `"pos": "top"` on top (drawn next). EmuDump mirrors these too;
no hook runs for the added card in the mirror, so a card that reacts to entering a pile would be a mirror gap.

Fix lab steering (`--mode targets`): `python tools/fidelity/lab_targets.py --lab LAB --out LAB/targets.json` weighs every
card, relic, potion, enemy, encounter, event, power and act by how often real Ironclad A10 runs meet it (Spire Codex
aggregate counts, built once into `LAB/cache/codex_profile.json` by an isolated interpreter) times its gap factors
(nothing recorded reaches it, unreached engine blocks, a low-accuracy reference-set cell, open backlog items or codex-walk
stops naming it; each is a `reasons` line). `scenarios.py --mode targets --targets LAB/targets.json --jobs <dir>` draws
each spec's target by weight and builds a realistic setup around it from the profile (the act, an encounter of that act,
its deck size, Strikes and Defends, cards, relics and potions held there and held with the target), shrinks `--shrink K`
diverging fights and traces up to `--trace N` (40) of them. `python tools/fidelity/lab_groups.py --lab LAB --out
LAB/groups.json --summary LAB/summary.md` merges every finished lab job's symptoms (and `--batch`, `--corpus-results`,
`--codex-walk`, `--refset-score`) into root-mechanism groups, flags `host_suspect` ones from `referee/host_gaps.json`
(known headless-host gaps, each citing its FINDINGS entry) and matches them to the newest backlog.

## Rule probes (`probe.py`: ask the game how a disputed rule works)

Before writing a fix that depends on how the game reads a rule, ask the game. A probe (`referee/probes.py`) is a setup
fight plus a `script`, one measured step, and hypotheses that each predict the measurement from the game's state before
that step (so the random opening hand does not matter):

```
python referee/probe.py list
python referee/probe.py run --name <n> [--only P01,P05] [--game D:/spireline/referee/game-probe] [--servers 2]
python referee/probe.py run --name <n> --no-game     # after an engine change: replay the recorded games again
```

- The game side runs in trace mode through `serve.py` (every safety check, at most 2 game processes, no hot spare);
  each step records what the game did in order, and the report prints the relevant part ("game did").
- The emulator's value is the same step re-anchored on the game's state (`EmuDump reanchor`), else the plain replay's
  while it still matched the game before the step; a step that ends the fight is read from the plain replay's run
  phase (the re-anchored dump says only "Over"). A probe with no emulator value is never AGREE.
- Per probe: the game's value and the hypothesis it matches (the rule), the emulator's and its hypothesis,
  AGREE/DIFFER, the whole-state verdict of that re-anchored step, and notes on representation differences the
  probe works around. Outputs in `D:\spireline\referee\probes\<n>.*` (local only).
- A new probe: put the factor in the setup (relic, potion, power, cards into the hand or on top of the draw pile,
  energy), measure one number or tuple, and write each hypothesis as a prediction from the pre-state. Name the game's
  class and method in `game_class`; never copy game code. A probe whose answer is an order inside one step (no state
  shows it) answers from the trace (`evidence`), and a second probe should make that order decide an outcome.

## Sweep table (one mechanism, all its callers, before the fix)

```
python -m tools.fidelity.sweep_table CardCmd.AutoPlay --out sweep-autoplay.md          # or PowerCmd.Apply, "CardSelectCmd.From*"
python -m tools.fidelity.sweep_table CardCmd.AutoPlay --keep sweep-autoplay.md --check  # exit 1 until every row has a status
```

Each game call in the local decompile gets a row: the class-qualified call anywhere, and the bare call inside the
command's own class (the decompile drops the class there: CardCmd.DiscardAndDraw calls AutoPlay so). A caller in the
game's Commands or Hooks namespaces is a wrapper and its callers are listed too, until no new wrapper turns up
(`--depth` caps it): CardCmd.AutoPlay's Sly discards come through DiscardAndDraw and Discard. A call through a
delegate or a virtual hook (Hook.AfterX calling each model's AfterX) is no textual call and is not found. Each row has
its decompile site, the emulator sites naming the caller (`match` strong / weak / none / pinned; pin sites and add
mechanism terms in `data/sweep_map.json`), the tests naming it, and a status the owner fills: `converted`, `tested`
or `unreachable` (with the reason in the note). Attach the checked table to the review. It prints identifiers and
line numbers only. Broken cases: `tests/python/test_sweep_table.py`.

## Diagnosis packets (`packet.py`: everything about one item, in one file)

```
python -m tools.fidelity.packet --top 10                  # the first 10 open items of the newest backlog (in its order: by impact)
python -m tools.fidelity.packet --item C3-F02 [--item ...] [--backlog backlog-c3.json]
python -m tools.fidelity.packet --fight <corpus fight id | corpus seed | refset id>   # a failing fight with no item
python -m tools.fidelity.packet --item-file item.json     # an item written by hand (id, title, evidence; optional setup, seeds)
   [--out DIR] [--refresh-probes] [--no-sweep] [--no-emudump] [--codex-walk DIR] [--run <corpus run results.json>]
```

Output: `backlog/packets/<ID>.md` (what the fixer reads), `<ID>/` (the case as a referee `out.json` you can replay with `EmuDump fight` or
`reanchor`, and the trace text when there is one), `sweeps/` (full sweep tables, cached per fork commit), `index.md` and `completeness.json`.
It never starts a game; the emulator runs only through EmuDump on recordings that exist. Each packet has seven sections, each with a status
(full, partial, missing, n/a) and a reason, tabulated at its head:

1. **Smallest failing case.** The item's own setup spec with a stored shrink (`jobs-setup/*.packs/minimal/<seed>.json`) when there is one, else the
   smallest recorded fight that still diverges on the current emulator and is tied to the item: the corpus (first divergences from the newest full run of
   this commit) and the reference set's regression classes (one minimal setup each, replayed now). Tied means named: the item cites its seed or id (or quotes the
   stop reason of the fight's stored result), or, as a guess that the packet labels as one, the diverging step involves model names the item also names
   (weighed by how rare they are in the corpus; the step's card counts most, a held relic least). Order: named before guessed, re-anchored
   divergences (self-contained: the rule is wrong from the game's own state) before ones that only diverge in the plain replay, earliest step, shortest
   fight. For run-layer items (lane `run`, or no fight) the case is a codex walk stop (`EmuDump codex`, the run histories walked with no game; results under
   `codex-walk/<label>/*.jsonl`): the run the item cites, else the earliest run of the stop signature (class, component, field) whose words match strongly,
   with the run's floors up to the stop; the walker line named is the `Check`/`Fail` call of exactly the stop's field (`cards (fight floor)` is not
   `cards`), and none is named for a field the walker builds at run time. Also: the one-step case (the game's state before the step, the action, game
   against emulator after it), recording flags, selection-screen problems. "First divergence, plain replay" is the plain replay's own first difference
   (from the replay, else the stored result's `fd`); for a fight tied by a quoted stop reason the case is the step the reason names, and the packet says
   so when that is not the first divergence. Power lists are shown whole; what the comparer's power mask left out (a game power the emulator has no buff
   for, the emulator's own helper buffs) is listed under them, and `differ` names what was compared after the mask.
2. **First differing event inside the step.** `trace_diff` of a traced game record (`jobs-trace/**/traced.out.json`, `jobs-checksum/*.trace.out.json`)
   against this checkout's emulator re-anchored with `--trace` (an EmuDump run, no game), preferring the recording of the seed that the trace reproduces; a
   game record the driver snapshotted early is flagged UNSETTLED (not an emulator difference). Without a trace: the re-anchored step's differing fields,
   the emulator's own marks and draws, and the `trace_divergent.py` command that makes the trace (needs a game copy of your own; the packet does not run it).
3. **The game's per-action checksum state.** `netstate_diff` when the fight was recorded with the checksum tap and its full state (traced re-runs now are),
   else missing, with the re-run named. n/a for run-layer cases.
4. **Game classes and methods, and the emulator sites.** The decompiled file and hook overrides of each model the item names, the emulator sites that
   look like the same behaviour (`sweep_table`'s grep), the game commands it names (`Class.Method` found in the decompile) and the engine's own methods it
   names, and a sweep for each game command and each model-level hook it names: callers for a command, every overriding class for a hook (cached in
   `sweeps/`, the full table linked), so the siblings are in front of the fixer. Tool items list the code they are about and the RUNBOOK sections.
5. **RNG draw difference.** The trace's first differing draw, else per stream the draws the game made and the emulator made from the game's state, with
   the emulator's callers; n/a when no stream differs. Run-layer cases name the streams the stopped component draws on.
6. **Rule probe.** The probes (`referee/probes.py`, reports under `probes/`) the item names, or ties by ledger tag, game class or model names: the game's
   answer, what it did, the emulator's value when probed and now (`--refresh-probes` judges the recorded probe games again with this EmuDump, no game).
   "Related" is not "settles": only a probe the item names counts as full.
7. **Related ledger lines.** `backlog/ledger.md`, `FINDINGS.md` and `fix-attempts.csv` entries about the same thing (a cited seed, an old item tag, a method
   or hook name that few entries use, or two rare model names): what was found, what was fixed, what earlier attempts were rejected for. Every bullet is an
   entry, nested ones too (the notes under a finding hold its root cause and the seeds it was checked on), matched on its own text and shown with the
   line of the finding it stands under.

Reading a packet: start with the completeness table; "partial" and "missing" say what is absent and whether only a game run would supply it. What a game
run would add (a trace for a fight that has none, a shrink of a large setup, a checksum record) is named as `NEEDS A GAME RUN (not run here)` with the command.
Packets are local (they quote the referee's recordings and the run histories) and are never committed. `tests/python/test_packet.py` has the
synthetic cases (a tiny corpus, decompile, probe report, ledger, traced record, codex walk and reference set, with a fake EmuDump).

## Crash reports (Windows Error Reporting) and H2

**Incident 2026-10-15.** A game process crashed during an aimed fights batch; Windows started `WerFault.exe` for it, WerFault opened
a TCP connection to Microsoft (:443) to send the crash report, and the network check counted it against the game tree: the lab
wrote `HALT`. (A whole-runs session had also exited `0xC0000005` earlier that day.) The check was right to halt; the crash report
should never have been started.

**Fix (our launcher only; no Windows setting is touched).** Every game process starts in `run_referee.launch`, which spawns through
`popen_no_wer`: for the length of the spawn the launcher's error mode has `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
SEM_NOOPENFILEERRORBOX` (`GAME_ERROR_MODE`, 0x8003) added, the child inherits it (and so do the processes it starts, e.g.
`crashpad_handler.exe`), and the launcher's own mode is put back. With `SEM_NOGPFAULTERRORBOX` an unhandled exception ends the
process without calling Windows Error Reporting. `CREATE_DEFAULT_ERROR_MODE` would undo this, so `popen_no_wer` refuses it and a test
fails if any other file under `referee/` or `tools/` names it or the game's exe. `serve.py`, `batch.py`, `scenarios.py`, `refset.py`,
`runs.py` and the lab start the game only through `run_referee.py` or `serve.Pool(launch=run_referee.launch)`.

**Second line of defence.** Each `netstat` tick (and at a server's exit and at close) the pool also lists any `WerFault.exe` that the
game tree started or whose command line names (`-p` / `-ip`) a game process: `wer_reports` in `run_referee.py`'s session result,
in `serve.py`'s `summary.json`, a `WER:` line in the pool log, and `, WER REPORTS [...]` on the scenario / batch summary line when
there are any. It does not halt the lab (the connection check does, unchanged); it makes an attempt visible when no connection was seen.
Command lines are read once per WerFault through PowerShell (WMI), and only when a `WerFault.exe` exists.

**Test** (`tests/python/test_referee_wer.py`, about 25 s): the launch flags; the error mode inside the spawn and restored after; the WerFault matching on
fake process tables; the pool summary; and a crash on purpose: a child started by `launch` prints its own `GetErrorMode`, **exits 99
instead of crashing if the mode is not there**, otherwise reads address 0. It asserts exit `0xC0000005`, child mode `0x8003`, no WerFault
naming it or started by it for 20 s, and no connection. The test process runs in error mode 0 for it, so the harness that started the
tests cannot supply the mode. To read a running game's mode from outside: `NtQueryInformationProcess(ProcessDefaultHardErrorMode = 12)`,
bit 0 inverted (`GetErrorMode` only reads the calling process).

**What is not proved.** The mode is per process: code in the game that calls `SetErrorMode` itself would replace it (live game servers
and their `crashpad_handler.exe` children read 0x8003 mid-session, so none does at that point). The crash test covers an unhandled access
violation in a native process; a .NET fatal error or a fast-fail (`__fastfail`, `RaiseFailFastException`) takes other paths in Windows,
and no such crash was provoked in a game process. The owner's optional Windows Firewall outbound block (addendum 1, H2) remains the
layer that does not depend on any of this.

## Speed settings check (`normal_settings`, `--windowed`)

A job with `"normal_settings": true` skips every speed change (Instant fast mode, Engine.MaxFps, `fast_patches`, `silent`,
`still_visuals`, `fast_tweens`, `sleep_usec`, `no_preload`, `fast_reset`, and run_referee's `skip_intro_logo` / `fps_limit`);
tutorials stay off and the settle rule is the job's. It is about 17x slower (r20v2: 24 min, against 83 s).
`run_referee.py --windowed` launches without `--headless`, every safety check unchanged; use it only with the owner's own go.

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
  one `netstat` call per second covers every server and every process a server started (its `crashpad_handler.exe`),
  and one more after the last server exits. It exits 1 if any of them opened a connection.
- The driver's watchdog arms once a server is ready and counts frames as well as seconds (`driver/Watchdog.cs`): it exits after
  `stall_s` seconds and 600 frames without progress, or when no frame ends while the main thread is blocked or burns CPU for 2 x
  `stall_s` seconds, so a slow, starved server is not killed. serve.py kills on a 10 x `stall_s` old status file only as a backstop.
- A server that dies or hangs is replaced; the spec it held goes back to the front of the queue once and is recorded as
  `stalled` if it kills a second server. One hot spare is kept booted (`--spares 0` turns it off; batch.py has `--spares`
  too). Servers run at BelowNormal priority.
- A server whose fight left the game's turn flags set retires itself (exit 5, after its result line and a `_retire`
  line; `driver/Server.cs`, `Driver.StaleState`) and is replaced like a dead one, losing nothing (`retired` in the
  summary). The flags (`CombatManager.EndingPlayerTurnPhaseOne/Two`, `_inPlayerTurnSetup`) are cleared only by the turn
  flow itself, not by `CombatManager.Reset`, so a turn flow cut short (an exception thrown out of it, or a fight broken
  off mid-turn) made every later fight in that process end with `idle_timeout entering the room` (sp-nodeless: 4 fights).
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
- Engine frame rate: every game starts with `--fixed-fps 60` after its own command line (`serve.engine_args`; serve.py,
  batch.py and so scenarios.py, the shrinker and the lab's fights; `--fixed-fps N` sets it, `--fixed-fps 0` and
  `--old-settle` leave the engine's real-time frame sync). Frames advance as fast as the CPU allows; the speed spike
  (`D:/spireline/referee/design/real-game-speed.md`) found the per-action game checksums and recorded steps equal on 50 of
  50 fights (1,994 checksums, 6-10 repeats of each in other processes) and +15-24% fights per minute. Game-time timers
  then advance 1/60 s per frame. The summary's `engine_flags` says what the games got. Not set for `runs.py`'s sessions
  (proven on fights only).
- `runs.py` plays its sessions with serve.py's defaults above (`runs.session_job`: flag settle, fast tweens, no padding,
  5% settle audit, `silent`, `still_visuals`, 3.4x faster on 4 act-1 runs with every decision and snapshot equal to the
  stock settle's); `runs.py --old-settle` plays the stock settle and pacing for an A/B.
- Nodeless as the lab's default was considered and left off: the spike measured it on one lab batch of setup fights
  (+20% fights per minute at 8 servers, +33% with `--fixed-fps` at 10, fallbacks included), but refset, trace, probes,
  hypotheses, retests and whole runs need the full host, so the host type stays the specs file's choice.
- `tools`: `python referee/linkcopy.py --src <game> --dst <game-name>` makes a private game copy out of hard links, so a
  session can install its own driver without waiting for the shared copy's `mods/*.dll` to unlock.
- Run files: `<game>/../serve/<ts>/{inbox/s*,out/s*.jsonl,scratch/s*,results.json,results.jsonl,summary.json}` (a batch:
  `serve/<name>`, with `chunks/` for the emulator's working files).
- Driver side (`driver/Server.cs`): job `{"out": "x.jsonl", "inbox": "<dir>", ...}` instead of `"runs"`; claim by renaming
  `<id>.json` to `<id>.json.taken.<pid>` in its own inbox (`inbox/s<NN>`, `driver/InboxClaim.cs`), append one line per
  fight, `STOP` in the inbox ends it; `hold.<pid>` makes it wait without claiming, and its status file then reads `hold`.

## Nodeless host (`"nodeless": true`)

The same game process and rules code without the NRun scene (`driver/Nodeless.cs`, `driver/NodelessPatches.cs`):
about 8x less CPU per fight. Put the keys in the specs file given to `serve.py` (it passes `nodeless`,
`nodeless_on_error`, `nodeless_audit`, `error_log`, `frame_sleep_usec`, `health` to every server).

- Fights that create a model on the fallback list (`NodelessPatches.FallbackNames`: rules code that needs a node) are
  replayed from the start on the full host; `cost.host` / `cost.fallback` say which. Two checks find them: the
  `MutableClone` postfix, and a scan of every recorded state (creatures, powers, piles, deck, relics). Either one stops
  the nodeless attempt at the driver's next frame check, in the flag settle too (before, the flag settle never checked,
  so the attempt played on into the model's node code: RollingBoulderPower and EntropyPower threw at the next turn start).
- Full-host lane (serve.py and batch.py, on by default for nodeless servers): the fallback and audit replays are not
  played in the nodeless server's own process. Its line comes back with `_full`, and the dispatcher plays the spec again
  with `"host": "full"` on a server of the full-host lane (`--full-lane N`, default about 15% of the servers, at least
  1; every server runs the same job, the lane is the dispatcher's preference, and any server takes the other lane's work
  when its own queue is empty), then merges both lines into the format below (`cost.fallback`, `cost.nodeless_attempt`,
  `cost.audit`). Nodeless servers then never load the full host's scenes, a long full-host replay no longer holds up the
  nodeless server's next fight, and a server that retires (above) loses no replay. `--full-here` keeps the old in-process
  replays. Cancel checks stay in-process (they compare the cancellation sites that process recorded).
  batch.py prints a `hosts:` line: fights per host, replays, audits that differ, retirements and CPU per fight by host.
- Any game error or exception in a nodeless fight fails it (`error` starts with `nodeless:`); `"nodeless_on_error":
  "full"` replays it on the full host instead. `"nodeless_audit": 0.05` also plays 5% of fights on the full host
  (`cost.audit`).
- Cancellations: every `OperationCanceledException` is recorded once in `cost.cancel_sites` with its throw site, token
  owner (this fight's combat, an earlier fight's, the action executor, other, none), phase (`fight`, `reset`, `teardown`,
  `between`), step and the methods it cut short. One thrown while a nodeless fight plays fails it unless its type and
  site are on `Nodeless.HarmlessCancels` (empty: nothing has been proven harmless) or the spec, replayed on the full host
  at once, throws the same one (`cost.cancel_check`); with `nodeless_on_error` "full" the full host's result is used.
  On the coverage fights no nodeless fight throws one; every cancellation is the full host's (VFX and UI nodes awaiting
  a frame, and `CombatStateTracker.CallCombatStateChangedDeferred` when NRun leaves the tree at teardown).
- Each fight's span runs from its start through its reset, the empty scene (NRun goes then, not at the next run start)
  and 2 frames, so its teardown is recorded with it. Errors after that count for the next fight. (Before, a full-host
  replay's teardown showed up as 8-237 "cancellations" of the next nodeless fight.)
- The game-over guard catches only the null dereference of `NRun.Instance.RunMusicController.StopMusic()` in
  `CreatureCmd.Kill` (its IL range is read from the game at start; the log names it), nothing else in Kill.
- `"frame_sleep_usec": 0` removes Godot's 6.9 ms headless frame pacing: safe nodeless (results unchanged), not on the
  full host, whose settle check then catches card tweens mid-flight. In a nodeless job it paces only the nodeless
  fights; fallback replays, audits and cancel checks run at the default (`cost.frame_sleep_usec`).
- `"error_log": true` on a full-host job records each fight's game errors in `cost.errors` and cancellations in
  `cost.cancel_sites`. On every full-host fight in these jobs and in nodeless servers, a game error while the fight
  played sets `error` (`full host: N game error(s); first: ...`), so a UI abort ("Hand size 11 is greater than 11!") is
  never a clean result; errors of its teardown go to `cost.teardown_errors`.
- The full host's "freeze" after a turn-start autoplay (Mayhem, Hellraiser, also Rolling Boulder) is a driver bug, not a
  game tween: `Idle()` took the turn as the player's as soon as player actions were enabled, before the turn-start hooks
  and autoplays had run, so the next play or end turn was parked by the action synchronizer and the steps recorded no
  change. Fixed on `fix/referee-stalls` (bd68419: wait for `PlayerTurnPhase.Play` and the synchronizer's `PlayPhase`).
  With it (and `feat/checksum-tap`'s per-step game checksums) the 300 coverage fights are 300/300 identical on both
  hosts, the 29 holding Mayhem or Hellraiser (17 play one) included, and none freezes. Without it, full-host results
  for these fights are not a valid reference (night04 has 23 such frozen fights).
- Latent: on the full host `AfterCardChangedPiles` is skipped when a card tween's owner leaves the tree at room exit;
  nodeless always runs it. Only matters once a run spans several fights.

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
- `reanchor_compare` exits 0 only when every recorded step was re-anchored and matched; 1 on a diverging or throwing step,
  an unusable result or a game fight without a result (results are paired by seed when the order differs); 3 when
  nothing differed but some steps were not judged (broken cases: `tests/python/test_reanchor_broken.py`).

## Trace mode (re-runs of fights that diverged)

A fight replays identically from its spec, so the costly recording is paid only on re-runs:

```
python referee/trace_divergent.py <batch out.json> --game <your own game copy> [--specs <job.json>] [--max 10] [--servers 1..3]
```

- It finds each diverging fight's step (first divergence of the plain replay, else the first primary re-anchored one), re-runs the spec through `serve.py` with `"trace": true` and `max_actions` = that step + 2, checks the re-run reproduces the batch up to the step, replays it with `EmuDump reanchor --trace-to`, and writes `<name>.packs/traces/<seed>.trace.txt` (+ `.json`, `index.md`, the traced records). `context_pack` folds each group's trace diff into its pack. `--reuse` redoes only the emulator side.
- Game side (`driver/Trace.cs`, off unless the spec asks): per step the game's CombatHistory entries, each GameAction run, each `Hook.Before*/After*` called, each RNG draw (stream, counter, two callers) and marks with the visible state (play/played, turn_end, enemy_turn, move/moved, turn_start), plus hidden state (card costs with modifiers, enchantments, afflictions, relic counters and fields, power amounts and fields).
- Emulator side (`EmuDump reanchor --trace`, `tools/emudump/Engines/Old/OldTrace.cs`): the same marks from `Core/CombatTrace` probes, every draw of a combat stream with its caller, and the hidden state. The probes cost one null check in normal runs.
- `python -m tools.fidelity.trace_diff <traced out.json> <traced reanchor.json> [--seed S] [--step J]` aligns the marks, measures each side from its own start state, and prints the first segment whose changes differ (with the game's log, hooks and draws there), whether the re-anchored start already differed (a write-back gap), the first draw one side makes and the other does not (per stream), and hidden-state differences at the step start.
- `"trace_listeners": true` (with `"trace": true`; off by default, the recording is unchanged without it; the L patch, `driver/ListenerTrace.cs`): the game's hook-listener order. Each step's `trace` also gets, per listener enumeration the game makes, `["LB", id, hook, "combat"|"run"]`, one `["L", id, model, kind, [side, slot], index]` per listener in the order the game yields it (index: power application order, relic obtain index, potion slot, orb position, or `[pile, position]` for a card and for the affliction and enchantment on it), and `["LE", id]`; the `Modify*`, `Should*` and `Try*` dispatchers also log an `H`. The metadata is the state when the game built its list. It only observes (a read-only wrapper on `CombatState` and `RunState` `IterateHookListeners`). Volume: a short fight with it on is 5 to 12 MB of output (about 600 L records a step on average, up to tens of thousands), so keep `max_actions` small and run one fight per job. The game copy's startup or menu reload can stall past the driver's watchdog (main thread blocked 40 s, exit 3221226505) under machine load: seen 6 times in about 40 sessions, with the key on and off, and at a session's start as well as after 4 to 5 fights; a session of 5 or 10 such fights in one process hit it both times it was tried with the key on. `python -m tools.kernel.listener_check <out.json>...` checks each enumeration against `hook_table.game_order_key` and prints, per hook family, dispatches, agreeing, disagreeing (with the first pair) and unplaced listeners.

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
A fight recorded without the tap cannot be checked and fails `--check` (UNCHECKED); in a tapped fight a step
without checksums is late. The compare modes exit 1 on any difference or gap: a fight the other side has no
result for, a driver artifact, a missing trace, an emulator replay that stops before the game's last step or
reports an error (broken cases: `tests/python/test_netstate_broken.py`).

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

- Taken as divergences, never as passes: a game fight the replay never played, an offer the replay did not export
  (an event's options are exported as its action mask, `enabled`, and compared at every index), a snapshot it did
  not export and a replay paired with another seed's run. With `--id-map ''` the fight steps are not compared
  (printed, never counted identical) while each fight's outcome still is (broken cases:
  `tests/python/test_run_compare2_broken.py`).

To compare a new relic counter, add it to `RunSnap.RelicProps` (getter and setter in the game's units;
`Absent` for a `SaveIfNotTypeDefault` property, mirrored in `run_snapshot.SAVED_IF_NOT_DEFAULT`; a
counter the game reads modulo n goes in `run_snapshot.PROP_MOD`).

`EmuDump runstart` + `run_compare2 start` compare run starts (all three acts' room sets, grab bags,
odds, counters) without playing.

Driver notes: `?` points are entered in Fast mode (PunchOff's endless animation loop freezes the game
in Instant mode), the chest opens through `NTreasureRoom.OpenChest`, the Crystal Sphere is played with
`SetTool` and `CellClicked`, the Fake Merchant's stall is bought from, potions are drunk in fights with
spec `"potions": true` (never outside fights: the emulator has no action for that).
## Oracle-free checks (flag emulator bugs without the game)

Checks that hold whatever the game does, run over every recorded fight (`jobs/*.out.json`, each distinct fight once) and over
the emulator alone. A violation in the game's own recording is a recording or driver issue; in the emulator it is a bug or a
tool artefact (triage below).

```
# invariants on the game's recording and on the emulator's replay of it (hp, energy, block, powers, piles, rng, card conservation,
# pile order), plus the engine's own live state after every replayed step (EmuDump invariants)
python -m tools.fidelity.invariants run --jobs D:/spireline/referee/jobs --emudump tools/emudump/bin/Release/net9.0/EmuDump.dll --work <scratch> [--json report.json]
# metamorphic relations on the emulator alone, from every distinct recorded start (determinism, clone, pad, twins, symmetry,
# streams, hand order, discard order, energy, monotone, upgrade, linear, invariants)
python -m tools.fidelity.invariants meta --jobs D:/spireline/referee/jobs --emudump tools/emudump/bin/Release/net9.0/EmuDump.dll --work <scratch> --seeds 4 [--relations twins,pad] [--json meta.json]
dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll meta <out.json> --relations clone --chooser <seed> --diff 40     # one failure, wide diff
```

- `data/card_flow.json` (which game models create cards, which powers may repeat or go negative) is made from the local decompile:
  `python -m tools.fidelity.card_flow build --decompiled D:/spireline/referee/decompiled --out data/card_flow.json`.
- The relations live in `src/Sts2Emulator.Tests/Support/Invariants/` and each is tested twice in `MetamorphicTests`: it must hold on the
  engine and must fail on a copy of the engine with one fault put in.
- A relation that fails is read before it is trusted: some cards read the very thing a relation changes, by design (Mind Blast and
  Seeker Strike read the draw pile, Aggression takes an `UnstableShuffle` of the discard pile, Stampede rolls an index into the hand).
  `TRIAGE` in `tools/fidelity/invariants.py` records each such reason, checked against the game's class.
## State lifecycle and the byte-safety of a refactor

**State lifecycle** (`tools/emudump.Tests/LifecycleTests.cs`, `--filter Category=Lifecycle`, ~10 s). Every field of
CombatState and RunState and what they hold, by reflection (`tools/emudump/Engines/Old/OldStateWalk.cs`), is set and checked at each
stage: Clone and CopyTo (copied, not shared), CombatFactory.Reset on a reused combat and RunEngine.Reset (fresh), the
dump and RunSnap.Export (shown), CombatSnapshot (loads back), Reanchor (written, or carried on purpose). Fine-by-design
gaps sit in `Allowed` with the reason; known defects in `tools/emudump.Tests/lifecycle-open.txt`. A stage fails on a
finding in neither (a new field nobody handles: handle it, or add the reason) and on an open line that is fixed (delete
it). The reasons that say a field is idle between steps rest on `Transient_fields_are_idle_after_every_step`: random
play with decks that auto-play cards (Hellraiser, Havoc, Knife Trap, Sly discards), open selections and end fights,
judged against the open list too (`transient` while the fight goes on, `transient-end` after a step that ended it; none is
open on main: a fight-ending step leaves no auto-play queued or card held now). `LIFECYCLE_STRICT=1` fails on the open ones too. The full
field x stage table is written to `tools/emudump.Tests/bin/Debug/net9.0/lifecycle-ledger.md`.

**Byte-safety of a refactor** (commit A, structure only): the same state after every step, every field, and the same
RNG draws by caller, not just the same dump bytes or counts:

```
dotnet build -c Release tools/emudump                  # in each checkout; keep the old build's DLL somewhere
python tools/fidelity/corpus.py identity --a <old EmuDump.dll> --b <new EmuDump.dll> [--limit N]   # EmuDump fight --state-hash
python -m tools.fidelity.rng_diff --a <old EmuDump.dll> --b <new EmuDump.dll> [--limit N]       # EmuDump fight --rng-log
```

`identity` names the first differing step's fields (`--state-hash=fields`); `rng_diff` names a draw that moved to
another caller, step or stream position under equal counts, or a draw one build makes and the other does not. Each
can miss what the other sees: two enemies' rolls swapped under one caller can leave the draw log the same, and a
draw moved ahead of another can leave every step's state the same, so a refactor needs both. Both exit 1 on any
difference or a replay error on one side; both builds need these flags. Both replay the same fights: every corpus
fight this checkout's EmuDump source can replay, the setup fights included (47 draw sites, by caller, occur only
there: potion rolls, the creation of Inklet, Fossil Stalker, Exoskeleton, Scrolls and others). A fight whose spec needs a feature the source lacks (`enemy_hp_mult`: 80 fights
today) is left out by both, and both say so: `skipped, ...` (identity) and `NOT COMPARED ...` (rng_diff, in its last
line too); that count is a gap in the check, not a pass. `rng_diff` compares draws by caller, so a refactor that only
moves a draw into another method (extract-method around a draw) fails it though nothing else changed: that is by
design, and the draw's new caller is what the diff names. Broken cases: `tests/python/test_build_identity.py`.

## Mutation testing (how strong are the recorded-fight checks?)

Seeds bugs into the combat core (CombatEngine, BuffSystem, EnemyAI, Effects/*) and counts how many the
recorded fights catch. Local data only; the run folder holds game ids, never commit it.

```
python -m tools.fidelity.mutate prepare --out-json <out.json> --emu <emu.json> --dir <run>   # identical fights, probes, hashes
python -m tools.fidelity.mutate schemata --dir <run> --reanchor <reanchor.json> --mutants 150   # ~3 min for 150 x 529 fights
python -m tools.fidelity.mutate report --dir <run> --results <run>/schemata/results.jsonl
```

- Killed = some fight now has an emulator error or a first divergence (batch.py's test) or a re-anchored step that
  matched no longer does; a hang or crash also kills. `silent` counts fights whose replay changed but passed both
  checks: the change is in state the checks do not compare.
- `schemata` builds every mutant into one engine behind `MutCov.Is(id)` and replays them in long-lived
  `EmuDump muthost` processes; `run` builds one EmuDump per mutant instead (slow: ~3 min per surviving mutant,
  but independent; `--ids` cross-checks schemata verdicts).
- Mutants are drawn from blocks the fights ran, weighted by sqrt(fights); a block no fight runs is a blind spot
  by construction (`prepare` prints how many mutants sit in such blocks per file).

## Hook table (every game hook override against the emulator's dispatch points)

The game dispatches every gameplay hook through `Hook.cs` over one listener list per pass; the emulator has a
hand-written dispatch point per hook. This lists all 951 (model, hook) overrides of the decompile (inherited ones
included) and holds each one to the emulator. Names and paths only: no game code is written anywhere.

```
python -m tools.fidelity.hook_table game  [--json game_hooks.json]     # overrides, Hook.cs passes and iterators
python -m tools.fidelity.hook_table check [--json table.json]          # exit 1 on a MISSING override or a stale entry
python -m tools.fidelity.hook_table order --work <scratch> [--json o]  # exit 1 on a new order inversion or a harness error
python -m tools.fidelity.hook_table mutants --run <mutation run dir> --work <scratch> [--json m]   # kill rate
```

- The decompile comes from `--decompiled` or `SPIRELINE_DECOMPILED` (default `D:\spireline\referee\decompiled`).
- `check`: an override is *registered* where the model's emulator id is named in one of its hook's dispatch
  points (`tools/fidelity/hook_dispatch.json`: methods, a private helper they call, or a region of `EndTurn` /
  `ResolvePlayRepetition` cut at the CombatTrace marks), or next to a comment citing `Model.Hook` in the same method;
  `generic` entries name a mechanism that serves a listener without naming it (costs derived in `EffectiveCost`,
  the per-turn play tallies). *Not modelled* needs a reason in `hook_unmodelled.json` (achievements, run
  modifiers, badges, multiplayer, test mocks, unreachable powers, feedback-only bodies, combat-local resets).
  Everything else must be in `hook_open.txt` as a known candidate bug, or the check fails; a fix deletes its line
  (an open line that is no longer true fails the check too). `tests/python/test_hook_table.py` runs `check` on
  this checkout whenever the decompile is there.
- `order`: probes go before every dispatch-point statement that names a listener (a scratch copy of the engine
  only), `EmuDump hookprobe` runs eight synthetic combats (combat start, attack, skill, power, exhaust, draw,
  potion, a full turn round) holding every registered listener of their hooks, twice (obtain and application
  order reversed), and the probe order is held to the game's: within one dispatcher pass, creature (player
  first), kind (powers, relics, potions, orbs, cards), then obtain order; across hooks, CombatManager's phase
  order. `ORDER` rows are gated (`order ...` lines in `hook_open.txt`); `commutes` (value hooks inside one pass)
  and `index` (obtain or application order within one kind) rows are reported only. A screen that pauses the turn
  start's setup (a relic's pick, Entropy's transform) lets the game run on to AfterSideTurnStart before it is asked,
  so the harness marks each screen it answers and an AfterSideTurnStart listener before a mark with a setup
  listener after it (no earlier in the game's phases than where the setup had reached) is in order
  (`hook_order.pause_floors`); the same pair with no screen between them is still an inversion.
- Limits: a probe fires where the listener is checked, not where it acts; a probe in a method listed for several
  hooks of one model makes no claim; per-event hooks (hits, draws, exhausts) are compared within one creature;
  monsters, afflictions, orbs and enemies' combat-start powers are not in the scenarios (one dummy enemy).
- `mutants` carries a mutation run's survivors onto this checkout by their line text and counts a kill when the
  table changes (the mutant applied as text) or the order run changes (all mutants behind `MutCov.Is(id)` in one
  instrumented build; the harness is deterministic, a no-op mutant gives the base log byte for byte).
- Broken cases: `tests/python/test_hook_table.py`; `SPIRELINE_HOOK_E2E=1` adds the build-and-run one (the turn
  start's AfterPlayerTurnStart powers moved ahead of BeforeHandDraw must fail the order gate).
- When the engine renames or moves a dispatch method, `check` reports `STALE dispatch point <hook>: <Method> is not a
  method of the engine`: point the hook at the method that now dispatches it (`generic` entries that start with
  `File.Method:` are held to the same test). Do not move the row to `hook_open.txt` unless nothing models it.
- Measured on main b03cefd8 (2026-10-15, the selection mechanism landed): 951 overrides, 704 registered, 136 not
  modelled, 111 open, 0 missing; 91 gated order rows (turn one's combat-start relics now come before AfterPlayerTurnStart's but still after
  BeforeHandDraw's, which run first up to the relic whose screen holds the turn; the turn start's AfterPlayerTurnStart
  powers run in list order after the draw); of the 344 survivors of the 2f5fb8b mutation run, 294 carried over (the rest were on lines the selection
  work changed) and 31 killed (10.5%: 15 by the table, 24 by the order run; 15 would fail the checked-in gates),
  delete-call 21/56, swap-calls 2/8, cmp-negate 7/102, cmp-boundary 1/64, literal changes 0/61.

## Nightly (main against the previous main)

`python -m tools.fidelity.nightly run` builds main's EmuDump in `D:\spireline\referee\nightly\wt-night`, replays the
corpus and runs `corpus.check` against last night's run (a first night: the nearest ancestor's corpus baseline),
bisects up to `--bisect` fights that got worse (`git bisect run`, one fight per commit, after checking that the fight is
worse at tonight's commit and not at the previous one under the bisect's own judge; otherwise it says so and blames
no commit) and writes `<date>-<commit>.md`: five numbers, the bisects, the gate's lines. Exit 1 when the gate fails, 2
when the night could not run or had nothing to compare with. Scheduling it (Task Scheduler, 03:00) is a standing
task: the owner's go first. Broken cases: `tests/python/test_nightly.py`.

## Debugging benchmark (`debugbench.py`: how good is our debugging?)

SWE-bench from our own history: past engine fixes, undone on today's main, for a fixer to find again. Local data only,
in two places, never committed. The **bench** (`D:\spireline\referee\debugbench`) holds each task's public half (its
base, how many tests are hidden), its signal and main's gate results; builds run in worktrees under `debugbench\wt`
and the main checkout is never touched. The **truth store** (`--truth-store` or `SPIRELINE_DEBUGBENCH_TRUTH`, no
default, outside the bench; here `D:\spireline\referee\debugbench-truth`) holds the answers: each task's fix, message,
hidden tests, sites, mechanism and diffs, the harvest log, the harvest and score logs and every score. Set the variable
in the scorer's shell only.

```
python -m tools.fidelity.debugbench --truth-store <dir> harvest [--jobs 3] [--only <sha>...]   # tasks from main's history (resumable)
python -m tools.fidelity.debugbench signal --all [--missing] [--gates corpus,invariants,refset,codex]   # what the gates show on each base
python -m tools.fidelity.debugbench list [--split dev] [--truth]       # --truth needs the store; held-out truth only with --heldout
python -m tools.fidelity.debugbench --truth-store <dir> run <task> --run <name> --sandbox D:\spireline\sandbox --fixer "<command>"
python -m tools.fidelity.debugbench --truth-store <dir> score <task> --repo <dir> --ref <branch> --run <name> [--verdict accept --round 1]
python -m tools.fidelity.debugbench --truth-store <dir> report [--run <name>] [--split dev|heldout|all]
python -m tools.fidelity.debugbench --truth-store <dir> migrate        # a bench made before the truth store
```

- **Harvest.** A candidate is a non-merge commit on main that changed `src/Sts2Emulator/` and a test file. It becomes a
  task when it is a fix (no `feat:`/`refactor:`/`style:`/... prefix, at most `--max-lines` 300 engine lines, not the same
  engine patch as another candidate), its engine change reverts cleanly on main (3-way, engine files only), a test of
  the classes it touched fails at the fix with its parent's engine and passes at the fix (fail-to-pass, built and run),
  that test still fails on the base, and the rest of main's suites pass on the base once it is out. **Base** = main with
  the engine change reverted and the fix's fail-to-pass tests (**hidden**) and any other main test that fails on the base
  (**collateral**: it would point at the site) taken out. Outside the engine it loses every line that names the answer:
  in Markdown docs the line (a list item with its sub-items), in test sources a whole-line comment, that names the fix's commit,
  a hidden or collateral test, or code the gold patch adds that the base lacks (a game name in a comment is no hint);
  a name left in code or another file is listed in the truth (`hints.left`). Its one commit says `base`. **Truth** = the
  gold patch (base to main, engine only), its files and sites (`file::Type.Member`, `[case X]` inside a switch section;
  a comment or blank line is no site), the **mechanism** (per site its callers and callees on main; the hook table's
  (model, hook) pairs whose dispatch points the base changes, `tools/fidelity/hook_table.py` with the local decompile)
  and the fix's commit message. The task id is a hash of the fix commit; the split is by that hash
  (`HELDOUT_PER_MILLE`), so a task never changes side. Every skip is logged with its reason (the store's
  `harvest/<main>.jsonl`).
- **Signal** (no truth), each gate against main's own result: the corpus check (the gate's rule), the invariants, the
  reference set v2 (`refset_score`'s gate per cell, fight, class and coverage) and the Spire Codex whole-run replay
  (`refset_score --codex`: every Ironclad A10 codex run walked; a run that completes on main and not on the base, or
  stops earlier, is worse). The refset run is the base's own `refset_score` under `python -I`, its scores in a private
  copy of the reference set's folder (`debugbench\main\<main>\refset`; the shared score history is never written),
  main's score made once; refset and codex share one run (about 2.5 min a task). Per gate: detected or not judged
  (an unsound comparison is never a detection), what it names, seconds to detect. A gate not run keeps its last result.
- **Fixer's loop (`run`, the harness).** Setup: a sandbox `<sandbox>\<run>\<random>\` clear of the fork and its
  worktrees (main is the answer), the bench, the truth store, `D:\spireline\src`, `D:\spireline\spireline` and the
  backlog (`SPIRELINE_DEBUGBENCH_DENY`); a one-commit repo of the base (`repo\`, branch `base`); `BRIEF.md` with the
  gates' lines and output only (no task id, commit, path or word of the fix; main is "the reference"); a Claude Code
  deny profile for those folders (`{settings}`, in the store). The fixer's command gets the prompt on stdin, the
  bench's variables stripped, `{sandbox} {repo} {brief} {report} {settings}` filled in, e.g.
  `--fixer "claude -p --output-format stream-json --verbose --settings {settings} --permission-mode acceptEdits"`;
  it fixes on branch `fix` and writes `REPORT.json` (`{"suspects": [...]}`, most likely first). Its wall time is its
  process alone (setup and scoring are timed apart), its tokens are read from its own output (the `result` events'
  usage; unknown is never 0) and what it says it cost is kept apart, never used. Its tool calls are audited: one that
  names a protected folder or the bench's tool contaminates the attempt (counted nowhere). Fixers run on detected
  tasks only; `--silent-ok` is a blind attempt, scored apart. Held-out tasks are for evaluations only: never `--heldout`,
  `list --heldout` or the store's held-out tasks during normal work.
- **Score**, by behaviour, never by the diff against the gold patch (that is kept as `differs_from_gold`): main's
  hidden and collateral tests (main's test files put back), main's other tests, the gates against main (`--gates`,
  default corpus), and the **mechanism credit**: a truth site counts whole when the candidate changes it or a member it
  calls, else by the share of its callers the candidate changes; a hook pair by the share of its dispatch points that
  are main's again. **fixed** = all tests pass, no gate worse than main, credit 1; **partial** = the hidden tests pass
  with no gate worse but a sibling is left (a collateral test fails, or credit < 1); **disputed** = no gate worse and one
  better than main while a test of main fails (a fix better than main against a test that pins main's own behaviour:
  a reviewer settles it, `--verdict accept` counts it fixed); **wrong** = the rest. Beats main = no gate worse, one
  better. The candidate is judged with the base's judge (`tools/fidelity`, `data/id_map.json` put back).
- **Report** per run: detection first; the fix rate is of the detected tasks only (silent tasks' blind attempts apart,
  contaminated attempts nowhere); partial, disputed and wrong; mean mechanism credit; beats main; localisation top-1/
  top-3; first-round acceptance; regressions added; cost as the harness measured it (self-reported cost is not
  counted). Without `--run`: the task set, what each gate detects, and what the added gates catch of what the first
  run's gates miss.
- First harvest (main 3fc51181, 2026-10-14): 453 candidates, 80 tasks (50 dev, 30 held out; 41 ours, 39 upstream
  picks). Skipped: 298 reverts conflict on main, 41 over 300 engine lines, 14 not fixes, 12 tests that need the API the
  fix added, 5 bases that do not build, 2 with no failing test, 1 root. The first signal (corpus check, invariants) saw
  22 of the 80 bases (corpus check 21, invariants 4; median 34 s). Harvesting took about 80 min with `--jobs 3`.
- Added gates (2026-10-15, the same 80 bases and main; the first run's corpus and invariants results kept): the
  reference set v2 sees 19, the codex whole-run replay 29 (main against its own score: 0 worse on either). With the
  first gates 46 of the 80 are detected (dev 32 of 50, held out 14 of 30). Of the 58 the first gates miss, the codex
  replay catches 23 and the reference set 1 (24 in all; the reference set mostly overlaps the corpus check, both
  being fights); 34 stay silent. The walk's added stops are mostly events, shops, the deck, ancients and card
  rewards: the run-level fixes. Median 165 s a task for both (one refset_score run, 16 processes), 3.7 h for 79.
  The truth store took the 80 tasks' truth (`migrate`): 20 lines naming an answer left their bases' docs and test
  comments; the hook table applies to all 80 and gives 9 of them (model, hook) pairs, 60 have callers to count.
- Limits: a fixer runs as the same Windows user as the scorer, so the boundary is the sandbox, the stripped variables,
  the deny profile and the audit, not the file system's permissions; text that describes the fix in words without a
  name the scrub knows stays in the base's docs; a fix whose tests need an API it added cannot be a task (its tests do
  not build without it); a fix later code covers (its tests pass on the base) is skipped; the mechanism's callers are
  found by name (a same-named member of another type is told apart only when called through its type). Broken
  cases: `tests/python/test_debugbench.py` (a synthetic fork).

## Fix lab (idle PC time turned into information that makes a fix cheap)

CPU is free; AI credits and fixing are the limit. While the PC is idle the lab does the work that makes each backlog item cheap to fix, so a cheaper fixer can take it: real-game fights aimed at the gaps, each divergence shrunk to a minimal setup with its first wrong step and first differing event, symptoms grouped into root mechanisms and matched to backlog items, and a fix kit per top item (the game's rule pointer, every sibling site, a generated test that fails on main). It makes no AI calls and uses no network. All of it lives in `LAB` = `D:\spireline\referee\lab` (outside every repo, never committed). The lab's emulator tree is the worktree `D:\spireline\fork\wt-lab`, detached at the fork's LOCAL main (the runner moves it when main changes and rebuilds EmuDump, the driver and the native library; it never fetches), its game copy is `D:\spireline\referee\game-lab` (`referee/linkcopy.py` from `game`).

The parts (each takes `--lab`; a step whose tool is not on the tree yet is skipped with a log line):

```
python tools/fidelity/lab_targets.py --lab LAB --out LAB/targets.json [--backlog <newest backlog-*.json>] [--codex <dataset>]   # weighted gaps: how often real Ironclad A10 runs meet it x how little covers it
python referee/scenarios.py --name <n> --mode targets --targets LAB/targets.json --fights N [--start S] [--servers K] --game <game copy> --shrink K [--jobs <dir>] [--aim all|bugs|gaps]   # aimed fights; shrink + first event for the diverging (bugs: only targets an open backlog item names; gaps: starter items weigh 1/20)
python referee/scenarios.py --name <n> --mode choices ...   # small-deck fights whose card screens are answered with edge answers (driver spec key choice_edges: none, one, max, last, unusual)
python referee/scenarios.py --name <n> --mode analyze --jobs <dir> --game <game copy> --shrink K --trace N   # judge, shrink, trace a finished batch.py batch (--shrink-cap, --shrink-minutes, --trace-minutes bound it)
python referee/runs.py --name <n> --runs N --procs 4 --game <game copy> --jobs <dir>   # whole seeded runs; a crashed session is logged and the rest compared (exit 3)
python tools/fidelity/lab_groups.py --lab LAB [--backlog <newest>] --out LAB/groups.json --summary LAB/summary.md   # divergences grouped by root mechanism, matched to backlog items
python -m tools.fidelity.packet --top N --groups LAB/groups.json --backlog <newest>   # fix kits -> backlog/packets/<id>.md + packets/grades.json (A: minimal repro + failing test + rule pointer + siblings)
python tools/lab/runner.py [--lab LAB] [--tree D:/spireline/fork/wt-lab] [--game ...] [--trial MINUTES] [--once presort|fights|runs|probes|mutation|invariants] [--dry-run] [--always-on]   # the runner (below)
powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1 [-Trigger Idle|Logon] [-WhatIf | -Unregister]   # the Windows task "Spireline fix lab" (Logon: always on)
```

- **Runner.** Started by the scheduled task (current user, Interactive, Limited, no password; idle trigger, 10 min idle, stops when the idle ends, `pythonw.exe` from the fork's main checkout, which it only reads) or by hand. Jobs by value: `presort` (when local main changed: move and rebuild the tree, then codex walk + refset score, corpus check, groups, kits), then a rotation of `fights` (one strategy per job, from `LAB/mix.json`, below: batches of 50 fights, every diverging one shrunk and traced within the job's time budget, groups, kits for the top 20; the whole-runs strategy is its own job type `runs`), `probes` (only when a kit names probe ids in `grades.json`), `mutation` (reach data from the newest fights batch, which `lab_targets` reads), `invariants`. Each job is `LAB/runs/<UTC yyyymmddThhmm>-<job>/`: `job.json` (inputs, main sha, steps, `outputs` = role to paths), `logs/<command>.log|.rc`, its outputs, and `done.rc` written last (0 ok, 1 a required step failed or every core command failed, 3 closed because main moved or it sat idle over 24 h). An interrupted job resumes at its first unfinished step (a killed command is rerun from scratch, finished commands are kept). `LAB/state.json` holds the runner's state, `LAB/status.txt` is one always-current line, `LAB/runner.log` its log. Paths in `job.json` and the logs are symbolic (`{tree}`, `{lab}`): no host names, user names or absolute user paths.
- **Mix and yield** (`tools/lab/mix.py`). `LAB/mix.json` (written with the defaults when missing, read at every choice) weighs the strategies: aimed-bugs 30, aimed-gaps 25, choice-edges 10, whole-runs 10, dense 10, under-tested 10, random 5, hypotheses 5. A weight is a share of the CPU time the fights slots of the rotation spend: the next job is the available strategy furthest below its share, 0 never runs, an unavailable one (tool, input or driver key missing on the tree) is passed over with a log line, and one that fails twice in a row backs off alone. `LAB/yield.tsv` gets a row per finished job: `utc job strategy main rc wall_min cpu_min fights mismatches unbuildable groups fix_ready screen_groups` (fix-ready = a group that is not host-suspect with a symptom of this job that has a traced first event and no mirror gap; screen_groups = groups with a symptom of this job that is a card screen the setup mirror answered differently from the game, which the choice-edges strategy looks for and which is never fix-ready because the setup has a mirror gap), so the next week's mix can be set from data. Strategy values measured 2026-10-15 (fix-ready groups per CPU-minute, 1000 fights each): aimed-bugs 10.5, aimed-gaps 8.7, random dense 2.8, under-tested 2.2; random natural/rich 97% identical.
- **Hypotheses strategy** (`tools/lab/hypotheses.py`, `scenarios.py --mode hypotheses --hypotheses DIR`, weight 5). Every static audit finding that a real-game setup fight can show is a setup file `D:/spireline/referee/hypotheses/<source>/*.json` (one object or a list; `<source>` = the audit, e.g. `order-audit`, `data-checks`): `id` (unique, no spaces), `claim` (one line), `spec` (the fight spec as the aimed fights take it: a non-empty `setup` list in the setup-fight commands of `driver/Setup.cs`, optional `character`, `ascension` (10), `neow`, `chooser_seed`, `max_actions`, `script`, `use_potions`; `seed` is derived per fight), `seeds` (a count, default 8, or a list of seed strings; use more when the situation is rare in a random hand), `compare` (`fields`: the field groups of `reanchor_compare.group`, e.g. `.hand`, `.draw_pile`, `.enemies`, `.player`; `from_step`, default 0). A finding no fight can show is `{"id", "claim", "testable": false, "why": "what it would need"}`. The strategy plays each pending setup once per job, in one batch whatever `--fights`/`--batch` say (shrink and trace off), and judges each verdicts file once (the job's `hyp-judged.json` lists them), so a not testable setup spends one attempt per job; a batch with nothing testable writes an empty verdicts file. The verdicts decide: **confirmed** = a fight first diverges in a compared field at or after `from_step`; **refuted** = every fight identical with no mirror gap; **not testable** = anything else (errors, mirror gaps, a divergence in another field, `testable: false`), with the reason. Confirmed and refuted are final; a not testable one is retried up to 3 attempts. A confirmed entry says the game and the emulator differ in that field in that setup, a lead to read with its evidence, not a proven cause. Results: `LAB/hypotheses.json`, id to `{status, reason, claim, source, fields, fights, evidence (the verdicts file, relative to the lab), utc, attempts}`; the fight's packs are beside the verdicts file. The strategy is passed over when no setup waits. Setups converted so far: `order-audit` (M1 Earring, M2 Power Cell and Aggression, M4 Summon Forth testable; M3, U1, U2 not) and `data-checks` (the card base damage and block rows, the Foul Potion target; the rest as not testable with the reason). Tests: `tests/python/test_lab_hypotheses.py`, plus the plan in `test_lab_mix.py`.
- **Guards** (`tools/lab/guards.py`): `LAB/PAUSE` or `backlog/main.lock` present: no step starts and a running one gets `--grace` (90 s) to end, then is stopped; `echo now > LAB/PAUSE` stops at once. Keyboard or mouse input in the last 60 s: children stopped, state saved, exit 0 (the task restarts at the next idle); with `--always-on` (the logon task) the runner waits instead of exiting, also for `LAB/HALT`. `PAUSE` and `HALT` are read as PowerShell writes them (UTF-16 or BOM). Other processes over 50% of the CPU for 2 minutes (machine busy time minus the runner's own Job Object, so its game servers never count): children stopped, wait. `LAB` over 30 GB or `D:` under 50 GB free: prune (below), else wait. `LAB/HALT`: written when a game batch reports a network connection or a touched real profile; the runner then stops until the owner has read the logs and deleted the file. The runner runs at BelowNormal priority and in a Job Object with KILL_ON_JOB_CLOSE, so every child (servers, EmuDump, dotnet) dies with it, also when the task is stopped or the runner killed.
- **Corpus import** (`tools/lab/corpus_import.py`; the last step of a `fights` job, or by hand `python tools/lab/corpus_import.py --lab <LAB> [--all] [--dry-run] [--corpus <dir> --no-marker] [--budget-gb N]`): the job's fights the emulator matched exactly go into the regression corpus (`tools/fidelity/corpus.py`) as they are, the ones it diverged on tagged `lab-diverged` (the baseline holds their verdict; `check` flags them only when they get worse); one fight per setup (`corpus.spec_id`), matches first, up to a size budget on the corpus fights folder (`--budget-gb`, 2 in the plan) and a cap on the corpus's fight count (`--max-fights`, 100000: a full replay costs time, not disk); error, mirror-gap, setup-first and broken-off fights are left out. The job gets `corpus-imported.json`: per `.out.json` its size, mtime, whether the corpus now holds every setup in it, and those setup ids. `--dry-run` counts and writes nothing; an import into a scratch corpus needs `--no-marker` so no job points at it.
- **Disk** (`tools/lab/prune.py`), cheapest first: tier 0, a `<name>.out.json` that `corpus-imported.json` lists as covered, unchanged since the import, read by a later `groups.json`, with every listed setup still in the corpus index (the corpus holds it compressed); then scratch folders (`serve/`, `chunks/`, `work/`) and shrink replays; then the raw entries of fights that matched, in `<name>.out.json|.emu.json|.reanchor.json`, become the stub `{"seed", "pruned": true, "steps": []}` in place (lists stay aligned with `<name>.verdicts.json`, which is never touched); then raw outputs of jobs over 7 days old that a later `groups.json` has read. Never removed: minimal packs, groups, summary, targets, state, `job.json`, `done.rc`, verdicts, summaries, and the kits (outside the lab).
- **Always on** (a week away). `register_task.ps1 -Trigger Logon`: starts at logon, no idle condition, no execution time limit, priority 7, one instance, restart on failure every 5 minutes up to 3 times, Interactive and Limited with no password, runner flags `--always-on` (and no `--exit-with-parent`) and `--lab-gb 100` (`-LabGb N` sets it). The idle trigger stays the default. The lab does not change the power plan: a PC that sleeps stops the lab.
- **Using it.** Look: `LAB/status.txt`, `LAB/yield.tsv`, `LAB/summary.md` (top groups), `backlog/packets/grades.json`. Plan without running: `python tools/lab/runner.py --dry-run`. Trial before registering: `python tools/lab/runner.py --trial 10`. Register after a trial: `register_task.ps1`; stop: `Stop-ScheduledTask -TaskName "Spireline fix lab"` or `echo now > LAB/PAUSE`. `--skip codex,refset,corpus` and `--no-guard user,load` are for trials. Give a fixer the kit path of its item (step 4b); a kit graded C is still evidence, not a verdict.
- Tests: `tests/python/test_lab_runner.py` (guards with fake clocks, pruning, plans, a runner on trivial commands, the Job Object kill of a real process tree, the task script under `-WhatIf`), `test_lab_mix.py` (the mix, the strategy plans, yield rows, always-on), `test_runs_batch.py` (a crashed session), `test_scenarios_targets.py` (aim, choices, analyze, trace chunks).
