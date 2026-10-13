# Ironclad A10 reference set

One versioned set of game recordings that every accuracy number is quoted from. Earlier batches
(night03, night04) were random decks in act 1's first row, recorded by older drivers; 23 of night04's
fights froze after a turn-start auto-play and 5 were snapshotted mid auto-play, so they are not
references. The set replaces them.

## What is in v1

| Part | Specs | Why |
|---|---|---|
| realistic | 600 | Real Ironclad A10 runs (Spire Codex, v0.107.1): the run's seed and Neow pick, then its deck (upgrades, enchantments), relics, potions and HP as they stood before a floor, then that floor's encounter. 200 per act: 110 normal (3 floor buckets), 55 elite (2), 35 boss; one fight per run. The headline. |
| coverage | 283 | The fewest plausible setups so that every Ironclad and colourless card, base and upgraded, is in enough decks to be played 5 times, every act 1-3 encounter is met 3 times, and every Ironclad-reachable relic (3) and potion (5 belts, about 3 uses) the emulator models is held. |
| regression | 117 | One minimal setup per known divergence class (FINDINGS.md, triage files, shrunk setup fights) and the original fight that found it, re-recorded with today's driver. |
| runs | 20 381 + 30 | Every Ironclad A10 codex run through the codex replay (run-level offers), and 30 seeded whole runs for the run referee. |

`manifest.json` lists every spec with its reason, strata and source hashes (codex file, export,
setupinfo). It is local (`D:/spireline/referee/refset/<version>/`) and never committed: realistic specs
are derived from per-run third-party data.

## Commands

```
python -I referee/refset.py build --version v1                      # the manifest (about 6 min: reads the codex once)
python -I referee/refset.py record --version v1                     # every fight, twice (the final record)
python -I referee/refset.py record --version v1 --parts runs        # the 30 whole runs (needs driver/RunPlay.cs)
python -m tools.fidelity.refset_score --version v1 --emu <checkout> [--codex --codex-emu <checkout>]
python -I referee/refset.py status --version v1
```

`--subset pilot` takes the pilot list (60 realistic, 60 coverage, all regression); `--limit N` the first N
of each part. Game copy `D:/spireline/referee/game-refset` (`referee/linkcopy.py`), 3 game processes.

## Setup

Items run through the game's own commands only (`driver/Setup.cs`, `driver/RefsetSetup.cs`): `relics_set`
(RelicCmd.Obtain in the run's order, RelicCmd.Remove of extras), `deck_set` (CardPileCmd.RemoveFromDeck,
CreateCard, CardCmd.Upgrade, CardPileCmd.Add, CardCmd.Enchant; up to 3 passes, a residual is reported),
`hp` (CreatureCmd.SetMaxHp, SetCurrentHp), `potions_set` (PotionCmd.Discard, TryToProcure), then `fight`.
Never godmode, kill, win, damage or heal shortcuts. EmuDump mirrors each item (`tools/emudump/RefsetSetup.cs`).
Known realism limits: the fight is entered on act 1's row-1 map point, so act 2-3 encounters run with the
act index of act 1 (read in combat only by Decimillipede segments and a few relics); relic counters and
gold start fresh; Goopy-like enchantment amounts are the run's final ones.

## References

Each fight is recorded twice (the second pass in reverse order on fresh servers). Excluded, and listed in
`record.json`: a fight whose recordings differ (unstable); a fight whose game code logged an error (each
server's log is cut at the driver's Neow line, one per fight; faults in the visual layer, `Core.Nodes.*`, are
kept and listed); a setup item the game refused; a fight not entered. A fight that only hit a driver wait limit
(idle_timeout), or that repeats the previous fight's fault on its server, is recorded twice more on fresh
servers and judged on that pair: a fight whose own game code threw leaves its server unable to settle the next
rooms (the fast reset does not clear a faulted turn task). Kept recordings are stamped with the set version,
sts2.dll sha256, driver commit and driver DLL hash. The safety checks are serve.py's; a breach writes no
references.

Game-side failures the pilot met (excluded, not emulator bugs): `MadScience.OnPlay` throws
ArgumentOutOfRange when a generated Mad Science is played; `RelicCmd.Obtain` of Archaic Tooth
(NullReference) and Dusty Tome (ArgumentNull) fail on the map outside their own event.

## Numbers

`refset_score` replays and re-anchors every reference with the given emulator checkout and stores
`scores/<subset>-<commit>.json` and a line in `scores/history.jsonl`:

- realistic: fights identical to the end %, steps verified % (a step the plain replay or the re-anchored
  replay reproduces), per act and room type;
- coverage: identical %, and the items not yet clean, counted over the whole set;
- regression: classes whose fights all pass;
- codex runs: floors matching per component, runs matching to the end;
- open bug classes: distinct first-divergence causes.

Quote the summary line, with the set version, subset and emulator commit it names.
