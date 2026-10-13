# Ironclad A10 reference set

One versioned set of game recordings that every accuracy number is quoted from. Earlier batches
(night03, night04) were random decks in act 1's first row, recorded by older drivers; 23 of night04's
fights froze after a turn-start auto-play and 5 were snapshotted mid auto-play, so they are not
references. The set replaces them.

## What is in v2

v2 holds the same fights as v1 (the same selection, seeds and ids); what changed is how they are set up and
recorded (below).

| Part | Specs | Why |
|---|---|---|
| realistic | 600 | Real Ironclad A10 runs (Spire Codex, v0.107.1): the run's seed and Neow pick, then its deck (upgrades, enchantments, saved properties), relics (with their saved state), potions, potion slots, HP and gold as they stood before a floor, then that floor's encounter in its own act. 200 per act: 110 normal (3 floor buckets), 55 elite (2), 35 boss; one fight per run. Played by the heuristic chooser. The headline. |
| coverage | 283 | The fewest plausible setups so that every Ironclad and colourless card, base and upgraded, is in enough decks to be played 5 times, every act 1-3 encounter is met 3 times (in its act), and every Ironclad-reachable relic (3) and potion (5 belts, about 3 uses) the emulator models is held. Random play. |
| regression | 117 | One minimal setup per known divergence class (FINDINGS.md, triage files, shrunk setup fights) and the original fight that found it, re-recorded with today's driver; each class names its trigger. Random play (originals: as first recorded). |
| runs | 20 381 + 30 | Every Ironclad A10 codex run through the codex replay (run-level offers), and 30 seeded whole runs for the run referee. |

`manifest.json` lists every spec with its reason, strata, chooser, realism flags and source hashes (codex
file, export, setupinfo). It is local (`D:/spireline/referee/refset/<version>/`) and never committed:
realistic specs are derived from per-run third-party data.

## Commands

```
python -I referee/refset.py build --version v2                      # the manifest (about 7 min: reads the codex once)
python -I referee/refset.py record --version v2                     # every fight, twice (the final record)
python -I referee/refset.py record --version v2 --parts runs        # the 30 whole runs (driver/RunPlay.cs)
python -I referee/refset.py rejudge --version v2 --record <dir>     # judge a record again from its saved passes
python -m tools.fidelity.refset_score --version v2 --emu <checkout> [--codex --codex-emu <checkout>]
python -I referee/refset.py status --version v2
```

`--subset pilot` takes the pilot list (60 realistic, 60 coverage, all regression); `--limit N` the first N
of each part. Game copy `D:/spireline/referee/game-refset` (`referee/linkcopy.py`), 3 game processes.

## Setup

Realistic and coverage fights: `act` (from act 2 on: the console's act command, `RunManager.EnterAct`, so the
fight is met in its own act), then `player_load`, then `fight`. `player_load` puts the holdings in place through
the game's save-load path (`driver/RefsetSetup.cs`): each card and relic in the save's own shape
(SerializableCard, SerializableRelic) read by the game's serializer and built by `CardModel.FromSerializable` /
`RunState.LoadCard` and `RelicModel.FromSerializable`, which restore the `[SavedProperty]` state (Mad Science's
type and rider, Archaic Tooth's and Dusty Tome's picks, Touch of Orobas' relics, counters); deck, relics, potions,
slots, HP and gold are replaced with the steps `Player.SyncWithSerializedPlayer` and `Player.LoadInventory` use.
No pickup effect runs, because the record's holdings already include every one, as a save does. The item fails
when what the game then holds is not what was asked. EmuDump mirrors it with the run replay's own anchoring
(`RunReplay.AnchorHoldings`), and a saved property it cannot hold is a mirror gap.

Coverage relics an event picks for (Archaic Tooth, Dusty Tome, Touch of Orobas, ...) carry a real run's pick and
what their pickup left (the transformed starter card, the added card, the upgraded starter relic).

Regression minimal setups keep the console path (`relics_set`: RelicCmd.Obtain, `deck_set`, `hp`,
`potions_set`, `power`), since some classes are about a pickup itself; a deck `deck_set` cannot build fails the
item.

### Saved state before a floor

The codex keeps each held item's saved properties at the run's END. The value before floor F comes from
`refset.STATE_RULES`: fixed picks (event picks, act indices, skins) are the end value; counters are rebuilt from
the floors between the item's floor and F (combats, elites won, turns, rest choices, cards gained) and trusted
only when the same rule run to the end gives the end value. Over 6000 codex runs the checks pass for, e.g.,
Happy Flower 572 of 649, Pendulum 562 of 595, Book of Five Rings 482 of 505, Sword of Stone 381 of 381. Anything
the record cannot give is FLAGGED, never silently freshened: `state_unknown:<ITEM>.<Prop>` (no rule; a counter
like Pen Nib's attacks that only grows and is not fresh at the end; an item that left the run before its end),
`state_unverified:<ITEM>.<Prop>` (the end check failed). Other per-spec flags: `deck_inexact`, `potions_inexact`
(the rebuild had an ambiguous step), `potion_slots_unverified`, `map_coordinate:FUR_COAT`.

Limits every realistic fight shares (stated with the numbers): the run's RNG streams stand where the seed's run
is after Neow, not where the real run had them; the fight is met on the act's row-1 room, not the real map
coordinate; the actions come from a chooser, not the player.

### Choosers

`"chooser": "heuristic"` (realistic): drink a potion at or below half HP; else play a card, never ending the turn
while one is playable: block when an enemy intends to attack, else an attack or a power (attacks aimed at the
lowest-HP enemy), else any card; ties to the seeded rng (`driver/Chooser.cs`). `"random"` (coverage,
regression): a seeded random legal action. Each spec and each reference names its chooser.

## Recording

`record` refuses a driver build that is not a clean build of the checkout's commit (the DLL is stamped with its
commit and changed sources at build time, `SpirelineReferee.csproj`), or that lacks a capability a reference
needs (`refset.REQUIRED_CAPS`: settle_flags, play_phase_gate, watchdog_v2, console_setup, refset_setup,
player_load, act_entry, deck_residual_fails, relic_flashes, heuristic_chooser), checked on the DLL before any game
starts and again on every server's job-start line (`[referee] driver {...}`, `driver/Capabilities.cs`), which
also has to show the flag settle. The stamp of every reference names the driver commit, its capabilities, the
tools commit and sts2.dll.

Each fight is recorded twice (the second pass in reverse order on fresh servers). Excluded, by cause, and
listed in `record.json`: `setup` (an item failed: a deck residual, a load the game does not hold, the fight met
in another act), `game_error` (the game's code logged an error; each server's log is cut at the driver's Neow
line, one per fight, and attributed by the result line's 0-based `_id`; faults in the visual layer,
`Core.Nodes.*`, are kept and listed), `unstable` (the two recordings differ), `driver_wait_limit`,
`neighbour_fault`, `log_uncut`, `no_result`. A wait limit, a neighbour's leaked fault or an uncuttable log is
recorded twice more on fresh servers and judged on that pair. The safety checks are serve.py's; a breach writes
no references.

## Numbers

`refset_score` replays and re-anchors every reference with the given emulator checkout (refused when it has
uncommitted changes, or when the record names no driver build; both can be scored anyway, labelled NOT
QUOTABLE) and stores `scores/<subset>-<commit>.json` and a line in `scores/history.jsonl`. The line states, per
part, specs, kept, excluded by cause, flagged, identical as a share of the kept fights AND counting exclusions as
failures, steps verified, the emulator commit and the driver build and capabilities; the table repeats it per act
x room type.

- regression: a class passes when its fights are identical AND its trigger was reached (its card played, its
  relic fired, its potion used, its power up when an enemy attacked); driver classes are counted apart;
- coverage: a relic counts only where its effect fired (a relic flash) at verified steps, not where it was held;
- codex runs: per component, floors matching among the floors reached before each run's first difference;
  stops by class (profile-dependent apart), components at 0%, the walker's skips against the set's run count.

"Identical" compares the player's HP, max HP, block, energy and powers, the four piles (card id and upgrade),
each living enemy's name, HP, block, powers and intent, nine RNG counters and the potion belt; not relic
counters, card costs or enchantments, gold, or the hidden counters the comparer skips.

Quote the summary line, with the set version, subset and emulator commit it names.
