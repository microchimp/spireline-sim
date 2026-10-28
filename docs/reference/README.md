# Human reference curve

`codex-a10-ironclad.json` and `.md` hold what human Slay the Spire 2 runs look like for one character at one Ascension,
so the AI's real-game numbers (G1, the week's review) have something to be placed against. They are built from the
Codex corpus (`data/spire-codex/runs-<version>.jsonl.gz`, one JSON object per human run) by
`tools/reference/codex_curve.py`. The files here are the committed result; the corpus itself never is.

## D13: aggregates only

Only counts, shares, means and quartiles leave the corpus. No per-run row, no player identifier, no seed, no run hash,
no start time and no run duration is kept past the pass, written to a file, logged or reported. The tool enforces it three ways:

- every cell that rests on fewer than 20 runs (`--min-cell`) is `null` in the JSON and `-` in the Markdown. A share is
  suppressed when its population, its group or the rest of the population is under the floor, so a small complement cannot
  be read off a published share;
- the writer refuses any output that holds a field from `FORBIDDEN_KEYS` (`seed`, `run_hash`, `player_token`,
  `start_time`, `run_time`, `player_id`, `players`, ...);
- `tests/python/test_codex_curve.py` builds a five-run fake corpus and checks the selection, the survival curve, the
  suppression rule and that no seed or player field reaches the output.

Context counts (runs by character, by ascension, the selection funnel) are floored the same way (`<20`).

## Who is in the reference

Ironclad, Ascension 10, single player, standard mode, no modifiers, the game's own acts and content (no mod ids),
duplicates dropped (by the corpus's own run hash, in memory only), and abandoned runs left out of the curve (an abandoned run
ends because the player left, not because the run was lost, so it is censored; the header and the context table count
them, and `abandoned_sensitivity` gives the mean floor with them in).

Floors are the AI's floors: map points counted across acts from 1, the ancient being floor 1, the Act 1 boss floor 17,
the Act 2 boss 33, the Act 3 boss 48 and the second Act 3 boss (the win) 49 (`rl/run_fields.ACT_BOSS_FLOORS`). The corpus keeps the room a
run died in, so a death on a boss has that boss's floor, as the AI's `obs.floor` does (`checks.death_room_is_last_floor` shows
the share of lost runs whose killer is the last room's fight).

## Regenerate

    D:/spireline/src/sts2/.venv/Scripts/python.exe -I tools/reference/codex_curve.py D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz

(about the cost of reading the corpus once; BelowNormal priority on Windows; one process; the bootstrap is seeded, so a
rerun on the same corpus gives the same file apart from the generation time). Options: `--out` (path without extension, default
`docs/reference/codex-a10-ironclad`), `--character`, `--ascension`, `--min-cell`, `--boot`. A different character or
ascension is the same command with other options and another `--out`.

## AI metric to corpus column

The AI side is the A7 fields (`rl/run_fields.py`), the real-game end event (`rl/gen0.py real_runs`) and
`rl/gen_loop.boss_kill`. The corpus side is a table in `codex-a10-ironclad.json` (its source column in the corpus in brackets).

| AI metric | where it comes from | compares with | notes and gaps |
|---|---|---|---|
| mean floor, `floor` of the end event | `runs.jsonl` end row, `summary.mean_floor` | `reference.floor` (map points in `map_point_history`, across acts) and `survival.reach` | same numbering. The AI's runs include the driver's own stops (a crash counts as floor 0); humans have none |
| win | `win` of the end event | `boss_kill.win` (`win`) | |
| Act 1 boss kill, `gen_loop.boss_kill` | win, or floor past 17 | `boss_kill.act1.killed_of_all` | the same rule, applied to the corpus's floors; `floor_rule_vs_act_lists_disagree_runs` counts runs where it differs from "reached the next act" |
| `boss_act1` (reached, won, kill rate) | summary `run_fields.boss_act1` | `boss_kill.act1.reached`, `killed_of_reached` | the AI's rate is among runs that reached the boss |
| Act 2 and Act 3 boss kills | not in the A7 fields (derive from `floor`) | `boss_kill.act2`, `act3_first`, `act3_both` | the AI has no wins yet, so these are one-sided |
| `boss_damage`, `boss_damage_acts` | fraction of the boss's starting HP dealt | none | **the corpus lacks it**: no enemy HP or damage dealt per fight. Nearest columns: `boss_hp` (the player's HP and max HP entering each boss, `player_stats.current_hp`, `max_hp` of the floor before) and `survival.ended.at_actN_boss` (died on the boss) |
| `cards` `[floor, name or skip]` | each card reward decided | `cards.by_floor` (screens, taken, skip rate), `cards.taken_per_run`, `cards.skip_rate` (`card_choices` with `was_picked`, on monster, elite and boss rooms) | the AI also records rewards outside those rooms if the referee offers them; rerolls and relic alternatives are in neither. The corpus also gives the deck size by floor (`deck.by_floor`, rebuilt from the final deck, `floor_added_to_deck` and the removals); the A7 fields have no deck size |
| `rests` `[floor, heal or smith]` | each rest site choice | `rests.1` / `2` / `3` / `all` (`rest_site_choices`) | the corpus also has other choices (dig, lift, cook, ...), shown as `other`. Leaving a rest site is recorded by neither |
| `nodes` (M E R S T B ? A) | map node types taken | `map_point_type` of each map point | the corpus has them; the tool does not tabulate them yet |
| end of run (`death`, driver stops) | `end` of the end event | `end_causes.top10` (`killed_by_encounter`, `killed_by_event`) | **the AI's end row names no killer**, so there is nothing to place next to the corpus's top ten; the driver stops (`treasure_without_relic_picking`, ...) have no human counterpart |
| shop buys | not recorded (A7 note) | `bought_relics`, `bought_potions`, `bought_colorless`, `cards_removed` per map point | corpus has it, AI lacks it; not tabulated here |
| potions used, gold, relics | not in the A7 fields | `potion_used`, `current_gold`, `relic_choices` | corpus has them, AI lacks them; not tabulated here |
| decisions, wall time, divergences, re-anchors | lockstep and driver bookkeeping | none | AI only: a human run has no decisions count or divergence |
| seed | the AI's seed | none | **never in the reference** (D13) |

Differences that are not field gaps: humans pick their own runs to upload and 27% of the Ascension 10 Ironclad runs in the
corpus were abandoned (left out of the curve), so the human curve is for completed runs of a self-selected, probably
stronger-than-average group; the corpus build may differ from the build the AI played; and the AI's per-run numbers have
sampling noise from its own seeds, which the corpus's whole-run bootstrap interval does not cover.

## Reading the numbers

- The early floors of the survival curve show `-` where fewer than 20 runs ended before the floor: a share whose rest is under the
  floor is suppressed as well, so the table cannot be read backwards to a handful of runs.
- 41 of the 14,811 runs have an Act 1 list that is not 17 floors long (cause not examined); the
  floor rule and the "reached the next act" rule disagree on them (`boss_kill.floor_rule_vs_act_lists_disagree_runs`). The floor rule is
  the one the AI's `boss_kill` uses, so it is the one reported.
- The mean-floor interval is the percentile bootstrap over whole runs (2,000 resamples, fixed seed). The AI's real-game interval is
  over its own seeds, so the two are intervals of different populations.
