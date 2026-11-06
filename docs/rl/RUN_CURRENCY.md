# The run currency at play time (training design section 3, brief 3a; big-step memo change B)

A fight is searched in HP: a win is worth its end HP plus 10 for each potion still held, a death -100. A potion is then worth 10 HP in a
hallway and at a boss alike, and the player drinks in 60% of the Act 1 hallway fights it wins holding one (humans 23%) and reaches the Act 1
boss with 0.8 potions (humans 1.6). The run currency prices the end of a fight the way the run does: by what the frozen run head
(generation 11's V_run, sha256 `f0a199fb`, the head generations 12 and 13 carry) says the run is worth from the post-fight state.
Nothing is switched: a config without `-r` plays as it always did.

## What is priced

| symbol | meaning |
|---|---|
| T | the head's expected final floor of the post-fight state (HP, max HP and the potions still held by identity; deck, relics and gold as at the start; the end-of-combat heal of Burning Blood 6 / Black Blood 12 added, which the run layer applies and the combat sim does not) |
| T_lost | the fight's own floor (a death ends the run there) |
| T_full | the head's value with full HP and every starting potion held |
| u | (T - T_lost) / (T_full - T_lost), clipped to [0, 1.5]; 0 is a death, 1 a full-HP win holding everything |
| v | -100 + u (max HP at the start + 100): the search's units, so the spread from a death to a full-HP win is today's and the exploration constant keeps its meaning |

`rl/run_currency.py` (numpy only): `FightCurrency(head, contexts)` with `u(hp, max_hp, potions)` and `__call__` (the units); `potions` is a
tuple of ids or a number (a judge leaf's count: the head is averaged over the subsets of the starting potions, interpolated between
neighbours). The search takes it as `terminal=` (`live_search.search_live`, `search_steps`, `_outcome`, `_judged`,
`judge_client.estimate`, `combat_mcts.terminal_value(sim, currency)`), so the live search, the judge leaf and the old label generator price a
fight with one module. A fight whose T_full - T_lost is under 0.25 floors has no scale and is priced as today (`fallback`, counted in the row).

## Configs

`a3-v1-r`, `judge-v0-r`, `a3-v1-k-r`, and `<any known base>-r` (`search_configs.with_currency`); the stakes and hold options go after it
(`a3-v1-r@h1e4b16+hold30`). `SearchConfig.search_kwargs(pos, terminal=...)` refuses a currency config without the fight's currency, so a
caller that cannot build one (the real-game player, the hindsight runner, the old dev-set tools) fails loudly instead of playing in HP. The
callers that can: `gen0.play_emu_run` (the run's exact view at each fight's first decision; its row gains `currency` and, for every config,
`boss_door`: [[act, potions, relics, HP, max HP], ...]) and `fight_gym.play_steps` (the start record; its row gains `u_end` and `currency`;
arm `a3-v1-r`, `fight_exam.py play --config a3-v1-r`).

## Where the view comes from

A run in progress has it exactly. A start record from a replay carries it (`run_view`, written by `fight_gym._replay_task`; the loop's
`starts.lite.json` leaves it out like the document). A record without one (the board's, the exam's, a human start) has the fight's snapshot
document, which holds HP, max HP, gold, relics, potions and the whole deck but no map: the view is built from it, an unknown boss and
region are every candidate and the value their mean, and the map-ahead block and the rows to the boss come from the mean over 300 of the
copy's Act 1 maps by floors to the boss and node type (`data/run_currency_map_mean.json`, `run_currency.py map-table`). Human starts need
nothing added: their documents carry the same fields. Not carried: gold and deck or relic changes made inside a fight, a fight inside an
event.

## The head and its audit

`python rl/run_currency.py audit --starts <exam starts>`: per act and kind the head's floors for 10 more HP at 30 / 60 / 90% of max HP, for
a potion (mean over the kinds, and each kind), for 50 gold, and the HP a potion is worth against today's 10; per (act, floor band) the mean HP
curve. It stops (exit 2) when the median potion held is worth under 3 or over 40 HP in an Act 1 hallway fight, or when V does not increase
in HP in some act and band. The frozen head is pinned by its sha256 (`FROZEN_HEAD_SHA256`); `SPIRELINE_RUN_CURRENCY_HEAD` names another.

## Fast tests

`python rl/currency_fast.py all --out DIR`: the audit; the board (200 Act 1 boss starts, `a3-v1-r` against `a3-v1` on the same draws, 400
games an arm and then 2,000 with `--games 2000`); the exam's Act 1 hallway and elite cells (100 starts, four draws, both arms: win rate, HP
lost, the share of won fights that began with a potion in which one was drunk); a 200-run copy batch of both arms on the same seeds (potions
and relics at the Act 1 boss door, the boss kill, the floor). It holds a `b-fast` QUIET.lock line, waits for a free machine, and ends with the
stop rule's verdict (`report.json`). The readings of the first run are in the HANDOFF line of the landing.

Unit tests: `tests/python/test_run_currency.py` (the arithmetic on a stub head, contexts, the map table, the head's batches, the audit and
each way it stops, the config grammar, the search's terminal function against the old units, the gym and a whole run) and
`tests/python/test_currency_fast.py` (the readings, the free-machine rule).
