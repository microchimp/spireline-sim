# The seed-matched run exam

Written 2026-11-12 (lane `feat/seed-exam`, D58 addendum 14). Code `rl/run_seed_exam.py`, tests `tests/python/test_run_seed_exam.py`. The human figures are
aggregates of the Spire Codex public run export for v0.107.1 (Ironclad, Ascension 10, standard, one player, vanilla acts); a cell over fewer than 20 runs
is not shown; the exam fifth (hash bucket 0) is never read here unless `--exam --milestone NAME` says so.

## 1. What the exam is

A human run's seed is a recipe the copy can cook again. The same seed gives our player the same maps for the three acts, the same first-room gift, the
same shop shelves and prices and the same card offers (the probe of 2026-11-05, `docs/diagnostics/2026-11-05-human-seed-replay.md`: the route of every
one of 400 human runs is a path of the regenerated map once the two items the record does not say are given; card offers match in 97.5 screens in 100,
shop shelves in 97.7, shop prices to the gold in 99, while the replay is in step). So we play the generation's player on the seeds of human runs and read
what it does against what the human did on the same seed, split by whether the human won the run. It is the run-level twin of the fight-level human
comparison (D57 addendum 6: the same 400 human fight starts, our player 64.6 in 100 against the humans' 79.8).

The question it answers: where a human is known to have won (or lost) from this start, how far does our player get, and how often does it kill the Act 1
and Act 2 boss, from the same start? The control for "are these seeds easier or harder than the lab's own": the same agent's figures on the lab's own
judgement seeds, in the last column.

## 2. What the seed fixes, and what it does not

Fixed by the seed (and by the earlier rooms' draws, not by the fights): the map of each act, which monsters, elites, boss and events each act holds, the
"?" room results, the first-room gift, card offers, shop shelves and prices, chest relics. The same seed puts our player at the same decision screens
the human had, at the start.

Not fixed:

- Everything the fights draw. Shuffles, monster actions, targets, monster HP rolls and the `niche` stream that random transforms, upgrades and curses use
  are carried through the run and depend on how the fights were played, so after the first fight where our move differs from the human's the run is its own.
  That is the point of a draw: the same seed with another stream of the player's own search (`--draws`) gives a different run.
- The route and the picks. Our player chooses its own path, cards, shops and events; the offers it sees match the human's only while it makes the same
  non-fight choices in the same order (the offers depend on the earlier draws of the same kind, not on which card was taken: control (a) of the probe).
- The human's profile. About 5 players in 100 had a different first-room gift set than the copy gives (the copy assumes a fully unlocked profile) and about
  1 in 100 a different early fight order or boss; the human's own unlocks are not known to the copy.
- Our player's potion, relic and card use are its own; the human's run only fixes where it began.

So this is a paired start, not a paired run: the humans' floor and boss figures on a seed are the result of that human's whole run, and ours the result of
ours. Read the split by "the human won" as "starts where a win was found" against "starts where one was not found", never as a difficulty rating of the seed.

## 3. The one command

```
python rl/run_seed_exam.py play --fifth 4 --seeds 400 --draws 2 --agent agent/current.json --out D:/spireline/referee/seed-exam/<date>-gen<N>
python rl/run_seed_exam.py report --out D:/spireline/referee/seed-exam/<date>-gen<N>
```

`play` reads the corpus once (about a minute: the exam fifth's lines are dropped before a record is seen, then only runs of the asked fifth are kept), picks
`--seeds` runs, half won by the human and half lost, the ones with the smallest salted seed hash first, one run per seed, and plays each seed `--draws`
times with the lab's own run loop (`gen0.play_emu_run`: the agent in `--agent`, its search schedule at the lab's config `a3-v1`, CPU prior, Ascension
10, Ironclad, the default engine, one emulator per process, a spawn pool of `--procs` workers (default 4) at BelowNormal). The first draw of every seed
is played before any second draw, so a stop leaves every seed with a first draw. A rerun into the same `--out` keeps its finished rows and plays the rest
(another agent, config or pick is refused unless `--force`). A worker that died is run again once; a run past its wall limit (the lab's, times the
agent's stakes factor) is a `lost` row, not run again. Crashed, timed-out and lost runs are counted and left out of the means.

Files in `--out`: `runs.jsonl` (one row per run: the seed's key, the draw, whether the human won the seed, `end`, `win`, `floor`, `boss_damage_acts` (the
lab's boss rows: act and the share of the boss's HP taken, 1.0 = killed), `door` (the HP and max HP the first decision of each boss fight began with),
the route, card picks and rest choices, decisions, wall and CPU), `human.json` (the humans' figures on the picked seeds, all and split), `summary.json`
and `table.md` (the table), `resume.json`.

`play --records DIR` also writes one derived record a run, `DIR/<seed key>-d<draw>.json.gz` (`rl/seed_exam_records.py`: the lab's `RunRecorder` with accounting on top; the run is the run the exam plays, and a
replay of a finished exam into another `--out` reproduces its rows exactly, floor, route, picks, decisions and simulations, when the tree, the engine build and the agent are the exam's). Per fight: the deck,
relics, potions, gold and HP it began with, its enemies, its turns (the enemies' HP at the start and the end of each of our turns, our HP, cards and potions played) and whether it was won; per decision: its kind,
the option taken, what was offered, HP, gold and deck size. The folder is handed to the spawn workers through the environment (`SPIRELINE_EXAM_RECORDS`). Note for a replay: `rl/data/card_tags.json` (the cache
`rl/card_tags.py` writes) changes the combat net's inputs, so a tree that has it plays another game than one without; a measurement worker must not create it in the tree it replays in.

What is never written: a seed (rows carry `seed_key`, `sha256("spireline-seed-exam|" + seed)[:16]`; an error text that names the seed is scrubbed to its
key), a run id, a player id (never read). The humans' per-seed figures stay in memory; the paired block is computed from them and only its aggregates are
written. `--exam` (with `--milestone NAME`, and only then) reads the exam fifth and appends one line to `referee/seed-exam/exam-use.log`.

Human door HP is the HP carried into the boss fight (the rest site's own figure before it); ours is the HP at the first decision of the boss fight. Floor
is map points entered, the human bar's definition. Boss "reached" is the share of all runs; "killed" is of the runs that reached it.

## 4. Settings of the first reading

Generation 13 (`gen13/candidate-stakes`: the combat prior of generation 11, V_run, the human-prior blend at lambda 0.5, thinking time x1 at hallways, x4
at elites, x16 at bosses; the lab's `current.json` at the time), config `a3-v1`, CPU prior (no prior server), the default engine (the lab's self-play runs
the old engine; `--engine core2` is there but is not what the lab plays whole runs on), Ascension 10, Ironclad, fifth 4 of the corpus: 2,901 usable runs, 323
won by the human (11.1 in 100), 2,578 lost; 200 + 200 picked by the smallest seed hash (2 repeated seeds skipped); 2 draws each, 800 runs, 4 processes (3 for
the first 270 runs) at BelowNormal beside the teacher's round 0, the corpus lane and a judgement. Every run ran to its end (800 deaths, 0 wins, no crash, no
timeout, no lost worker). Cost: 34,135 CPU-s (9.5 core-hours, 42.7 CPU-s a run, the stakes schedule's x16 bosses; the brief's 18 CPU-s was the plain
player's), about 4 hours of wall on the shared machine. Folder `D:/spireline/referee/seed-exam/2026-11-12-gen13/`.

## 5. The first table

seed-matched run exam (Spire Codex public run export for v0.107.1; Ironclad, Ascension 10; fifth 4; 400 seeds x 2 draws; agent generation 13, config a3-v1, engine old)

| figure | our player | humans, same seeds | our player, human won | humans, won | our player, human lost | humans, lost | copy on its own seeds (gen 13) |
|---|---|---|---|---|---|---|---|
| runs | 800 | 400 | 400 | 200 | 400 | 200 | 6494 |
| win rate | 0.0% [0.0, 0.0] | 50.0% | 0.0% [0.0, 0.0] | 100.0% | 0.0% [0.0, 0.0] | 0.0% | 0.0% |
| mean floor reached | 25.1 [24.3, 25.9] | 35.7 | 25.8 [24.7, 26.9] | 49.0 | 24.4 [23.3, 25.6] | 22.3 | 24.4 |
| median floor | 24.0 | 49.0 | 25.0 | 49.0 | 24.0 | 17.0 | 23.0 |
| Act 1 boss reached (of runs) | 90.5% | 81.8% | 92.0% | 100.0% | 89.0% | 63.5% | 90.0% |
| Act 1 boss killed (of those reaching it) | 66.3% | 86.2% | 68.8% | 100.0% | 63.8% | 64.6% | 63.5% |
| Act 1 HP at the door | 63.6 (76.7% of max) | 55.1 (67.3% of max) | 64.0 (77.3% of max) | 55.8 (67.5% of max) | 63.3 (76.2% of max) | 54.1 (66.9% of max) | - |
| Act 2 boss reached (of runs) | 31.6% | 64.2% | 33.5% | 100.0% | 29.8% | 28.5% | 27.4% |
| Act 2 boss killed (of those reaching it) | 30.8% | 90.3% | 34.3% | 100.0% | 26.9% | 56.1% | 33.8% |
| Act 2 HP at the door | 60.9 (70.9% of max) | 63.4 (73.5% of max) | 63.3 (73.2% of max) | 64.1 (73.7% of max) | 58.1 (68.4% of max) | 60.7 (73.1% of max) | - |

Ours: runs that ran to their end; brackets are 95% intervals over seeds (two draws of a seed are not independent). The humans' columns are the records of the same seeds, so by construction their win rate is 100 in 100 on the seeds the human won and 0 on the ones it lost; the 'all' columns are half and half. The copy's own seeds are the lab's judgement blocks of the same agent (no HP at the door there).

Weighted to the fifth's own mix (11.1 in 100 of its usable runs were won by the human; the pick is half and half), the two splits of each side mixed at that share:

| figure | our player | humans |
|---|---|---|
| win rate | 0.0% | 11.1% |
| mean floor reached | 24.6 | 25.3 |
| Act 1 boss reached (of runs) | 89.3% | 67.6% |
| Act 1 boss killed (of those reaching it) | 64.3% | 70.4% |
| Act 2 boss reached (of runs) | 30.2% | 36.5% |
| Act 2 boss killed (of those reaching it) | 27.8% | 69.5% |

On the same seed (our draws against the human's run, aggregates):

| figure | all seeds | human won | human lost |
|---|---|---|---|
| seeds | 400 | 200 | 200 |
| floor, ours minus the human's | -10.5 [-12.2, -8.9] | -23.2 [-24.4, -22.1] | +2.2 [+0.2, +4.1] |
| our best draw reached at least the human's floor | 37.2% | 1.0% | 73.5% |
| seeds where some draw of ours won | 0 | 0 | 0 |
| Act 1 boss: our draws that killed it, where the human did | 63.8% of 282 seeds | 63.2% of 200 seeds | 65.2% of 82 seeds |
| Act 2 boss: our draws that killed it, where the human did | 12.7% of 232 seeds | 11.5% of 200 seeds | 20.3% of 32 seeds |

A cell over fewer than 20 runs (or 20 seeds) is not shown (-).

## 6. What it says

- **No win was found.** In 800 runs on 400 seeds, including the 200 seeds a human won, no draw of ours won; the lab's own blocks of the same agent have 0 wins
  in 6,494 runs too. The exam reads floors and bosses until the player starts winning.
- **The seeds are not easier.** On all 400 seeds the player reaches floor 25.1 [24.3, 25.9] against 24.4 on the lab's own seeds; the Act 1 boss is killed by
  66.3 in 100 of the runs reaching it against 63.5; the Act 2 boss by 30.8 against 33.8. On the seeds the humans lost the floor and the Act 1 boss kill are the
  lab's own (24.4 and 63.8 against 24.4 and 63.5).
- **Perhaps a small seed effect.** On the seeds a human won the player does a little better than on the ones a human lost: floor 25.8 against
  24.4, Act 1 boss killed 68.8 against 63.8, Act 2 boss killed 34.3 against 26.9. The floor intervals overlap and the difference of 1.4 floors is about
  one interval wide, so at 400 seeds it is not established; if it is real the start (the map and the offers a human found a win in) explains a floor or
  so, and the rest of the gap is in how the run is played.
- **Against the humans on the same seeds.** On the seeds the human won, the human reached 49.0 floors and both bosses every time; ours reaches the Act 1 boss in
  92.0 in 100 runs, kills it in 68.8 of those, reaches the Act 2 boss in 33.5 and kills it in 34.3 (the human's 100 is by construction: a won run killed
  both). Seed by seed, our mean floor is 23.2 below the human's [-24.4, -22.1] and our best draw gets as far as the human in 1.0 in 100 seeds; where the human
  killed the Act 2 boss, our draws kill it on 11.5 in 100 of the seeds' draws. On the seeds the human lost, ours gets 2.2 floors further [+0.2, +4.1] and as far
  in 73.5 in 100.
- **Weighted to the fifth's own mix** (11.1 in 100 won; the humans' record on all their runs, ours as the two splits mixed the same way) the floors are level, 24.6
  against 25.3; the player reaches the Act 1 boss more often (89.3 against 67.6) and kills it less (64.3
  against 70.4), and the Act 2 boss is where it falls furthest behind: 27.8 in 100 killed of those reaching it against 69.5. At the doors the player is not short of HP: at the
  Act 1 door ours carries 63.6 against the humans' 55.1, at the Act 2 door 60.9 against 63.4.
- The fight-level exam says the same from the other end (D57 addendum 6: bosses 18 and 27 points behind on the same fight starts); this one adds that the
  deficit is not a matter of luckier seeds, and that no start the humans won to the end was won by ours in two draws.

## 7. What is not verified

- That each run began where the human's did. The probe found the first-room gift equal in 379 of 400 runs and the map a path of the regenerated one in all
  400 once two items were given; this reading did not check either per seed (the exam is a paired start, not a replay).
- Engine: the lab's default (old) engine, not core2. The own-seed column is the lab's judgement blocks of the same agent (the GPU prior server, several days,
  6,494 runs, no door HP), not a re-play under this driver; a matched control (a few hundred fresh seeds through this driver) is a further 40 CPU-s a run.
- Two draws separate the player's own search streams only: they are not two independent runs of the seed (the table's brackets are over seeds).
- The exam fifth was not read.

