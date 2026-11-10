# Labels searched to convergence (D55 addendum 15; 2026-11-09)

Lab code: `rl/converge.py`, `tests/python/test_converge.py`. Nothing here changes the engine, the search, the loop or the night.

## 1. What it measures

The owner's idea: instead of a fixed search depth, take the positions human A10 players were in, with their decks, and search each until
we are satisfied the move is the best one, so a training set holds solutions we know are good. Before building such a corpus this
measures what convergence costs and how often the current teacher is already converged. The search's view at the end of a line is the
current network's, so "converged" means **stable under the current judge**, not perfect play.

The teacher is player `v2` under `a3-v1@h1e4b16` (the budget is a3-v1's simulations times 1 at hallway fights, 4 at elites, 16 at
bosses), with the CPU prior in the process.

## 2. The sample

`select` reads the fight exam's human stratum (`exam/starts-human.lite.json`, fixed seed 20261109) and writes `sample.json`: turns wanted
are 100 Act 1 boss, 100 Act 2 boss, 50 each of Act 1 elite, Act 2 elite, Act 1 hallway and Act 2 hallway (400). A cell with more fights
than turns takes its fights by hash order (a boss cell stratified over its bosses); the exam holds 31 elite and 31 hallway fights an act
and 91 Act 2 boss fights, so those cells give a **second turn** (a different decision of the same fight) from the first 19, 19, 19, 19 and
9 fights in hash order to reach their 50 and 100. Bosses are played first, so a partial result is the boss table.

`run --kinds boss` plays only the boss fights (the first measurement was cut to the bosses, 2026-11-09: the sample file lists every kind and
`run` takes the kinds it is asked for, in the order boss, elite, hallway); the elites and hallways were played the next day on the same
code (`run --kinds elite,hallway`), 124 fights, 200 turns.

The bosses were timed on the engine build before the engine-speed change (`854a6baf`), the elites and hallways on the build after it.
`calibrate` plays 24 stored boss fights again on the current build and searches their turns at x16, x32 and x64 with the stored seeds:
all 24 games and all 72 searches came out alike (same moves, simulations, visits and values), and the CPU ratio of the same searches
was 0.371. `report` reads `calibration.json` and prices a corpus that mixes kinds with the boss CPU-s times that ratio (the table
says so and prints the unscaled figures beside it).

`run` plays each fight with the teacher (draw 0 of the exam's namespace `x`), records every decision (the move, the distinct root moves,
the observation's hash, the search's statistics), and keeps turns chosen by a hash coin uniformly among the fight's decisions with two or
more distinct root moves (a decision with one legal move is not a choice). The turn is re-found by replaying the fight's moves from its
fresh world (the fight is a function of its start, its draw and its seeds) and checked against the observation's hash.

## 3. The ladder

Each turn is searched at the kind's teacher budget times 1/2, 1, 2, 4, 8 and 16 (boss 8 to 256, elite 2 to 64, hallway 1/2 to 16 in the
recipe's units; the 1/2 rung only gives the 1x rung something to be compared with). One search a rung, its own fixed seed (a function of
the start, the draw, the decision and the rung), the CPU prior, every simulation from a resampling clone. The clock of a rung is the
process's CPU time with a cap of 300 CPU-s: a search stopped there is marked `capped` and says nothing in the rule.

## 4. The rule and the figures

Converged at a rung (1x and up) when its move equals the move of the rung below and the two most visited moves' values differ by at least
two standard errors. The error is the search's own `value_stderr`: the spread of the per-simulation returns over the visits, each
simulation one determinization and one playout; the gap's error adds the two in quadrature. Tree search does not draw its simulations
independently, so these errors are the closest estimate the search exposes, not a guarantee.

The tables (`report`) give, per kind and act: the agreement of the move between 1x and 4x and between 4x and 16x of the teacher's budget
(and 1x against 16x, and the game's own search against the 1x rung, which is the same budget on another seed); the share of turns for
which the rule holds at 1x, 4x, 16x and the share converged by each; the median, 90th percentile and mean CPU-s to convergence (the sum of
the rungs up to the converged one; a turn that never converges pays the whole ladder); the share of the teacher's game moves the
converged move overturns (a turn that never converged counts with the top rung's move; the converged-only share is beside it); the cost
of a 10,000-turn corpus in core-hours at the human fights' own mix of kinds (`exam/human-meta.json` pool counts, by fights and by turns)
and for bosses alone; and the machine's busy share while the rungs ran (`GetSystemTimes` around each search), since CPU seconds on a
loaded machine include the other hyperthread's pressure.

    python rl/converge.py select --out DIR
    python rl/converge.py run --out DIR [--shard LABEL] [--kinds boss,elite,hallway] [--limit N]    # several processes share DIR by claim files
    python rl/converge.py report --out DIR [--md FILE]

## 5. Tests

`tests/python/test_converge.py`: the sample's quotas and second turns, the turn pick, the ladder (the 1x rung is the teacher's budget
for every kind of decision; each rung doubles the simulations), the rule on fixed numbers (including a gap of exactly two errors and a
capped rung), the convergence budget and its CPU, the tables' arithmetic and the 10,000-turn cost on a small fixed set of turns, the
claim files, and on a real emulator fight that a replay of the moves reaches the same observation and that a rung's search is a function
of its seed.

## 6. The comparison in shared worlds (2026-11-11, for the judge corpus)

The rule's gap used the spread of the simulations' returns, which carries the draw's and the enemies' luck that both moves share. `search_rung(...,
pairs=N)` now follows its search with a **duel** of the two most visited moves (`duel_pair`): N rounds, each one sampled world (a resampling clone
with the round's seed, `duel_seed`) in which both moves are played, then the search's own rollout policy to the fight's end with the same random
stream; the value is the search's (end HP plus 10 a potion held, a death -100). The rounds' differences give the gap and its standard error
(`duel_stats`), stored on the rung's row as `duel` (and its CPU as `duel_cpu_s`). `se_mode="paired"` (`gap_stats`, `rule_holds`, `convergence`)
reads that instead of the visits: a rung with no duel has no gap in this mode, so the rule cannot hold on it. The other two modes are unchanged and
read the rows that carry no duel as before. It is the search's own confirmation round (`_Search._confirm`, which a3-v1 leaves off) for the two
moves the rule is about. Measured on one Act 2 boss fight (v3, three turns, x32 and x64): a duel of 128 pairs costs 0.1 to 0.4 CPU-s against
1.0 to 3.9 for the search it follows; at one turn the unpaired gap read 42 HP (z 2.4) where the duel read 0.9 +- 6.9, at another the duel put
the runner-up 2.1 +- 0.5 HP above the most visited move at x64: the tree's value of a move includes the best play after it, the duel's does not, so
a duel is the arbiter of "is the gap clear", not of which move is right (a turn where it disagrees stays unsettled). The corpus
(`JUDGE_CORPUS.md`) uses it with 128 pairs at the first rung searched and 256 at the cap. `quiet_reason` and `pause_point` take `ignore=` (a
caller's own tag). `play_teacher` keeps the game's `potions_held`, `potions_used` and `L0` as well.
