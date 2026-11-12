# The judge's corpus (D58 C1, D57 addendum 7; 2026-11-11)

Lab code: `rl/judge_corpus.py` (the store, the record layout, the loader, the pure parts), `rl/judge_corpus_build.py` (plan, play, compact,
self-play, coverage, manifest), `rl/converge.py` (the ladder and the rule, with the comparison in shared worlds added: `CONVERGE.md` section 6),
`tests/python/test_judge_corpus.py`, `test_judge_corpus_build.py`, `test_converge.py`. Nothing here changes the engine, the search, the loop or the night.
Status words as in GPU_INFERENCE.md: **measured**, **estimated**, **decided**.

Source of every human position: Spire Codex public run export for v0.107.1 (the four training fifths; aggregates and derived game positions only).

## 1. Why, and what it is

The judge (the leaf estimate the search relies on) reads boss positions worse than noisy play-outs (D55 addendum 21) and is trained on the loop's
buffer with raw outcomes. Two things were missing: positions from the decks and fights humans reached (players survive the Act 1 and Act 2 bosses 72
and 74 in 100), with the human's own result of that fight; and better targets than one outcome, namely the search settled on a ladder of budgets
(the convergence measurement: the teacher's x16 move is overturned in 30% of boss and 34% of elite turns). The corpus holds both, balanced by
stratification (D57 addendum 7), in a store the judge lane trains from. Store: `D:/spireline/referee/judge/corpus/` (not in git; `manifest.json`
says what it holds, how it was made and the sha256 of every shard and source).

## 2. Sources

- **Human** (`human-*` shards). Every start of `starts/human-starts.*` (`HUMAN_STARTS.md`: 46,410 rebuilt fight starts of the four training fifths)
  except the fight exam's own (the 400 human starts and every run in `exam/exclude_runs.json`, the boards' start lists) and any run of the exam fifth
  (`human_starts.assert_no_exam`, in the plan and again in every compaction; the loader refuses a human row of bucket 0). The teacher plays the
  fight: player **v3** under `a3-v1@h1e4b16` (x1 at hallways, x4 at elites, x16 at bosses), the CPU prior, the fight's seeds `x` / draw 0 (the exam's
  own namespace, so a game is a function of its start). **The door** (the first decision, the human's own position) and **three turns** are kept: the
  fight's decisions that have two or more distinct moves are cut in thirds, and one turn of each third (early, middle, late) is drawn by a hash coin
  (`pick_stage_turns`; a fixed function of the start). Every kept position can be reached again: `replay` holds the game's seed namespace and draw and the
  teacher's moves up to the position (`converge.replay_to`), and `oh` the observation's hash it must reach.
- **Self-play** (`selfplay-*` shards). Every usable position of the always-on loop's replay buffer (`buffer/shards`): one position per decision of
  the x16 teacher's fights (the loop's night starts, rebuilt human starts and pre-loss starts; players v0 to the current candidate), excluding capped
  fights and fights of cut segments (no value label), and every run and start in the exam's list, the boards' start lists (every `meta-main.json`
  under the lab's fight-gym results) and the M2 board's. `selfplay` keeps a ledger of the buffer shards taken, so a later run adds the newer shards and
  keeps the ones the loop's retention has deleted.

Split: by **run**, one in five held out (`split_of`: the first 32 bits of the sha256 of the run id mod 5), the same rule for both sources and for
`judge_data.split_of` on a run (a run's fights share a deck, so a split by start would put one deck on both sides). The exam fifth is never inside either
side. No row carries a player id (the corpus's single-player runs have none) and no raw run leaves the corpus folder.

## 3. Labels

Every row carries three kinds of label; the judge lane picks:

- `human` (human rows): the human's own result of that fight from `human-outcomes.json` (`HUMAN_STARTS.md` has the reading): survived, end HP (before the
  run engine's post-fight heals), HP lost, turns, potions used and held, `l0` (the search's currency) and `l0_unit` (the gym's L0). It is the result of the
  fight from the door; for a later turn it is a label of the fight, not of the position (the loader's `label="human"` gives it as given).
- `teacher`: the result of the very game this position was in (won, end HP, potions held, L0): one draw from the outcome distribution at the position.
- `label`: the search. `tier` `settled`, `soft` or `played`; `move` and `children` (each root move's visits `n`, mean value `v` in HP units, error `se` and
  death rate `pd`), `root_value` (the visit-weighted mean of the moves' values), `best_value`, `p_death`, `rel` and `budget` of the rung it comes from, the
  rung's `duel`, and `rungs` (a line a rung searched: the move and the visits it chose, its CPU, its duel). Self-play rows carry the search as played
  (tier `played`, the loop's x16) with the moves of the buffer's action slots.

**The ladder** (boss and elite turns, and the door when it has a choice; `settle`). The teacher's own search at that decision is the first rung (x16 at a boss, x4 at an elite: the very
budget of `converge.py`'s 1x rung, on another seed). Then the **screen**: one search at twice that (x32 / x8), followed by a **duel** of its two most visited
moves in 128 shared worlds (`converge.duel_pair`: both moves in the same sampled draws, then the search's rollout policy; its mean and error are the gap
and its error). The turn is **settled** at that rung if its move equals the first rung's and the gap is at least two standard errors
(`converge.rule_holds`, `se_mode="paired"`); otherwise the **cap**: a search at four times the teacher's budget (x64 / x16) with a duel of 256 pairs, settled
if its move equals the screen's and the gap holds. A turn the rule never held for is **soft**: it keeps the last rung that was not cut by the 300 CPU-s cap
(normally the cap's), marked so. A **hallway** turn, and any position with a single distinct move, keep the teacher's own search at its budget (`played`; the brief: hallways
rarely matter by 5 HP and cost little either way). CPU in the rows is the rungs' and duels' (the game's is in the fight's total).

The comparison in shared worlds replaces the spread of the unpaired values in the rule (`CONVERGE.md` section 6; the brief allowed one day for it, it took
a few hours). It is the tree's value that is stored as the label; the duel decides whether the gap is clear. A turn where the duel puts the runner-up
ahead stays unsettled.

## 4. Strata, minimums, sampling, weights

Strata (D57 addendum 7): kind, act, enemy (a boss and an elite a cell), turn stage (early, middle, late, and the door), deck type, relics and cards
(presence), health band (healthy 2/3 and up, hurt 1/3 to 2/3, critical under 1/3 of max HP) and potion state (none, one, two or more). **Deck type**
(`deck_tag`) is the first rule that holds over the deck's core (cards that are not basic, curse, status or quest): `starter` three or fewer; `power` three
powers and a quarter; `exhaust` three exhaust cards and 30%; `block` three cards that give block and 35%; `attack` attacks 55%; else `mixed`.

**Cell minimums the build aims at** (positions; `MINIMUMS` in the builder): a kind-and-act cell 3,000, an enemy 400, a deck type 300, a relic 150 and a
card 150. A relic or card in under 20 runs is not listed in `coverage.md`; the sampling below gives every start with a rare feature a better place in the
order, but a feature the pool lacks cannot be made up.

**Sampling of the human starts** (`plan`, `boost_weights`, `order_key`, `inclusion`). A cell (act and kind) is drawn by sequential Poisson sampling: each
start has a **boost**, the largest over its relics, cards, enemy and deck type of 200 / (the sampling rate of its kind times the starts of its kind that have the feature), at least 1; its place
in the cell's order is `u / boost` (u a fixed uniform of the start id); the plan's work order takes 100 fights of each cell in turn, up to the cell's quota
(`QUOTA`: Act 2 boss 3,900 (the pool has 3,701: all of them), Act 1 boss 1,500, elites 2,700 each act, hallways 1,400 each act; a fixed sampling rate a kind, `SAMPLE_RATE`: boss 0.5, elite 0.2,
hallway 0.65, so a longer quota extends the same order), Act 2 first. Any prefix is a
balanced sample, so the corpus is usable while it is still being built. **The weight** of a start is 1 / its inclusion probability given the fights built
in its cell (`inclusion`: `min(1, lam * boost)` adding up to the cell's built count), stored as `w` on every row of the start; `w_cell` is the same divided by
the cell's mean. The loader's `weight="pool"` (the default) brings the oversampled strata back to the human pool's mix and its cell sizes; `"cell"` brings
back the oversampling only; `"none"` keeps the sample as drawn. Self-play rows weigh 1. **The weights in the finished corpus** (`w`: min / mean / max): Act 1 boss 1.0 / 5.1 / 22.5, Act 1 elite 1.0 / 7.2 / 36.7, Act 1 hallway 1.0 / 1.6 / 5.3,
Act 2 boss 1.0 / 1.0 / 1.7 (the whole pool is in), Act 2 elite 1.0 / 3.2 / 23.4, Act 2 hallway 1.0 / 1.7 / 9.0. A weight of 20 or more is a rare start standing for twenty like it; a loss
weighted that way is noisy, so clip it or use `weight="cell"` (mean 1 in each cell) or `"balanced"` (every kind-and-act cell of the selection the same total).

## 5. Coverage and counts

**Built 2026-11-11 to 2026-11-12** (complete: 13,401 of 13,401 planned human fights have a result; `manifest.json` has `complete: true`). Positions and runs
a cell, from `coverage.md` (a cell under 20 positions or 20 runs is not shown; the cell minimums are those of section 4).

| source | cell | positions | runs |
| --- | --- | --- | --- |
| human | Act 1 boss | 5,805 | 1,497 |
| human | Act 2 boss | 14,338 | 3,692 |
| human | Act 1 elite | 10,255 | 2,056 |
| human | Act 2 elite | 10,218 | 1,947 |
| human | Act 1 hallway | 5,198 | 1,314 |
| human | Act 2 hallway | 5,277 | 1,248 |
| self-play | Act 1 boss | 438,136 | 4,598 |
| self-play | Act 2 boss | 76,783 | 1,082 |
| self-play | Act 1 elite | 86,555 | 1,568 |
| self-play | Act 2 elite | 47,486 | 1,104 |
| self-play | Act 1 hallway | 155,616 | 3,215 |
| self-play | Act 2 hallway | 112,492 | 1,980 |

**Against the targets** (the brief: at least 20,000 boss turns, 20,000 elite turns and 10,000 hallway turns from human positions): **20,143 boss turns**
(both acts, every one with a label: 10,988 settled, 9,155 soft), **20,473 elite turns** (9,285 settled, 11,188 soft) and **10,475 hallway positions** (the teacher's
own search), 51,091 human positions in all from 13,371 fights of 6,418 runs (30 fights are missing, below), plus **917,068 self-play positions** of 7,392 runs. Held out
by run: 10,345 human positions (20.2%) and 192,463 self-play positions (21.0%). The doors (first decisions) are 13,371 of the human positions, 11,030 of them as
their own stage and the rest as the sampled early turn.

Human positions by stage: early 13,371, middle 13,358, late 13,332, door 11,030. By boss (positions; every cell over its 400 minimum): Insatiable 4,778, Kaiser Crab 4,729,
Knowledge Demon 4,831 (Act 2: the whole pool of each), Vantom 1,034, Soul Fysh 1,024, Lagavulin Matriarch 1,025, Waterfall Giant 935, Ceremonial Beast 894, The Kin 893 (Act 1:
a part of the pool, 1,500 fights). By elite: from 1,610 (Phantasmal Gardeners) to 3,461 (Decimillipede). By deck type: starter 2,875, power 2,536, exhaust 4,775, block 5,141,
attack 11,509, mixed 24,255 (all over their 300 minimum). By health band: healthy 19,621, hurt 21,910, critical 9,560. By potions held: none 29,759, one 11,804, two or more 9,528.

**Relics and cards.** 220 relics appear in the human positions and **214 of them in at least 150 positions and 20 runs** (the other six are in fewer than 20 runs: the
pool does not have them); of the 465 cards that appear, **222 are in at least 150 positions** and 229 in at least 20 positions and 20 runs; the other 236 are in fewer than 20 runs of
the pool (a long tail of cards seen in a handful of runs). Self-play: 204 relics and 198 cards in at least 150
positions. The sampling put the rare features first (section 4): on the plan, the share of the pool's relics and cards that reach 50 starts in the sample rose from 175 to 189
of 210 relics and from 126 to 157 of 187 cards at elites against a plain draw of the same size (bosses: cards 131 to 143 of 143); measured on the plan before the sampling
rates were fixed, so approximate. **What the sampling cannot do**: a start has many features and the cells' quotas are fixed, so a feature of 100 to 200 pool starts reaches
about half of them; and nothing is made up for a feature the pool lacks. The weights (`w`) undo the oversampling.

`ai_assumed` is on 11,393 human positions (built on 2026-11-11 after 08:21Z and before the field was withdrawn). It is the search flag `ai_state_assumed`, which a fight loaded from
a saved position never clears: it is sticky and carries no information about the position. Do not read it.


## 5b. The manifest's flags (for the judge-training lane)

`manifest.json` is rewritten by every compaction (the supervisor compacts every 15 minutes until the corpus is usable, every 45 after, and once at the
end with `finalize`; each file is written whole, so a reader never sees half of it). Keys the lane polls: `usable_for_training` (sticky: true once the
human boss and elite turns carry their labels, settled or soft: at least 500 of each kind and 150 of each kind and act; `usable_counts` has the counts at
that moment and the time) and `complete` (true when every fight of the plan has a result and the shards, `coverage.md` and the manifest were made from
them; `complete_at`). `human_build` says where the human build stands: fights planned, built and failed, the labelled boss and elite turns by act, and
`paused_by`, the QUIET.lock line or night block the work processes are waiting on, if any. The self-play shards are in from the start; the human shards
appear with the first compaction after fights finish.

## 6. Using it

```python
import judge_corpus as jc
c = jc.Corpus()                                     # D:/spireline/referee/judge/corpus
for pos in c.iter_positions(src="human", kind="boss", split="train", tiers=["settled", "soft"]):
    pos.row["label"]["children"], pos.row["human"], pos.weight, pos.tokens["types"]  # tokens: types, ids, owners, feats [T, 18]
z = c.arrays(kind="boss", split="train", label="auto", weight="pool")   # the npz layout judge.Positions reads, plus weight and run
```

`arrays` gives `lens`, flat `types`/`ids`/`owners`/`feats`, `labels` [n, 4] (won, end HP over max HP when won, potions held when won, L0), `meta` (judge_data's
`META`: k, turn, hp, max HP, root value, best value, best P(death), sims), `group` (the start), `start_ids`, `start_heldout` (by run), `start_cat`, `cats`,
`weight`, `run`. `label="auto"` is the human's result for human rows and the teacher's for self-play rows; `"teacher"` and `"human"` give one kind of
label for both sources (a row with none of that kind is left out). The judge's own validation tenth (`judge.val_of`) can be cut by run from `run`.
`Corpus.shards(src, kind, act)` lists the files; a shard `human-boss-act2-0003.jsonl` has its tokens in `human-boss-act2-0003.npz` (the same rows).

## 7. Extending it

- **More fights**: raise `QUOTA`, run `plan` (the order is a fixed function of the pool and the quotas, so the first fights are the same), then `run`; `compact`
  makes the shards again from every fight on disk (it can run while `run` goes on). **More turns a fight**: `pick_stage_turns(..., per_stage=2)` and
  `build_fight`; a second pass replays the same game (the teacher is a fixed function of the start). **Another player or engine**: a new store (the manifest
  names the player's policy hash and the engine tag; v3 and the engine of this build are in it).
- **Self-play**: `selfplay` again adds the buffer shards the ledger has not taken. New self-play positions beyond the buffer need the lab's `selfplay`
  job (`LAB/jobs.json`, off since 2026-11-10) or the loop's own play; the corpus builder does not start either.
- **Soft turns**: more rungs (x128, x256) with the same rule if a later lane wants the unsettled turns settled; the doubling ladder's cost is in `CONVERGE.md`.

## 8. What was built and measured

**The build.** Eight work processes at BelowNormal beside the practice loop and the other lanes' jobs, from 2026-11-11 00:40Z (the first fights; the work waited on the design lane's
`run-teacher` QUIET.lock line from 21:24Z to about 00:40Z until the coordinator said to go on beside it, and on `lab-train` and `handover-judgement` lines for about three hours in
all) to 2026-11-12 06:32Z; the supervisor compacted every 15 then 45 minutes, so the manifest said `usable_for_training` from 2026-11-11 01:36Z (765 labelled boss and 759 labelled
elite turns, 2,270 human positions, 917,068 self-play positions) and `complete` at 2026-11-12 06:40Z. **CPU: 148 core-hours of process time for the human fights** (13,371 fights;
the machine was shared with the loop, the lab and the design lane, so process time reads high: two Act 2 boss games timed with the machine to themselves took 17 and 27 CPU-s against
54 on average in the build) and a few CPU-minutes for the self-play pass. Mean CPU a fight (s; the teacher's game plus the ladder): Act 1 boss 40 + 29, Act 2 boss 54 + 37, Act 1 elite 5.7 + 8.6, Act 2 elite
7.6 + 11.8, hallways 1 to 2. The brief expected 50 to 100 core-hours; the quotas were cut to its counts once an Act 2 boss fight was seen to cost 90 to 110 s.

**Labels** (human positions with a ladder, `report.md`):

| cell | positions | settled at the first rung | settled at the cap | soft |
| --- | --- | --- | --- | --- |
| Act 1 boss | 5,805 | 48% | 9% | 43% |
| Act 2 boss | 14,338 | 46% | 8% | 46% |
| Act 1 elite | 10,255 | 38% | 10% | 52% |
| Act 2 elite | 10,218 | 33% | 10% | 57% |

So **55% of boss turns and 45% of elite turns are settled** at x32 / x8 or at the cap, and the rest keep the cap's values marked `soft` (the convergence measurement:
60% of boss and 54% of elite turns converge by 4 times the teacher's budget, and a turn that never converges there is what `soft` is). The screen settled 82% of the turns that
settled at all, so the cap (x64 / x16 with its 256-pair duel) was searched for 54% of the boss turns and 64% of the elite turns.

**The comparison in shared worlds, on the same searches** (the rule applied again to each turn's rung lines under the unpaired reading, `report.md`): the paired rule settles
57 / 54% of Act 1 / Act 2 boss turns and 48 / 43% of elite turns, the unpaired one 47 / 46% and 40 / 33%; 39 / 38% (boss) and 30 / 25% (elite) settle under both, 16 to 18% settle
only under the paired rule (the unpaired spread is what keeps them open) and 8 to 9% only under the unpaired. At the last rung the duel puts the runner-up at least two errors
ahead of the most visited move in 4% of boss turns and 7 to 8% of elite turns (a tree move that rollouts disagree with: those stay unsettled). The duel cost 0.4 CPU-s at the first rung and 0.9 at the cap
against 4.1 and 8.8 for the searches they follow (the first 27 Act 2 boss fights).

**The teacher against the human on the same fights** (the door's result: the teacher's game against the human's run record): Act 1 / Act 2 boss 62 / 60 in 100 against 72 / 74, elite
85 / 82 against 88 / 87, hallway 98 / 93 against 98 / 96 (v3 at the recipe's budgets, the CPU prior). This is the practice loop's human bar on 13,371 fights; the exam's 400
starts are not in it.

**What did not go in.** 30 of the 13,401 planned human fights (0.22%: 3 Act 1 and 9 Act 2 bosses, 4 and 13 elites, 1 hallway) have no row: the emulator's fail-fast killed
the work process on them twice (exit code 0xC0000409), so they were recorded as failed and not tried a third time. Their start ids are in
`referee/backlog/judge-corpus/crash-starts.txt` for the engine side. The loader and the manifest count what is there.

**Checks** (`manifest.json` `checks`, counted on the 968,159 rows; every figure 0): rows of a run or start in the exam's or the boards' lists, human rows of the exam fifth, rows
whose split is not their run's, runs on both sides of the split.

**Unverified or approximate.** (1) The weights `w` follow the sampling plan (inclusion probabilities from the boosts), not an end-to-end check of a trained judge on reweighted data.
(2) A self-play row's `hp` is the HP fraction of its tokens times its fight's max HP (the buffer keeps no HP), and its stage is a third of the fight's decisions (a human row's is a
third of the decisions with two or more moves). (3) The human positions are the teacher's play from the human's start (the deck, relics, potions and HP going in, a fresh hand and
draw order from the copy): the human's own hand and moves are not in the record, and `human_starts.py`'s approximations (enchantments, relic counters, event potions) apply. (4) The
paired duel judges the gap under the rollout policy after the first move, not under the tree's play, so it can put a tree move behind (4 to 8% of turns). (5) The comparison of the two
readings uses the rung lines' top two moves; a move's merged copies are not kept in them, so a rung that chose the same move through another copy reads as a change (rare). (6) The
teacher's search sees the true enemy moves in a fight built from a saved position (`ai_state_assumed`); the diagnosis lane found the flag sticky and the in-run difference a measurement
artifact, so the labels are unaffected. (7) Self-play positions are the loop's, including pre-loss starts; their labels are the search as played (x16) and their outcome, not settled values.


## 9. Tests

`test_judge_corpus.py`: the split and stages, the stage turns (one a stage, in its third, fixed, uniform), the deck tag, the strata read off tokens, the boosts,
the order, the inclusion probabilities and the weights (a rare group over-sampled and weighted back to its share in the pool), the work order, the ladder's
screen and cap (settled at the first rung, the cap, soft, a capped rung, no duel), the label, the shards and the loader (filters, the arrays, the labels, the
weights, the refusal of an exam-fifth row and of a split that is not the run's), the coverage's suppression of small cells and, with the emulator, that the
token constants and vocabulary inverses are the encoder's. `test_judge_corpus_build.py`: the exclusions, the plan on a small set of starts, a fight's assembly
(door and turns, the rungs a boss, an elite and a hallway turn get, the replay check, a capped game), the compaction (weights, the human's result, stale
shards, the exam-fifth refusal, the manifest) and the self-play rows from a buffer shard (stage, labels, the search as played, exclusions, the ledger).
`test_converge.py` has the paired comparison's tests, one on a real fight (the duel is a function of its seeds).
