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
(`QUOTA`: Act 2 boss 3,900 (the pool has fewer: all), Act 1 boss 3,000, elites 3,350 each act, hallways 1,800 each act), Act 2 first. Any prefix is a
balanced sample, so the corpus is usable while it is still being built. **The weight** of a start is 1 / its inclusion probability given the fights built
in its cell (`inclusion`: `min(1, lam * boost)` adding up to the cell's built count), stored as `w` on every row of the start; `w_cell` is the same divided by
the cell's mean. The loader's `weight="pool"` (the default) brings the oversampled strata back to the human pool's mix and its cell sizes; `"cell"` brings
back the oversampling only; `"none"` keeps the sample as drawn. Self-play rows weigh 1.

## 5. Coverage and counts

(Filled in from `coverage.md` and the manifest when the build ends; see section 8.)

## 5b. The manifest's flags (for the judge-training lane)

`manifest.json` is rewritten by every compaction (the supervisor compacts every 15 minutes until the corpus is usable, every 45 after, and once at the
end with `finalize`; each file is written whole, so a reader never sees half of it). Keys the lane polls: `usable_for_training` (sticky: true once the
human boss and elite turns carry their labels, settled or soft: at least 1,000 of each kind and 300 of each kind and act; `usable_counts` has the counts at
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

(Filled in when the build ends.)

## 9. Tests

`test_judge_corpus.py`: the split and stages, the stage turns (one a stage, in its third, fixed, uniform), the deck tag, the strata read off tokens, the boosts,
the order, the inclusion probabilities and the weights (a rare group over-sampled and weighted back to its share in the pool), the work order, the ladder's
screen and cap (settled at the first rung, the cap, soft, a capped rung, no duel), the label, the shards and the loader (filters, the arrays, the labels, the
weights, the refusal of an exam-fifth row and of a split that is not the run's), the coverage's suppression of small cells and, with the emulator, that the
token constants and vocabulary inverses are the encoder's. `test_judge_corpus_build.py`: the exclusions, the plan on a small set of starts, a fight's assembly
(door and turns, the rungs a boss, an elite and a hallway turn get, the replay check, a capped game), the compaction (weights, the human's result, stale
shards, the exam-fifth refusal, the manifest) and the self-play rows from a buffer shard (stage, labels, the search as played, exclusions, the ledger).
`test_converge.py` has the paired comparison's tests, one on a real fight (the duel is a function of its seeds).
