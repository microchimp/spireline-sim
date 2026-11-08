# The round: play, label, refit the prior, refit the judgement, build, smoke, block (D55 addendum 11, memo row 5)

One round of the deck loop, as the big-step memo (`docs/design/2026-11-06-big-step.md` 2.1, row C) and the strategy memo (3.4, 3.5)
define it, with today's player: the player plays exploring runs; a sample of its deck decisions is labelled by playing every option
forward at the player's own search schedule (K 2); the deck prior `pi_h` is refit on human decisions, the player's blended choices and
the labels' clear winners behind a KL anchor to the current prior; the run judgement `V` is refit on the new fighter's runs plus the
human chosen-option rows behind a source input; a candidate is built from the two and judged by the keep test v2 against the current
player (`current.json`) on a smoke block, then a block. Code: `rl/round1.py` (steps, chain, fragment), `rl/round1_data.py` (config, labels,
rebuilt decisions, the label leg), `rl/round1_prior.py`, `rl/round1_judgement.py`; labels run through `rl/hindsight.py` (three new
options: `--own-schedule`, `--run-config/--macro/--stakes/--prior-sha`, `--rates`). Lab code only.

## The files (one round folder, `paths.round_dir`)

| step | reads | leaves |
|---|---|---|
| `play` | `agent/current.json`, `LAB/runs` | `play/play.json`: the player's identity, search config, the exploring job folders and the count of their runs |
| `label` | `play/play.json` | `labels/labels.jsonl`, `labels/manifest.json`, `labels/label-summary.json` (+ `labels.d/`, `tasks.jsonl`, `report.json` of hindsight) |
| `refit-prior` | `play/play.json`, `labels/labels.jsonl` | `prior/weights.npz` (+ sha256), `prior/refit.json` |
| `refit-judgement` | `play/play.json`, `labels/labels.jsonl`, `prior/weights.npz` | `judgement/run_value.pt/.npz`, `source_input.npz`, `vrun_fit.json`, `refit.json` |
| `candidate` | `prior/weights.npz`, `judgement/vrun_fit.json` | `candidate/candidate.json` and `agent/gen<N>/candidate-<tag>` (`gen_loop.build_candidate`) |
| `smoke` | `candidate/candidate.json` | `smoke/smoke.json` (the `gen0.py promote --sequential --keep-rule v2` command and its result) |
| `block` | `smoke/smoke.json` | `block/block.json` |

`chain.json` records each step's time, the sha256 of every input and output. A step refuses to start without its inputs.

**play.** The filler already plays the current player exploring (tau 0.1, eps 0.02, `--record`). `play` counts those runs (rows whose
`config` is the player's own schedule name, e.g. `a3-v1@h1e4b16`, from summaries naming its macro, stakes tag and combat prior) and plays
only the shortfall to `play.runs`.

**label.** `hindsight.pilot_cmd` with the config: each exploring run is replayed on the copy under the player's own schedule (the same
streams and search as `gen0.play_emu_run`; a replay that leaves its record is refused by reason), and at a drawn decision every option of the
shortlist (at most `max_options`) is played K times to the end of the act, boss included, with common random numbers across the options.
Resumable (`labels.d/<seed>.jsonl`, `tasks.jsonl`); stops for the night, `main.lock` and others' QUIET lines. **Labels file schema**
(`labels.jsonl`, one JSON object per decision, `schema` 2): `id` (`<seed>|d<ordinal>|<kind>|f<floor>`), `kind` (card, rest, route, shop), `type`,
`act` (0-based), `floor`, `hp`, `max_hp`, `gold`, `deck`, `relics`, `potions`, `k`, `options` (kind, action, card, `idx`), `played` (screen indices
played), `taken` (the player's own pick), `head_values` (the run head's value of each option), `continuations` (`o`, `k`, `G`, `end`, `floor`,
`head`, `fights`, `moves`, `seconds`, `cpu`, `door`), `Q` and `A` (mean and centred mean of `G` per played option, floors), `run`
(`seed`, `ix`, `job`, `final_floor`), `schedule` (the search config name) and `heldout` (the run is in the held-out part). A refused
decision (`refused`) is dropped. `G` is the death floor, the run head's expected final floor at the next act, or the won-run floor.

**refit-prior.** Starts from the current prior's weights. Loss = `weights.human * CE(human choice) + weights.agent * CE(player's blended
choice) + weights.label * CE(label clear winner) + kl_beta * KL(old || new)`, the KL over the options of every decision of every batch.
Human decisions: `decisions-v1` (true maps), training fifths for the fit, fifth 4 for early stopping, `assert_no_exam` on both. Agent decisions:
card and rest screens of the exploring runs' shards. Label winners: the best played option when its mean beats the next by more than
`label_gap_se` standard errors (pooled paired-difference sd over the train labels, divided by sqrt K). Runs are split train/held-out by
a hash of the seed (`label.heldout_fraction`); held-out runs enter neither term. Early stopping on the sum of the human validation NLL, the
held-out agent-choice NLL and the held-out label-winner NLL.

**refit-judgement.** The incumbent's FightNet and run head are the start (`fight_net` is `keep`: H of a human row was computed by that
FightNet, so self and human rows read one H; retraining it needs the human rows rebuilt and is not built). Self rows: the exploring blocks
of the current player (`gen_loop.fighter_manifests`, `exploring_only`, same stakes tag), pure Monte Carlo targets (`beta` 0), a tenth of the
runs held out whole. Human rows: the chosen option's row of each human card, ancient and route decision in the training fifths, target the
run's final floor class (won: the won class), no end-HP term. A source input (a first-layer column times 1 for human, 0 for self; play is
0) lets the net absorb the level difference; the exported `run_value.npz` is the plain network. For each `human_weights` value (per-row weight of
human rows against self rows) `seeds` fits start from the incumbent; the weight with the lowest mean held-out log-loss of the clears-Act-1
and clears-Act-2 heads (read off the 53-class floor distribution, rows at floors 1-16 and 18-32) is kept and its seeds' weights averaged. The three
binary heads of memo 3.4 are not separate outputs: the floor distribution gives them. The D33 keep record (`keep.evaluate_value_net`) and
the ranking check against the incumbent (`ranking_check`) decide `kept` in `vrun_fit.json`; `candidate` refuses a fit that is not kept.

## The parameters (`rl/round1_config.json`; the file is `round1_data.DEFAULTS`, a test pins them equal)

Pass `--config FILE` with only the keys to change (unknown keys are errors). Memo defaults where the memo gives one; **builder** marks my choice
where it does not (the Opus weights call may reset these).

| key | default | source |
|---|---|---|
| `label.k` | 2 | memo 2.1 (K 2) |
| `label.quota` | card/a1 450, shop/a1 300, rest/a1 180, route/a1 150, card/a2 60, shop/a2 30, route/a2 20, rest/a2 10 (1,200) | memo 2.1 size; the cell mix is the sizing memo's, scaled, Act 2 down (builder) |
| `label.per_run`, `max_options`, `own_schedule` | 2, 6, true | hindsight's; own schedule per the brief |
| `label.heldout_fraction` | 0.2 | sizing memo (a fifth of runs held out) |
| `prior.lambda` | 0.5 | the candidates' setting (generation 12/13) |
| `prior.lambda_rule` | `fixed` | memo 3.3 lets the labels choose; `labels` takes the best of `lambda_grid` only if it beats the fixed one by a standard error. Left `fixed` until the labels-against-blocks decision |
| `prior.weights` | human 1.0, agent 0.5, label 1.0 | memo 3.5 names the three sources, no weights (builder) |
| `prior.kl_beta` | 1.0 | memo 3.5 names the anchor, no strength (builder) |
| `prior.label_gap_se` | 2.0 | memo 3.5 (gap over 2 SE) |
| `prior.lr`, `epochs`, `batch`, `patience` | 1e-4, 8, 512, 3 | a fine-tune of a fitted prior (builder) |
| `prior.max_agent_decisions` | 40,000 | memory (about 160k rows) (builder) |
| `judgement.human_weights` | 0.25, 1, 4 | memo 3.4 |
| `judgement.seeds` | 1, 2, 3 | memo 3.4 (three seeds, averaged) |
| `judgement.beta` | 0 | memo 3.4 (pure Monte Carlo) |
| `judgement.lr`, `epochs`, `batch`, `end_hp_weight` | 3e-4, 6, 1024, 0.25 | train.py's end-HP weight; a fine-tune (builder) |
| `smoke.seeds`, `block.seeds`, `keep_rule` | 200, 3,000, v2 | memo 2.1; keep test v2 |
| `candidate.tag`, `target_generation`, `marker` | round1, 14, false | `marker` false: the chain judges the candidate itself; true also lets the lab's train job do it |

## Running one by hand

Python for the steps that train is the GPU venv (`paths.gpu_python`), at BelowNormal; the others use the emulator venv. All from a fork tree
with `out/` built:

    python rl/round1.py config --write my-round.json          # edit the keys you want to change
    python rl/round1.py play   --round-dir RD --config my-round.json
    python rl/round1.py label  --round-dir RD --config my-round.json     # resumable; hours
    python rl/round1.py refit-prior       --round-dir RD --config my-round.json
    python rl/round1.py refit-judgement   --round-dir RD --config my-round.json
    python rl/round1.py candidate|smoke|block --round-dir RD --config my-round.json
    python rl/round1.py dry-run --scratch SCRATCH                        # all steps on a tiny sample, nothing real touched
    python rl/round1.py <step> --dry-run --round-dir SCRATCH             # one step of it
    python rl/round1.py fragment --round-dir RD --out jobs-fragment.json

The dry run builds a scratch agent folder (the current player's files with the stakes schedule's base replaced by 4 simulations), plays two runs
there, labels one decision at K 1, refits both artefacts on one human shard for one epoch, builds a candidate in the scratch folder and parses the
smoke and block commands with gen0's own parser. It starts no block, no real game, and writes nothing under the agent or lab folders.

`jobs-fragment.json` holds one disabled `round-<step>` entry per step (command, cores, `needs`, `leaves`). The runner's kinds are fixed
(`tools/lab/jobs.py`), so these are not runnable by it yet; the orchestrator merges them with a `jobs.json` backup at a quiet time and either
adds the kinds to the runner or runs the steps by hand.

## What each held-out report means

* `labels/label-summary.json`: decisions per cell, the pooled single-draw noise (sd of the difference of two options on one draw), the held-out share.
* `prior/refit.json` `fit.heldout`: **human** (fifth 4) top-1 and NLL per kind before and after: how far the prior moved from human choices;
  **agent** (held-out runs) NLL/top-1 of the player's own pick: how far it moved toward the player; **label** the NLL of the label winners.
  `label_leg`: on held-out labelled decisions the mean centred value A (floors) of the pick among the played options by the old blend (V plus lambda log
  pi_old), the new blend, the judgement alone, the new prior alone and each grid lambda, with `*_minus_old_blend` paired differences (mean, se, 95%
  interval); `players_own_pick` and `best_option` bracket them. This is the arbiter of memo 3.3: a positive new-minus-old difference whose interval
  excludes zero says the refit helps under the player's own search. It is not an exam (K 2 is noisy: one label is a fifth signal).
* `judgement/refit.json`: `fit.grid` held-out log-loss of the clears heads by human weight and seed; `incumbent_heldout` and `kept_heldout` the same for the
  incumbent and the kept fit (lower is better); `keep` the D33 record; `label_leg` the picks of the incumbent blend, the new judgement with the old prior, the
  new judgement with the new prior, and the new judgement alone.
* Cells under 20 decisions are not shown. Human data enters as aggregates only; no run id is written; attribution "Spire Codex public run export for v0.107.1".

## Open, for the weights call

The labels-against-blocks memo (`docs/design/2026-11-08-labels-vs-blocks.md`) may reset `prior.lambda`, `lambda_rule`, `prior.weights`,
`kl_beta`, `judgement.human_weights`; each is one key. Not built: FightNet retraining (`fight_net` retrain), the three binary heads as outputs,
shop and rest-site-target rows in the agent and label terms (card and rest screens only), the judge-leaf and run-currency parts of the first stack.
