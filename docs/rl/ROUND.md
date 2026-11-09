# The round: play, label, refit the prior, refit the judgement, build, smoke, block (D55 addendum 11, memo row 5; weights and mix: addendum 12)

One round of the deck loop, as the big-step memo (`docs/design/2026-11-06-big-step.md` 2.1, row C) and the strategy memo (3.4, 3.5)
define it, with today's player: the player plays exploring runs; a sample of its deck decisions is labelled by playing every option
forward at a K 2 search (the player's own schedule, bosses at x8); the deck prior `pi_h` is refit on human decisions behind a KL anchor to the
current prior (and, at weights above 0, on the player's blended choices and the labels' clear winners); the run judgement `V` is refit on the
new fighter's runs plus the human chosen-option rows behind a source input; a candidate is built from the two and judged by the keep test v2
against the current player (`current.json`) on a smoke block, then a block. The labels-against-blocks memo
(`docs/design/2026-11-08-labels-vs-blocks.md`, section 6; D55 addendum 12) sets the defaults below. Code: `rl/round1.py` (steps, chain,
fragment), `rl/round1_data.py` (config, labels, rebuilt decisions, the label leg), `rl/round1_prior.py`, `rl/round1_judgement.py`; labels run
through `rl/hindsight.py` (options added for the round: `--own-schedule`, `--run-config/--macro/--stakes/--prior-sha`, `--rates`, `--per-act`,
`--cont-boss-mult`). Lab code only.

## The files (one round folder, `paths.round_dir`)

| step | reads | leaves |
|---|---|---|
| `play` | `agent/current.json`, `LAB/runs` | `play/play.json`: the player's identity, search config, the exploring job folders and the count of their runs |
| `label` | `play/play.json` | `labels/labels.jsonl`, `labels/manifest.json`, `labels/label-summary.json` (+ `labels.d/`, `tasks.jsonl`, `report.json` of hindsight) |
| `refit-prior` | `play/play.json`, `labels/labels.jsonl` | `prior/weights.npz` (+ sha256), `prior/refit.json` |
| `refit-judgement` | `play/play.json`, `labels/labels.jsonl`, `prior/weights.npz`, `prior/refit.json` | `judgement/run_value.pt/.npz`, `source_input.npz`, `vrun_fit.json`, `refit.json` |
| `candidate` | `prior/weights.npz`, `prior/refit.json`, `judgement/vrun_fit.json`, `judgement/run_value.npz` | `candidate/candidate.json` and `agent/gen<N>/candidate-<tag>` (`gen_loop.build_candidate`) |
| `smoke` | `candidate/candidate.json` | `smoke/smoke.json` (the `gen0.py promote --sequential --keep-rule v2` command and its result) |
| `block` | `candidate/candidate.json`, `smoke/smoke.json` | `block/block.json` |

**`chain.json`** records every step: when, the sha256 of the config, and every input and output file as `{"sha256", "bytes"}` (an input also
names the step that left it, `from`), `sha256` the first output's, and `artifacts` for the files outside the round folder the step made or
read (the candidate step, the smoke and the block record the candidate folder's files). A step refuses to start without its inputs, and on an
input whose sha256 is no longer the one its producer recorded (edited since, or left by an older run of the step); the smoke and the block also
refuse a candidate folder that differs from the one the candidate step recorded (a judgement's own `promotion.json` and `arms/` are not part of
its identity).

**play.** The filler already plays the current player exploring (tau 0.1, eps 0.02, `--record`). `play` counts those runs (rows whose
`config` is the player's own schedule name, e.g. `a3-v1@h1e4b16`, from summaries naming its macro, stakes tag and combat prior) and plays
only the shortfall to `play.runs`. The exploring play, the kept player and the gym's teacher stay at the player's own x16 (a judgement fitted
on x8 rows would learn a weaker player).

**label.** `hindsight.pilot_cmd` with the config: each exploring run is replayed on the copy under the player's own schedule (the same
streams and search as `gen0.play_emu_run`; a replay that leaves its record is refused by reason), and at a drawn decision every option of the
shortlist (at most `max_options`) is played K times to the end of the act, boss included, with common random numbers across the options.
The continuations search the player's own stakes schedule with the **boss multiplier lowered to `label.cont_boss_mult` (8)**, the schedule of
`gen14/candidate-stakes-b8` (`a3-v1@h1e4b8` for a player at `a3-v1@h1e4b16`): about 1.7 times the labels for the same core-hours, and a label is
mostly noise (memo section 5). The replay stays at the player's own schedule so that it is the recorded run. Resumable (`labels.d/<seed>.jsonl`,
`tasks.jsonl`); stops for the night, `main.lock` and others' QUIET lines.

*What is labelled* (D55 addendum 12): **two labels a run per act** (`label.per_act`; at most one per kind and act, and `label.per_run` is read only
when `per_act` is 0), of the kinds card, rest, **ancient** (the relic screen: Neow in Act 1, the Act 2 ancient), shop and route, with a quota per
cell (`label.quota`) of which **at least half sit in Act 2** (`label.act2_min_share`, checked when the config loads). The memo's finding behind
it: 1,437 of the first 1,449 labels were Act 1 decisions, where the human prior costs a little, while its whole gain at the frontier comes from its
Act 2 choices. Default quota: Act 1 card 200, rest 120, ancient 40, shop 140, route 100; Act 2 card 240, rest 140, ancient 100, shop 70, route 50 (600
and 600). `label.rates` is the chance a decision of each kind is taken (a kind with no rate is never labelled).

**Labels file schema** (`labels.jsonl`, one JSON object per decision, `schema` 3): `id` (`<seed>|d<ordinal>|<kind>|f<floor>`), `kind` (card, rest,
ancient, route, shop), `type`, `act` (0-based), `floor`, `hp`, `max_hp`, `gold`, `deck`, `relics`, `potions`, `k`, `options` (kind, action, card, `idx`),
`played` (screen indices played), `taken` (the player's own pick), `head_values` (the run head's value of each option), `continuations` (`o`, `k`,
`G`, `end`, `floor`, `head`, `fights`, `moves`, `seconds`, `cpu`, `door`), `Q` and `A` (mean and centred mean of `G` per played option, floors), `run`
(`seed`, `ix`, `job`, `final_floor`), `schedule` (the search config name the exploring play and its replay ran under, e.g. `a3-v1@h1e4b16`),
`cont_schedule` (the one the continuations ran under, e.g. `a3-v1@h1e4b8`; a record without it is read as the play schedule) and `heldout` (the run
is in the held-out part). A refused decision (`refused`) is dropped. `G` is the death floor, the run head's expected final floor at the next act, or
the won-run floor. Schema 2 differed by lacking `cont_schedule` and the ancient kind. `label-summary.json` adds `by_act` (counts and `act2_share`) and
`schedules`; the manifest's `policy.continuation` names `schedule` and `play_schedule` and its `plan` the `per_act`.

**refit-prior.** Starts from the current prior's weights. Loss = `weights.human * CE(human choice) + weights.agent * CE(player's blended
choice) + weights.label * CE(label clear winner) + kl_beta * KL(old || new)`, the KL over the options of every decision of every batch.
**With the defaults (human 1, agent 0, label 0) the refit is `pi_h` on the larger human file behind the anchor**: a source with weight 0 is not
trained on, is not read for training (the agent decisions), and does not steer the early stop; its held-out figures are still in the report. Human
decisions: `decisions-v1` (true maps), training fifths for the fit, fifth 4 for early stopping, `assert_no_exam` on both. Agent decisions: card,
ancient and rest screens of the exploring runs' shards. Label winners: the best played option when its mean beats the next by more than
`label_gap_se` standard errors (pooled paired-difference sd over the train labels, divided by sqrt K). Runs are split train/held-out by a hash of the
seed (`label.heldout_fraction`); held-out runs enter neither term. Early stopping on the sum of the human validation NLL and the held-out NLL of the
sources the fit trains on.

**refit-judgement.** The incumbent's FightNet and run head are the start (`fight_net` is `keep`: H of a human row was computed by that
FightNet, so self and human rows read one H; retraining it needs the human rows rebuilt and is not built). Self rows: the exploring blocks
of the current player (`gen_loop.fighter_manifests`, `exploring_only`, same stakes tag), pure Monte Carlo targets (`beta` 0), a tenth of the
runs held out whole. Human rows: the chosen option's row of each human card, ancient and route decision in the training fifths, target the
run's final floor class (won: the won class), no end-HP term. A source input (a first-layer column times 1 for human, 0 for self; play is
0) lets the net absorb the level difference; the exported `run_value.npz` is the plain network. **The human weight is chosen by a selection step**
(`round1_judgement.select_human_weight`) over the grid `judgement.human_weights` (0.25, 1, 4 and any you add): `seeds` fits start from the incumbent
for each weight; the weight with the **lowest held-out clears-Act-2 log-loss** (rows at floors 18-32) among those whose clears-Act-1 log-loss (floors
1-16) is not worse than the incumbent's by more than `judgement.act1_tolerance` is kept, and its seeds' weights averaged. If no weight keeps clears
Act 1, the one with the lowest clears-Act-1 loss is kept and the report says so (`selection.fallback`). The chosen weight, the grid and every weight's
figures are in `judgement/refit.json` (`selection`) and the weight and grid in `vrun_fit.json`; `judgement.weight_rule` `mean` is the first draft's
mean of both heads. The three binary heads of memo 3.4 are not separate outputs: the floor distribution gives them. **Kept** is the D33 keep record
(`keep.evaluate_value_net`, keep test (1)). The ranking check against the incumbent (keep test (2), with its rest, heal and human-agreement checks),
the card **skip guard** (a fall of more than `skip_guard_points` per 100 in card skips fails unless the label leg favours the candidate) and the
**label bound** ((3): the new judgement with the new prior against the incumbent's blend on the held-out labelled decisions, the lower end of the paired
95% interval above `label_bound_floors`, a floor a decision) are **report only** (`judgement.guards` `report`, in `refit.json` and `vrun_fit.json`
under `guards`, each with its figures and `ok`): the label bound refuses a truly level candidate about 83 times in 100 at 240 held-out K 2 decisions,
and the Act 1 skip guard would have refused generation 12, whose gain came in Act 2. `guards` `gate` makes (2) and (3) decide `kept` again.
`candidate` refuses a fit that is not kept. The keep test v2's routes A and F (the smoke and the block) stay the rule.

## Lambda by act (the recipe option, D55 addendum 12)

The blended chooser's lambda may differ by act: the recipe string takes optional parts `:act<N>=<lambda>` after `:pi_h=...` (`macro_recipe`), e.g.

    vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0

decides Act 1's deck screens (card, ancient, rest) with lambda 0, which is exactly `vrun` (no prior is read), and every later act with the base
`lambda=` 0.5. The act of a decision is its floor's (`macro_recipe.act_of_floor`: floors up to 17 are Act 1, 18 to 33 Act 2, later ones Act 3; the
Act 2 ancient is floor 18), the switch of the labels-question probe. One canonical string per chooser: the acts in order, none twice, none equal to
the base lambda. agent.json records it as `human_prior.lambda_acts` (`{"1": "0"}`) beside the recipe, and the loader refuses an agent.json that
disagrees with its recipe. A recipe without the parts is the recipe it always was, byte for byte (a test pins it).

Build: `gen0.py build ... --human-prior W --lambda 0.5 --lambda-act 1=0 --stakes a3-v1@h1e4b16 --tag prior-act2` (or `gen_loop.build_candidate(...,
lam_acts={1: 0.0})`). **Generation 13's recipe with lambda 0 in Act 1 and 0.5 from Act 2** is the folder
`D:/spireline/referee/agent/gen14/candidate-prior-act2` (recipe `vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0+stakes:a3-v1@h1e4b16`; every weight file
identical to `gen13/candidate-stakes`, only model.json's `iteration` differs), to be judged by the lab's train job on the keep test v2 against
generation 13 (marker `LAB/train/ready-11-g14-prior-act2`). The probe measured the arm at a3-v1 with generation 11's fighter: deep +3.38 per 100 runs
(SE 0.62), Act 1 boss +3.94 and floors +0.63 over generation 12's blend; the block under generation 13's stakes fighter decides (the memo expects it to
pass the floor rule, route F, not the Act 2 rule). The round's candidate carries the act gate when `prior.lambda_acts` is set (`{"1": 0.0}`) in the
config and the block holds.

## The parameters (`rl/round1_config.json`; the file is `round1_data.DEFAULTS`, a test pins them equal)

Pass `--config FILE` with only the keys to change (unknown keys are errors). Memo defaults where the memo gives one; **builder** marks my choice
where it does not; **addendum 12** the labels-against-blocks decision.

| key | default | source |
|---|---|---|
| `label.k` | 2 | memo 2.1 (K 2) |
| `label.quota` | Act 1: card 200, rest 120, ancient 40, shop 140, route 100; Act 2: card 240, rest 140, ancient 100, shop 70, route 50 (1,200) | addendum 12: at least half in Act 2, ancients a kind; the cell split is the builder's |
| `label.per_act` (`per_run`) | 2 (2, read only when `per_act` is 0) | addendum 12: two a run per act |
| `label.act2_min_share` | 0.5 | addendum 12; checked against the quota |
| `label.cont_boss_mult` | 8 (0: the player's own) | addendum 12: the continuations on `candidate-stakes-b8`'s schedule |
| `label.rates` | card 0.30, rest 0.50, route 0.07, shop 0.60, ancient 1.0 | hindsight's rates; the ancient's is the builder's |
| `label.max_options`, `own_schedule` | 6, true | hindsight's; own schedule per the brief |
| `label.heldout_fraction` | 0.2 | sizing memo (a fifth of runs held out) |
| `prior.lambda` | 0.5 | the candidates' setting (generation 12/13); addendum 12 |
| `prior.lambda_rule` | `fixed` | addendum 12: the labels do not choose lambda (`labels` takes the best of `lambda_grid` only if it beats the fixed one by a standard error; report only) |
| `prior.lambda_acts` | `{}` | addendum 12: `{"1": 0.0}` once the act gate's block holds |
| `prior.weights` | human 1.0, agent 0.0, label 0.0 | addendum 12 (memo section 4) |
| `prior.kl_beta` | 1.0 | addendum 12: the refit stays near `pi_h-v0`, the prior that gained |
| `prior.label_gap_se` | 2.0 | memo 3.5 (gap over 2 SE) |
| `prior.lr`, `epochs`, `batch`, `patience` | 1e-4, 8, 512, 3 | a fine-tune of a fitted prior (builder) |
| `prior.max_agent_decisions` | 40,000 | memory (about 160k rows) (builder) |
| `judgement.human_weights` | 0.25, 1, 4 | memo 3.4; the grid of the selection step |
| `judgement.weight_rule`, `act1_tolerance` | `clears_act2`, 0.0 | addendum 12: lowest held-out clears-Act-2 log-loss, clears Act 1 not worse |
| `judgement.guards`, `label_bound_floors`, `skip_guard_points` | `report`, -0.1, 5.0 | addendum 12: keep test (2) and (3) report only; the strategy memo's -0.1 and 5 points |
| `judgement.seeds` | 1, 2, 3 | memo 3.4 (three seeds, averaged) |
| `judgement.beta` | 0 | memo 3.4 (pure Monte Carlo) |
| `judgement.lr`, `epochs`, `batch`, `end_hp_weight` | 3e-4, 6, 1024, 0.25 | train.py's end-HP weight; a fine-tune (builder) |
| `smoke.seeds`, `block.seeds`, `keep_rule` | 200, 3,000, v2 | memo 2.1; keep test v2 |
| `futility.enabled`, `futility.min_seeds` | true, 500 | D55 addendum 18: the block stops as rejected (`futility@<seeds>`) when no route can reach its promotion gain; passed to `promote` as `--no-futility` / `--futility-min-seeds` (docs/rl/KEEP_TEST_V2.md) |
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
there, labels one decision at K 1 (the continuations on the lowered boss multiplier, `flat-4@h1e4b8`), refits both artefacts on one human shard for one
epoch, builds a candidate in the scratch folder and parses the smoke and block commands with gen0's own parser. It starts no block, no real game, and
writes nothing under the agent or lab folders.

`jobs-fragment.json` holds one disabled `round-<step>` entry per step (command, cores, `needs`, `leaves`). The runner's kinds are fixed
(`tools/lab/jobs.py`), so these are not runnable by it yet; the orchestrator merges them with a `jobs.json` backup at a quiet time and either
adds the kinds to the runner or runs the steps by hand.

## What each held-out report means

* `labels/label-summary.json`: decisions per cell, by act (and Act 2's share), the schedules, the pooled single-draw noise (sd of the difference of two options on one draw), the held-out share.
* `prior/refit.json` `fit.heldout`: **human** (fifth 4) top-1 and NLL per kind before and after: how far the prior moved from human choices;
  **agent** (held-out runs) NLL/top-1 of the player's own pick: how far it moved toward the player (not trained on at weight 0); **label** the NLL of the label winners.
  `fit.trained_on` and `fit.source_weights` say which sources the fit used. `label_mix`: the train and held-out labels by act and cell.
  `label_leg`: on held-out labelled decisions the mean centred value A (floors) of the pick among the played options by the old blend (V plus lambda log
  pi_old), the new blend (with `prior.lambda_acts` when set), the judgement alone, the new prior alone and each grid lambda, with `*_minus_old_blend` paired
  differences (mean, se, 95% interval); cells `all`, `card`, `rest`, `ancient`, `act1`, `act2`; `players_own_pick` and `best_option` bracket them. **A report,
  not an arbiter**: at 240 held-out K 2 decisions the paired SE of a chooser comparison is about 0.15 floors a decision, too weak to choose a lambda or a
  weight that moves the frontier (the blocks do). It is not an exam (K 2 is noisy: one label is a fifth signal).
* `judgement/refit.json`: `selection` the human weight chosen, the grid and each weight's held-out figures; `fit.grid` the same by seed; `incumbent_heldout` and
  `kept_heldout` the held-out log-loss of the incumbent and the kept fit (lower is better); `keep` the D33 record; `guards` the ranking check, the skip guard and
  the label bound as numbers; `label_leg` the picks of the incumbent blend, the new judgement with the old prior, with the new prior, and alone.
* Cells under 20 decisions are not shown. Human data enters as aggregates only; no run id is written; attribution "Spire Codex public run export for v0.107.1".

## Not built

FightNet retraining (`fight_net` retrain), the three binary heads as outputs, shop and rest-site-target rows in the agent and label terms (card, ancient
and rest screens only), the judge-leaf and run-currency parts of the first stack. The option-by-schedule interaction of x8 labels is an argument, not a
measurement (memo section 5).
