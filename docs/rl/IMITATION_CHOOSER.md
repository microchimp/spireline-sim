# The imitation chooser (`vrun+pi_h`)

Written 2026-11-05 for milestone 4 of the strategy memo (D55; `docs/design/2026-11-05-strategy-rethink.md` in the Spireline repo, sections
3.3 and 3.5 step 1). The human prior `pi_h` (`docs/rl/HUMAN_PRIOR.md`, `rl/human_prior.py`) becomes a chooser the player can play, and the
first imitation candidates are built for the lab to judge.

## The rule

The `vrun` macro scores each option's afterstate with the run judgement `V` (expected final floors, a won run worth 53) and takes the best.
The blended macro takes, on the deck decisions the prior was fitted on,

    argmax_o [ V(o) + lambda * log pi_h(o) ]

* `lambda = inf` is the prior alone (ties by `V`); `lambda = 0` is `vrun` (the prior is not consulted); the lab's candidates are
  `lambda = inf` (`candidate-imit`) and `lambda = 0.5` (`candidate-imit-l05`; the memo: a 3-to-1 preference is 0.55 floors).
* **Screens decided by the blend:** a card reward (the cards and the skip), an ancient's relic (`neow` screens), a rest site's kind and the
  smith's target. **Left to `V` alone:** routes, events, shops, potions, relic rewards, deck selections (`pi_h-v0` has no shop fit), and any
  deck screen holding an option the prior has no row for (a card reward with a reroll or sacrifice, a rest site with its leave or hatch).
  Fights are not the macro's and are unchanged.
* **Guard (lambda > 0):** an option the prior gives under 0.02 is taken only when `V` prefers it to the blend's pick by more than one floor.
  `lambda = 0` is `vrun` and has no guard.
* **The rest site is two choices**, as the fit has it: the kind (heal, smith, or the relic-given others) by the prior's marginal over the
  kinds, then within smith the target by the prior's softmax over the smith options. A kind's `V` is its best option's. Within the
  relic-given others `V` decides (the fit has no softmax there). The flat softmax over all options would heal in a case where the kinds'
  marginal says smith (heal 0.45, two smith targets 0.30 and 0.25), which is why the stages are kept apart.
* At play the true map is in the state, so the map-ahead block and the distance to the boss are the run's own (the fit drew a donor's).
  The fight block H is the judgement's FightNet on the afterstate, as the fit had it (the candidate carries generation 9's run
  judgement, the file the prior was fitted from).
* **Exploring self-play** (`--explore-tau/--explore-eps`, the filler) samples around the blend: `softmax(score / tau)` over the options the
  guard leaves, plus `eps` uniform; the rest site is sampled on its flat joint score. Evaluation plays the argmax.

## Where it is

| file | what |
|---|---|
| `rl/human_prior_infer.py` | `pi_h` in numpy (no torch): `PriorScorer.load(weights.npz)`, `logits`, `log_probs(view, options, nets)`, `option_rows` (the judgement's input row of each option's afterstate, H appended) |
| `rl/imitation_chooser.py` | `blend_pick`, `blend_rest` (pure), `screen_of`, `BlendedPolicy` (a `LookaheadPolicy`; `last` says which screen it decided and whether the prior's top, `V`'s top or the guard decided) |
| `rl/macro_recipe.py` | the recipe strings: `macro_name`, `parse_macro`, `is_blended`, `recipe_of` |
| `rl/gen_loop.py` | `Agent.human_prior/lam`, `load_agent` (a blended macro needs `human_prior.npz` of the sha the macro names), `build_candidate(human_prior=, lam=, tag=)`, the recipe in the ledger (below), tagged markers |
| `rl/gen0.py` | `build --human-prior W --lambda L --tag T`; `train` takes `ready-<base>-g<N>-<tag>`; the run row gains `decided` |
| `tools/lab/jobs.py` | the runner reads `ready-<gen>-g<N>-<tag>` (marker order, `params_for`: `tag`) |
| `tests/python/test_imitation_chooser.py` | 49 tests |

## Identity, attempts, markers

* The candidate's `macro` is the recipe: `vrun+pi_h:lambda=<inf|number>:pi_h=<8 hex of the weights' sha256>`, e.g.
  `vrun+pi_h:lambda=inf:pi_h=2ebfe6d7`. `summary_matches`-style comparisons (`gen0.emu_arm_match`) read the `macro` of an arm's summary, so an
  arm played by another lambda or another prior does not match.
* `promotions.tsv` gains a last column `macro` (the recipe; empty for `rules`/`vrun` and in every older row, so the file is read as before).
  **Attempts are counted per recipe:** `attempts_before(rows, gen, sha, head, recipe)` counts the distinct (fight player, run head) pairs of
  the generation judged under the same recipe; `block_used_by_another` and `resolve_e_block` take the recipe too, so two recipes of one
  generation number never share a block. Generation 12's closed self-play line (two refusals, recipe "") does not count against the
  imitation line; a lambda or a prior of its own is a recipe of its own with its own cap of two (the review guard's `ATTEMPT_CAP`).
* Tagged candidates: `gen<N>/candidate-<tag>` (the tag starts with three letters, lowercase words joined by hyphens, never beside
  `--attempt`) and the marker `ready-<base>-g<N>-<tag>` (`done-...` once consumed). `gen0.py train` reads the built candidate's recipe for the cap check
  before it asks `promote` to judge it as it is.

## The first candidates (`gen12`, built on `current.json`: generation 11's fighter, generation 9's run judgement)

```
python rl/gen0.py build --prior <agent>/gen11/candidate --base <agent>/current.json --agent-root <agent> --candidate-generation 12 \
    --tag imit --human-prior <human-prior>/pi_h-v0/weights.npz --lambda inf --why "D55 milestone 4: pure imitation of the human prior pi_h-v0 on the deck decisions"
python rl/gen0.py build ... --tag imit-l05 --lambda 0.5 --why "D55 milestone 4: the human prior pi_h-v0 blended with the run judgement on the deck decisions, lambda 0.5"
```

Both carry `human_prior.npz` (sha256 2ebfe6d7f25e...) and the base's `run_value.npz` (sha256 f0a199fb82c2...); the markers
`ready-11-g12-imit` and `ready-11-g12-imit-l05` are written (`--marker-dir`) once the change is on fork main.

## Checks

* The numpy scorer against the torch net: 200 stored validation decisions (661 rows, each with its fixed donor map), largest difference of a
  logit 1.9e-6 and of a decision's log-probability 1.9e-6; on the 570 rows of 111 live decisions of the sanity run below, 1.9e-6 as well.
  `tests/python/test_imitation_chooser.py` pins the same bound at 1e-4 where the shards are on the machine, and that the rows scored at
  play (`option_rows` and `models.value_inputs`) equal the inputs the fit assembled from its stored rows.
* @@SANITY@@

## Caveats

* Pure imitation plays what an average human picks with our fight play and our judgement of the rest of the run: the memo's own caveat. Whether
  it is worth anything is the lab's to say (paired floors against generation 11, the Act 2 boss kills, the label leg).
* The prior was fitted with donor maps and H from generation 11's run judgement; at play the true map enters and H comes from the candidate's
  own FightNet (the same file for these two candidates).
* The shop is not imitated (`pi_h-v0` has no shop fit); a fit with prices verified would add `shop.*` screens here.
