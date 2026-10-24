# D24 promotion guard: the independent review's section 4 and the plan's promotion rule (snapshot 2026-10-24)

From microchimp/spireline `docs/reviews/2026-10-24-cloud-plan-and-week3.md` section 4 and appendix A, and `docs/PLAN.md` section 5 / `docs/DESIGN.md` 5.4, copied for the cloud item that implements the guard in `tools/lab`.

## Review section 4

## 4. D24: the boss-kill guard

### 4.1 Power at the current rates

The promotion block is 2,000 paired seeds (`resolve_e_block`, n = 2000; D24's 53 discordant pairs match it), not
500; the 500-run figure is the real-game scoring block. Both are tabulated. The test is the exact one-sided McNemar
on discordant pairs, as implemented in `rl/gen_loop.py` (`mcnemar_exact_one_sided`, `decide`). *Estimated* under
independence of the two arms' kills on a seed (discordance = p1 + p2 - 2 p1 p2); real arms are positively
correlated (D24 saw 53 discordant pairs where independence predicts 66), so the true power is lower still.

Guard mode: the probability that a candidate whose kill rate fell is stopped (incumbent at 2%).

| true drop | seeds | one-sided 0.01 | one-sided 0.05 |
|---|---|---|---|
| 2.0% to 1.5% (a quarter) | 500 | 2% | 10% |
| 2.0% to 1.3% (a third) | 500 | 4% | 16% |
| 2.0% to 1.0% (a half) | 500 | 9% | 28% |
| 2.0% to 1.5% | 2,000 | 11% | 29% |
| 2.0% to 1.3% | 2,000 | 24% | 49% |
| 2.0% to 1.0% | 2,000 | 57% | 81% |
| 2.0% to 0.5% | 2,000 | 98% | 100% |

At 500 seeds there are about 15 discordant pairs and the 0.01 guard needs 13 of them one way: a formality. At
2,000 the guard catches a collapse to a quarter and usually a halving; it does not see a 25% drop. The compounding
case is the one that matters: six generations each losing a quarter of the kill rate take it from 2% to 0.4%, and
the probability that the 0.01 guard never fires along the way is about 0.89^6, one half. At 0.05 it is 0.71^6,
about one in eight, and the guard weakens further as the rate falls (fewer discordant pairs).

Co-primary mode: the probability that a real improvement passes (one-sided 0.01).

| improvement | 500 seeds | 2,000 seeds | 4,000 seeds | seeds for 80% power |
|---|---|---|---|---|
| 1.35% to 2.0% (generation 1) | 4% | 20% | 44% | about 7,700 |
| 2% to 2.5% | - | - | - | about 17,600 |
| 2% to 3% | 6% | 35% | 68% | about 4,800 |
| 2% to 4% | 26% | 91% | 100% | about 1,400 |
| 10% to 13% | 18% | 73% | 97% | about 2,300 |
| 10% to 15% | 49% | 99% | 100% | about 900 |

### 4.2 Verdict on D24

- **Dropping the co-primary at 1 to 2% is right.** A co-primary with 80% power against a 50% relative gain needs
  about 4,800 paired seeds (2.4 blocks); against a 25% gain, about 17,600 (nine blocks, a day of self-play per
  generation). The lab cannot pay that per generation. The 10% switch is well placed: at 10% and 2,000 seeds a 30%
  relative gain has 73% power, a 50% gain 99%.
- **The guard at 0.01 is nearly free of false fires and nearly free of power.** Its cost model is asymmetric: a
  false fire costs one retrial on a fresh block (the loop already grants another budget after a rejection); a
  missed collapse costs generations and shows up only in the weekly real-game block. A guard should lean toward
  firing. Run it at one-sided 0.05: power against a halving rises from 57% to 81%, against a one-third drop from 24%
  to 49%, at the price of about one spurious retrial in twenty generations.
- **Re-judging generation 1 on its existing block under the new rule** is post hoc, and D24 says so; it is
  defensible because the primary passed at p = 5e-5 and the rule change was argued from power, not from the result.
  It should not become a habit: rule changes apply from the next block.

### 4.3 A better guard

Keep the McNemar guard (at 0.05) and add two checks that cost no extra runs:

1. **A drift check across generations.** Pool the promoted lineage's last three blocks (6,000 seeds) and compare the
   candidate's kill rate on its block with the pooled rate (unpaired, same seed distribution; Wilson intervals). Flag
   the promotion for the weekly review when the candidate's 95% upper bound is below the pooled lower bound; block it
   when that happens twice in a row. This catches the slow trade of kills for floors that no per-generation test at
   these rates can see.
2. **A continuous boss-fight measure as a paired guard.** Damage dealt to the Act-1 boss as a fraction of its HP, 0
   when the boss is not reached, paired by seed, cluster bootstrap exactly as the floor primary, not-worse at
   one-sided 0.05. It measures the intent of the boss-kill check (the ability to win the boss fight) with the power
   of a continuous variable, and becomes informative long before kills reach 10%. It needs one field in `runs.jsonl`.

Two side findings from the promotion code and log. (a) The loop grants another budget after each rejection with
no cap: generation 3 was rejected at 23:29Z and promoted at 00:54Z on a fresh block. The retried candidate is
retrained, so it is not the same test twice, but each attempt is another 1% draw; over 20 generations with two
attempts each the expected false promotions are about 0.4, not 0.2. Cap attempts at two per generation and record
the attempt in `promotions.tsv` (the block name already carries `-r1`). (b) Promotions at the margin (lower bounds
+0.09 and +0.06 floors) are real under the rule but subject to the winner's curse: the promoted agent's true gain is
smaller than its block showed. The real-game block is the unbiased number, which is why the headline must stay there.


## Appendix A: how the numbers in section 4 were computed

Exact McNemar power: for n paired seeds with kill rates p1 (incumbent) and p2 (candidate), discordance
d = p1 + p2 - 2 p1 p2 (independence); the number of discordant pairs m is Binomial(n, d); given m, the count of
incumbent-only kills is Binomial(m, (p1 - p1 p2) / d); the guard fires when that count reaches the smallest k with
P(X >= k | m, 1/2) <= alpha. Power is the mixture over m. Sample sizes for 80% power use the normal approximation on
the discordant pairs, one-sided alpha 0.01. The independence assumption overstates discordance and so overstates
power for real arms. The arithmetic is 60 lines of standard-library Python and can be re-run by anyone.


## PLAN.md section 5 (promotion rule)

## 5. Lanes

Each lane names its deliverables, its round-the-clock CPU jobs, what agent rounds do, and its kill rule.

### Lane A: the agent

**Deliverables.** A PLAY agent (IS-MCTS with hidden-state resampling, `DESIGN.md` section 4.1) that plays whole A10
Ironclad runs on the emulator and, through lockstep, in the referee; expert-iteration generations that improve it; a
weekly real-game score; from P2, an offline run review (`spireline review`) that re-grades a logged run decision by
decision.

**The bridge, decided.** For training and scoring, the emulator runs in lockstep with the referee: both start from the
same seed, the emulator replays every decision the agent makes in the referee, and at each decision the search runs
from the emulator's lockstep state through `clone(resample_hidden=True)` plus the AI-state particles of `DESIGN.md`
section 1.1. The search never sees the lockstep state's RNG counters or draw order; the leak tests of `DESIGN.md`
section 7.6 (L2, L3, L8 and a clairvoyance test: resampling on against off must not be value-equivalent) are gates.
When the lockstep state's hash differs from the referee's, that decision is logged as an agent-weighted divergence
(Lane B) and the emulator is re-anchored from the referee's record where `Reanchor` can (fight start, room entry);
otherwise the run continues on the referee alone and its later decisions are scored but not ranked. This reuses the
lab's `runs` replay, `Reanchor` and the structural comparer; no new state format. The observation-only bridge
(`DESIGN.md` section 3.2, for a live advisor that must never hold the seed) is a P3 item.

**CPU jobs (new lab job kinds, section 7).**
- `selfplay`: emulator self-play runs with the current PLAY agent and generation; the filler that takes every idle core.
- `train`: on the GPU, after each generation's self-play budget: the combat prior (Gumbel ExIt) and V_run (DMC),
  `DESIGN.md` section 5.4, with promotion on a fresh 2,000-seed emulator E-block (paired mean floor, cluster bootstrap,
  one-sided alpha 0.01; Act-1 boss-kill rate a not-worse guard, one-sided 0.01, until both arms reach 10%, then
  co-primary; wins join once both arms reach 10%; D24).
- `score-real`: the promoted agent plays a fixed real-game block (E-block seeds), lockstep on, at 12-16 game processes.
  Yield: win rate, mean floor, crashes (counted as losses), decisions/s, and the agent-weighted divergence list for Lane B.

**Agent rounds.** Port and run the fork's `rl/` on Windows (A0); the lockstep adapter for `envproto.py` (A1); the
generation loop as a lab job with checkpoints and promotion (A2); the search budget schedule and Gumbel settings re-tuned
on this PC (A3, Opus); Acts 2-3 after the run-clone tests F3/F4 pass (A4); the offline review command (A5, P2); labels
and the regret benchmark (A6, P2, `DESIGN.md` sections 4.3 and 7.1 adapted to Ironclad A10, with the 2% referee spot
check).

**Kill rule.** If after six generations the promoted agent does not beat generation 0 on the real-game block (paired mean
floor, CI excluding 0), stop training, keep the searcher, and spend the lane's budget on search quality (candidates,
racing, budgets) until the regret benchmark exists.

### Lane B: fidelity on demand

**Deliverables.** A weekly ranked list of emulator divergences weighted by agent play; fixes for the top of it landed
through bug process v2; the measured transfer gap; the G1 decision on the kernel rebuild.

**CPU jobs.** `score-real` (Lane A) produces the lockstep comparisons. The existing `runs` repro step shrinks each first
divergence to a setup fight, and `lab_groups` groups them by mechanism. A new `rank` step (part of `score-real`'s
analysis) weights each group by the number of agent decisions it affected, its first-divergence floor, and whether the
run's outcome differed between engines. Existing strategies continue at reduced weight (section 7), with `aimed-bugs`
pointed at the agent-ranked groups.

**Agent rounds.** Fix kits for the top-ranked groups, Sonnet fixers with the section 10 checklist, one reviewer, landed
per commit behind the gates. Opus only for mechanism classes the audits named (listener order, action queue, rounding
order) and for amends. The driver and host lane of `rebuild-plan.md` D11 (~10 rounds) runs only for host-suspect groups
that appear in agent-ranked lists.

**The transfer gap.** `G = W_emu(pi) - W_real(pi)` on paired seeds, plus the per-decision state-divergence rate (share of
agent decisions whose lockstep hash differs from the referee's), both from `score-real`. Reported weekly with intervals.

**Kill rule and gate.** Section 8.3 (G1).

### Lane C: the referee as environment

**Deliverables.** A hardened full-host env: process recycling, crash as episode truncation, the ten-minute cliff named (rerun the 12-process, 31-minute conditions with the bench's new system-stats logging) and
the per-run memory growth bounded by recycling,
the missing action types added (choose 0, card-reward alternatives, potion use and discard outside fights), a
`character` key, a whole-run A/B of the fast settings against stock pacing over a few hundred runs with run hashes
(review m3), and the H6 patch list current. Throughput measured per phase and per process count.

**CPU jobs.** `bench-env` nightly: 20 minutes at 12 processes, sustained decisions/s, crash count, memory curve, into
`readiness.json`.

**Agent rounds.** C0 one measurement round (profile per phase, what limits scaling past ~12 processes, long episodes with
the PLAY agent); C1 hardening and the action-space gaps; C2 the `character` key and the A/B. The nodeless run host
(`nodeless-run-host.md`, first slice size M, run-level fallback for the 27 encounter models) and the Godot-free host are
built only if section 8.1's bar says so. Addendum 2 for the Godot-free host is drafted (owner reading, no code) only
then.

**Kill rule.** If the env cannot sustain 50 decisions/s with recycling, or crashes exceed 5% of 700-decision runs,
scoring moves to fewer, longer-spaced blocks and the shortfall is reported; no host build starts on that basis alone.

### Lane D: the CPU scheduler

**Deliverables.** The lab rotation extended with `selfplay`, `train`, `score-real`, `bench-env` and `bench-emu`; weights
in `mix.json`; yield columns for the agent metrics; a utilisation report (CPU minutes used over wall minutes, excluding
owner-use windows) in `readiness.json`. Target: 90% of wall-clock busy.

**Agent rounds.** D0 job kinds and guards (2-3 rounds, Sonnet); D1 the weekly mix re-weighting rule from `yield.tsv`
(Haiku, mechanical).

### Lane E: security and ops (owner actions, no rounds)

- H2: add the Windows Firewall outbound block for the Godot host executable and `crashpad_handler.exe` now. Continuous
  training use multiplies the exposure; the rule stops being optional.
- H3: checked 2026-10-22: no profile folder exists on this PC (the game is installed but has never been played here,
  so the folder the referee watches was never created); nothing to back up. The referee's per-session check that the
  folder is not created or touched (`run_referee.py`, `real_profile_state`) is the control. If the game is ever played
  on this PC, back the folder up before the next `score-real` block and compare file lists after it.
- H6: refresh the patch list with every env patch and the trace-mode `L` patch (rebuild-plan D2, approved under D-d);
  record whether a live-client check is possible on this PC (no readable profile is noted in `SIMULATOR.md`); if it is
  not, say so in the addendum rather than leaving H6 silent.
- Version policy: stay on v0.107.1 through P2. A port is priced when a later build is archived, never assumed.

### Lane F: the public in-fight mod (side product, D22)

**What it is.** A game mod, our own code, that reads the live fight through the game's modding hooks and shows the
agent's recommended play with a short reason. It runs the **distilled combat policy network alone**: no search, no
emulator, no referee, so nothing private ships. Answers are instant and the network is a few MB. Its advice is weaker
than the searching agent by a measured margin that the release notes state.

**Why a distilled network and not the searcher.** The searcher needs the emulator, and the simulator stays private
(Amendment 1a). A network trained on the searcher's own play (the combat prior of Lane A, `DESIGN.md` section 5.3)
captures most of its in-fight strength and can be exported (ONNX) and run inside a C# mod on any player's PC.

**Release gate (pre-registered).** All of: the network's top action agrees with the searching agent's on at least 85%
of decisive benchmark positions and is in the near-best set on at least 95%; the network playing alone (no search)
loses less HP per fight than the human Codex median per floor on paired setups; the mod reads only displayed state
(the allow-list of `DESIGN.md` section 3.2) and never the seed or hidden piles, checked by the leak tests; the mod's
patch list is published with reasons and contains no game text; the game's and the modding framework's terms allow it
(checked and recorded before release); and the owner approves the release.

**CPU jobs.** `distill`: after each promoted generation, export the combat prior and score it against the searcher
on the benchmark (agreement, near-best share, HP per fight). Its numbers go to `readiness.json` under `mod`.

**Agent rounds.** F0 (P2): the export and the agreement scorer (2-3 rounds, Sonnet). F1 (P2): the mod itself, built
on the game's modding framework in a public repo of its own, reading state through the allow-list, with the patch
list (6-10 rounds, Sonnet; Opus for the state reader). F2 (P3): release once the gate passes: the mod, the weights, a
short write-up of what it is and is not, and the seed block for the head-to-head.

**Not in scope.** Autoplay, Daily runs, leaderboards (`CONTRIBUTING.md` rule 6); run-level advice (map, shop, rewards)
until the run value network passes its own gate; any server.

**Kill rule.** If the network alone cannot pass the agreement gate within three generations after the searcher beats
reference (1), the release waits; the mod is never shipped weaker than the human median in fights.


## DESIGN.md 5.4

### 5.4 Training loop (`learn/exit.py`)

```
for gen in 0..G:                                   # each generation capped at ~8 h wall-clock
  # R: run level (Act 1 only until F3/F4 pass)
  self-play A0 Defect runs on T-seeds until the 5-h self-play budget is spent
      combat = PLAY(prior_gen, V_run_gen, 120/500/1000, Gumbel noise)
      macro  = V_run afterstate argmax, eps 0.15 -> 0.05, capped tempo deck-fights on close calls
               (NO pi_ref Monte Carlo verification inside self-play)
  train V_run, FightNet on last 3 generations (validation split BY RUN)
  # C: combat ExIt
  6,400 fights sampled from this generation's runs, 300 sims, crn_fraction 0.2 -> Gumbel completed-Q targets
  train combat net (per-head checkpoints, 167k-935k params; capacity was never the limit)
  # promotion, on a FRESH 2,000-seed E-block each generation
  primary:     paired mean floor (cluster bootstrap), one-sided, alpha 0.01
  guard:       Act-1 boss-kill rate not worse (exact McNemar, one-sided 0.01) while either arm < 10%;
               co-primary once both arms >= 10% (D24). Wins join as co-primary once both arms >= 10%
  AND F7 audit clean; else revert and log in DECISIONS.md
  regenerate 200 rotating silver labels under the new pi_ref; report argmax stability across generations
```

Promotions are engineering decisions, made on independent fresh blocks. At α = 0.01, about 0.2 false promotions are expected over 20 generations. Headline claims use only the F pool, once per phase gate.

