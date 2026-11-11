# The learning system, judge first: the design (D56, D57, D58)

Written 2026-11-10 and 2026-11-11 by the design lane (`referee/backlog/run-self-play/BRIEF.md`; D56, D57 with addenda 1 to 8, D58 with
addendum 1). Figures are read from the file named, counted from saved rows, measured here (section 9 and the probes under
`referee/backlog/run-self-play/`), or marked *est.* Human figures are aggregates from the Spire Codex public run export for v0.107.1,
training fifths only; the exam fifth is never read; no cell under 20 runs or starts is shown; no player id appears anywhere.

## 0. For the owner

- Today only the fight player has a way to get better on its own. Everything between fights (which room next, which card, rest or
  upgrade, what to buy) is decided by a quick guess from a learned scorer nudged toward what people pick. That guess cannot get better
  than its training, and the gap to people at the second boss is as much the hand we arrive with as how we fight it.
- The plan builds the same "coach and apprentice" loop one level up. At each choice between fights a slower coach tries each option
  for real on copies of the run, with the unknowns redrawn, plays on to the next boss's door, and asks the fight judge how that boss
  fight looks from there. An apprentice network learns to pick what the coach picks; it replaces the people-taste nudge one kind of
  choice at a time, but only where it still agrees with people as well as the old nudge did.
- Measured on 50 runs: the coach costs about 1,400 processor-seconds a run (the player alone about 25), so the machine time the old
  background self-play used makes about 440 coached runs a week. One choice at a time its guesses are noisy (two halves of its own
  practice games name the same best option 56 times in 100, where a coin would 36), so it overrides the player only when it is sure,
  3 times in 100 choices; the runs it played that way went about two floors further and beat the first boss 10 more times in 100, a
  promising sign that 50 runs cannot confirm. The apprentice learns from all of its numbers, where the noise averages out over
  thousands of choices.
- What it sees best is which room to go to and when to rest; it barely tells card rewards apart, where people's taste is known to help.
  So the apprentice takes over routes and rest sites first and card picks last, after the judge has improved.
- Like a chess coach who can only afford a few practice games per position: one game says little, but a student who sees thousands
  of positions learns the pattern.
- The judge comes first because the coach leans on it at every door. A side find: fights started from saved positions (all practice
  and all exams) let the search see the enemy's real move behind the shown intent, yet the same boss fight played from a saved position
  ends about 8 HP worse than played inside the run, so every saved-position measure reads our player a little low; section 5.

## 1. The parts today

| part | what it is | learns from | how much | judged by |
|---|---|---|---|---|
| fight player | the combat network as the search's prior (`a3-v1`, stakes schedule x1/x4/x16) | the fight gym: x16 search as teacher | buffer 977,619 rows (2026-11-10), 17,993 fights on 11-10 | the gym gate; the fight exam's human stratum (v3 64.6 against the humans' 79.8 in 100) |
| fight judge | the network's value at a fight position (P(win), HP if won, potions if won) | v0: 131,359 positions of 3,539 starts, raw outcomes; v1 (not switched): 416,279 positions of 6,840 starts | Act 2 boss: 57 starts (v0); 53 held-out starts (v1) | the board of D55 addendum 21: loses to x16 at the Act 2 boss (j480 15.8, j960 17.0, x16 19.5) |
| run value V | expected final floors of an afterstate (`RunValueNet`, 1,863 inputs incl. FightNet's fight block H) | the window's self-play runs, Monte Carlo floor targets | not refit since generation 12 (`run_value` f0a199fb in gen12 to gen14) | the D33 keep test (Brier skill +0.044 rows4, VRUN_SWEEP.md) |
| human prior pi_h | a softmax over a screen's options (`PriorNet`) | human choices, four training fifths | 108,115 card, 38,339 rest, 15,052 ancient decisions | held-out fifth 4 (card top-1 51.6) |
| run chooser | `argmax V(o) + 0.5 log pi_h(o)` on card, ancient and rest screens; V alone elsewhere | nothing of its own | | the lab's paired blocks (keep test v2, futility, screens) |

The run level has no loop: V is a one-step guess with little skill, pi_h tops out near the average human, and nothing measures which
option is better by playing it. The play-or-deck diagnosis (`referee/backlog/play-or-deck/tables.md`) puts the Act 2 boss gap half in the
hand: generation 11 beats the Act 2 boss 44.4 in 100 from human starts and 14.7 from its own (+29.8 points from the hand alone); at the
door humans hold 1.68 potions against 0.61, 11.6 relics against 8.2, 71.5% of max HP against 63.3%.

## 2. Part 1: the judge (checkpoint C2, by 2026-11-18)

**Why it reads bosses worse than play-outs.** (1) *Few boss decks.* v0 saw 385 Act 1 and 57 Act 2 boss starts; v1 saw the gym's buffer,
6,840 starts of 2,654 runs, with 53 Act 2 boss starts held out (JUDGE.md section 7). Positions of one fight share one outcome, so the
effective sample at a boss is the number of fights, a few thousand, against a board of 400 human decks per act it has never seen (human
Act 2 boss decks are 25 cards or more in 2,535 of 3,701 starts; the loop's own in 285 of 358). (2) *One-draw labels.* A position's label
is the fight's one outcome: at a boss position near P(win) 0.5 the label carries one bit, while x16's root value averages 16 times
a3-v1's simulations of the same position; that is why the root value ranks held-out positions better than v1 does (Spearman 0.902
against 0.860). (3) *Leaf use.* In the search the judge answers for every leaf, blunders included, which it never saw played.
(4) *Features.* The tokens carry the cards by pile, buffs with their owner, relics, potions, the enemies' current and next intents and
the encounter; they do not carry the enemy's move history beyond those two intents (a boss's phase) or any run context (the same HP is
priced alike after a hallway and before a boss). The hand effect above is all in what the tokens see (HP, potions, relics, deck).

**The two new sources** (the corpus lane builds them; section 7 is the composition): (a) the human boss and elite positions with the
human's outcome (pool: 11,539 human boss starts, 27,989 elite, 4,706 hallway after the exam stratum and its runs are out; about three
positions a start), (b) settled search values as the target: converge.py's screen capped at x64 for bosses and x16 for elites (D55
addendum 15 outcome 2), hallways at the teacher's budget.

**Design decisions for the judge lane (lane 1, section 10; running since 2026-11-11 as `feat/judge-train`, so these are the checks it
meets rather than a new lane).**
- *Targets.* The leaf reads one number, `P(win)(HP + 10 potions) - 100 P(death)`. Add a direct regression head on that number trained
  on the settled values (the search's improved estimate, a much lower-variance target than one outcome), keep the three outcome heads
  as auxiliaries on every position. The human outcome is a label of *human* play from that position, not of ours: it enters through a
  source input (as the run judgement's human rows did, ROUND.md) and trains the trunk and the outcome heads; the leaf head reads the
  self-play source. A judge whose leaf learned human outcomes would price boss positions 18 to 27 points too high for our player (the human stratum's gaps at the two bosses).
- *Weights.* The stratified weights of section 7; held-out by run (one in five by seed hash), the exam fifth and the exam's runs never in.
- *A cheap screen before the board.* On held-out boss turns with settled values (by run), the judge's rank correlation with the settled
  value must beat x16's own root value's on the same turns; a judge that ranks worse than x16's root cannot win a board that pits it
  against x16. The screen costs about a core-hour; the board about 48 core-hours on the old engine (*est.* 18 on the new, 0.371 CPU ratio).
- *The test:* addendum 21's board under rule 1.3 (j960 at least +3 per 100 over x16 with p < 0.05, or j480's one-sided 95% lower bound
  above -2). Failing it with the human and settled labels in means the features, not the data (D58 C2's named response): a features lane
  (the boss's move history, the run context) before any more training.

## 3. Part 2: the fight gym's gate (checkpoint C3, by 2026-11-23)

The gym-gate lane (`referee/backlog/gym-gate/BRIEF.md`) builds D57's rule: promote on the old clear win, or when not worse (lower bound
above -1.0) and the pooled paired difference of the last five stage-2 candidates is positive at one-sided p < 0.10. Two refinements:

- **The five candidates are correlated.** They are all paired against the incumbent's same 2,000 games, so their differences share the
  incumbent's draws: with candidates and incumbent exchangeable at correlation rho, `Var(d) = 2 s^2 (1 - rho)` and `Cov(d_i, d_j) =
  s^2 (1 - rho) = Var(d) / 2`. The mean of five differences then has variance `0.6 Var(d)`, not `0.2 Var(d)`: an independent-SE test
  understates the pooled SE by a factor of 1.7 and promotes level candidates far more often than its nominal 10%. The statistic should
  average the five candidates' outcomes per start (and draw) first and then take one paired test over starts against the incumbent, which
  carries the correlation by construction; the false-promotion rate should be read on a simulated level sequence with that correlation,
  not on independent draws. **Message for the orchestrator** (gym-gate lane).
- **A ratchet on the human comparison.** Every promotion plays the 400-start human stratum (already in the brief). Add: when a promoted
  version's comparison is below its predecessor's by more than two paired standard errors, the loop reverts to the predecessor and logs it.
  "Not worse" then stays honest when the trend test lets a level candidate through. Optional; it changes no gate input.

C3's measurement is the weekly human comparison on the same 400 starts (64.6 to at least 66, Act 2 boss 56.3 to at least 58). Section 5's
find bears on it: those 1,600 games start from documents.

## 4. Part 3: the run-level loop (checkpoints C4 by 2026-11-27, C5 by about 2026-12-10)

### 4.1 The teacher

At each run-level screen of a self-play run on the copy (route, card reward, rest site, shop, event, the ancient's relic, potion and relic
rewards, deck selections), on floors before the Act 2 boss (Act 3's fidelity is barely verified, D38): the player's own chooser gives its
pick and its score of every option; the teacher shortlists M options (the player's pick, the unchanged ones such as skip or leave, the
player's best-scored others), and plays each forward K times on resampled clones of the run (`hindsight.clone_session`: the map, the boss,
shown intents and known cards kept, everything unseen redrawn; section 5), the same K clones for every option (common random numbers), with
the player's own chooser and fight search at its stakes schedule, **to the next boss door**. The leaf there is the fight judge's estimate
of the boss fight from that exact position: `G = P(death) x boss floor + (1 - P(death)) x B[act]`, with `B` the mean final floor of the
player's runs that got past that boss (generation 13 on E14-2026-11-08: 30.93 past the Act 1 boss of 1,516 runs, 42.30 past the Act 2 boss
of 241); a death on the way scores its floor. `Q(o)` is the mean of G over the K clones, in floors (V's unit). The teacher's pick is the
player's own rule with Q in place of V: `argmax Q(o) + lambda log pi_h(o)` on the blended screens with the same guard and the rest site's
two stages, `argmax Q(o)` elsewhere (`rl/run_teacher.py`, `teacher_index`).

**Why the door and not the act's end** (the 11-05 labels rescored, `run_teacher.py relabel`, `probes/door_leaf.out`; 1,449 decisions, K 4,
gen11 at a3-v1, judge v0's estimate recorded at every door): the boss fight is most of the noise and most of the cost of a continuation.

| leaf | paired sd of one draw's option difference | noise sd of a centred option value | true spread between options | SNR a decision at K 4 | split-half agreement of the best option (chance) | over the player's pick at 2 SE |
|---|---|---|---|---|---|---|
| the boss played (hindsight labels) | 5.80 | 3.67 | 1.11 | 0.61 | 35.1 (24.6) | 4.2 in 100 |
| the door, judge v0 | 3.11 | 2.00 | 0.86 | 0.86 | 37.3 (24.6) | 11.7 in 100 |

By kind (door leaf): card 3.41 / spread 0.64, rest 2.76 / 0.82, route 4.83 / 1.91, shop 2.94 / 0.72. The door halves the noise sd
(a 3.4 times smaller variance: the information of 3.4 times as many boss-played draws) and keeps 77% of the spread (the centred values
under the two leaves correlate 0.67); an Act 1 continuation at the player's own schedule spends most of its CPU in the x16 boss fight
(66.5 of about 75 s on the old engine, labels-vs-blocks memo section 5), so per CPU-s the door is worth roughly ten times the boss played.
The judge at the door is miscalibrated (v0's mean P(death) 0.604 against a realized 0.487 for a3-v1 continuations; Brier skill 0.237): it
ranks better than it levels. The teacher's lane recalibrates P(death) per act on the player's own door outcomes (Platt), and the leaf
gains whatever the judge lane gains (C2).

**What it plays: guarded.** One decision's values are noisy (SNR under 1), so a raw teacher changes 40 screens in 100 (40.5 in the measurement; 60 in the 11-05 labels) and most of the
changes are noise; in the two-run smoke at K 2 the raw teacher died on floor 9 and 28 where the player reached 30 and 43. The teacher
takes its own pick over the player's only when the measured gain (Q difference plus the prior's term on a blended screen) exceeds two
standard errors, the decision's own variance of the paired difference pooled with the kind's door-leaf prior (`guard_se`: 4 prior degrees
of freedom, so two draws that happen to agree do not make a tiny error). Its raw pick and every draw are recorded: the student learns from
the numbers, the run follows the guarded pick, so the self-play runs stay at least as good as the player's and V's refit data stay near
the next generation's play.

**What "deeper" costs** (section 9): the player's own decision is a forward pass of V and pi_h (milliseconds); the teacher's is
33.5 CPU-s a screen at K 4 and M 4, 43.8 screens a run, 1,383 CPU-s a run against 25.3 for the player's own run (55 times), measured
while the machine ran about ten of our processes; on the machine shared with the corpus builder after 00:42Z the same figures read
48.7, 2,342 and 42.2 (contention inflates CPU-s by about 1.45 on this machine, both arms alike).

### 4.2 The student

A run-policy network `pi_s` with the human prior's architecture and inputs (`PriorNet`: the afterstate rows the chooser already builds,
`human_prior_infer.option_rows`, 1,863 wide with FightNet's H), started from pi_h's weights, one softmax per screen as pi_h's fit has it
(card and ancient one softmax, rest site's kind then the smith's target), with the screens pi_h has no fit for (route, shop, event, reward,
deck selection) as new kinds. Loss per batch:

    CE(p_T, pi_s) on teacher decisions  +  beta x CE(human choice, pi_s) on human decisions of the training fifths  +  kl x KL(pi_h || pi_s)

`p_T` is the teacher's distribution over the options it played: `softmax((Q(o) + lambda log pi_h(o)) / tau_T)` with `tau_T` the noise sd
of a centred option value at the decision's K (about 2.0 / sqrt(K) floors by kind, section 4.1), so an option the teacher cannot tell apart
gets an even share and the student learns the teacher's clear preferences across thousands of decisions, where the noise averages out
(per decision the SNR is under one; over N decisions in a feature region the error of a learned advantage falls as 1/sqrt(N)). Options off
the shortlist take the student's own mass (no target). `beta` and `kl` are chosen on the held-out legs below; the human term keeps the
student a human-shaped prior where the teacher is silent.

**What it replaces, and lambda as a measurement.** The chooser becomes `V(o) + lambda_k log pi_s(o)` on the kinds where the student passes
the human bar, pi_h's blend stays on the others. `lambda_k` (per kind and act) is measured, not set: on held-out teacher decisions,
regress the teacher's centred values less V's centred values on the student's centred log-probabilities,
`lambda_k = sum (A_T - A_V) l / sum l^2` with its standard error; it says how many floors the teacher's play-outs credit to one unit of the
network's log-probability beyond what V already sees. The same regression with pi_h measures today's fixed 0.5: 0.09 overall, about 0 at card rewards, 0.13 to 0.27 at rest sites (section 8). A kind whose
interval includes 0 gets no student term. "Never below what the bar allows": a kind failing the human bar keeps pi_h at its current lambda.

**The human bar for the student** (D58 C4): on held-out human choices of fifth 4, by kind, the student's log-loss on won runs within +0.002
of pi_h's (the screen:prior rule of D55 addendum 15), the all-runs guard 0.02; kinds with no human fit (events, rewards, deck selections)
have no human bar and hand over on the teacher leg and the lab's judgement alone.

### 4.3 The data

- **Self-play with the teacher**, the current generation on the copy, Acts 1 and 2 screens. Budget: the filler's 168 core-hours a week
  (D58 addendum 1 stopped the filler; this job takes its place with its own targets). At the measured 1,383 CPU-s a run that is
  62 runs a day, 437 a week, about 19,100 teacher decisions a week (43.8 a run); on a machine shared as on 11-11, about 260 runs and
  11,300 decisions. Sampling the screens where the teacher sees no spread (rewards, Act 2 shops: 17% of its CPU) at a low rate buys
  the rest back. pi_h was fitted on
  about 160,000 human decisions; the student's first round (10,000 to 20,000 teacher decisions) is a fine-tune behind the human anchor.
- **The first corpus**: the 2026-11-05 to 11-08 shards hold 21,033 exploring runs, 6,889 of them generation 13's own (vrun+pi_h,
  a3-v1@h1e4b16). Their decisions are replayed exactly (the replay in section 9 matched the lab's recorded generation 13 runs floor for
  floor and decision for decision) and labelled by the teacher at the replayed screens (the run is paid once): the student's round 0
  needs no new self-play runs (the labels cost the same per screen as new play: the continuations are the cost). The same runs refit V (not refit since generation 12) before the first student round.
- **Recorded per decision** (one line per screen, a shard per job): the decision id (seed, ordinal, kind, act, floor), the option rows
  (npz, as `option_rows` builds them), V of each option, pi_h's log-probabilities, the shortlist, every draw's G, end, door HP and the
  judge's P(death), Q, the raw and the guarded pick, the player's pick, the gain and its SE, the K, the engine tag, the judge's and the
  agent's sha256, and the run's outcome (final floor, bosses reached and beaten, door HP). The run itself is recorded as the lab records a
  self-play run (`gen_loop.RunRecorder`), so V refits from the same runs.
- **Where:** a new store `referee/run-teacher/<generation>/` (jobs, shards, manifest) beside `LAB/runs`, not inside it: LAB/runs holds the
  full-run records V's window reads; the teacher job writes its run records there too, tagged, and its decision shards to the new store.
- **The human exam fifth** is never read: the teacher reads no human record at all; the student's human term and its human bar read the
  training fifths and fifth 4 through `human_prior`'s readers, which drop the exam fifth (id list plus hash bucket 0) in the reading
  process and assert on every shard (`assert_no_exam`); the student lane's tests pin that.

### 4.4 The gate

As the gym judges a network, then as the lab judges a generation, the human bar at every stage:
1. **Held-out imitation of the teacher** (runs held out by seed hash, a fifth): the student's cross-entropy against `p_T` and its top-1
   agreement with the guarded pick must beat pi_h's and the incumbent student's on the same decisions (paired bootstrap over runs, lower
   bound above 0).
2. **The human bar** per kind (4.2), reported always, gating the hand-over per kind.
3. **Play against the incumbent:** the candidate generation (generation N's recipe with the student on the passing kinds at the measured
   lambda) judged by the lab under the sequential judgement, the 500-seed futility stop and the screens of addenda 18 and 19. **The
   lab's `screen:prior` must not judge a student:** it asks a prior-only candidate to predict human won-run choices better by 0.002, which
   a student built to move away from human choices would fail by design; a student candidate takes a `screen:student` (leg 1's gain, leg
   2's guard) instead. **Message for the orchestrator** (the lab screens; the student lane builds it).
4. **The run-level human bar** reported beside every verdict: floor reached and HP at the Act 1 and Act 2 boss doors on the copy against
   the human records' averages (C5's measure; computed once from the corpus, aggregates only).

### 4.5 The loop

- **Who runs it:** the lab runner, as job kinds beside `selfplay` and `train`: `teacher` (self-play with the teacher on the current
  generation, the filler's slot and rules: BelowNormal, judgement first, no filler within an hour of a presort), `student` (refit when at
  least 10,000 new teacher decisions exist since the last fit or weekly; the held-out legs; builds the candidate and writes its marker
  `ready-<gen>-g<N>-student`), then the existing `train` judgement.
- **Without a builder:** a promotion writes `current.json`; the next `teacher` job reads it at start, so the new generation's self-play
  and labels follow by themselves (addendum 20's "close the loop"). V refits from the window at each `student` step (D33's keep test).
- **What stops it:** a student that fails the human bar twice in a row on a kind freezes that kind; failing on every kind twice stops the
  loop (`LAB/alerts/run-student.txt`); a fidelity drop in the next real block stops it; the plateau rule of D57 addendum 2 at the run level:
  no promotion in ten days writes `LAB/alerts/run-plateau.txt` with the last verdicts.

### 4.6 Risks and the plateau

- **Teacher noise.** Measured SNR under 1 a decision; the guard keeps play safe and the student averages. Signal that it fails: leg 1's
  held-out gain over pi_h not above 0 after two rounds (C4's named response: the lookahead's depth or the run features). The remedies in
  order: adaptive K (K 2 first, more draws only where the top two are within 2 SE: the convergence lane's screen, which bought 60% of the
  boss turns at a third of the uncapped cost), then a longer horizon for Act 1 decisions (the head's value at the post-boss state in
  place of the constant B).
- **The leaf's bias.** The teacher is as good as the judge at the door and the player in the continuations. The judge reads x16's boss
  positions 11 points low (v0; JUDGE.md section 7) and is recalibrated per act; C2's judge lifts the teacher with it (judge first).
- **The horizon.** The door sees the next boss only; the deck's value after it is the constant B. The labels-vs-blocks probe found the
  Act 1 prior choices' value nil past the Act 1 boss (D55 addendum 12), so the loss is modest for Act 1; for Act 2 screens the door is the
  frontier itself.
- **The plateau, as the gym had it.** Per round the run level may gain a point or two of deep kills; a 3,000-seed block resolves about
  +-1.2 (SE 0.6), the same trap as the gym's +1-point candidates. Use the gym's new trend rule at the run level (consecutive student
  candidates pooled, with the correlation of section 3), and read C4 as leg 1's gain round over round.
- **What the convergence lane says about depth** (D55 addendum 15 outcomes): x16 is not converged at any kind (30% of boss turns, 34% of
  elite and 30% of hallway turns overturned by the settled move; 10%, 7% and 1% by 5 HP or more), and the same budget on another seed picks
  another move in a third of the turns. For the run teacher that means the continuations' fights are themselves noisy; the door leaf
  removes the boss fight, the noisiest and dearest, from every continuation. For quantity against depth: the student learns across
  decisions, so its targets want many decisions at small K (K 2 to 4) more than few decisions at large K; the guard wants precision where
  the top two are close, which adaptive K buys.

## 5. Part 4: hidden information in the search

**What holds (code and tests).** A fight fork resamples every roll still to come (each stream rebuilt from a new seed at its call count),
reshuffles only the part of the draw pile the player has not been shown (`ReshuffleUnknownDrawOrder` keeps `KnownTopCount` and
`KnownBottomCount`), and rolls the enemies' current moves again behind the intents they show, keeping a roll only when every enemy shows
the same intent (`IntentParticles`: rejection sampling from the AI's own prior, up to 4,096 tries). A run clone redraws the normal and elite
encounters not yet entered, the unseen act contents and rewards; the boss and the map are kept. Every simulation of the search forks its
own world (`live_search`: one resample seed per simulation). Tests: `ResampleLeakTests`, `KnownDrawOrderTests`, `IntentAmbiguityScanTests`,
`SolverLeakTests`, the leak probes L2, L3, L8 (`referee/leakprobes.py`, `tests/python/test_lockstep.py`). The run teacher's clones are
resampled the same way, so it has no foresight (D6: no clairvoyant teacher).

**What does not: fights started from a document.** A fight loaded from a position document (`native.load_combat_snapshot`) has no
intent roll to start from, so its forks keep the true moves behind the shown intents (`IntentParticles` returns false and flags the fork
`ai_state_assumed`, "for good": every later fork of it carries the flag). Measured from the saved games: in the fight exam's human stratum
(v3, 1,600 games) 81.9% of searched decisions at the Act 1 boss and 87.1% at the Act 2 boss stand on flagged forks, 80.3% in the AI
stratum (v2, 4,372 games); in generation 13's whole runs on the copy 6.9% (28,708 of 413,368 combat decisions; summons and forced moves).
Because the flag is sticky, those shares are an upper bound on the decisions that actually read a kept move; the certain part is the
first intent pass of every document start. Every practice start of the fight gym, the fight exam and its human comparison (C3), the
convergence lane's labels and the judge corpus's positions start from documents.

**The cost** (`probes/doc_vs_run.py`): the same boss doors of generation 13's runs, the fight played J times from a resampled clone of
the run and J times from the exported document, same player at its schedule. Result (`referee/backlog/run-self-play/docvsrun/`, 30 doors of 30
seeds, 24 at the Act 1 boss and 6 at the Act 2 boss, four draws each way, x16, 7,673 CPU-s, on the shared machine): from the document the
player survives **63.3 in 100 against 70.0 from inside the run (-6.7, 95% interval -16.7 to +3.3)** and ends the boss with **7.7 HP less
(-10.5 to -4.9)**; 81.7% of the document side's searched decisions carry the flag, as in the exam. So the kept enemy moves do not flatter
the player: whatever the search gains from them is outweighed by something else that makes a document-started fight play worse than the
same fight in the run. Candidates, untested: the search's resolved budget or stakes multiplier on the document path (`play_steps` with
`position_of` on a three-key info against the run's full info), what the prior's tokens read on a loaded fight (encounter, relics,
potions), and state the document does not carry (an export or load fidelity gap).

**The fix** (an engine item, full gate): the position document carries the last intent roll (the roster before and after the pass, as
`IntentRoll` holds it), or the loader rolls the first intents again from the encounter's pre-pass state, so a loaded fight resamples like
an in-run one; the identity proof is that in-run fights do not change and document fights change only where a flag was set.
**Recommendation: no engine lane for the intent roll now.** The probe finds no foresight gain to remove and a cost in the other direction;
the first step is a short Python-side diagnosis of why document starts play worse (`referee/backlog/run-self-play/DOC-START-BRIEF.md`),
and the intent-roll fix stays a later item, built only if the diagnosis ends in the engine. Every document-started measure (the gym's
practice targets, the fight exam and the human comparison of C3, the convergence lane's and the corpus's labels) reads our player
about 8 HP low at the boss until it does. **Message for the orchestrator** (judge-corpus lane): record `ai_assumed` per position, so the corpus's share of
positions labelled on kept moves is known and can be relabelled after the fix.

## 6. Part 5: the data rate

| what | rate now | source |
|---|---|---|
| fight gym | 17,993 fights on 2026-11-10 (9,845 on 11-09, under 1,000 a day 11-06 to 11-08) | `gym-loop/games/segments.jsonl` |
| a player run on the copy | 25.3 CPU-s (generation 13, a3-v1@h1e4b16; 42.2 on the shared machine; 124.7 s of wall on the old engine under load) | section 9; the E14-2026-11-08 arm |
| a lab judgement | 3,000 paired seeds, about 42 core-hours (two arms at 25.3 CPU-s a run); a loser stops at 500 seeds, about 7 | futility, D55 addenda 18 and 19 |
| a teacher run | 1,383 CPU-s (2,342 shared); 437 a week on 168 core-hours (260 shared) | section 9 |
| the judge corpus | 50 to 100 core-hours once (the corpus lane's estimate); 20,000 boss turns at the x64 cap about 50 core-hours | D55 addendum 15 outcome |

The machine's 16 cores make about 2,690 core-hours a week. Under judgement-first (addendum 19) a judgement takes the machine when it
runs; the practice loop takes six cores; the teacher job takes the filler's 168. The convergence report's depth-against-quantity reading is
in 4.6: for the fight judge, capped settled values on many positions (the corpus lane's recipe) beat uncapped values on few; for the run
student, many decisions at small K beat few at large K, with adaptive K for the guard.

## 7. The corpus's composition (the judge's training set)

Balanced by stratification, not equal counts (D57 addendum 7). The corpus lane chooses the exact figures; this section fixes the rules and
refines its brief.

- **Primary strata:** source (human, self-play) x act (1, 2) x kind (hallway, elite, boss), each boss its own cell: 2 x (2 + 2 + 9) = 26
  cells. **Secondary** within a cell: turn stage (early: the first two turns; middle; late: the last third of the turns the teacher played)
  x health band at the fight's start (healthy at 70% of max HP or more, hurt 40% to 70%, nearly dead under 40%) x potions held (0, 1, 2 or
  more) = 27 sub-cells.
- **Minimums:** every boss cell at least 1,500 positions per source where the pool allows, every elite cell 3,000, every hallway cell
  3,000; every sub-cell at least 50 (or all there are, listed short); every card and relic seen in at least 20 human starts in at least 100
  positions. Items seen in fewer than 20 starts are not oversampled (too few distinct runs to learn from without memorising them).
- **Oversampling:** a sub-cell or item under its minimum is filled by drawing more starts that hold it, up to four times its natural
  share; the self-play side fills its thin cells (Act 2 bosses and elites) from pre-loss starts and, if still short, new self-play (the
  corpus lane tells the orchestrator what to set).
- **Reweighting:** each position carries `w = pi / q`, q its stratum's share in the corpus, pi the loss's target share: within a primary
  cell pi is the cell's natural mix (so the oversampled bands and items count at their natural rate in the loss), across cells pi is half
  equal per cell and half the mix the search meets in the player's runs, so bosses keep more say than their share of fights ("boss fights
  much more complex") and hallways are not ignored. Weights clipped to [0.25, 4] and renormalised to mean 1.
- **Held out** by run seed (a fifth); the exam fifth, the exam's human stratum and every run on `exam/exclude_runs.json` never in.
- **Coverage table before any training.** The pools as they stand (`probes/coverage.py`, `probes/coverage.out`; starts, about three
  positions a start; human: the four training fifths with the exam stratum and its runs out, 44,234 starts; self-play: the loop's night
  set, 17,632 starts, and 4,506 pre-loss starts):

| cell | human starts | self-play starts (nights) | pre-loss |
|---|---|---|---|
| Act 1 hallway | 2,374 | 10,596 | 155 |
| Act 1 elite | 19,396 | 1,140 | 233 |
| Act 1 boss (six bosses, 1,211 to 1,434 each human; 257 to 337 each self-play) | 7,838 | 1,880 | 2,662 |
| Act 2 hallway | 2,332 | 3,145 | 442 |
| Act 2 elite | 8,593 | 513 | 336 |
| Act 2 boss (three bosses, 1,223 to 1,248 each human; 112 to 124 each self-play) | 3,701 | 358 | 678 |

| health band at the start (share of the cell) | human Act 1 boss | human Act 2 boss | self-play Act 1 boss | self-play Act 2 boss |
|---|---|---|---|---|
| healthy | 3,289 | 1,997 | 1,236 | 155 |
| hurt | 4,125 | 1,588 | 523 | 152 |
| nearly dead | 424 | 116 | 121 | 51 |

| potions held | human Act 1 boss | human Act 2 boss | self-play Act 1 boss | self-play Act 2 boss |
|---|---|---|---|---|
| 0 | 944 | 372 | 778 | 168 |
| 1 | 2,783 | 1,109 | 806 | 157 |
| 2 or more | 4,111 | 2,220 | 296 | 33 |

  Cards in the human starts: 412 distinct, 227 of them in fewer than 20 starts, 70 in 20 to 99, 71 in 100 to 999, 44 in 1,000 or more.
  Relics: 220 distinct, 2 in fewer than 20 starts, 10 in 20 to 99, 125 in 100 to 999, 83 in 1,000 or more. Thin cells to oversample: nearly
  dead at the Act 2 boss (116 human, 51 self-play), self-play Act 2 bosses holding two potions (33), self-play Act 2 bosses at all (358
  starts: fill from pre-loss and new self-play). The deck-type tag (an archetype from the deck's dominant cards) is the corpus lane's; its
  cells take the same minimum of 50.

## 8. What the measurement says about the design

1. **The teacher is affordable.** 1,383 CPU-s a run, 55 times the player's 25.3 (section 9): the filler's 168 core-hours make about 440
   teacher runs and 19,000 labelled screens a week, enough for a first student in about a week (lane 2) and a second round before C4.
   Its cost is the continuations (about 34 CPU-s a screen: 3.1 options x 4 draws to the door), not the run.
2. **Its values are noisy one decision at a time and carry signal.** The noise sd of a centred option value is 1.94 floors a draw against
   a true spread between options of 0.87 (SNR 0.90 at K 4, a little above the 11-05 labels' 0.86 with the door leaf: the player's own
   schedule plays steadier continuations than a3-v1); two halves of the draws name the same raw best option 56 times in 100 where chance
   is 36. A student that sees thousands of decisions can learn that; a teacher that acted on one decision's values could not.
3. **Where it sees and where it does not.** Routes carry the largest real differences (spread 1.71 floors in Act 1 and 1.65 in Act 2,
   SNR 1.3 and 2.0), then rest sites (0.98 and 0.57; SNR 1.4), ancients (1.1 and 1.0) and Act 1 events (0.76). Card rewards are nearly
   flat to it (0.56 in Act 1, 0.16 in Act 2; SNR 0.5 and 0.2), as are rewards (0.3 to 0.4) and Act 2 shops (0). One card's effect on
   the next boss fight, read by a judge trained on few boss decks, is small against a draw's noise.
4. **What the guard plays.** 3.1 screens in 100 (1.3 a run; 36 of the 50 runs had at least one), most at Act 1 rest sites (8.4 in 100),
   Act 1 shops (8.4) and ancients (11 to 12), almost none at card rewards (2.3 and 0.5). The runs it played went further: +2.4 floors
   (SE 1.4), the Act 1 boss +10 per 100 (SE 5.9), the Act 2 boss +4 (SE 6.4), HP at the Act 1 door 69.9 against 65.0. A sign the
   direction is right; 50 seeds cannot confirm it, and the lab's judgement is the arbiter.
5. **Lambda, measured.** The teacher credits the human prior 0.09 floors a nat beyond V overall (SE 0.025), 0.27 at Act 1 rest sites
   (SE 0.05) and 0.13 at Act 2 rest sites (SE 0.03), and about nothing at card rewards (-0.03 +- 0.06 in Act 1, 0.04 +- 0.03 in Act 2),
   against the recipe's 0.5. The blocks found the human prior's Act 2 card choices worth +3.4 deep kills per 100 runs (D55 addendum
   12): the teacher cannot see that gain, so at card screens its silence is its limit, not the prior's lack of worth. The human bar
   per kind (4.2) is what keeps a student from unlearning it.
6. **What changes in the plan.** (a) The student's first kinds are routes (V alone today; pi_h v1's route fit, 76 in 100 top-1, gives
   the human bar), rest sites and events; card rewards keep pi_h at its lambda until the judge (C2) gives the door leaf a deck reading
   the teacher can use, and the first measurement after C2 repeats this one's card rows. (b) Lane 2 samples the screens where the
   teacher sees no spread (rewards, Act 2 shops: 17% of its CPU) at a low rate. (c) Adaptive K where the SNR is under one (cards,
   shops, selections), K 4 where it is over (routes, rests). (d) Lambda for the shipped blend is a lab question this measurement
   cannot settle for cards (it sees only to the next door); for rest sites it reads 0.13 to 0.27, under the recipe's 0.5.

## 9. The first measurement (50 runs, the teacher against the player's own depth)

`rl/run_teacher.py play --runs 50 --procs 4 --k 4 --max-options 4` (BelowNormal; the QUIET line `run-teacher` until 00:42Z, then `--share`), output
`referee/backlog/run-self-play/measure/` (`runs.jsonl`, `decisions.d/`, `report.json`, `manifest.json`). Generation 13 (`gen13/candidate-stakes`,
`vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7+stakes:a3-v1@h1e4b16`), CPU prior in process, judge v0 at the door, engine of fork main `68589879`.
Seeds: the first 50 of generation 13's E14-2026-11-08 arm by index, with the arm's own run indices, so the player arm is the lab's own
record: it reproduced the recorded runs exactly on all 50 seeds (floor and run decisions). Played 2026-11-10 21:24Z to 2026-11-11 06:49Z.

| arm | runs | CPU-s a run, mean / median / p90 | screens a run (teacher's) | mean floor | Act 1 boss beaten | Act 2 boss beaten | HP at the Act 1 / Act 2 door |
|---|---|---|---|---|---|---|---|
| player's own depth | 50 | 32.7 / 24.33 / 71.42 | 44.3 (0.0) | 26.04 | 32 | 8 | 65.0 / 61.4 |
| teacher (K 4, M 4, guarded) | 50 | 1,882.1 / 1,852.73 / 2,992.56 | 47.9 (43.8) | 28.44 | 37 | 10 | 69.9 / 58.8 |

Paired by seed (teacher minus player, 50 seeds): floor +2.40 (SE 1.361); Act 1 boss +10.0 per 100 (SE 5.9); Act 2 boss +4.0 (SE 6.4).

| screens | n | CPU-s a screen | options played | raw pick differs | guarded (played) differs | split-half agreement, raw (chance) | noise sd / spread sd of a centred value | SNR at K 4 |
|---|---|---|---|---|---|---|---|---|
| all | 2190 | 42.0 | 3.07 | 40.5 | 3.1 | 56.1 (36.1) | 1.942 / 0.872 | 0.898 |
| act1 | 1373 | 37.9 | 3.07 | 41.8 | 3.9 | 53.5 (36.2) | 2.154 / 0.952 | 0.884 |
| act2 | 817 | 49.0 | 3.07 | 38.4 | 1.6 | 60.5 (36.1) | 1.521 / 0.717 | 0.943 |
| ancient/a1 | 50 | 60.8 | 3.0 | 50.0 | 12.0 | 40.0 (33.3) | 3.363 / 1.139 | 0.677 |
| ancient/a2 | 37 | 72.8 | 3.0 | 40.5 | 10.8 | 56.8 (33.3) | 2.202 / 0.964 | 0.876 |
| card/a1 | 354 | 50.1 | 4.0 | 47.2 | 2.3 | 45.8 (25.0) | 2.193 / 0.564 | 0.514 |
| card/a2 | 194 | 57.4 | 4.0 | 41.2 | 0.5 | 51.5 (25.0) | 1.603 / 0.162 | 0.202 |
| event/a1 | 140 | 26.9 | 2.09 | 41.4 | 3.6 | 62.9 (48.6) | 1.759 / 0.758 | 0.862 |
| event/a2 | 94 | 34.5 | 2.15 | 45.7 | 4.3 | 50.0 (47.5) | 1.233 / 0.473 | 0.766 |
| rest/a1 | 154 | 26.9 | 3.99 | 36.4 | 8.4 | 66.9 (25.1) | 1.374 / 0.975 | 1.419 |
| rest/a2 | 82 | 38.1 | 4.0 | 22.0 | 0.0 | 80.5 (25.0) | 0.837 / 0.566 | 1.353 |
| reward/a1 | 187 | 23.7 | 2.1 | 33.7 | 2.1 | 61.5 (48.4) | 1.717 / 0.372 | 0.433 |
| reward/a2 | 121 | 28.4 | 2.1 | 38.8 | 0.8 | 70.2 (48.3) | 1.058 / 0.297 | 0.561 |
| route/a1 | 308 | 35.1 | 2.32 | 33.4 | 1.6 | 59.4 (45.3) | 2.711 / 1.709 | 1.261 |
| route/a2 | 165 | 42.6 | 2.27 | 19.4 | 1.2 | 75.8 (45.8) | 1.666 / 1.648 | 1.979 |
| select/a1 | 56 | 40.6 | 3.62 | 58.9 | 5.4 | 41.1 (29.2) | 1.963 / 0.556 | 0.567 |
| select/a2 | 33 | 58.3 | 3.79 | 54.5 | 0.0 | 27.3 (27.3) | 1.742 / 0.376 | 0.431 |
| shop/a1 | 119 | 46.8 | 3.57 | 55.5 | 8.4 | 31.9 (30.3) | 2.01 / 0.778 | 0.774 |
| shop/a2 | 91 | 81.9 | 3.7 | 67.0 | 1.1 | 45.1 (28.7) | 1.744 / 0.0 | 0.0 |

| lambda measured on the teacher's values (floors a nat of pi_h beyond V) | decisions | lambda | SE |
|---|---|---|---|
| act1 | 558 | 0.0896 | 0.0413 |
| act2 | 313 | 0.0908 | 0.0229 |
| all | 871 | 0.0901 | 0.0247 |
| ancient/a1 | 50 | -0.5987 | 0.3 |
| ancient/a2 | 37 | 0.3956 | 0.2572 |
| card/a1 | 354 | -0.0299 | 0.0564 |
| card/a2 | 194 | 0.0384 | 0.0319 |
| rest/a1 | 154 | 0.2739 | 0.0527 |
| rest/a2 | 82 | 0.125 | 0.0309 |

The machine was shared: until 00:42Z on 11-11 the measurement held the `run-teacher` QUIET line (about ten of our processes ran);
from 00:42Z, at the coordinator's word, it held none and ran beside the corpus builder's eight processes and the practice loop's six
(`play --share`). The table pools both phases; the cost figures quoted in the text are the first phase's (24 teacher and 28 player
runs), the shared phase's are given beside them. The player arm equals the lab's recorded generation 13 runs on all 50 seeds (floor
and run decisions), and the 14 teacher runs in which the guard never fired equal the player's runs floor for floor (the teacher's
streams never touch the run's). Outcomes over 50 paired seeds are a sign, not a verdict (SE 1.4 floors, 6 points a boss).

## 10. The build plan: four Sonnet lanes, judge first

The full plan, with each lane's files, tests, machine time and dependencies and the first student's judgement cost, is
`referee/backlog/run-self-play/BUILD-PLAN.md` (delivered first, 2026-11-11, and accepted by the coordinator with two changes: lane 1 is
the judge-training lane already running, `feat/judge-train`; lane 3 started against the fixed shard schema). `rl/run_teacher.py` landed on
fork main as `c1c8d245` so lane 2 extends it. The corpus lane and the gym-gate lane run beside them and are not counted. Each lane lands with
tests through the select-mode gate (the full gate where it touches the engine); usage is in Sonnet worker-days (*est.*) and core-hours.
The first student's judgement on the new engine: the smoke look 200 paired seeds about 3 core-hours, the futility look at 500 about 7, the
whole 3,000 about 42 (25.3 CPU-s a player run, both arms; about 1.45 times that on a shared machine; half if the incumbent's stored record
is reused), plus card-minutes for the fit and the V refit.

| # | lane | lands | tests | judged by | checkpoint | cost |
|---|---|---|---|---|---|---|
| 1 | **Judge v2** (from `feat/judge-v1` `9c119fa0` and the corpus loader `rl/judge_corpus.py`) | the settled-value leaf head, the source input, the stratified weights, the held-out-by-run split; the rank screen against x16's root value; `judge.py eval` by stratum; the board run only when the screen passes | weights and clipping, no exam run and no shared run across the split, the source input read only by the auxiliary heads, the screen's arithmetic on fixtures, the leaf value as the search reads it | the screen, then addendum 21's board under rule 1.3 | C2 (11-18) | 1 to 1.5 worker-days; GPU about an hour; screen 1, board about 18 core-hours |
| 2 | **The teacher job and the run corpus** (from `rl/run_teacher.py`) | lab job kind `teacher` (`tools/lab/jobs.py`) with the filler's rules and a daily budget; the decision shards with option rows and the run records; the replay-labelling of the 11-05 to 11-08 generation 13 runs as round 0; P(death) recalibration per act; adaptive K | option rows equal to the chooser's at play; a replayed run equal to its record; a teacher that agrees everywhere is the player's run; no human record read; held-out split by seed; the job's QUIET and judgement-first rules | data checks and the report of section 9 on the job's first day (cost a run, changed share, split-half agreement within the measured intervals) | C4 | 1 worker-day; 24 core-hours a day of the machine |
| 3 | **The student and its gate** | `rl/run_student.py`: the fit (teacher CE, human CE, KL to pi_h), the held-out legs (teacher imitation, human bar per kind), lambda measured per kind and act, the candidate builder (recipe with the student on passing kinds), `screen:student` in the lab's screens; V refit from the window | soft targets and their temperature, kinds without a human fit, the human term never reads the exam fifth (`assert_no_exam`), lambda's regression on synthetic decisions, the recipe string and the loader's refusals, the screen | legs 1 and 2, then the lab's judgement (keep test v2, futility, sequential) | C4 (11-27), C5 | 1.5 worker-days; GPU minutes; a judgement 7 (futility) to 42 core-hours |
| 4 | **Close the loop** | the runner chains `teacher` -> `student` -> `train`; the next generation's teacher reads `current.json`; the stops (two human-bar failures, a real-block fidelity drop, the ten-day plateau alert); the daily line of runs, decisions and CPU; the run-level human bar (floor and door HP against the records) beside every verdict | the chain on a scratch lab folder, each stop's trigger and reset, the alert files | the first automatic round end to end | C4, C5 | 0.5 to 1 worker-day |

**The engine item of section 5** (the document's intent roll) is not counted among the four: the probe found no foresight gain (section 5), so it is not built now; a Python-side diagnosis of the document starts' 8-HP
cost comes first (`DOC-START-BRIEF.md`), and an engine item follows only if that diagnosis ends in the engine.

Order: lanes 1, 2 and 3 run in parallel from day one on disjoint files (lane 1 `rl/judge*.py`; lane 2 `rl/run_teacher.py` and the
`teacher` kind in `tools/lab/jobs.py`; lane 3 `rl/run_student.py`, the chooser, the recipe, `rl/screens.py`, `rl/gen_loop.py`); lane 3's first
real fit waits on lane 2's first 10,000 to 20,000 screens (four to seven days of the teacher's budget); lane 4 after lanes 2 and 3 land.

## 11. Messages for the orchestrator

1. **Gym-gate lane:** pool the five candidates per start before the paired test (section 3); their differences correlate about 0.5 through
   the incumbent's shared games, and an independent-SE test understates the pooled SE 1.7 times. Optional: the human-comparison ratchet.
2. **Judge-corpus lane:** record `ai_assumed` per position (section 5); every position it labels starts from a document, and document
   starts play about 8 HP worse at the boss than the same fight inside a run (the probe); the diagnosis brief says why that matters.
3. **The lab's screens:** `screen:prior` must not judge a student candidate (4.4); lane 3 adds `screen:student`.
4. **The run loop's first kinds** (section 8, item 6): lane 3 hands over routes, rest sites and events first; card rewards wait for C2.
   And the measurement shared the machine from 00:42Z on 11-11 without a QUIET line, as told; CPU-s per run rose about 1.45 times under
   that load, so the costs quoted are the first phase's.

## 12. Not verified

- The door leaf's constant B ignores the deck's value past the boss; its effect on Act 1 card values is argued (D55 addendum 12's probe),
  not measured here.
- Judge v0 at the door, uncalibrated; v1 and the new judge v2 were not tried as the leaf.
- 50 paired seeds: the outcome differences are not significant; the changed share and the costs are firm (2,190 screens).
- The CPU-s figures come from two machine phases (held, then shared); per-floor and per-screen costs differ by about 1.45 between them.
- The student's design (soft targets, the human term, lambda's regression) is specified and its arithmetic tested (`lambda_fit`), not fitted.
- The gym-gate correlation argument (section 3) is arithmetic on an exchangeable model, not a replay of the ledger.
- The relabel's figures are for the 11-05 labels (generation 11 at a3-v1), not for generation 13 at its own schedule.
