# A fight started from its document, against the same fight inside the run (2026-11-11)

Branch `feat/doc-start`. Written for the run-level design lane's question (`referee/backlog/run-self-play/DOC-START-BRIEF.md`): generation 13's runs, replayed
exactly to each boss door, played the boss fight 70.0 in 100 from a resampled clone of the run and 63.3 from the position exported at the door
(`doc_vs_run.py`, 30 doors, 4 draws each; the gap -6.7, 95% interval -16.7 to +3.3), and ended it with 7.7 HP less (-10.5 to -4.9). Every gym
practice start, the fight exam and its human comparison, the convergence lane's labels and the judge corpus start from documents, so the question
was whether a document start is a worse place to fight from.

## Answer

**A fight from its document is the same fight.** Nothing the player or the search reads differs, the document is an exact round trip, hidden
information is redrawn alike from either start, and the three ways of playing the same door (below) come out level. The probe's HP gap was a
measuring difference, and its survival gap is inside what four draws on 30 doors cannot tell from zero.

- **HP**: the in-run side read HP from the run block after the fight, which holds the rewards screen's post-fight heal (`RunCombat.combat_fork`:
  "a won playout ends at the fight's own HP, before the rewards screen's heals"). Burning Blood heals 6 after every won fight: on the rerun's 79
  in-run wins the run block was 6 above the fight's own end HP in all 79. The document side reads the fight's own end HP (the gym's currency
  everywhere: `fight_gym.play_steps`, the exam's `hp_lost`, the human sidecar's `human_end_hp`). Read on one basis, the HP gap is +0.8 (-1.9 to +3.7).
- **Survival**: the rerun on the same 30 doors, fresh draws: in-run 65.8, document 64.2, gap -1.7 (-10.0 to +6.7). At a3-v1's own search (the plain
  config, 24 Act 1 doors x 24 draws, three ways): run clone 61.1, run fork through the gym loop 61.3, document 60.4; document minus run clone -0.7
  (-4.5 to +3.1), minus run fork -0.9 (-3.5 to +1.7). The sign is negative throughout and the size about a point; four draws a door cannot resolve
  that, and no mechanism for it was found.

## What was compared, and what it said

All on the probe's doors (generation 13, E14-2026-11-08 seeds, exact replays; `probes/`, results kept beside the brief).

1. **The position** (`doc_start_diag.py`, 10 doors): the observation (276 integers), the encounter id on every handle, `combat_tokens.encode` with
   and without the end-turn probe (types, ids, owners, features), the legal moves, `search_configs.position_of` and the resolved
   `cfg.search_kwargs` (simulations, stakes multiplier, hold, kill check, time budget) are identical from the in-run fork and from the loaded
   document, at every fight kind and config (`tests/python/test_doc_start.py` pins it). The first decision's search runs the same budget over the
   same root moves, and `search_steps` driven by `_drive` (the gym's driver) is `search_live` (the run's) call for call, children and all.
2. **The search at decision 0, many seeds** (`doc_start_root.py`, 5 doors x 24 seeds): best-move counts, every move's visit share and mean value from
   the two roots agree (all |z| under 2.6 over 60 comparisons). At one seed the two need not agree (the resampled streams stand at other call
   counts), and they do not; over seeds they do. At a boss's first decision most simulations die, so the move is mostly the prior's.
3. **The document round trip** (`doc_start_roundtrip.py`, 10 doors): export, load, export again: no difference (the draw pile's unordered middle is
   a multiset and compared as one).
4. **A fixed move sequence** (`doc_start_diag.py`, `doc_start_endturn.py`): end the turn until the fight ends. Oracle forks (the true future) agree
   turn by turn, enemy moves and HP, on 8 of 10 doors. On the other two the boss has 221 HP and random moves, the oracle's stream is the run's
   and the document's is seeded, so only distributions can agree: over 300 resampled forks each (6 doors, 44 per-turn comparisons) the player's
   HP after every turn agrees; two comparisons (turns 7 and 8 of one door) differ at |z| 2.6, in the document's favour (it survives 0.14 turn
   longer). After one real turn the forks of a document and of a run both give a single outcome (the displayed intent fixes the move).
5. **Hidden information from either start** (`doc_start_fork.py`, 8 doors x 300 forks, the player ending its turn at once or playing a random
   turn): the hand drawn (each card's frequency), the enemies' next intents, HP, block and energy agree (z under 1 on the means, intent
   distributions identical).
6. **Three ways to play one door** (`doc_start_three.py`): A the run clone stepped on the run handle by the exam agent's fight player (the in-run way),
   B a resampled combat fork of the run's fight played by `fight_gym._play` (the gym's loop on a position that never left the run), C the document
   played by `fight_gym._play` (the gym's way). A against B isolates the run engine's stepping and the HP basis; B against C isolates the document.
   Plain a3-v1, 24 doors x 24 draws: 61.1 / 61.3 / 60.4 in 100; HP on the fight's own basis B 14.5, C 14.3 (C minus B -0.1, -1.2 to +0.9); A's run-block HP
   is 3.4 above C's, which is the heal (0.61 survival x 6 = 3.7). At the agent's own x16 boss schedule on the first four doors x 8 draws: 90.6 / 81.2 / 71.9,
   noise at that size, which is why the rerun above has 30 doors.

## What the document start does differ in

- A fight loaded from a document has no intent roll, so its forks keep the document's true enemy moves for the **first turn only** (the engine's turn-start
  intent pass records its roll on the combat whatever started it, so later turns redraw the move behind the intent as in the run). That favours the document side slightly; it is not a
  deficit.
- `ai_state_assumed` is never cleared once set (`CombatState.AiStateAssumed`, clones carry it), so on a document-loaded fight every search reports
  it: the probe's "82% of searched decisions" is the flag's stickiness, not 82% of decisions on kept moves. The search does not read the flag (it
  only reports it). The flag is honest on a resampled fork of a run's fight (0 to 5 decisions a game against 20 to 46 from a document). If the
  figure is ever wanted as a measure of kept information, the engine would have to clear the flag at the first intent pass of a loaded fight
  (an engine change; not done).

## Which saved measures need re-reading

- The probe's own `docvsrun/read.json` HP gap (-7.7): re-read on one basis (above). Its survival gap stands as a noisy zero.
- The fight exam, its human comparison (the player's 64.6 baseline), the convergence lane's labels and the judge corpus: **no re-reading**; their
  positions are the run's positions to the last token, and their HP is the fight's own end HP on both sides (`HUMAN_STARTS.md`: the human's end
  HP is `current_hp` less the relics' post-fight heal).
- Any comparison that sets a gym fight's HP beside an HP read from the run block after the fight (a run's own HP after a boss, the lab's HP-after
  figures) is 6 HP too kind to the run for an Ironclad with Burning Blood (12 with Black Blood, 12 with Meat on the Bone at half HP or less).

## Reproduce

```
python probes/doc_start_three.py --out DIR --doors SEED:IX,... --draws 4 --chunk 4 --ways AC         # the rerun (x16, the agent's own schedule)
python probes/doc_start_three.py --out DIR --doors ... --draws 24 --chunk 12 --config a3-v1 --act1-only # the plain-config three ways
python probes/doc_start_three_read.py DIR [--json OUT]                                                  # door-level bootstrap intervals
python probes/doc_start_diag.py --out DIR --doors SEED:IX,... --draws 1 --act1-only                     # position, tokens, kwargs, searches, end-turn traces
```
Doors are `E14-2026-11-08` seeds with their stream index (`run_teacher.seeds_from`); the engine is the worktree's `out/`
(`publish-wt.sh`); the emulator's Python, `-I`. Each door's replay costs 10 to 60 seconds; a played boss fight at x16 about 25 CPU-seconds.
