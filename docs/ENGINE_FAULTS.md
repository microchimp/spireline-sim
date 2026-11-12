# Engine faults

What happens when the engine's own code throws in the middle of a step, the four classes that did so in 2026-11 (the
teacher's round 0, the judge corpus, the first judgement of generation 14), their mechanisms and their tests. D58
addenda 16 and 20 (Spireline `docs/DECISIONS.md`) are the inventory; the lane's brief is
`referee/backlog/engine-crashes/BRIEF.md`.

## The refusal

Before: every step export ran the engine with nothing between it and the native boundary, so an unhandled exception
ended the worker process (NativeAOT fails fast). The pool replaced the worker; the run or fight was written as lost, or
not at all, and the class was visible only in the worker's log.

Now (native API 40, run API 33):

- `Sts2_Step`, `Sts2_StepTargeted` (`NativeExports.StepOrRefuse`) and `Sts2Run_Step` (`RunNativeExports`) catch any
  exception out of the step other than the two they already answered (the kernel guard's unbound fight and a refused
  answer, both `-1`) and return **`-2`** (`EngineFault.Code`). `Sts2_GetLastError` then holds the exception's class,
  the top frame inside the engine (`CardEffects.DrawCards`, the runtime's and the list's frames above it skipped) and
  its message, tab-separated. `Sts2_EngineFaults()` counts the faults the process has met.
- **The state is not usable after a fault.** The exception left the combat or run wherever it was, part way through a
  step (a Pillage half drawn, a queue with an action executing). The handle keeps the fault and refuses every later step
  with the same `-2` and text; a clone of a faulted combat handle carries it; a reset clears it. The drivers end the
  fight or run at the first fault anyway.
- `sts2_gym.native.step` and `native.run_step` raise `sts2_gym.faults.EngineFault` (a `RuntimeError`, not a
  `StepRefused`) with `cls`, `frame`, `detail` and `reason` = `engine_fault:<class>`. `sts2_gym/faults.py` is pure, so
  drivers and tests name a fault without the library.
- What the copy cannot follow it says so with its own classes (`Core/EngineFaults.cs`): `GameHangs`, a state where the
  game's own code never finishes the step, and `EngineLimit`, a chain deeper than the copy's stack allows. Both reach
  the drivers the same way.
- A stack overflow is not an exception .NET can catch: a process that overflows still dies. `EngineLimit` is there so
  the one recursion found to be unbounded (Unceasing Top's chain, below) stops before it.

## Where faults are counted

Every fault is a row with its reason, counted by class in the job's result; none is retried.

| Driver | Row | Count |
|---|---|---|
| Judgement arms (`rl/gen0.py` `_run_safe`) | `end: "engine_fault"`, `crash: true`, `reason`, `engine_fault`, `fault_frame` | `engine_block(...)["engine_faults"]` in each arm's summary |
| Seed exam (`rl/run_seed_exam.py` `play_task`) | the same fields on the run's row | `run_block(...)["engine_faults"]` |
| Teacher (`rl/run_teacher.py`, through the gym pool) | the capped row, `cap_reason: "engine_fault:<class> at <frame>"` | the job's `engine_faults`; `fault_seeds` keeps a faulted run from being asked for again |
| Gym pool (`rl/fight_gym.py` `run_tasks`, `guarded_run`) | a task that raised a fault gets the capped row above | `guarded_run`'s done line; the arm tables' `cap_reasons` |
| Judge corpus workers (`rl/judge_corpus_build.py work`) | the fight's result with `error`, `reason`, `engine_fault`, `fault_frame` (a fight with a result is not built again) | `status`'s `engine_faults` |

`faults.engine_fault_counts(rows)` reads any of these rows.

## The four classes

### 1. `ArgumentOutOfRangeException` in `CardEffects.DrawCards` (index)

Round 0 of the teacher, 2 runs: through `CombatEngine.EndTurn`'s hand draw and through
`RelicEffects.ApplyAfterHandEmptied`. A Strike Hellraiser plays out of a draw is filed on its result pile (the top of
the draw pile under Nostalgia, Rebound or Shining Strike), and the draw then looked for it by position, the top of the
draw pile or the end of another pile, to finish its draw hooks. The play's own empty-hand check could move it first: the
copy's Unceasing Top drew after every play, nested or not, and in the turn's start, and its draw took the Strike back off
the draw pile, so the draw read the top of an empty pile.

The game checks the empty hand after a card's play or a potion's use only when no other card's or potion's effect is
running, its Unceasing Top draws only in the AutoPrePlay, Play and AutoPostPlay phases, and its later draw hooks act on
the card itself wherever it is. The copy now keeps the effect depth and marks the turn's end, the enemies' turn and the
next turn's setup (`CombatEngine.CheckForEmptyHand`), checks after a potion's use too, and finds a card by its uid
(`CombatEngine.WhereNow`: the draw's Hellraiser path and Decisions Decisions' repeat, which read the top of the draw pile
the same way). Test: `EngineFaults/DrawnCardWhereaboutsTests` (both engines).

### 2. `OutOfMemoryException` at `StepInProgress.OnTake` inside `DrawUntilNonAttack`

Judge corpus 8 traces (4 fights, each twice), the first judgement of generation 14 2 traces (one run written as lost).
Not the machine's memory: the step's record of its draw-pile takes grew without bound. Pillage draws until it draws a
non-Attack, the hand fills, both piles are empty or the fight ends. A Strike Hellraiser plays never reaches the hand.
When a card-play limit refuses every auto-play (Sloth, Ringing, Normality: every one of the corpus's fights had one),
each Strike is filed in the discard unplayed and drawn again after the next shuffle, and with only Strikes left in the
piles nothing ever changes. The game's loop has the same exit test and its refused auto-play files the card the same way,
so **the game itself hangs there**; the copy drew until memory ran out.

The copy now watches the loop (`CardEffects.EndlessDraw`) and throws `GameHangs` once a whole pass over both piles, and at
least 64 draws, changed nothing that could end it (the hand's, piles' and exhaust pile's sizes, a play started, any
creature's HP and block), or once one Pillage has drawn 20,000 cards. Pillage is the game's only loop on a draw; the
other draw loops are bounded by their count and the hand. One recursion was found unbounded with it: Unceasing Top's
draw, a Strike Hellraiser plays, its empty-hand check, the next draw (the game's chain, ended by the Strikes' kills): the
copy stops it at 100 deep with `EngineLimit`. Test: `EngineFaults/EndlessDrawTests` (both engines; a loop that ends is
drawn to its end).

### 3. `InvalidOperationException: Collection was modified` in `CardEffects.ApplyDefectCard`

Judge corpus 22 traces, every one a Chill handed to an Ironclad. Chill channelled one Frost per living enemy by walking
the live enemy list; a channel into full slots evokes the oldest orb, a Lightning's evoke can kill, and under core2 the
kernel's damage command takes the corpse out of that list at once. The game counts its hittable enemies as the card is
played (a copy of the list); so does the copy now, and Tesla Coil counts its Lightning orbs before its passives. Every
other walk of a live combat list was read for a body that can change it; none can. Test:
`EngineFaults/EnumerationSnapshotTests` (Chill on both engines, and every card played into the same hazard under core2).

### 4. `InvalidOperationException: PlayCard#N / UsePotion#N Executing at the front of queue 0 is in an invalid state`

Judge corpus 27 traces (19 card plays, 8 potions) at `Core2.ActionQueueSet.GetReadyAction`. Under core2 an answer to a
screen raised inside a card play or a potion resumes the paused action under a new id and the executor starts it before
the body replays the step. When the replay was refused (`StoppedStepMismatch`, a loaded document's step in a world where
it does not reach its screen), the combat went back on its screen from a copy taken inside the body, with the action
already executing: the caller's next step met that front and threw. `KernelGate.Run` now copies the kernel state before
it acts on an answer and puts it back when the answer is refused. Test: `EngineFaults/RefusedAnswerQueueTests`
(Armaments' pick and Gambler's Brew's).

## Not done here

- The corpus's 26 failed fights are not rebuilt (their fight results exist; a rebuild deletes them first). Fights an
  engine fault ends from now on are written with their class and not built again.
- A search could treat a faulted child as a lost branch and go on; today a fault anywhere in a fight's search ends the
  fight. After these fixes the known remaining fault is the game's own hang (class 2), which the corpus met in 4 of
  11,063 fights.
