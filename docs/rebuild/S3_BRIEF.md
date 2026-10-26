# Fight solver S3 brief (starts at the weekly reset)

S3 builds the exact layer on top of S1 (the state key, `solver/s1-key`, landing now) and S2 (chance, fork main 480d5b29),
as `docs/rebuild/FIGHT_SOLVER.md` designs it. Estimate 3.4 units (2 Opus rounds) plus a review.

Pieces, one commit each behind the gates (C# and EmuDump suites, the old-engine byte proof at 0 differ, the Python suites,
corpus 0 regressed, Codex walk 0 worse):

1. Expectimax over the S1 key with S2's chance tree: decision nodes over the legal actions (card plays with targets, potion
   uses, end turn, screen choices), chance nodes from S2's hook (the draw model, enemy move odds with the shown intent
   fixed, other rolls). Expected value only; the hindsight export stays separate (FIGHT_SOLVER.md section 4a).
2. The transposition table keyed by S1's key (expected mode), with the lossless prunings of FIGHT_SOLVER.md section 5:
   transposition merging, multiset hands, interchangeable enemies, subtree reuse across decisions.
3. The reference solver: a plain, unpruned expectimax for small fights, and the equal-value harness that runs every
   pruning against it (equal values to the last bit on a test set drawn from corpus fights and self-play positions).
4. The node budget and the in-search size estimator from the week-3 sizing measurement (W4-S1-report.md in the shared
   notes): estimated rest-of-fight state count under 10^7 for 58% of decisions (act 2 47%, boss 33%, last three turns
   of lost fights 72%, one turn left 100%, four or more turns left about 1%); a depth-limited, turn-by-turn exact solve
   with D4's L0 plus the network at the cut is the shape that fits.

Known inputs from S1/S2 reviews: S2's seven draw sites are classed; Perfect Fit is modelled for one card value and
refuses two or more; the key refuses a busy kernel queue; King's Courage is an order reader listed by hand (its read is
in core2 code behind OldState).
