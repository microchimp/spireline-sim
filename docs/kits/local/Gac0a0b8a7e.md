# Kit Gac0a0b8a7e (rank 85, weight 9.2, family F16 / rule R17: treasure, .gold). Same mechanism as G96df219a44

Short kit: full body in `G96df219a44.md` in this folder; implement together (batch K1-T1).

## Symptom
At the treasure question the emulator has exactly 300 more gold than the referee: 173 against 473 (floor 6, a chest reached from an unknown node) and 33 against 333 (floor 10; this run lists earlier divergences at decisions 58 and 92, but the delta is exactly 300 so it is this artifact). 2 divergences, 4 decisions unranked, 1 confirmed outcome-changing. 300 is the Old Coin pickup (`ApplyRelicPickup`, case `RelicOldCoin`, through `GainGold`). `acted` (a map step, the Sunken Statue grab-sword option) is the step before the question and is not the cause.

## Expected behaviour (game)
As in G96df219a44. Chest gold is paid when the chest is opened (before the question) and so is in both sides gold; the relic gold appears on the referee side only after the take.

## Emulator mechanism and its other uses
Same as G96df219a44. `GainGold` is the run gold chokepoint (Bloody Idol listens), so the deferral must stay in the comparison, not in the emulator gold code.

## Scenario test
`tests/python/test_lockstep.py`, `TreasureTests`: test 2 of G96df219a44.md, gold case: referee `obs()` gold 99, emulator view gold 399, relic OLD_COIN in both the take label and the emulator relics; fails before (diverged 1 on `.gold`), passes after. Guard: the same gold gap with a different relic id in the label is still a divergence.

## Fix sketch
`.gold` is in the deferred set of G96df219a44.md. Nothing group specific.

## Gate counts
As in G96df219a44.md (C# 6,755 / 0; EmuDump 246; Python no new failing name; corpus and byte proof local).

## Confidence
High for the mechanism (the exact 300 matches the Old Coin case); the sample has two records only.