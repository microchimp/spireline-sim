# Kit Gda7c3abc9d (rank 115, weight 3.6, family F16 / rule R17: treasure, .floor .gold .relics). Own and unresolved: the emulator did not walk into the chest room

## Symptom
One record (run WBA8PXQS5T, decision 77), 2 decisions unranked, not outcome-changing. Referee against emulator: floor 10 against 9, gold 50 against 14, relics include POCKETWATCH on the referee side only. The relic is the chest offer the comparer adds to the referee view and the 36 gold is the poverty-scaled chest gold the referee has already been paid, so the real difference is: the referee is in the chest room at floor 10 and the emulator is still at floor 9. HP and deck agree, so unlike Gbc5227da53 the emulator is alive. `acted` is The Legends Were True event, nab-the-map option (adds the Spoils Map card to the deck), at floor 9. The run lists an earlier divergence at decision 1 (the opening), so this may be a consequence of an earlier map or position drift. A second run with the same event option and a chest at floor 10 (in G96df219a44) walked in correctly, so the event option alone is not the cause.

## Expected behaviour (game)
After the event the referee had no map choice (a single path), so it walked to the chest without being asked; the chest is entered through the normal map step.

## Emulator mechanism and its other uses
`EmuLock.sync`, treasure branch (referee/lockstep.py about line 581): it calls `_forced("map")` and then walks one step only when the emulator is at a map decision with exactly one option and its floor is below the referee floor. `_forced` applies single-option screens (event result pages, rewards, card skip) up to 30 times and returns at any other screen. So the walk is skipped when the emulator map at floor 9 offers two or more nodes (a map or position difference from the referee), or when `_forced` stops on a screen it does not clear (a pending reward, a second event page). The same preconditions serve every `_forced` caller, so a change here must keep `ReanchorWalkTests` (test_lockstep.py) green.

## Scenario test
Cannot be written from the record alone. Once the cause is known, a `FakeRun` test in the style of `ReanchorWalkTests` (the `FloorRun` fake in test_lockstep.py: every map node leads onward and the floor counts rooms) that puts the emulator on the screen found by the probe below and asserts it ends at the referee floor.

## Fix sketch
Depends on the probe. If the emulator floor-9 map had two options while the referee had one: a map generation or position difference after the decision-1 divergence (then a consequence of that; no kit of its own). If `_forced` stopped on a screen: add that screen to its clearing list.

## Gate counts
C# 6,755 / 0 (if C# changes) and EmuDump 246; Python no new failing name; corpus (4,298 fights, 0 regressed) and byte proof local.

## Confidence
Low. Probe for the landing session (it can run the game): replay run WBA8PXQS5T to decision 76 and print, at the failing `sync`, the emulator `dec.type`, `len(dec.options)`, floor and map coordinates beside the referee map node and its children at floor 9; also check whether decision 1 left the emulator on a different map. Skip this group until the probe says; its weight is 3.6.