# Kit ShivDamage (no lockstep rank or weight; rebuild-lane finding; family: fidelity tooling vocabulary; a decision kit)

## Symptom
The emulator keeps the Silent power Accuracy as the buff `BuffId.ShivDamage` (bonus damage for Shiv cards). The game's power is `AccuracyPower`. The
shared name table `tools/fidelity/power_aliases.json` ("emu_rename": emulator buff key -> game power key, both UPPER_SNAKE without `_POWER`) has no
entry for it: `SHIV_DAMAGE` -> `ACCURACY` is kept only in code (`src/Sts2Emulator/Core2/OldState.cs`, the dictionary `RenamesNotInAliases`, with a
comment saying it is left out of the file so that `fight_compare` and Reanchor, which read the file, do not change).
Effect today: `tools/fidelity/fight_compare.py` renames the emulator's powers through `EMU_RENAME` and finds no entry, so in a fight where Accuracy is in
play the emulator side lists `SHIV_DAMAGE` and the game's side `ACCURACY` (the game's `ACCURACY_POWER` loses its suffix in `power()`). That is a
difference of vocabulary only (the amounts agree; `Cards/Silent/AccuracyTests.cs` pins the rules). Expected to show as a first divergence at the
Accuracy play in every such fight and to hide the next real one (to be measured, see Gate counts). Reanchor's write-back
(`tools/emudump/Engines/Old/OldAnchor.cs`, `OldReanchor.cs`) uses the same table in the other direction; whether it drops the game's `ACCURACY` power or
reports it unknown should be read from `OldAnchor` before landing. A judging error, not a rules bug.

## Expected behaviour (game)
`Models.Powers.AccuracyPower` adds its Amount to the damage of a powered attack whose card carries the Shiv tag (only Shiv does); its id is
`ACCURACY_POWER` in the recorded power lists. The emulator's buff already reproduces that; what is missing is that the comparison tools know both
are the one power.

## Emulator mechanism and its other uses
Readers of `tools/fidelity/power_aliases.json`: `tools/fidelity/fight_compare.py` (`EMU_RENAME`, `FOLDS`); `tools/emudump/Engines/Old/OldAnchor.cs` and
`OldReanchor.cs` (the id tables); `OldGameView.cs` (old buff key -> game key); `Core2Engine.cs`; `KernelOrder.cs`; `src/Sts2Emulator/Core2/OldState.cs`
(`OldModelIds` builds its rename dictionary from the file and then `TryAdd`s `RenamesNotInAliases`, so a file entry makes the code entry a harmless
duplicate); and the copiers `tools/fidelity/kit.py`, `mutate.py` (COPIED_FILES) and `hook_order.py`.
The emulator-side key is `snake("ShivDamage")` = `SHIV_DAMAGE`; the game side is `power("AccuracyPower")` = `ACCURACY` (`data/id_map.json` has `AccuracyPower`
in its powers). `src/sts2_gym/live/powers.py` `BUFF_ALIASES` already maps the live game's `ACCURACY_POWER` to `ShivDamage` for the Python gym.

## Scenario test
Project: `tests/python`, file `test_fight_compare.py` (copy the helper `emu_powers` and `PowerListTest.test_alias_table_is_the_one_reanchor_reads`, which
pins `EMU_RENAME["NEXT_TURN_ENERGY"]`).
- `test_accuracy_is_the_shiv_damage_buff`: `self.assertEqual(game_powers([["ACCURACY_POWER", 4]]), emu_powers([["ShivDamage", 4]]))` and
  `EMU_RENAME["SHIV_DAMAGE"] == "ACCURACY"`. Fails before (the emulator side stays `SHIV_DAMAGE`), passes after.
- A table-shape test: every `emu_rename` key is the snake form of a `BuffId` member name (parse `src/Sts2Emulator/Core/BuffState.cs`) and every value is
  the `power()` form of a name in `data/id_map.json` "powers" (passes before and after; guards a typo in the new entry).
- Optional C# side (`tools/emudump.Tests/Core2GameViewTests.cs`, which reads the file): `OldModelIds` built from the real file maps `ShivDamage` to a Game
  mapping for `ACCURACY` (passes before too, because of the code entry; it protects the later removal of `RenamesNotInAliases`).

## Fix sketch
Option A (recommended): add `"SHIV_DAMAGE": "ACCURACY"` to "emu_rename" in `power_aliases.json`; leave `RenamesNotInAliases` in `Core2/OldState.cs` (the
rebuild lane's file; it can delete the line later). What changes: `fight_compare` stops reporting the vocabulary difference, so fights with Accuracy get
their real first divergence (group lists, ranks and the diverged count move; fights that only diverged at the Accuracy play go clean); Reanchor's
write-back gains (or stops mishandling) the player's Accuracy; the byte proof changes only for fights whose recording lists `ACCURACY_POWER`, in the modes
that read the table; `OldModelIds` and Core2 are unchanged. Cost: a judging step in the baselines (refset v2, the lab corpus score) for those fights;
landing must record before and after counts.
Option B: keep the file as it is. Nothing moves and the baselines stay comparable, but `fight_compare` keeps a false difference for every Accuracy
fight, the same mapping lives in two places that can drift, and the kernel's listener order keeps a rename the tools do not share.
Recommend A: the entry only removes a difference that is not a rule difference, the baseline break is small (fights with the Accuracy power) and should
be taken once, together with the corpus baseline for the commit that lands it, rather than carried as a permanent footnote.

## Gate counts
Python: the new tests pass and no new failing name; C# `Sts2Emulator.Tests` 6,755 passed / 0 failed (unchanged); `tools/emudump.Tests` 246. Locally by
the landing session: the corpus (4,298 fights, 0 regressed) with `fight_compare` run before and after on every fight whose game power list contains
`ACCURACY_POWER` (count them first; expect the diverged-fight count down or equal, never up), the Reanchor outputs for the same fights, and
`byte_proof.py`: identical except those fights.

## Confidence
Medium-high that A is correct and safe (one table entry, the code already assumes it). Medium on the size of the judging change: not measured here (no
game data). Reading `OldAnchor`'s unknown-power handling, and running `fight_compare` on one Accuracy fight before the change (it should show
`SHIV_DAMAGE` against `ACCURACY`), would raise it to high.
