# Engine-neutral scenario tests

Plan of record: `referee/design/rebuild-plan.md` section 3 (acceptance oracle, item g) and section 5 "Tests (D10)". A scenario is
a test whose state is in the game's schema and whose expectation cites the game (a decompile method or a recording). It runs on
every engine through `IEngine` only, so the same file judges old, core2 and later the new core.

    EmuDump scenario <file|dir> [--engine old|core2] [--json] [--dump]
    python tools/fidelity/scenarios.py [--dir tests/scenarios] [--engines old core2]

`EmuDump scenario` prints one line per scenario (PASS, FAIL, XFAIL, XPASS, SKIP, INVALID) and the differing paths under a fail;
exit 1 when a result is unexpected (a fail, an xpass, an invalid file). `--json` writes the report, `--dump` adds the game view at
every point to it (the way to find the value an expectation should hold, then check it against the game, not against the engine).
`tools/fidelity/scenarios.py` runs the directory on both engines, one EmuDump process per engine; `tests/python/test_scenario_runner.py`
puts it in the Python suite together with broken cases the runner must catch; `tools/emudump.Tests/ScenarioTests.cs` tests the path
language, the check, the format check and the setup translation.

## The format

One JSON object per file under `tests/scenarios/` (a list of objects also works). Unknown keys are errors.

    {
      "name": "victory-heal/meat-at-threshold-then-burning-blood",
      "source": "game: Hook.AfterCombatVictory (...)",     // game: <Class>.<Method> | recording: <id> | ledger: <id>
      "doc": "free text: the rule, in ModelIds and numbers",
      "engines": ["old", "core2"],                          // optional: the engines it runs on (others skip)
      "xfail_engines": {"old": "the open row it waits on"}, // optional: engine -> reason it is known to fail
      "setup":   { ... },
      "start_answers": [ ... ],                             // optional: screens the fight's start opens
      "actions": [ ... ],
      "expect":  [ ... ]
    }

**setup** (the game's terms; `character`, `ascension`, `seed`, `encounter`, `relics` and `deck` are required, so a scenario does
not lean on what Neow or the seed happens to give):

| key | meaning |
|---|---|
| `character`, `ascension`, `seed` | `IRONCLAD` (the only character the engines build), 0..10, the run seed |
| `encounter` | `{id, act, kind, index}`: the game's encounter id (a label for messages) and its place in the game's act lists (`act` OVERGROWTH/UNDERDOCKS/HIVE/GLORY, `kind` weak/normal/elite/boss, `index` in `EmuDump tables` `acts[act][kind]`); the fight is met there |
| `relics` | ModelIds in obtain order. The held set becomes exactly this list (the starting relic is kept only when listed) |
| `deck` | ModelIds, or `{id, upgraded}`; the deck becomes exactly this |
| `potions` | ModelIds; the belt becomes exactly this, slot 0 first (the belt fills from the first free slot, so a gap cannot be written) |
| `hp` | `{max, current}` of the player; default 80/80 |
| `block` | the player's block in the fight becomes exactly this (an integer, 0 or more) |
| `enemy_hp` | `[{slot, max, current}]`: the living enemy in that creature slot (the slot `enemy:<slot>` names) gets this max HP, then this current HP, in the fight. Either may be left out, not both: the rolled max or the held current stays, and a held current above a lowered max comes down to it |
| `enemy_block` | `[{slot, amount}]`: that enemy's block in the fight becomes exactly this (0 or more) |
| `powers` | `{id, target, amount}`, `target` `player` or `enemy:<slot>`: a power applied with no applier, in the fight |
| `energy` | energy the player gains at the start |
| `hand`, `draw`, `discard` | cards added to that pile in the fight (`{id, upgraded, top}`, `top` for a draw-pile card on top) |

The setup runs as the setup items `Setup.cs` runs for the fuzzer and the referee's setup fights (relics_set, deck_set, potions_set,
hp, fight, block, enemy_hp, enemy_block, power, energy, card), in that order: relics, deck, potions, HP, the fight, then the combat
items: the player's block, the enemies' HP, the enemies' block, powers, energy, cards. An item an engine cannot build fails the
scenario on that engine (xfail-able), it is never skipped.

`block`, `enemy_hp` and `enemy_block` are the scenarios' own items (the referee has no counterpart yet). They write the creature's
fields as they stand, after the fight's start: no hook runs for them, and an intent the enemy already rolled stays. They **refuse**
rather than cap, since a value the fight cannot hold as written is a drifting assumption a cap would hide: a slot with no living
enemy (out of range, or dead), a block below 0, an HP below 1, or a `current` above the max HP in force (the `max` given beside it,
else the rolled one) fails the scenario at that setup item ("did not take", with the reason: `no living enemy in slot 3`, `current HP
50 above max HP 44`). What shows in the file itself is **invalid** before any engine runs: a negative value, an `enemy_hp` `max` or `current` of 0, a `current` above its own
`max`, a slot set twice in one list, an `enemy_hp` entry with neither `max` nor `current`. To set a current HP above the rolled max,
give `max` too.
Not expressible yet: a potion belt with gaps, other characters.

**actions** run in order against the open fight; each may carry `name` (default `a<n>`, 1-based) and `answers`:

| `do` | fields |
|---|---|
| `play` | `card` (a ModelId, the first copy in hand; `upgraded` narrows it) or `hand_index`; `target` slot (optional when the move has one legal target) |
| `potion` | `slot`, `target` as above |
| `end_turn` | |

An action must be legal at that moment (`IEngine.LegalActions`) or the scenario fails with the legal list. `answers` (and
`start_answers`) answer the screens the action opens, in order: `{"pick": [ModelId, ...]}` takes the first candidate with that id for
each, `{"decline": true}` refuses a skippable screen. A screen still open when the answers end is a failure.

**expect** items read the game view at a named point. Points: `start` (the fight open, start answers given), `after:<action name>`,
`end` (after the last action, the default). The view is `{snapshot, net, private, screen}`: `IEngine.GameView()` plus `screen`, the open
screen (`kind`, `candidates` as `[ModelId, upgraded]`, `max_picks`, `skippable`, `source`) or null. A path is dot-separated keys,
`[n]` a list index, `{ID}` the amount of a `[ID, amount]` entry of a pair list (`snapshot.player.powers{STRENGTH_POWER}`), `#len` last
for a length. A value is checked with exactly one of `equals`, `one_of`, `at_least`, `at_most`, `absent`; `delta_from: <point>` checks the
change since that point instead of the value (an RNG counter, an HP loss). `why` is free text. An absent path reads as absent; `snapshot.enemies`
lists the living enemies (a killed lone enemy is `#len` 0); after a won fight `snapshot.player.hp` is the run's HP, heals included.

## Outcomes

pass; fail (a difference, an engine error such as an unknown model, an illegal action, a setup item that did not take); xfail (a difference
the scenario lists for this engine, with the reason: the open `hook_open.txt` row or the unlanded fix it waits on; a run that stopped, by an
illegal action or an engine error, never reached its expectations and is a plain fail whatever the list says); **xpass** (listed and
passing: unexpected, take the row out so a fixed engine cannot go back unnoticed); skip (`engines` leaves the engine out); invalid (the
file breaks the format: always unexpected; that includes an `at_least` or `at_most` that is not a number, an `absent` that is not true or false,
and an `absent: true` on a path whose parent is not in the game view, which would pass for any typo). `xfail_engines` may name only engines the scenario runs on, with a reason of 8 characters or
more.

## What is in `tests/scenarios` (24)

Converted from the batch-3 reference tests (`referee/design/game-shaped-dispatch.md` section 6; they hold the game's order, so each
fails on an engine that has the old order). The setups are reached through the fight, not through engine internals, so a few are
restated: a lone-enemy kill is `JUGGERNAUT_POWER` 99, enemy HP the slimes' and a Strike's damage are tuned with `STRENGTH_POWER`, an
incoming hit is the lone Nibbit's first attack (13 at A10 for the seed) with `STRENGTH_POWER` on it. They were written before
`block`, `enemy_hp` and `enemy_block` existed and stay as converted; `regression/block-absorbs-first-and-zero-hp-kills` is the
setters' own cover (an enemy set to 8 HP and 4 block takes two Strikes: the first breaks the block and leaves 6 HP, the second kills).

| family | scenarios | old | core2 (F08+F10 on, as checked in) |
|---|---|---|---|
| `exhaust-order/` (ExhaustListenerOrderTests) | juggernaut-from-feel-no-pain-kills-before-forgotten-soul, juggernaut-picks-before-charons-ashes-kills | xfail (2) | xfail (2) |
| `play-order/` | curl-up-blocks-charons-ashes-after-the-card (the played card's after-play listeners before its pile: Curl Up's block before Charon's Ashes' hit) | xfail (1) | xfail (1) |
| `victory-heal/` (VictoryHealOrderTests) | meat-at-threshold-then-burning-blood, ...-black-blood (xfail); meat-above-threshold-does-not-heal, meat-below-threshold-then-burning-blood (pass) | 2 xfail, 2 pass | 2 xfail, 2 pass |
| `rounding-order/` (RoundingOrderTests) | slow-joins-the-multiplicative-chain, slow-is-decimal-exact-90x3, intangible-caps-a-halved-30-to-1 and -3-to-1, intangible-colossus-halves-a-vulnerable-4-to-1 (xfail); slow-is-decimal-exact-30x9, hard-to-kill-cap-runs-after-slow, diadem-halves-15-to-7, intangible-cap-leaves-a-halved-1-at-0, intangible-colossus-halves-a-vulnerable-1-to-0, halving-truncates-once (pass) | 5 xfail, 6 pass | 11 pass (the F08+F10 switch fixes them) |
| `regression/` | strike-damages-the-enemy-and-costs-1, vulnerable-multiplies-by-1-5, strength-adds-before-vulnerable-multiplies, weak-truncates-once, feel-no-pain-blocks-per-exhaust, burning-blood-heals-6-after-victory, block-absorbs-first-and-zero-hp-kills | pass (7) | pass (7) |

Slow's multiplier is `1 + 0.1 x SlowAmount`, and SlowAmount is a separate count of the cards played this turn (zero when the power lands, reset when its owner's side turn starts), not the power's amount. `SLOW_POWER` applied with an amount N therefore multiplies by 1 in the game too, so the four Slow scenarios apply it with amount 1 and play the cards (`DEFEND_IRONCLAD`) before the hit; the hand holds at most 10, so the count tops out at 9.

`xfail_engines` is the per-engine expectation: it names an engine only while that engine fails the scenario, so a scenario the
switch fixes lists old alone. core2 reads `migrated_families.json` beside the DLL (F08+F10 as checked in), so under the switch the
five `rounding-order` rows above pass on core2 and old keeps its 9 xfails; core2 lists only its 4 remaining ones (the F11 and F17 rows,
whose families are not switched). When a switch makes one more pass, the runner reports it as an XPASS and its core2 row comes out.

A test about core2 with no switch must say so itself and not lean on the repository's file: a `KernelOptions.None` engine in C#, a scratch
build whose `migrated_families.json` is `{"migrated": []}` (`fuzz_core2.dll_for(dll, [], work)`) in Python, or `--switches none` where a
tool takes it. With no switch core2 runs old's rules, so it fails what old fails (the five rows above then show as plain fails, since
core2 no longer lists them); `tests/python/test_scenario_runner.py` checks both views.

Not converted: `RoundingOrderTests.HalvingMultipliersTruncateOnce`'s second half, which passes a Doom-style extra multiplier straight to
the damage function (no game action reaches it with the setup members); and the direct-call forms in the other tests (a fight built by
hand, `CardEffects.ExhaustCard` called without a card play): the scenarios reach the same hook through Burning Pact and a Strike.
`JuggernautPicksBeforeCharonsAshesKills` looped 8 RNG seeds and asserted both outcomes occur; the scenario fixes one seed whose
game-order pick is the slime Charon's Ashes would kill (found with the relic absent, which leaves the pick order unchanged).

## Adding a scenario

1. State the rule's source (`game: Class.Method`, or a recording id) and work the expected number out from it, not from an engine.
2. Write the setup and actions; `EmuDump scenario <file> --dump --json` shows the views to read paths from. Pin a setup assumption you
   rely on (an enemy's HP, a tuned damage) with an `expect` at `start` or an earlier `after:` point, so a drifting assumption fails there.
3. Run both engines. An engine that fails for a known reason gets an `xfail_engines` row with that reason; the file lands with the
   fix that makes it pass, or with the xfail that names the open row. Each old run-layer fix lands with its scenario (D10), and the
   area's port must then pass it.
4. No game text in the file beyond ModelIds: `doc` and reasons name models by id.
