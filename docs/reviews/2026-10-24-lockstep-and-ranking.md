# Review: the lockstep bridge and the divergence ranking

Date 2026-10-24. Independent, read-only. Scope: `referee/lockstep.py` on `main` (4a4b15f), the driver host
under `referee/driver/`, the native re-anchor path (`src/Sts2Emulator/Core/FightAnchor.cs`,
`src/Sts2Emulator/Interop/RunNativeExports.cs`, `RunEngine.DebugSetPlayer` / `DebugResolveCombat`),
`tools/lab/rank.py`, `tools/lab/by_family.py`, their tests under `tests/python/`, `referee/RUNBOOK.md`,
`referee/FINDINGS.md`, and the two snapshots on `review/input-2026-10-24` (`docs/review-input/README.md`,
`by_family-rank-s2-m12.md`; the F08 diagnosis there was read for context).

What was run: `uv run python -m unittest` over `test_lockstep`, `test_lockstep_actions`,
`test_lockstep_anchor_state`, `test_lab_rank`, `test_lab_by_family`: 110 tests, 6 skipped (they need the
native library), 1 failure. The failure (`test_rank_outputs_are_unchanged_and_by_family_is_written_beside_them`)
is a golden sha256 of `ranked.json` and `top20.md` captured on a machine that writes CRLF: the same bytes with
`\n` replaced by `\r\n` match both goldens. It is not a ranking drift. Small Python probes were run against
`rank.classify` and `by_family.assign` to confirm the classifier claims below; nothing that needs the game was
run, and the block's own `divergences.jsonl` is not in the repository, so every number here is recomputed from
the snapshot's tables.

Numbering below uses the snapshot's rule ids (R01 to R18) and group ids. Note that the snapshot predates
commit 5657cd5 (R03 is F17 on `main`, F10 in the snapshot).

## 1. Re-anchoring: what still seeds the emulator from somewhere other than the game

Where the message carries `anchor_state`, the M1 claim holds. At a fight start `EmuLock._anchor_fight`
(`referee/lockstep.py:772`) calls `run_reanchor_fight_hidden` with both flags; `FightAnchor.PatchRng`
(`FightAnchor.cs:341`) builds every combat stream from the game's seed and counter outright,
`FightAnchor.PatchPiles` (`:215`) takes the draw pile in the game's order when `drawOrderKnown`, and
`ReanchorFight` (`RunNativeExports.cs:862`) then moves the run's streams with `SetStreams`. At a room entry
`_set_streams` (`lockstep.py:763`) moves the run's and the player's streams. `Env.AnchorPoint` (`Env.cs:311`)
and `reanchor_plan` (`lockstep.py:240`) define the anchor points identically, and `_take_hidden` strips the
key before anything else sees it (tests: `test_lockstep_anchor_state.FirewallTests`, `EmuLockAnchorTests`,
`NativeFirewallTests`). That is what convinced me for stream counters, draw order, pile contents by card id
and upgrade, the roster paired by kind, enemy HP, block and powers, and the player's HP, block, energy and
powers.

The paths that remain, each with where the state comes from:

1. **The walk of a suspended emulator is played by the agent** (`EmuLock._to_map`, `lockstep.py:717`). While the
   lockstep is lost the emulator stands still; at the next room entry `_to_map(floor)` walks it to the
   referee's floor, choosing map nodes with `self.rules.pick(..., self.rng)` (the rules policy on the agent's
   RNG), leaving other screens by `reward.proceed` / `card.skip` / `shop.leave` / `rest.leave` or again by the
   rules, and winning fights unplayed through `run_debug_resolve_combat`. `_set_player` and `_set_streams`
   then reset holdings and counters, and nothing else. What the walk leaves as the emulator's own: the map
   position and the path taken, the reward odds (`CardRarityOffset`, `PotionRewardOdds`, the four
   unknown-room odds), the relic grab bags, the event pool, the shop inventory, `LastResolvedRoomType`, the
   elite-sequence index, the gold and removals owed, relic counters, wax relics, Guilty and Mad Science
   properties. Every one of these is something the whole-run replay's anchor does set from the game's save
   (`tools/emudump/RunReplay.cs` `AnchorHoldings:676` and `AnchorStreams:754`). So after any resume, the next
   relic reward, card reward rarity, potion drop, unknown-room roll and event are drawn by the game's counters
   but from the emulator's own pools and odds. This is agent-side state (the policy and the agent seed decide
   the path) seeding emulator state that the comparer later reads as F15 and F16 divergences (see 2.8).

2. **`_set_player` zeroes what the observation does not carry** (`lockstep.py:745`). The deck is set as
   `(id, upgraded, 0, 0)`: enchantments and tinker state are dropped because the driver's `Env.Obs`
   (`Env.cs:269`) reports a card as `[entry, upgraded]` only. A relic the emulator did not hold is created
   fresh by `RunEngine.DebugSetPlayer` (`RunEngine.cs:1891`), so its counter starts at zero whatever the game's
   is; a relic it did hold keeps the emulator's own counter. Neither is the game's.

3. **`_set_streams` cannot tell a no-op from a success.** `SetStreams` (`RunNativeExports.cs:807`) moves only
   the sixteen streams `StreamNamed` maps, and only when the emulator's `RawSeed` equals the game's; any other
   name, or a seed derivation that differs, is left at the emulator's counter and merely not counted in
   `moved`. `_set_streams` tests `>= 0` and sets `anchored_with = "streams"` before the call, so a re-anchor
   that moved nothing is reported as a stream anchor. The names the game actually sends are in every
   `anchor_state` (`Driver.RngCounters` enumerates the whole set, `Env.cs:323`), so the gap is measurable but
   is not measured.

4. **A fight start after a walk rolls the fight from the emulator's lagging streams** (`_reanchor_with`,
   `lockstep.py:824`). The branch walks to the map below the fight's floor, sets the holdings, applies the lone
   map option and enters the fight, and only then anchors it. The encounter, the enemies' HP rolls and the
   opening shuffle come from the emulator's own run streams, because the hidden counters at a fight's first
   decision are the post-setup counters and cannot be applied before the setup. `FightAnchor.PatchEnemies`
   (`:292`) repairs HP and powers when the roster pairs by kind and refuses otherwise
   (`anchor:the game has X where the emulator has none to pair with it`), which is what
   `reanchor_fail_reasons` in `summary.json` would show. Even on success the encounter id, the enemy count
   when kinds repeat, and the AI memory are the emulator's.

5. **What the fight anchor cannot say stays the emulator's** (header of `FightAnchor.cs:1`). Enemy AI memory
   beyond `MoveIndex` (`SetMove:439` moves the cycle position; `LastMove`, `MoveHistory`, branch state stay),
   relic counters, per-card modifiers for cards matched in place (`PatchPiles:238`), enchantments of cards it
   has to build fresh (`new CardInstance(id, up)`, `:269`), the `MonsterAi` stream when `AiRng` is not a
   `CountingRandom` (`:361`), `KnownTopCount`, and the potion belt (the combat record is `obs["combat"]`,
   `Combat.cs Snapshot:430`, which has no `potions` key, so `PatchPotions` never runs in the lockstep).
   `ReanchorFight` also passes `_ => []` as the later moves, so `SetMove` ranks positions on the current
   intent alone.

6. **The observation-only anchor is still reachable and is invisible in the block summary.** With no hidden
   state `_anchor_fight` falls back to `run_reanchor_fight` (`lockstep.py:789`, `anchored_with =
   "observation"`), which keeps the emulator's streams and, where the cards agree, its draw order. That
   happens with `SPIRELINE_ANCHOR_STATE=0`, an emulator side without `wants_anchor_state`, or a game copy
   whose installed driver ignores the `anchor_state` spec key. `envproto.play` aggregates `reanchors`,
   `reanchor_by_plan` and `reanchor_fail_reasons` into `summary.json` (`referee/envproto.py:697`) but not
   `anchored_with`, which lives only in each run's `stats` in `runs.jsonl`. A block run entirely on
   observation anchoring has the same summary as an M1 block.

7. **`_auto_combat` can end the emulator's turn before the fight-start comparison** (`sync`,
   `lockstep.py:620`). If exactly one combat action is legal on the emulator (only End Turn, because a card
   it costs differently is unplayable) it is taken before `_equal` runs; the anchor then writes the record's
   turn, piles and HP back, but the AI memory, per-turn counters and relic per-combat counters of that extra
   turn remain.

None of these is a path by which the game's hidden state leaks to the agent; that direction is covered by
the firewall tests and by the per-simulation resampling the search does from the `combat_fork(None)` root
(`rl/run_train/engine.py:259`). They are the paths by which the emulator's lockstep state is not the game's.

## 2. Comparer artefacts: what the comparer still counts that is the bridge's doing

The comparer is `project` plus `structural_diff` (`lockstep.py:130`, `:169`): act, floor, hp, max_hp, gold,
the deck, relics and potions as sorted multisets, and in a fight the sorted hand and `enemy_hp` as an ordered
list. `tools.fidelity.fight_compare.diff` compares lists whole, so one differing element is one diff entry
with both full lists. The two known artefacts are handled: `deck_pending_pick` with `_deck_within_picks`
(`:208`, `:219`) and `TREASURE_PICKUP_PATHS` (`:41`, `:395`). The further candidates, strongest first, each
with how to confirm it from `divergences.jsonl` without the game:

| # | candidate | where it comes from | shape in the records | how to confirm |
|---|---|---|---|---|
| 2.1 | A corpse at a mid-card selection screen (R04, group G2891c25e0d, 2277 weight, 32% of the block) | The emulator keeps dead enemies in its roster for stable observation indices and sweeps them once per card (`CombatEngine.ResolvePlay`, `CombatEngine.cs:629` and `:1118`); the game removes a creature inside the kill. A screen opened by the card's own effect is compared before the sweep. | `type == select_cards_combat`, `ctx.pick == 1`, diff `.combat.enemy_hp` with one more entry, value 0, on the emulator side; `acted` is a play of a card that damages then selects (the diagnosis names HEADBUTT and NEOWS_FURY). | Count R04 records by `type` and by which side is longer. All at `select_cards_combat` with the extra 0 on the emulator side = the screen-timing case; any at `type == combat`, or with the extra entry on the referee side, is a real roster difference (a `ShouldCreatureBeRemovedFromCombatAfterDeath` refusal, or a sweep the emulator missed). Note the by_family R04 text still describes the pre-b876d5f comparer and names the wrong side. |
| 2.2 | The deck mid-fight (R16 at `type == combat`: five single-divergence groups at floors 19 and 20, 175 weight) | The game changes the master deck at once when a card is stolen (`SwipePower.Steal`) or a fight card reward is taken; the emulator's run deck changes at `FinishCombat` (`TakeStolenCardsFromTheDeck`, `RunEngine.cs:1724`) or at the selection. | diff `.deck` at a `combat` decision; direction tells which: referee shorter by one card = a steal, referee longer = a card gained in the fight. | For each R16 record at `combat`: the sign of the multiset difference, the card id in `display.entities`, and the record's `encounter` field (a hopper in the roster). |
| 2.3 | HP at the first reward question (R03, 32.5 weight) | Heals at the victory. `combat_fork`'s own docstring says a won playout ends "before the rewards screen's heals (Burning Blood)"; if the game applies such a heal after the rewards screen opens while `FinishCombat` applies it before, the first `reward` decision differs and the next decision would not. | `type == reward`, diff `.hp` only, referee lower. | The distribution of `emu - ref` across R03 records: a constant equal to a held relic's heal (6 for the starting relic) is timing; a spread is a rule. The relics are in the run's start or in the nearest re-anchor record's `.relics` values. |
| 2.4 | Floor at the chest question (group Gbc5227da53: `treasure: .floor, .gold, .hp, .potions, .relics`, 108 weight) | `sync("treasure")` walks the emulator to its map and applies a lone map option when its floor is below the referee's (`lockstep.py:630`); the chest itself is auto-taken by the engine (`engine._treasure`). A `.floor` difference at the question means the walk stopped on the wrong side of the room. | `type == treasure` with `.floor` in the diff. | The record's `.floor` values (emulator ahead or behind by one) and its `acted` (the map choice and its room type). |
| 2.5 | `screen:None` classed as run flow (R02) | `_decision()` returns None in combat and at run end (`lockstep.py:554`). `screen:None!=rest` (group G1b02fc9e96) means the emulator was still fighting, or dead, when the referee reached a rest site. R02 matches any `screen:` prefix. | `reason` starts with `screen:None`. | Trivial from the `reason` string; these belong with R01. |
| 2.6 | `phase:0` and `phase:6` lumped under one cause (R01) | Phase 0 is `PHASE_COMBAT`, 6 is `PHASE_COMPLETE` (`src/sts2_gym/run_constants.py:21`). `phase:0!=rewards` is the emulator still fighting; `phase:6` at a `combat` decision (group G499831a0f5) is the emulator's player dead where the game's is alive. R01's text names only the first. | `reason` carries the number. | Parse it; the two are different F10 mechanisms (a win check against a death check). |
| 2.7 | A lone End Turn the emulator did not take, or took when the game did not (part of R12 and R13) | `_auto_combat` (`lockstep.py:664`) takes an action only when exactly one is legal. A playability difference (F07 costs, energy) leaves the emulator mid-turn while the game ended its turn, or the reverse; the next `combat` decision then shows a hand-size gap or a hand plus HP diff. | R12 or R13 whose referee `turn` advanced since the previous decision while `acted.type == "play"`. | Not confirmable today: the record carries `turn` but not the previous decision's turn. `_track` has `self.round` from the step before; one field (`turn_advanced`) would make it so. |
| 2.8 | Reward and event differences after a resume (R15, R14, R17, and the two `event:X!=Y` groups) | Section 1 item 1: the walk leaves pools and odds the emulator's own. | Any of these rules in a run that has an earlier record with `reanchored: true`, or whose `not_ranked_span` covered the decisions before. | Per run, order the records by `decision`; split each rule's weight into "after a resume" and "never lost". A rate several times higher after a resume is the walk. The walk's length is not recorded; it should be (section 5). |
| 2.9 | Enchantment and counter state after a re-anchor | Section 1 items 2 and 5: cards rebuilt without their enchantment, relics added with a zero counter. | Damage and block differences in later fights of a run that re-anchored. | Not confirmable from the lockstep records: the observation has no enchantment. The re-anchor record's `.relics` diff does list the relics that were new to the emulator; later R06 records in that run can be checked against those relics' mechanics. |
| 2.10 | Living enemies missing from the emulator's list in spawner fights | `emu_enemy_hp` reads `MAX_ENEMIES` slots of the raw roster (`lockstep.py:121`); the divergence catalogue (E444, E474) records that corpses the emulator keeps can push living creatures past the sixth slot, where the observation drops them. | `.combat.enemy_hp` length diff at `type == combat` with the referee listing more, in a Fabricator or Ovicopter encounter. | The `encounter` field and the direction. This is an observation artefact, not a death rule. |
| 2.11 | First-in-span masking | Not a wrong record but a missing one: after a loss nothing is compared until the next re-anchor (`Lockstep.decide:306`), so every divergence inside the span is unrecorded. | Visible only as `not_ranked_span.count`. | The span distribution per group; a group with long spans is masking an unknown amount of other families' divergences. |

Checked and found not to be artefacts: gold, potions, hp and max hp during a fight (the emulator's view
reads the combat copy, `RunEngine.WriteObservation`, `RunEngine.cs:1051`); the potion belt at a reward
question (both sides offer a potion only into a free slot); the gold reward (claimed before the first
question on both sides, `_flush_rewards`, `lockstep.py:575`).

## 3. The ranking

### 3.1 What "agent weight" is

`rank.weigh` (`tools/lab/rank.py:202`): weight = affected × floor_factor × outcome_factor, where
affected = 1 + `not_ranked_span.count`, floor_factor runs from 2.0 at floor 0 to 1.0 at floor 50, and the
outcome factor is 1.5 for a confirmed outcome change. The span is the number of decisions between the loss
and the next successful re-anchor, that is the distance to the next map choice or fight start. So the weight
measures how long the bridge stayed lost, which is bridge geometry (where in the floor the decision sits),
not how much the agent's choices depend on the divergence. The snapshot shows the effect directly: weight per
divergence is 33 for R04 (a first-play screen in the first fights), 26 for F08, 9 for F15 (reward screens,
one to three decisions from the next map choice) and 6.4 for F16. Families whose errors surface late in a
floor are weighed a fifth of those whose errors surface early, independently of how many errors they hold.

Two further properties follow from the span. The weight is attributed wholly to the first divergence of the
span and nothing inside the span is recorded (2.11), so group weights are not additive: fixing the top group
will re-expose whatever its 68 spans hid, and the next ranking will show new mass rather than less. And the
same emulator error weighs differently depending on whether the referee happened to take a lone path (no
map decision, so no room-entry re-anchor until the fight start).

The outcome factor: `outcome_changing` (`rank.py:169`) confirms only when a record or run end carries
`emu_outcome`. Nothing in this repository writes that key (`envproto.play_worker` builds the run event
without it, `envproto.py:599`); only `rank.py` and its test mention it. The snapshot nevertheless reports
172 confirmed outcome-changing divergences. Either the block's records come from a writer outside this
repository, or the column is mislabelled; which one decides whether the 1.5 factor ever fired.

### 3.2 The architectural tag

`rank.classify` (`rank.py:129`) tags by decision type and field name, first match: a pile path →
`rng_stream` or `action_queue` by size equality; a reward type with a card or potion path → `rng_stream`;
`type == "combat"` with a listener word → `listener_order`, with an HP word → `damage_block_pipeline`;
everything else `single_site`. Confirmed by probe: an `enemy_hp`-only diff at `select_cards_combat` is
`single_site` (the type string is not "combat"), the same diff at `combat` is `damage_block_pipeline`; a
hand diff at `select_cards_combat` is `rng_stream` (piles are checked before the type); `.relics` at
`combat` is `listener_order` because "relic" is a listener word; `.deck` at `combat` is `single_site`.
`by_family.COMBAT_TYPES` does include `select_cards_combat` (`by_family.py:49`), so the two modules disagree
on the same record. The "architectural share" is therefore the share of weight at `combat`-typed decisions
with a hand or HP diff, plus reward-content diffs; it is not a measure of hook-family concentration.

### 3.3 What the gate rule actually sees

The snapshot: total 7043.8, architectural 3860 (54.8%), top families by architectural weight F08 1795.9,
F11 161.2, F15 146.5 = 2103.7, 54.5% of the architectural weight, "neither rule fires"; unassigned 3886.8
(55.2%), of which 1753.5 architectural (45.5% of the architectural weight). Only five families carry weight
at all, and F10 and F16 carry none that is architectural (their decision types are never "combat"), so "top
five by architectural weight" is three families, and the rule is vacuous as a concentration test.

Recomputed from the same tables with the one-line classifier fix of 3.2 (G2891c25e0d becomes
`damage_block_pipeline`, 2277 moves to architectural):

| reading | architectural weight | top-five architectural share | verdict |
|---|---|---|---|
| snapshot as published | 3860 (54.8%) | 54.5% | neither |
| classifier fixed, R04 unassigned (primary) | 6137 (87.1%) | 36.6% | Tier 2 fires |
| classifier fixed, R04 read as F10 (reading C) | 6137 (87.1%) | 71.9% | Tier 1 fires |
| upper bound, every unassigned diff in the top five | 6137 | 100% | |

The gate's answer is decided by a type string in `classify` and by which reading of R04 is taken, not by the
data. The lower and upper bounds (36.6% and 100%) straddle both thresholds under every reading, so with the
unassigned mass where it is the rule cannot fire and cannot be excluded.

### 3.4 How the family table can mislead the rebuild order

- F08 shows 27.5% but the F08 diagnosis attributes 52% of the weight of the four F08-bearing groups to death
  timing (its mechanism A), which is R04 and unassigned here, and F10 or F08 by reading. The table under-
  states the damage/death pipeline by about the whole of G2891c25e0d, and misplaces its coarse family.
- F11 (card piles) shows 2.3%, but R12 and R13 (1700 weight, hand diffs with or without HP) are the hand
  shapes the rules cannot place; they could be F03, F05, F06 or F11. The table says nothing about which, so a
  decision to defer the pile and turn families has no support either way.
- F16 (9.0%) and F15 (4.6%) are inflated by walk artefacts (2.8), the chest floor case (2.4) and the
  `screen:None` misclassification (2.5), all of which are the bridge's.
- The span weighting favours whichever family's errors surface earliest in a floor (3.1); a rebuild order
  that follows it will keep finding that the next family "grew" once the previous one stops masking it.
- R03 moved from F10 to F17 after the snapshot was taken (5657cd5); the published table is already one
  rule-table version behind.

### 3.5 The smallest change that makes the unassigned share informative

Every record since cccff9e carries `acted` (the step before: `play`, `end_turn`, `potion`, `choice`, with
the label), `turn`, `encounter` and `room_type` (`Lockstep._track`, `lockstep.py:351`). `by_family._match`
reads none of them. Two sizes of change, no game needed, rerun on the existing records:

- Smallest (reporting only, about fifteen lines in `by_family.table` and `md`): a breakdown of the unassigned
  weight by rule × `acted.type` × decision type, with the extra-corpse side for R04. This says at once how
  much of the 55% sits after a play, after an end turn, or at a selection screen, without asserting a family.
- Slightly larger (rules, about thirty lines plus tests): R04 split by side and `acted` (emulator-only corpse
  after a play → F10 kill and removal timing, evidence `shape`; referee-only corpse → F10 removal refusal);
  R12 and R13 split by `acted.type` (`end_turn` → coarse "start- and end-of-turn hooks", `play` → F06 with the
  card id as the ingredient, `potion` → F14), evidence `shape`. With the classifier fix this places about
  3860 of the 3887 unassigned weight with shape evidence, and the top-five bounds stop straddling the
  thresholds. The evidence column already exists to keep "shape" apart from "diff"; the gate should be read
  with evidence weights beside it.

## 4. Recommendations, ranked

| # | recommendation | expected effect | cost |
|---|---|---|---|
| 1 | Do not act on the published verdict; treat the gate as undecidable until 2 and 3 are done. | Avoids a rebuild order set by a type string and a reading. | none |
| 2 | `rank.classify`: count `select_cards_combat` and `card_reward_combat` as combat (one line, reuse a shared `COMBAT_TYPES`); regenerate `ranked.json` and `by_family.md` on the block. | Architectural share 55% → about 87%; the verdict becomes computable, and visibly reading-dependent. | one line, one test, minutes to rerun |
| 3 | `by_family`: read `acted` and `turn`; split R04 by side, R12/R13 by acted type; move `screen:None` to R01; distinguish `phase:0` from `phase:6`; correct R04's text. | Unassigned 55% → under 10%, with shape evidence; the F08/F10 and turn/pile questions get a first answer from existing records. | about fifty lines plus tests, no game |
| 4 | Record the bridge's provenance per re-anchor and aggregate it in `summary.json`: `anchored_with`, streams moved against streams named, walk length (floors walked, rules picks taken, fights resolved), relics and cards `_set_player` added. | Every artefact in section 2 becomes separable in the next block; a block that silently ran on observation anchoring is caught. | about forty lines in `lockstep.py` and `envproto.py`, tests on `FakeRun` |
| 5 | Add `turn_advanced` to the record context (`_track` has the previous round). | 2.7 becomes confirmable; R12/R13 split further into lone-end-turn cases. | three lines, one test |
| 6 | Publish the ranking under three weights (divergence count, span-weighted, span capped at five) and report rank stability; state the expected unmasking after the top fix. | A fix order that does not move with bridge geometry; honest expectations for the next block. | about twenty lines in `rank` and `top_md` |
| 7 | Anchor the run state beyond holdings and counters at room entry: have the driver put the odds, grab bags, relic properties (`RunSnap.RelicProps`) and card enchantments into `anchor_state`, and apply them through a native export built from `RunReplay.AnchorHoldings` and `AnchorStreams`, which already do this from a save. | Removes the walk artefacts (2.8) and the counter and enchantment artefacts (2.9); raises resume quality after every loss. | medium: driver, a native export, the binding, tests, a native rebuild |
| 8 | Fight start after a walk: include the room-entry counters in the fight-start `anchor_state` (the driver records them when the room is entered), so a walked emulator rolls the same encounter before the fight anchor; until then, count `fight_start:anchor:` failures in the report. | Fewer failed fight-start re-anchors; the roster is the game's rather than paired by kind. | small driver change plus `lockstep.py`; the counting alone is trivial |
| 9 | Once 2.2 and 2.3 are confirmed from the records, defer `.deck` during a fight and `.hp` at the first reward question the way the chest fields are deferred (compare at the next decision). | Removes two small artefact classes without hiding a wrong result for more than one decision. | ten lines each plus tests |
| 10 | Add enchantment and tinker state to the driver's card entries (`Env.Card`) and read them in `_set_player` and `FightAnchor.Cards`. | Enchanted decks survive a re-anchor. | driver, lockstep, FightAnchor; medium-low |
| 11 | Housekeeping: compare parsed JSON or normalised line endings in the golden test; regenerate the snapshot after 5657cd5; state where `emu_outcome` comes from or drop the "confirmed" column from reports. | A green gate on every platform; reports that mean what they say. | trivial |

## 5. Proposed tests (none added)

- `test_lab_rank`: an `enemy_hp`-only diff at `select_cards_combat` classifies as `damage_block_pipeline`;
  `.relics` at `combat` is not `listener_order` unless that is intended.
- `test_lab_by_family`: an R04-shaped record with the extra 0 on the emulator side after a `play` is placed in
  F10 with evidence `shape`, the referee-side case apart; `screen:None!=rest` goes to R01; a hand-size gap
  after `end_turn` lands in the turn coarse family, after `play` in F06 with the card as ingredient; the
  verdict is reported with its lower and upper bound and marked undecidable when they straddle a threshold.
- `test_lockstep`: `_set_streams` with seventeen names of which one is unknown reports one unmoved stream and
  counts it in `stats`; a walk of three floors records `walked: 3` with the picks taken; `turn_advanced` is
  true on the first combat decision of a new round; `_set_player` keeps an enchantment when the observation
  carries one; a run with no hidden state ends with `stats["anchored_with"] == {"observation": n}` and the
  block summary carries the same.
- `test_envproto` (or the play summary's own test): `summary.json` carries `anchored_with` and the walk and
  stream counters summed over runs.
- A records-level checker, no game: given `divergences.jsonl` and `runs.jsonl`, prints the confirmations of
  section 2 (R04 by type and side, R16 by direction and encounter, R03 delta distribution, every rule split
  by after-resume against never-lost). It is the same reading of the records the review did by hand and
  would make the next block's artefact share a number rather than an argument.

## 6. What this review could not check

The block's `divergences.jsonl` and `runs.jsonl` are not in the repository, so section 2's confirmations are
stated as procedures, not results, and section 3's numbers are arithmetic over the snapshot's tables (group
tags may move a few units when `by_family` is rerun per record). The game and the native library are absent
here, so the six native tests were skipped and nothing in section 1 was traced; it was read from both sides'
code. The PLAN 8.3 thresholds are known only through `by_family.verdict` (`by_family.py:239`).
