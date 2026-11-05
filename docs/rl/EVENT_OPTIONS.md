# Event options: the record's option names to the copy's option index (D55, 2026-11-05)

The human run record (Spire Codex public run export for v0.107.1) names the option a player took at an event by a text key,
`<EVENT>.pages.<PAGE>.options.<OPTION>.title`. The copy takes an option by its place on the page (the action number, 0 to 3). The fork
had option **counts** per event (`src/Sts2Emulator/Generated/EventOptions.g.cs`) and no table from one to the other, so the record's
event decisions could not be taught and a whole-run seed replay lost step at the first event (the 2026-11-05 seed-replay probe: 111 of
400 replays stay in step to the end). This is that table.

* `data/event_option_titles.json`: per event, its pages and the options of each page in index order, with the record's option name, the
  title the game shows, `source` (`capture`, `engine`, `record-inferred`), `mapping` (`exact`, `inferred`, `unresolved`) and the
  evidence in words.
* `rl/event_keys.py`: `resolve(event_id, record_title, state=None) -> index | None`, `coverage(...)`, `EventTable.followup(...)`.
* `scripts/build_event_option_titles.py`: `scan` (one pass over the training fifths of the corpus, aggregates only), `build` (writes the
  JSON), `check` (the file agrees with the captures). The curated rows (the names no title carries, the later pages, the pages rolled
  per visit) are written in that script with their evidence.
* `tests/python/test_event_keys.py`: 31 tests (every capture title resolves to its own index, the patterns, unresolved is None, the
  coverage count, the shown sets and the locks). `shown_for` gives the options displayed together with a chosen one and `available`
  the engine's locks; they feed the event rows of the human prior (`docs/rl/HUMAN_PRIOR.md`, "Event rows").

No decompiled text is used or copied: the sources are the live captures and our own engine code. The decompile is not on this machine
and `scripts/decompile.sh` and `scripts/generate_event_options.py` were not run. Anything derived from the record is from the Spire
Codex public run export for v0.107.1, the training fifths only (the exam fifth is left out with `human_starts.is_exam`), aggregates only,
counts under 20 not shown.

## 1. What the three sources gave

1. **The live captures** (`tests/fixtures/events/*-options.json`, the `-opt<n>` files: the page before a choice and the page after it).
   52 events have a first page, 99 `-opt` files show the page after a choice (Abyssal Baths' Linger and Exit Baths, Tablet of Truth's
   Continue Deciphering and Give Up, Trial's DECIDE pages, Slippery Bridge's second Hold On, the conveyor's Leave). The record's option
   name is matched to a captured title by its words: `TOUCH_A_MIRROR` is "Touch a Mirror", `RIP` is "Rip the Leech Off",
   `SETTING_2` is "Setting 2". A name is `exact` when its words (less a count and the small words) are in exactly one title of that page.
   A locked option's captured title is the generic "Locked", so a locked slot gives the index and a lock reason, not a name.
2. **The fork's own engine code** (`RunEngine.cs`'s event step and `WriteEventActionMask`, `RunNonCombatEffects.cs`, `RunConstants.cs`):
   which action number each option is, the lock rules and their prices, the order, and the events no capture covers (Colossal Flower,
   Grave of the Forgotten, Round Tea Party). `EventOptions.g.cs` bounds the indices (a test pins every index below the engine's count).
3. **The record's own keys and outcomes**: the set of (event, page, option name) pairs that the training fifths hold (11,961 vanilla
   Ironclad Ascension 10 runs of fifths 1 to 4, 45,623 event visits), and for visits with one choice what that choice led to (gold, HP,
   cards gained or removed, relics). Where no title carries the name (`PLAIN` and `ORNATE` for "This" and "That"; `EAT` for "Consume"),
   the captured description and the engine's semantics are set against the outcome the record shows for that name (PLAIN costs 6 HP
   and pays about 55 gold in every visit; ORNATE adds one card and one relic in every visit), and the row is `inferred` with that
   evidence written in it.

A stale capture: `JungleMazeAdventure-options.json` holds Endless Conveyor's page; the `-opt` captures of the same event name Solo
Quest (0) and Join Forces (1), which is what the table uses (the file notes it). Round Tea Party's warning page has one option the
record does not list (`.ThatWontSaveToChoiceHistory()` in the game), so the table carries it as an unrecorded follow-up
(`EventTable.followup("ROUND_TEA_PARTY", "PICK_FIGHT") == 0`).

## 2. Coverage

Over the training fifths (fifths 1 to 4, vanilla Ironclad Ascension 10, exam fifth out): 11,961 runs, **45,623 event visits**
(not counting the ancient screens, which are their own decision kind). The counts file's 66,577 event-choice points are 44,815 event
visits with a recorded option and, by difference, 21,762 ancient screens (Neow and the later ancients; not recounted).
Event decisions are 53,786 recorded choices (40,368 visits have one choice; the longest, a conveyor, 33).

A visit is as strong as its weakest key; an event's class is its weakest key among keys chosen at least 20 times.

| class | what it means | events | visits | share of all visits | decisions |
| --- | --- | --- | --- | --- | --- |
| `capture` | every key's words are in a captured title | 35 | 34,112 | 74.8% | 40,820 |
| `engine` | no captured page; the index is the engine's option order, the record's outcomes agree | 5 | 1,875 | 4.1% | 2,951 |
| `inferred` | a captured lock text or the record's outcomes name the index (no title carries the name) | 12 | 6,494 | 14.2% | 7,236 |
| dynamic (unresolved) | the page's options are rolled per visit; the index needs the offered list | 4 | 2,328 | 5.1% | 2,775 |
| unresolved (other) | a name no source knows (Reflections' STARE, an Ironclad option for an Ironclad) | | 4 | 0.0% | 4 |
| no option choice | Fake Merchant (the stall is its own screen) | 1 | 755 | 1.7% | 0 |
| other mods | events from other mods' content | | 55 | 0.1% | |

So **94.8% of the visits with a recorded option (42,481 of 44,813) resolve with no help from the run**, 93.1% of all event visits.
Events by class (visits in the training fifths):

* `capture` (35): Room Full of Cheese 2,159, Brain Leech 2,092, Sunken Statue 1,670, The Legends Were True 1,640, Slippery Bridge 1,505,
  Potion Courier 1,085, Symbiote 1,071, Doors of Light and Dark 897, Sunken Treasury 887, Spiraling Whirlpool 835, Drowning Beacon 832,
  Byrdonis Nest 803, Trash Heap 796, Abyssal Baths 795, Aroma of Chaos 765, Wellspring 760, Crystal Sphere 732, Wood Carvings 722,
  Jungle Maze Adventure 706, Waterlogged Scriptorium 655, Infested Automaton 645, The Lantern Key 643, Bugslayer 632, Spirit Grafter 631,
  Field of Man-Sized Holes 621, Whispering Hollow 560, Amalgamator 553, Morphic Grove 516, Trial 506, Hungry for Mushrooms 504,
  Reflections 499, Battleworn Dummy 496, Unrest Site 328, Punch Off 309, War Historian Repy 166.
* `engine` (5): Dense Vegetation 721 (the Rest page's Fight), Colossal Flower 593, Grave of the Forgotten 479, Round Tea Party 472,
  Luminous Choir 136 (Offer Tribute).
* `inferred` (12): This or That 2,548, Self Help Book 2,470 (Read the Entire Book), Sapphire Seed 793, Tablet of Truth 758, Relic Trader 743
  (Top, Middle, Bottom are places on the page), Ranwid the Elder 680, Lost Wisp 578, Stone of All Time 575, Tea Master 564, Welcome to Wongos 421,
  Endless Conveyor 390, Zen Weaver 358.
* dynamic (4): The Future of Potions 931, Colorful Philosophers 580, Doll Room 564 (373 reach the second page), Tinker Time 447.

Rolled pages, in decisions: The Future of Potions 931 (the record's key is `POTION` for every potion; which potion is not named),
Tinker Time 894 (a card type, then a rider, two of three and two of the type's riders, in a rolled order), Colorful Philosophers 580 (one
pool per other character, at most three, rolled), Doll Room 373 (the dolls offered are a shuffle from the event's own stream), Reflections 1.

## 3. Checks

* **The captures** (the test): every unlocked, non-proceed title of every captured page resolves to the index it has in the capture,
  except the rolled titles (the dolls, Tinker Time's types), which have none. 52 events. One `-opt` capture shows the first page again
  after the choice (Dense Vegetation's Rest) where the engine and the record have the fight page; see the owner's list.
* **Against the probe's own search** (`validate_table.py` in the probe folder): on the same 400 runs, wherever the clone search found one
  path with no deficit (894 of 1,373 searched visits, 1,086 keys), the table's index is the search's: **1,086 of 1,086**, 0 disagree,
  0 unknown. The events the search cannot settle (the option opens a card selection the search does not answer: Amalgamator, Aroma of
  Chaos, Battleworn Dummy, Field of Man-Sized Holes, Self Help Book, Symbiote, Wood Carvings, Potion Courier, Reflections) rest on the
  capture titles. The `inferred` and `engine` rows are checked by this search where it applies: This or That 73 of 73, Sapphire Seed 30
  of 30, Lost Wisp 21 of 21, Tea Master 20 of 20, Jungle Maze 19 of 19, Stone of All Time 13 of 13, Colossal Flower 26 of 26,
  Grave of the Forgotten 11 of 11, Welcome to Wongos 16 of 16, Ranwid 5 of 5, Round Tea Party 4 of 4.
* **Not checked by it**: Relic Trader's places (15 searched visits, all tied; the replay shows 0 gold or HP differences in 17
  table-resolved floors), Self Help Book's third page, Zen Weaver's second price (one visit).

## 4. The whole-run replay with the table

The probe's 400-run sample (the 400 training-fifth runs with the smallest hash; same library, same replayer, same cleanliness test: no
screen the copy could not answer the human's way, no event option that was ambiguous or did not reproduce the record, no gold or HP
difference after an earlier floor). The event step takes the table's index; a visit with a key the table cannot resolve goes the
old way (clone search) for that visit and is counted as unresolved; a relic-titled choice (a doll) is taken by trying each option on a
clone and keeping the one that gives that relic.

| | before (clone search) | after (table) |
| --- | --- | --- |
| replays in step to the last floor | **111 / 400** (152 if ties between equal options are allowed) | **184 / 400** (no ties) |
| ended by an ambiguous event option | 122 | 25 (the visits with a rolled page) |
| ended by an event option that did not reproduce the record | 53 | 0 |
| ended by a different event than the record's | 14 | 17 |
| ended by a gold or HP difference after a floor | 71 (28 shop, 16 "?", 14 fight, 10 elite, 3 other) | 129 (48 shop, 37 "?", 24 fight, 15 elite, 5 other) |
| ended on an event screen the replay cannot answer (Crystal Sphere's minigame: its clicks are not in the record) | 4 | 10 |
| rest option Hatch (the record does not name its relic) | 7 | 11 |
| not clean in all | 289 | 216 |

More replays reach the later floors, so the later causes (shops, fights) now end more of them. The table's index was never illegal on the
copy (0 of 1,591 table steps), and the table-resolved floors with a gold or HP difference are concentrated where the event is a fight or a
minigame, which the replay does not play (Crystal Sphere 22 of 24 floors, Battleworn Dummy 12 of 13, The Lantern Key 12 of 22, Dense
Vegetation 9 of 25, Punch Off 6 of 13); the events that are one choice show 0 or 1 in 15 to 76 floors each (This or That 1 of 76, Self
Help Book 1 of 73, Symbiote 1 of 37). The four rolled-page events are what remains: 62 visits in the 400 runs take the old way (Future of Potions 30,
Tinker Time 16, Colorful Philosophers 15, one visit with another mod's pool key) and they end the 25 replays counted as ambiguous; the 17 "different event" and the Hatch rests are the copy's profile and rest modelling, not the table's.

## 5. For the owner

1. **Rolled pages** need one thing from the engine: the open event page's options in display order (as option names), exported with the
   run info (the engine holds them as `EventRandomOffer`, the Colorful Philosophers pool order, `DollRoomOffer` and the trade stock; none
   is in the native lists today). With it `resolve(..., {"offered": [...]})` answers the 2,328 visits (5.1%, 2,775 decisions): Future of
   Potions 931 visits (and the potion traded must be read from the potion belt, the key does not name it), Colorful Philosophers 580, Tinker
   Time 447 (894 decisions), Doll Room 373. The replay's clone trick already settles the dolls (the relic is named); the other three
   need the export.
2. **Names to make `exact`** (one captured title or a decompile-written key list for each): This or That (PLAIN, ORNATE), Self Help Book (READ_ENTIRE_BOOK),
   Sapphire Seed (EAT), Lost Wisp (CLAIM), Stone of All Time (LIFT), Tea Master (EMBER_TEA), Ranwid (POTION, RELIC), Wongos (the three tags),
   Zen Weaver (the two prices), Luminous Choir (OFFER_TRIBUTE), Relic Trader, Tablet of Truth (the Continue page), the conveyor's other dishes, and the
   three events no capture covers (Colossal Flower, Grave of the Forgotten, Round Tea Party; 1,544 visits). They are validated by the
   search above where it applies, so this is for certainty, not for coverage: 8,369 visits sit on `inferred` or `engine` rows.
   `scripts/generate_event_options.py` extended to write option keys per page, or a live capture of each locked slot unlocked, would do it.
3. **Stale captures**: `JungleMazeAdventure-options.json` holds Endless Conveyor's page, and `DenseVegetation-opt1.json` shows the first page again
   after Rest (the engine and the record say the page after Rest offers only the fight). Re-capture both (`capture_event.py`, live game).
4. **Not the table's**: the Crystal Sphere minigame (the record does not list the clicks; 732 visits), the fights inside events
   (Battleworn Dummy, Lantern Key, Punch Off, Dense Vegetation's Rest) and 55 visits of other mods' events.

## 6. What is unverified

* A name matched by its words is `exact` for the page it is on; the record's page names (`INITIAL`, `ALL`, `HOLD_ON_3`, ...) are taken as
  the record gives them, and an option name found on several pages must give the same index there (the resolver returns None when it
  does not; none does today).
* Relic Trader's TOP, MIDDLE, BOTTOM are read as places 0, 1, 2 with three relics offered; every visit in the record is at a point where
  the player held five or more relics (checked), so fewer than three offers does not occur in the sample.
* The Wongos and Zen Weaver prices and the other gold gates are read from the engine's constants, not the game's; the captures agree
  at the one level they were taken (Ascension 8).
* The record's page names for the pages the captures do not show (the later Hold On pages, Tablet of Truth's 3rd and 4th Decipher) are
  taken from the record; the engine's page counters were not compared.
* Mod events (55 visits) and the two Colorful Philosophers keys of another mod's characters are not in the table.
