# The human deck prior (`rl/human_prior.py`)

Written 2026-11-05 for milestones 2 and 3 of the strategy memo (sections 3.1 to 3.3 and 3.6). Source of the human record: Spire Codex
public run export for v0.107.1. Aggregates only: no run is named, no cell under 20 runs is shown (every cell below is far over it).
The shards, donor maps and fitted weights are in the referee's `human-prior` folder, not in any repository.

**v1 (2026-11-05) extends this file**: the real map of each run replaces the donors, routes are a fifth kind, and shop prices the copy vouches for are in the fit; see the section "v1" at the end. The text below is v0 as it was written.
**v2 (2026-11-11)** gives the net the whole act map as a grid and a map tower (section "v2: the map grid", the last section): it matches v1 on every kind with the regularisation it needs, and does not beat it.

## What the record teaches

The corpus holds, for 14,974 finished Ironclad Ascension 10 solo standard runs (11,961 vanilla runs outside the exam fifth), the floor
by floor HP, gold, cards gained, removed, upgraded, enchanted, relics and potions picked up, and for each reward screen the offers.
That is enough to rebuild the state a human stood in and to read four kinds of decision, each with the options the screen listed:

| kind | options | the human's choice | built |
|---|---|---|---|
| card reward | the offered cards, and a skip | the card taken, or skip | yes |
| ancient (an act's first room) | its three relics (the record names them by title key) | the relic chosen | yes |
| rest site | heal, one smith option per distinct upgradable card, and the relic-given extras (dig, lift, cook, kindle, clone) | heal / smith / other, and the smith's target card | yes |
| shop | each offered card, relic and potion: leave it or buy it | the purchases (cards from `cards_gained`) | yes, **left out of the fit** (prices unverified, below) |

Not built: events (the record names the chosen option's title only), routes (the record has no map), potion rewards, card removal
targets at the shop, and the rest site's Hatch option (the relic that gives it is not named in the record).

## The build (`python rl/human_prior.py donors`, then `build`)

* **Reads** the corpus with `human_starts.is_exam` before any worker sees a line: a run in the exam fifth (the ids file united with hash
  bucket 0) is dropped in the reading process, a worker handed one stops, and `assert_no_exam` runs on every shard's run ids. Fifth
  `k` is the hash bucket 1 to 4; fifths 1 to 3 train, fifth 4 validates.
* **Rebuilds** the state before each decision with the `human_starts` rules (`apply_point`, `apply_potions`, `relics_at`): deck with
  upgrades and enchantments, relics held, potions held, HP, max HP and gold. A card reward uses the HP and gold after the room, a rest
  site those of the floor before, a shop the gold it was entered with, an act's first room the start state (64 of 80 HP, 99 gold, as
  the run judgement's own Act 1 start) or the HP carried in. The act's boss is the one the run met; a run that died before it gets one
  of the region's three bosses by a fixed hash (flagged `boss_guessed`: 16 in 100 of the card rewards, 25 in 100 of the ancient screens).
* **Features** are the run judgement's own: `run_train.features.apply_option` and `encode_afterstate` give the 1,855-wide base vector of
  every option's afterstate (a taken card is in the deck, a smith card upgraded, a heal applied with its Dream Catcher rider, a
  shop buy paid for); FightNet of generation 11's run judgement fills the 8-wide fight block H, so a row is the judgement's 1,863-wide
  input. Rows are stored compactly (counts as bytes, the dense part without the map block) in shards: 1,790,515 rows.
* **The missing map.** The record has none. Each decision draws a donor from a table built from the run judgement's own self-play
  states of generations 11 and 9 (`donors`): for each region and row of the act, up to 64 states' map-ahead block and distance to the
  boss (a row of the act is a floor, as the exam reads it). The block is redrawn every epoch at fit time and from a fixed hash at
  validation; the fight block H is computed once, with the first donor of the key. Region and row keys: 67; Act 3's last rows have too
  few self-play states, so 2 keys borrow the nearest row of their region (no decision falls back to a zero map).
* **Splits and merged screens.** A card-reward point listing six (nine, ...) offers is two (three, ...) merged screens of three and is
  split in order (11,502 decisions, flagged); a point with five, seven or other offers is dropped; a screen with two picks is dropped.
* Runs: 4 `spawn` processes at BelowNormal priority, 272 s for 11,961 runs; donors 162 s in one process.

### Counts against `counts.json`

The screens seen per kind and act equal the reference's in all twelve cells (card rewards 90,204 / 41,974 / 19,321 = 151,499; rests
31,845 / 15,504 / 8,447 = 55,796; ancients 11,954 / 6,491 / 3,441 = 21,886; shop visits 9,297 / 6,159 / 3,927 = 19,383). Decisions built
(over fifths 1 to 4):

| kind | Act 1 | Act 2 | Act 3 | total | note |
|---|---|---|---|---|---|
| card reward | 90,347 | 43,856 | 19,826 | 154,029 | 151,499 screens, less the drops below, plus 11,502 split halves |
| ancient | 11,882 | 6,115 | 3,381 | 21,378 | 97.7 in 100 of the screens |
| rest site | 31,276 | 15,262 | 8,280 | 54,818 | 98.2 in 100 of the points |
| shop offers | 120,763 | 80,105 | 51,399 | 252,267 | one buy-or-leave per offer, 19,123 visits (98.7 in 100) |

By fifth the card rewards are 38,251 / 39,330 / 38,937 / 37,511 (rests 13,722 / 14,022 / 13,816 / 13,258; ancients 5,310 / 5,465 / 5,417
/ 5,186). 31,507 card, 11,941 rest, 3,931 ancient and 54,805 shop decisions are from winning runs.

### Drops, and why (decisions; none is patched)

| kind | reason | count |
|---|---|---|
| card | screen with 5, 7 or more offers that is not a multiple of three (merged screens whose halves cannot be told apart) | 1,948 |
| card | a relic held that the copy has no id for | 1,518 |
| card | a deck card or enchantment the copy does not know | 221 |
| card | an offered card the copy does not know / hp or gold missing | 47 / 14 |
| ancient | an option that is not a relic the copy knows | 368 |
| ancient | a relic held unknown to the copy / a deck card unknown / hp or gold missing / no single chosen option | 105 / 20 / 14 / 1 |
| rest | a relic held unknown to the copy | 635 |
| rest | choice Hatch (the relic is not named in the record) | 219 |
| rest | a deck card unknown / Cook not offered by the relics held / Godremove / Starfusion / Nothing / Fuse | 95 / 4 / 15 / 5 / 4 / 1 |
| shop | a relic held unknown to the copy | 234 (visits) |
| shop | a deck card unknown to the copy / an offered card or relic unknown | 26 / 4 |

573 runs (4.8 in 100) whose deck replay disagrees with the run's final deck are kept in the shards, flagged `replay_mismatch`, and left
out of the fit and of every agreement figure (28,380 decisions of the training fifths).

### Rebuild checks (against the record's own fields)

* Rest sites: the HP walked in plus the HP healed equals the HP after, at 100.0 in 100 of 55,066 sites.
* Shops: the gold entered with (gold after plus gold spent) equals the previous floor's gold at 95.4 in 100 of 19,123 visits.
* **The run judgement itself, scored on the rebuilt validation decisions, takes the human's card 30.4 times in 100** (the memo's
  figure for today's player on the exam fifth is 31; a coin is 25.1). That agreement on an independent fifth is the end-to-end check of the
  rebuild and the features.

### Shop prices (`price_check`)

Prices are rebuilt from the copy's rules: a card's base by rarity (50 / 75 / 150, colourless x1.15) jittered by 0.95 to 1.05; a relic's by
rarity (175 / 225 / 275, shop 200) jittered by 0.85 to 1.15; a potion's (50 / 75 / 100) by 0.95 to 1.05; the Courier and Membership
Card take their share; the card-removal service is 100 plus 50 per earlier removal (Inflation, Ascension 10); one coloured card is on sale
at half and the record does not say which, so the check lets one bought coloured card be halved. The record lists the cards left on the
shelf in `card_choices` and the cards bought in `cards_gained`.

Rebuilt spend against the record's gold spent, over the 19,123 visits built (17,638 with a purchase):

| measure | share of visits |
|---|---|
| within 10 gold, all visits | 76.9 in 100 |
| **within 10 gold, visits with a purchase** (the share the memo's gate reads) | **75.4 in 100** |
| ... without the sale allowance | 53.3 in 100 |
| ... no relic bought (10,737 visits) | 98.4 in 100 |
| ... a relic bought (6,901 visits) | 39.6 in 100 |
| inside the rules' own jitter band, visits with a purchase | 96.2 in 100 (relic visits 92.4, others 98.6) |

**Under 9 in 100 of 10, so the shop rows are written, marked `price_unverified` and left out of the fit.** The misses are the relic
jitter (plus or minus 15 percent of 175 to 275, up to 41 gold, which no rule rebuilds); where no relic was bought the rules hit the
record to within 10 gold 98.4 times in 100. An informational fit with the shops in (below) is kept apart.

## The fit (`python rl/human_prior.py fit`)

* **Model.** `PriorNet`: `RunValueNet`'s embeddings and two 256-wide layers on the judgement's 350 dense inputs plus the 104 embedded
  counts, with one logit in place of the floor and end-HP heads. The same inputs and afterstates as the judgement, so it drops into the
  chooser.
* **Loss.** A softmax over the options of a decision against the human's choice. Card rewards and ancients: one softmax (a card reward
  includes the skip). Rest sites: the kind's marginal likelihood (the options of the chosen kind against all) plus, when the record names
  the smith's card, a second softmax over the smith options. Shops: buy-or-leave per offer (only in the informational fit).
* **Training.** AdamW (learning rate 3e-4 from the judgement's trunk, 1e-3 from scratch; weight decay 1e-4), 256 decisions a batch, kinds
  mixed, donors redrawn every epoch, dropout 0.1, gradient clip 1; early stopping at patience 3 on the sum of the mean validation
  negative log-likelihoods of the fitted kinds (fifth 4, fixed donors). Variants: initialised from generation 11's run judgement or
  from scratch, each with decisions of winning runs weighted 1x or 2x. Trained on 108,115 card rewards, 38,339 rests and 15,052 ancients
  (fifths 1 to 3, less the replay mismatches); validated on 35,229, 12,393 and 4,873 (fifth 4).
* **Wall time.** 266 s for the four variants on one RTX 3080 (4 to 5 s an epoch).

| variant | best epoch | validation score | card top-1 | ancient top-1 | rest kind top-1 | smith target top-1 |
|---|---|---|---|---|---|---|
| **from generation 11, winners 1x (kept)** | 17 | **3.962** | **51.6** | 54.4 | 75.6 | 48.6 |
| from generation 11, winners 2x | 14 | 3.987 | 51.0 | 54.4 | 75.0 | 48.7 |
| from scratch, winners 1x | 8 | 3.969 | 50.7 | 53.8 | 75.8 | 49.0 |
| from scratch, winners 2x | 10 | 3.987 | 50.9 | 54.4 | 75.9 | 48.8 |

Validation favours the generation 11 start with no winner weighting (the lowest score; the four are within 0.025 of each other, the
scratch fit as good on top-1). The 2x weighting is not kept.

## Agreement with the human choice on the validation fifth (`agreement`)

Top-1 in 100 on fifth 4 (never the exam fifth), the prior and the generation 11 run judgement's own choice (argmax of expected final
floors, W = 53) on the same decisions and donors. A coin over the options is 25.1 for card rewards.

| measure | n | prior | run judgement (gen 11) |
|---|---|---|---|
| **card reward, all** | 35,229 | **51.6** | 30.4 |
| ... Act 1 / Act 2 / Act 3 | 20,996 / 9,888 / 4,345 | 50.9 / 50.8 / 56.9 | 33.6 / 27.3 / 21.7 |
| ... human took a card | 26,743 | 51.6 | 36.3 |
| ... human skipped | 8,486 | 51.8 | 11.6 |
| ... whole screens only (no split halves) | 32,548 | 51.9 | 30.8 |
| skip rate: predicted / human | | 22.6 / 24.1 | 10.6 / 24.1 |
| ancient, all | 4,873 | 54.4 | 30.2 |
| ... Act 1 / 2 / 3 | 2,763 / 1,378 / 732 | 55.2 / 53.5 / 52.9 | 29.4 / 30.3 / 33.1 |
| rest kind (heal / smith / other), all | 12,393 | 75.6 | 59.3 |
| ... Act 1 / 2 / 3 | 7,209 / 3,406 / 1,778 | 75.6 / 73.9 / 78.6 | 58.9 / 59.3 / 60.7 |
| smith target, all | 6,893 | 48.6 | 22.3 |
| ... Act 1 / 2 / 3 | 4,158 / 1,747 / 988 | 54.4 / 41.0 / 37.6 | 25.8 / 18.7 / 14.0 |
| smith target, in the prior's top three | 6,893 | 83.4 | 47.0 |

The bar (45 in 100 on card rewards; under 40 would have meant a wrong rebuild) is cleared at 51.6. Where the judgement and the humans part
is clearest in the skips: humans skip 24 in 100 of screens (15 / 32 / 47 in 100 by act) and the judgement 11.

### Informational: with the shops in the fit

`fit --with-shops --name pi_h-v0-shops` (generation 11 start, winners 1x, best epoch 13, 133 s): card rewards 50.7, ancients 54.2, rest
kind 75.4, and shop buy-or-leave 83.7 in 100 against a majority rate of 83.1 (16.9 in 100 of the available offers are bought; negative
log-likelihood 0.401 against the base rate's 0.454). Fitting the shops costs the other kinds about one point and the shop signal is weak,
so the kept prior leaves them out until prices are verified.

## Event rows (`python rl/human_prior.py build --events`)

An event is the decision kind the record names by an option key (`<EVENT>.pages.<PAGE>.options.<OPTION>.title`) and the copy takes by an
index. `rl/event_keys.py` joins them (`docs/rl/EVENT_OPTIONS.md`); with `--events`, `build` adds a fifth kind, `event` (kind 4), and
the default build is unchanged (no event row, the counts contract and the other four kinds as before). The fit does not train the
kind yet (`kinds = card, ancient, rest[, shop]`); the rows are in the shards for the next step.

* **A decision** is one option page of an event visit. A visit with a series (Abyssal Baths: Immerse, Linger, Linger, Exit; Tablet of
  Truth; Trial; the conveyor; Slippery Bridge) gives one decision per page. The state is the one walking in (the floor before's HP and
  gold, the deck and relics rebuilt as for the other kinds).
* **The options** are the page's displayed indices (`shown` in the table: the first page's captured options, later pages' listed in the
  builder) that the state leaves on offer; each is the run judgement's own `event.option` afterstate, `(event id, index, step)` with the
  step the page's place in the visit (cut at 3, as `_live_event` does), through `apply_option` and the same encoder as every other kind
  (the event's own effect is the judgement's learned pending input, not a modelled one). The chosen option is the index the table gives
  the recorded key. Locks come from the engine's masks (`event_keys.available`): gold prices (Zen Weaver 125 and 250, Tea Master 50 and 150,
  Wongos 100, 200 and 300, the Scriptorium's quill and sponge), a potion held (Stone of All Time's Lift, Ranwid), a card type in the
  deck (Self Help Book's three pages, Symbiote, Stone's Push), a card to upgrade (Aroma of Chaos), an exhaust card (Grave of the
  Forgotten) and a relic to give (Ranwid); only the first page is gated. Three options are locked by a price the event rolls
  from its own stream (Luminous Choir's tribute, Crystal Sphere's divination, Whispering Hollow's exchange): they are listed and the decision
  carries the flag `gate_unknown` (bit 32).
* **Dropped and counted** (the decisions of the training fifths and the validation fifth, by act 1 / 2 / 3): a page whose options are
  rolled per visit, which needs the offered list (381 / 1,203 / 1,146: Future of Potions, Tinker Time's two later pages, Colorful
  Philosophers, Doll Room's second page); a page with one option (369 / 298 / 438: the fight pages, Tinker Time's first page); an event of
  another mod (13 visits); a name the table does not place (3); and the points the other kinds also lose (a relic held that the copy has no
  id for: 231 / 189 / 132 visits; a deck card unknown: 20 / 34 / 18).
* **Built over the training fifths and the validation fifth** (11,961 runs, 44,867 event visits with a recorded option): **49,243 event
  decisions**, 28,566 / 13,310 / 7,367 in Acts 1 / 2 / 3. By fifth: fifths 1 to 3 (the fit's) 37,314 (21,644 / 10,063 / 5,607) and the
  validation fifth 11,929 (6,922 / 3,247 / 1,760). 43,244 of them have 2 options, 5,886 have 3 and 113 have 4. 854 carry `gate_unknown`
  and 3,788 `replay_mismatch` (a run whose deck replay disagrees with its final deck; left out of a fit as for the other kinds).
* **Not in the rows**: what an event pays or costs (the afterstate carries the slot, as the run judgement's own event input does); the rolled
  pages (they wait on the offered list: `docs/rl/EVENT_OPTIONS.md`, section 5); a gate on a later page of a series.

`tests/python/test_human_prior_events.py` pins the switch (off by default), the option sets, the steps of a series, the conveyor's pages, a
locked option, the drops by reason, the rows' shape and a shard's round trip.

## Caveats

* Humans' choices assume human fight skill (the average human wins 11 in 100 here); the prior is what a human picks, not what works for
  our player. The chooser blends it with the judgement (`V + lambda log pi_h`), decided on the hindsight labels.
* The map is a donor's: the prior learns nothing from a route, and at play the true map enters instead. Donors come from the AI's own
  runs (generations 11 and 9), which are shallow in Act 3 (two keys borrow a neighbouring row).
* 16 in 100 of the card rewards (25 in 100 of the ancient screens) have a guessed boss (the run died first); the boss enters through the one-hot and H.
* The fight block H is computed with the first donor's distance to the boss, not the redrawn one.
* Relic effects the afterstate does not model (a relic with an effect on pickup) are the judgement's own approximations.
* The smith target is the first card in `upgraded_cards`; a site with two choices (Miniature Tent) uses the first.
* Not done here: the exam's `boss_entry_exam.py agreement` extension to ancients, smith targets and shops (the exam fifth is read once
  per frozen candidate by the orchestrator); the removal target at the shop; shops in the fit (waits on the prices); routes; the event kind in the fit.

## Rerun

```
python rl/human_prior.py donors                      # donor table (one process)
python rl/human_prior.py build --procs 4 --clean     # shards, counts.json (with the counts contract), price check
python rl/human_prior.py build --procs 4 --clean --events   # the same plus the event rows (kind 4)
python rl/human_prior.py fit                         # GPU (the venv with torch and CUDA); pi_h-v0/ weights.npz and fit.json
python rl/human_prior.py agreement --model <folder> --baseline <run_value.npz> --out <file>
```

`tests/python/test_human_prior.py` pins the exam rule, the option sets, a decision's round trip from record to shard to fit step (and
the numpy export against the torch net), the counts contract and the price rules; it runs in seconds.

## v1 (2026-11-05): the true map, routes, shop prices the copy vouches for, and a win-conditioned variant

Written 2026-11-05 on branch `feat/human-prior-v1`. Everything above is v0 and stays as it was written; where v1 changes a statement, this section says so.
The files are `rl/human_prior_replay.py` (new: the copy's half), `rl/human_prior.py` (extended; v0's commands and shard format still work),
`tests/python/test_human_prior.py` (extended). Source of the record: Spire Codex public run export for v0.107.1; aggregates only, no cell under
20 decisions is shown (the smallest cell below is far over it), the exam fifth is never read (`assert_no_exam` on every chunk, shard, fit and
agreement), the shards, truth files and fits stay in the referee's `human-prior` folder.

### What v1 changes, in plain words

A human run's seed is a recipe card the copy can cook again: from it the copy draws each act's map, the shelves and prices of every shop, and
the card offers (the seed-replay probe, `docs/diagnostics/2026-11-05-human-seed-replay.md`). v0 gave every decision a *donor's* map, a stand-in
taken from the player's own self-play, so it could learn nothing about routes. v1 puts the **real map of the human's own act** into every row
(what lies ahead of the room they stood on, and how far the boss is), adds a **fifth kind of decision, the route** (which room to go to next, among the
rooms the node they stood on leads to), and prices the **shop offers with the copy's own prices** where a replay of the run stays in step with
the record, so those shops are in the fit. Event rows come from the event option table (`build --events`, section "Event rows" above); v1 does not fit them, and with `--truth` they carry the true map like every other row. Potion rewards are still not built.

### The true map (`python rl/human_prior_replay.py run`, then `build --truth`)

* **Regenerated on the copy**: Ironclad, Ascension 10, the record's seed; Act 1's map at start, Acts 2 and 3 by the copy's own act transition, nothing
  played. Two things the record does not say the copy is told: a Spoils Map card in the deck before Act 2 is drawn (when the record gained one in
  Act 1; the engine then draws a different Act 2 map) and Winged Boots' three free steps (after the floor they were picked up on). The copy's library is the
  lockstep worktree's build at the engine source of fork main (native API 39).
* **The identity check, per run and act**: the record's route (the room types it went through) must be a path of the regenerated map. A run with a
  Golden Compass (its acts' maps are one column) is left out and counted (`golden_compass`); a run whose route is not a path is `map_mismatch` and left
  out whole; both are counted, none is patched. 11,961 usable runs outside the exam fifth were read:

| | runs |
|---|---|
| read (fifths 1 to 4) | 11,961 |
| **with a true map (built from)** | **11,764** (fifths 1 / 2 / 3 / 4: 2,952 / 2,948 / 3,016 / 2,848) |
| left out, Golden Compass | 126 |
| left out, `map_mismatch` | 71 (first failing act: Act 1 62, Act 2 8, Act 3 1; 2 hold Winged Boots) |
| left out, map could not be drawn | 0 |

| act | reached | tested (no compass) | route is a path of the map | the only path | region equals the record's | boss equals the record's (where reached) |
|---|---|---|---|---|---|---|
| 1 | 11,961 | 11,959 | 11,895 (99.5 in 100) | 11,666 | 11,867 | 8,636 / 8,791 |
| 2 | 6,528 | 6,402 | 6,364 (99.4) | 6,138 | 6,364 | 4,347 / 4,421 |
| 3 | 3,452 | 3,381 | 3,363 (99.5) | 3,282 | 3,363 | 2,844 / 2,937 |

  Winged Boots were held in 238 of the 11,764 runs with a map (the matching path uses the fewest free steps the record needs; 156 / 60 / 32 acts use one).
  The Spoils Map reading (Act 2 first) matched in 634 Act 2 maps. The 71 mismatches and the region and boss misses look like the player's unlocks (the copy
  assumes a fully unlocked profile); that is inferred, not shown.
* **Where the human stood.** The record names a room by its type only; the node is read from the path. When several paths of the same room types fit, the node is
  not known (521 runs have such a step somewhere): a decision on an unknown node is dropped and counted, not guessed (card 718, rest 75, shop 17, route 1,151).
  That keeps only the steps the record's later rooms pin down, which favours steps whose rooms differ in type.
* **The row.** The base vector's map block (25 numbers: per room type the fewest, mean and most on the paths to the boss, and the nearest one; the log of the
  number of paths) and the distance to the boss are those of the act's true map at the node the human stood on (`run_train.features.map_ahead`, the same
  function the player uses), and the fight block H is computed from that vector, so a row is the player's input at that state. v1 shards carry the block (113
  stored columns, `rows x 113`, `N_DENSE_V1`); v0 shards (88) still load, with donors.

### Route decisions

At each step of an act with more than one room to go to, one decision: the state after the previous floor (HP, gold, deck, relics, potions the human walked
out with), and as options the rooms the node leads to, each as the player's afterstate for a map step (`apply_option` with the map option, as the `vrun` macro scores
routes: the room's type pending, one floor on, the position moved, so the map block ahead of it is that room's). The chosen row is the room the record went to next.

| | Act 1 | Act 2 | Act 3 | all |
|---|---|---|---|---|
| steps of the record (a room entered after the first) | 166,498 | 81,680 | 46,288 | 294,466 |
| steps with only one room to go to (no decision) | 103,194 | 50,305 | 29,474 | 182,973 |
| **route decisions built** | **60,988** | **29,689** | **15,971** | **106,648** |
| ... of them the rooms differ in type | 43,781 | 21,038 | 11,395 | 76,214 |
| dropped: node unknown (several paths fit) | 435 | 564 | 152 | 1,151 |
| dropped: after Winged Boots were picked up | 1,304 | 670 | 454 | 2,428 |
| dropped: a relic or a deck card the copy has no id for | 577 | 452 | 237 | 1,266 |

Fifths 1 / 2 / 3 / 4: 26,731 / 27,175 / 26,883 / 25,859 decisions (80,789 train, 25,859 validation); 21,055 are from winning runs. About 9.1 route decisions a run,
6.5 where the rooms differ in type (the probe said 6.7). Winged Boots let the player step to any room of the next row, which the option set does not model, so
the floors after their pick-up are left out; a free step to a room that is not a child never appears as a choice.

### Shop prices

The probe's replay (`HistoryReplayer` with the changes that probe made) walks the copy through each run floor by floor, fights resolved not played, and at each shop
records the copy's shelf with a price on every item. A visit is graded:

* **strict (the probe's criterion, and the only grade in the fit)**: the replay was *clean* on entry (nothing before had left the record's path: no event option
  ambiguous or unreproduced, no gold or HP difference), the shelf equals the record's (cards left plus cards bought, relics, potions), and what the human bought adds
  up, to the gold, to what the record says was spent; no price-changing relic (Membership Card, Courier, Smiling Mask) was bought in the visit. Its rows lose
  `price_unverified` and carry the copy's price of each item (the sale item at its sale price), so which card was on sale is no longer guessed.
* **shelf only**: the same evidence (shelf equal, prices add up) but the replay had already left the record's path. The shelf coming out equal says the shop stream was in the
  right place; these rows keep `price_unverified` (they are out of the fit) and carry `F_SHELF_ONLY` (flag bit 64) so a later fit can take them (`_bool_unavailable(..., shelf_only_ok=True)`).
* otherwise the rules' price and `price_unverified`, with the reason counted.

| grade | visits | share of the 18,681 built | Act 1 | Act 2 | Act 3 |
|---|---|---|---|---|---|
| **strict (price verified)** | **7,249** | **38.8 in 100** | 6,133 of 9,036 (67.9) | 1,034 of 5,902 (17.5) | 82 of 3,743 (2.2) |
| shelf only | 5,863 | 31.4 | 1,734 | 2,678 | 1,451 |
| shelf differs | 4,705 | 25.2 | 929 | 1,851 | 1,925 |
| bought items' prices do not add up | 113 | 0.6 | 24 | 44 | 45 |
| price-changing relic bought | 294 | 1.6 | 104 | 126 | 64 |
| an offer the copy's shelf lacks | 116 | 0.6 | 17 | 67 | 32 |
| no replay of the visit | 341 | 1.8 | 95 | 102 | 144 |

The verified share falls by act because the clean path ends early (the median clean path ends at floor 7, probe section 3.2): most Act 1 shops are verified and almost
no Act 3 shop is. Among clean visits with a matching shelf and a purchase, the prices added up to the gold in 7,721 of 7,728. By rows: 94,237 of 246,362 shop rows are verified; 48,452 of
the training fifths' rows are in the fit (the rest are unavailable offers, replay mismatches or unverified); 15,630 rows validate. The rules' price check of v0 (75.4 in
100 within 10 gold) is still computed and reported in `counts.json` for comparison.

### The fit (`fit --decisions decisions-v1 [--win-input]`)

The v0 recipe unchanged: generation 11's run judgement as the start (`gen11`), decisions of winning runs weighted 1x, AdamW 3e-4, batches of 256 decisions of mixed kinds, dropout 0.1,
gradient clip 1, early stopping at patience 3 on the sum of the mean validation negative log-likelihoods of the kinds fitted (now card, ancient, rest kind, smith target,
**shop and route**), fifth 4 validating. No donors are drawn: the map block is in the row. One run each on one RTX 3080.

| | v1 | v1-win |
|---|---|---|
| decisions trained on (card / ancient / rest / shop / route) | 105,624 / 14,774 / 37,413 / 48,452 / 74,988 | the same |
| decisions validated on | 34,323 / 4,775 / 12,090 / 15,630 / 24,298 | the same |
| best epoch (of the epochs run) | 13 (16) | 15 (18) |
| validation score | 4.917 | 4.912 |
| wall time (data load 22 s included) | 171 s | 157 s |

The decisions of runs whose deck replay disagrees with the final deck (546 runs of the four fifths; 33,118 decisions in the training fifths) are left out of the fit, as in v0. Runs: 8,916 train, 2,848 validation (true-map runs of fifths 1 to 3 and 4).

### Agreement with the human choice on the validation fifth (fifth 4, `agreement`)

Top-1 in 100. **v0 with true maps** is v0's kept weights (`pi_h-v0/weights.npz`, trained with donors, never on routes or shops) scored on the v1 rows, i.e. with the true map
in place of the donor, on the same decisions as the other columns. v1-win is scored at its play setting (the input "this run won" is 1 for every row); the next column gives the
same weights with each decision's own run outcome as the input. The last column is generation 11's run judgement (argmax of expected final floors, W = 53) on the same rows.
Agreement is *not* how the variants are chosen (that is the head-to-head and the labels).

| kind | n | v0, true maps | v1 | v1-win (input 1) | v1-win (input as recorded) | run judgement (gen 11) |
|---|---|---|---|---|---|---|
| **card reward, all** | 34,323 | **51.7** | **50.7** | **50.9** | 51.2 | 30.5 |
| ... Act 1 / Act 2 / Act 3 | 20,587 / 9,523 / 4,213 | 51.1 / 50.9 / 56.9 | 49.8 / 49.9 / 57.0 | 50.1 / 50.2 / 56.8 | 50.6 / 50.3 / 56.3 | 33.7 / 27.4 / 22.1 |
| ... human took a card / skipped | 26,087 / 8,236 | 51.7 / 51.9 | 48.8 / 56.7 | 48.4 / 58.7 | 49.4 / 56.8 | |
| ancient | 4,775 | 54.4 | 54.5 | 54.8 | 55.0 | 30.3 |
| rest kind (heal / smith / other) | 12,090 | 75.4 | 75.8 | 76.6 | 76.3 | 59.1 |
| smith target | 6,751 | 48.4 | 48.7 | 48.9 | 48.7 | 22.0 |
| shop, buy or leave (verified rows; 12,717 / 2,692 / 221 by act) | 15,630 | not fitted | 83.6 | 83.9 | 84.0 | not scored |
| **route, all** | 24,298 | 59.9 (not fitted) | **76.0** | **75.7** | 75.8 | 62.5 |
| ... Act 1 / Act 2 / Act 3 | 14,135 / 6,707 / 3,456 | 55.1 / 66.0 / 68.0 | 75.4 / 76.7 / 76.6 | 75.0 / 76.6 / 76.7 | 75.0 / 76.8 / 76.8 | 60.9 / 67.4 / 59.9 |
| ... rooms differ in type | 17,371 | 63.7 | 79.1 | 78.8 | 79.0 | 64.5 |

Mean negative log-likelihood: card 1.109 (v0) / 1.128 (v1) / 1.131 (v1-win); route 0.540 (v1) / 0.545 (v1-win); shop 0.404 / 0.399. Skip rate of the card prior 27.2 (v1) and 28.7 (v1-win)
against the humans' 24.0 (v0 22.5; the judgement 10.3).

**Route baselines** (the same 24,298 decisions; 17,371 where the rooms differ in type):

| baseline | all | rooms differ in type | Act 1 / 2 / 3 | mean NLL |
|---|---|---|---|---|
| a coin over the rooms | 46.2 | 48.5 | 46.3 / 45.9 / 46.0 | 0.791 |
| the room whose column is nearest the boss node's (ties share the credit) | 47.8 | 49.2 | 47.8 / 48.1 / 47.2 | n/a |
| the room type humans took most often when offered it, per act (rates from fifths 1 to 3) | 64.0 | 73.4 | 64.1 / 63.7 / 64.0 | 0.710 |
| **v1 prior** | **76.0** | **79.1** | 75.4 / 76.7 / 76.6 | **0.540** |

"Nearest to the boss" is read as the column: every room of a step is one row nearer the boss, so only the column tells them apart. The type-frequency baseline is added because
most of the humans' choice is which kind of room they want; the prior beats it by 12 points overall and 6 where the rooms differ in type, so it learns more than a type preference
(HP, gold, the deck, what lies ahead of each room). The player's own judgement takes the human's room 62.5 in 100, between the type baseline and the prior.

Winning runs apart (the decisions of the 6,687 card screens of winners; routes 4,658): card 56.2 (v1), 57.0 (v1-win), 57.0 (v0); route 78.0 / 78.1; rest kind 75.4 / 77.5 / 73.9. The win variant's gain over v1 is
0.2 on all cards and 0.8 on winners' cards at the play setting; the files carry both.

**Shops.** 15.8 in 100 of the verified offers are bought; always leaving is right 84.2 times in 100, above the model's 83.6 / 83.9 top-1, while its negative log-likelihood
(0.404 / 0.399) is under the base rate's 0.436: a weak signal, as in v0's informational fit (83.7 against 83.1, which had unverified prices).

### The win variant (`pi_h-v1-win/`)

The net takes one extra input per row, "this run won" (1 for the rows of winning runs, 0 otherwise), a column of 256 weights added to the first layer's pre-activation (it starts at zero, so
the net starts as the plain one). At play the input is 1: the prior then says what humans who won picked. **Weights format**: `weights.npz` has every array and the `meta` JSON of v0's, so the v0 loader
(`NumpyPrior`) loads it unchanged, plus one array `extra.won.weight` (256) and one `meta["extra_inputs"]` entry:

```
{"name": "won", "weight_key": "extra.won.weight", "added_to": "hidden1 pre-activation (times the input)", "value_at_play": 1.0,
 "folded_at": 1.0, "values_in_training": "1 for a decision of a run that won, 0 for one of a run that did not"}
```

`hidden1.bias` already includes `extra.won.weight * folded_at`, i.e. it is stored at the play setting, so a loader that ignores `extra_inputs` scores at the play setting without any change (checked: the v0 loader of
main and the extended one give identical scores on 3,000 validation rows of both v1 files). A loader that wants another input value `x` adds `(x - folded_at) * extra.won.weight` to the first layer's
pre-activation. `NumpyPrior(path)(dense, counts, extra=None)` does that (`extra` one number per row); `export_net` writes the format. `fit.json` lists `win_input: true`; `agreement.json` reports the prior at input 1
(`prior`), with each run's own outcome (`prior_input_as_recorded`) and on the winners' decisions (`prior_winning_runs`). The plain v1 file carries no `extra_inputs`.

### Caveats

* **Map and position.** 98.4 in 100 of the runs have a true map; the 197 left out (compass, mismatch) are 1.6 in 100 and the mismatches look like the player's unlocks (inferred). The node a human stood on
  is read from the room types of their route; steps where several paths fit are dropped, which favours steps whose rooms differ in type, so the route set has slightly more type choices than the record.
* **Routes are the human's, from a human's play.** The prior says where a human goes (HP, deck and the map ahead as inputs), not where it works for our player; the chooser blends it with the judgement and
  that is decided on the labels. Winged Boots runs lose their route decisions after the pick-up (2.0 in 100 of runs hold them).
* **Shops.** Verified for 38.8 in 100 of visits and for 2.2 in 100 of Act 3 visits, so the shop rows in the fit are mostly Act 1 and early Act 2; the buy-or-leave signal is weak (above). The shelf-only rows (31.4 in 100 of visits) are
  evidence-equal to the strict ones apart from the replay's clean flag and could be added; they are not in this fit. A price-changing relic bought mid-visit leaves its visit out.
* **v1 against v0 on the same cards**: 50.7 against 51.7. The true map instead of a donor's changes v0's own score on cards by 0.1 (51.7 here, 51.6 on v0's decisions with donors): the map block matters little for
  card rewards. The multitask fit with routes and shops costs about a point on cards (v0's informational fit with shops cost about one too); no variant was tuned.
* **The win variant** needs the flag at play to be 1: no run is known to have won at play, so the input is an assumption about the target, and 11 in 100 of the runs won.
* The fight block H uses the true distance to the boss (v0: the first donor's). The 16 in 100 card rewards with a guessed boss (the run died first) are as in v0.
* Not built: potion rewards, the removal target at the shop, Hatch; events are rows (kind 4) but not fitted.
* Ids in the shards: `kind` 0 card, 1 rest, 2 ancient, 3 shop, 4 event, 5 route; flag bits 1 split screen, 2 price unverified, 4 unavailable, 8 replay mismatch, 16 guessed boss,
  32 gate unknown (events), 64 shelf only (shops).

### Rerun (v1)

```
python rl/human_prior_replay.py run --procs 2 --resume             # the copy: truth-NNNNN.json.gz in <out>/truth (680 s on 2 processes for 11,961 runs)
python rl/human_prior.py build --truth <out>/truth --procs 2 --clean   # <out>/decisions-v1 and counts.json (577 s)
python rl/human_prior.py fit --decisions decisions-v1 --name pi_h-v1 --variants gen11 --winner-weights 1                # GPU, 171 s
python rl/human_prior.py fit --decisions decisions-v1 --name pi_h-v1-win --variants gen11 --winner-weights 1 --win-input
python rl/human_prior.py agreement --decisions decisions-v1 --model <out>/pi_h-v1 --baseline <run_value.npz> --out <out>/pi_h-v1/agreement.json
```

The tests (`tests/python/test_human_prior.py`, 16 now, seconds) add: the route-is-a-path check (unique, ambiguous, no path, Winged Boots' jumps, another seed's map), the runs a true map cannot stand for, a route decision's option set
and afterstates, the dropped cases, the price grades and flags and the fit's use of them, a v1 shard through a fit step and the numpy scorer, the baselines, the win input (its fold, the v0 loader's identical scores at play, the loss
reaching the weight) and the truth chunk join.

## Strong sets by achievement (2026-11-09, D55 addendum 15; branch `feat/best-players`)

Source of the record: Spire Codex public run export for v0.107.1 (the 2026-11-06 download, 840,381 lines). Aggregates only: no run is named, no
cell under 20 runs is shown (a set or cell under 20 runs is written "under 20"), the exam fifth is never read, the working files and shards
stay in the data store, never in a repository.

**Why.** The prior of generation 13 (`pi_h=2ebfe6d7`) is fitted on every solo Ironclad Ascension 10 run, the average player included. The
export's per-player key is blank on solo runs, so a player's strength cannot be read; a run's achievement can. Owner decision 2026-11-08:
fit the coaching on strong runs; never on co-op runs; other characters' runs are not used for any decision (they depend on the character).

### The sets (`python rl/strong_sets.py filter`, `rl/strong_sets.py`)

One pass over the export, Ironclad lines of Ascension 10 to 20 only, with the 11-06 filter's rules (standard mode, not abandoned, one player,
no modifiers, build v0.107.1, one line per run hash) and the fork's own (the three vanilla acts, as `human_starts.vanilla_acts`); the exam
fifth (the ids file united with hash bucket 0, `human_starts.is_exam`) out of every set, runs new to the pipeline joining the fifths by the
same hash rule. A written run drops the player token, the badges and the player id.

* **All-player pool** (the reproduction baseline): Ascension 10, any outcome.
* **S-win**: Ascension 10 runs that won (a subset of the pool).
* **S-high**: Ascension 11 to 20 runs, any outcome.

| | pool (Ascension 10) | S-win | S-high |
|---|---|---|---|
| runs | 14,725 | 1,608 | under 20 |
| wins | 1,608 (10.9 in 100) | 1,608 | under 20 |
| fifths 1 / 2 / 3 / 4 (4 is held out of every fit) | 3,671 / 3,648 / 3,791 / 3,615 | 409 / 410 / 404 / 385 | under 20 |
| runs by the last act entered, Act 1 / 2 / 3 | 6,638 / 3,854 / 4,233 | 0 / 0 / 1,608 | under 20 |

Of 67,354 Ironclad lines of Ascension 10 to 20: 33,095 have more than one player, 13,128 are abandoned, 2,213 are not standard mode, 475 carry
modifiers, 3,653 are in the exam fifth, 48 do not have the three vanilla acts. **S-high is empty for a plain reason**: the game defines ten
Ascension levels, and the export's Ironclad lines at levels 11 to 20 are about two hundred in any mode (a modded ladder), under 20 runs after the filter.
The set is not a stronger crowd, it is not a crowd. Its fit (under 20 training runs) is reported because the brief asks for it and means nothing.

Decisions the fit reads (card rewards, ancient screens, rest sites; replay mismatches out, as in v0), training fifths 1 to 3 / held-out fifth 4,
by act 1 / 2 / 3:

| | pool | S-win |
|---|---|---|
| card rewards, training | 79,451 / 37,202 / 15,901 | 9,249 / 8,616 / 6,396 |
| ancient screens, training | 10,493 / 5,256 / 2,725 | 1,025 / 979 / 1,018 |
| rest sites (kind), training | 27,518 / 12,914 / 6,662 | 3,349 / 3,116 / 2,773 |
| smith targets, training | 15,846 / 6,444 / 3,537 | 2,486 / 2,086 / 1,760 |
| card rewards, held out | 26,011 / 12,208 / 5,383 | 3,023 / 2,808 / 2,121 |

The wins' decisions fall evenly over the three acts (a winner plays all three); the pool's fall off with the runs that die early. **Published
cells under 20 runs**: the prior's table is read as 16 cells (card, ancient, rest kind, smith target; each for all acts and for Acts 1 to 3);
for the pool, S-win and S-win plus S-high none of the 16 is under 20 runs, in the training fifths or in the held-out fifth; for S-high all 16
are (and are never shown). The finer splits of the v0 table (take and skip screens, whole screens) were not recounted.

### The fits (`rl/strong_sets.py fit`)

The code path of `pi_h-v0` (`human_prior.build` with its donors, then `fit_one`): generation 11's run judgement as the start, winners 1x,
AdamW 3e-4, batches of 256, dropout 0.1, early stopping at patience 3 on the sum of the mean validation negative log-likelihoods of card,
ancient and rest (kind and smith target), fifths 1 to 3 train, fifth 4 validates, the same seed. CPU only (the GPU belongs to the lab's
judgement; the fit refuses to run when a GPU is visible), one process. `human_prior.py` gained `build --ascensions`, the loader's parts
(several shard folders, a won-only part) and a warm start from a fitted prior (`init_from_prior`); the defaults are as they were.

| fit | training runs | training decisions (card / ancient / rest) | best epoch (run) | validation score | wall |
|---|---|---|---|---|---|
| baseline (pool) | 11,065 | 132,554 / 18,474 / 47,094 | 13 (16) | 3.9465 | 1,173 s |
| S-win | 1,205 | 24,261 / 3,022 / 9,238 | 15 (18) | 4.0135 | 293 s |
| S-high | under 20 | under 20 each | 1 (4) | 4.9967 | 14 s |
| S-win plus S-high | 1,217 | 24,380 / 3,040 / 9,276 | 18 (21) | 4.0206 | 341 s |
| informational: S-win, warm start from the baseline | 1,205 | as S-win | 2 (5) | 3.8512 | 58 s |

The baseline reproduces the shipped prior: the validation score 3.9465 against v0's 3.962 (a different, larger export: 14,725 usable runs
outside the exam fifth against 11,961; CPU against GPU). Scores of different rows are on different validation sets and do not compare.

### The held-out leg (`rl/strong_sets.py evaluate`)

Fifth 4 of the pool is trained on by no fit. **(a)** its won runs: 11,962 decisions of 336 runs (card 7,952, ancient 983, rest kind 3,027, plus
2,086 smith targets); **(b)** all of its runs: 65,111 decisions of 3,437 runs. Log-loss is the mean negative log-likelihood (nats) of the human's
choice over the card, ancient and rest-kind decisions (one softmax each, as `metrics_from_scores`); top-1 the share where the prior's first
choice is the human's. The difference to the baseline is paired by decision, with a 95% interval from resampling whole runs (2,000 draws).
**(c) the ranking check.** This file defines no ranking check beyond the smith-target "in the prior's top three" row (the ranking check of
`ROUND.md` ranks value heads, not priors), so (c) is that row and its companion on cards: the share of smith targets inside the prior's top
three and the mean rank of the human's card among the offers and the skip (1 = first; a screen has 4 options).

| fit | training runs | (a) log-loss | (a) top-1 | (a) vs baseline | (b) log-loss | (b) top-1 | (b) vs baseline | (c) card mean rank, a / b | (c) smith top-3, a / b |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 11,065 | 0.8731 | 60.8 | | 0.9459 | 57.4 | | 1.679 / 1.782 | 82.6 / 83.7 |
| S-win | 1,205 | 0.9192 | 59.2 | +0.0461 [+0.0372, +0.0547] | 1.0363 | 53.7 | +0.0904 [+0.0858, +0.0949] | 1.743 / 1.895 | 82.5 / 82.6 |
| S-high | under 20 | 1.2225 | 40.8 | +0.3494 [+0.3311, +0.3672] | 1.2978 | 36.4 | +0.3519 [+0.3443, +0.3593] | 2.318 / 2.396 | 39.7 / 42.4 |
| S-win plus S-high | 1,217 | 0.9147 | 59.6 | +0.0416 [+0.0326, +0.0504] | 1.0341 | 54.1 | +0.0882 [+0.0834, +0.0928] | 1.727 / 1.882 | 81.9 / 82.3 |
| *informational:* S-win warm start | 1,205 on the baseline | 0.8625 | 62.0 | -0.0106 [-0.0149, -0.0066] | 0.9616 | 57.3 | +0.0157 [+0.0136, +0.0180] | 1.665 / 1.793 | 83.4 / 83.5 |
| *shipped prior* `2ebfe6d7` (older data) | | 0.8704 | 60.9 | -0.0027 [-0.0058, +0.0006] | 0.9464 | 57.4 | +0.0005 [-0.0011, +0.0021] | 1.674 / 1.779 | 83.2 / 83.6 |

By kind on (a), log-loss / top-1: baseline card 1.033 / 56.2, ancient 0.958 / 52.9, rest kind 0.427 / 75.7, smith target 1.474 / 46.8; S-win 1.090 /
53.3, 0.954 / 54.0, 0.460 / 76.4, 1.508 / 46.5; warm start 1.022 / 56.8, 0.950 / 55.1, 0.414 / 78.0, 1.466 / 47.1.

**Selection rule** (`select_fit`): the fit with the best held-out log-loss on the won runs (a), provided its all-run log-loss (b) is not worse
than the baseline's by more than 0.02 nats; fits within 0.002 nats of the best tie and the larger set wins; no strong fit beating the baseline
on (a) means no candidate. **Outcome: no strong fit beats the baseline, so no candidate was built.** S-win and S-win plus S-high are worse on
the won runs by 0.046 and 0.042 nats and on all runs by 0.090 and 0.088 (over the 0.02 limit); S-high is noise.

What the table says: the winners' choices are easier to predict than the average player's (the baseline's log-loss 0.873 on the won runs against
0.946 on all; top-1 60.8 against 57.4), and a prior fitted on the 1,205 winning runs alone is worse than the one fitted on all 11,065 runs even
on the winners: a ninth of the data costs more than the sharper target gives back. Fitting on winners from scratch is data-starved, not
refuted. The warm start (the baseline's weights fine-tuned on the winners, an arm the brief did not list and the rule does not cover)
gains 0.011 nats on the winners' choices (interval above) and loses 0.016 on all runs, inside the 0.02 limit; had it been one of the fits
the rule would have picked it. Two cautions: the epoch was chosen on fifth 4's winners, the same runs (a) scores, so the 0.011 is optimistic
(checked next); and 0.011 nats is small beside what the lab's blocks resolve.

**Half-split check of the warm start (2026-11-09).** Fifth 4 is split in two halves by a fixed hash of the run id (A and B, about equal); the
warm start is fitted again with early stopping on one half only (the winners of that half; `fit --val-half`) and scored on the other half,
both ways, beside the baseline on the same runs (the baseline fit of `evaluate`, whose epoch was chosen on all of fifth 4, which if anything
favours the baseline). The check holds when on both swaps the warm start is below the baseline by more than the rule's tie margin (0.002) on the
won runs and not above it by more than 0.02 on all runs.

| scored half (early-stopped on the other) | won runs: runs, decisions | log-loss baseline / warm start | top-1 | warm start minus baseline | all runs: runs, decisions | log-loss baseline / warm start | top-1 | warm start minus baseline |
|---|---|---|---|---|---|---|---|---|
| B (stopped on A) | 176, 6,278 | 0.8768 / 0.8689 | 60.4 / 61.3 | -0.0078 [-0.0144, -0.0008] | 1,749, 33,445 | 0.9451 / 0.9634 | 57.3 / 57.2 | +0.0183 [+0.0148, +0.0217] |
| A (stopped on B) | 160, 5,684 | 0.8691 / 0.8565 | 61.4 / 62.7 | -0.0126 [-0.0186, -0.0066] | 1,688, 31,666 | 0.9467 / 0.9639 | 57.6 / 57.4 | +0.0171 [+0.0138, +0.0205] |

**It holds**: the gain on the winners' choices survives (0.008 and 0.013 nats, the intervals exclude zero on both swaps, a little under the
0.011 of the optimistic reading) and the cost on all runs stays under the 0.02 limit on both (0.018 and 0.017; the upper end of one interval,
0.022, is over it, so the margin is thin). The warm start gives up about 0.017 nats on the average player's choices for about 0.010 on the
winners'. The candidate `gen14/candidate-best-players` is built from the warm start's weights (the whole-fifth-4 fit, `56afab28`, not the two
half fits): generation 13's recipe with only the prior replaced, `train.json` marked "warm start on winners; half-split checked", no marker, no
`promotion.json`, no arms (`python rl/strong_sets.py candidate --set s-win-ft --label "warm start on winners; half-split checked"`).

### Caveats

* One held-out fifth is both the early-stopping set and the scoring set of every fit (the v0 recipe: fifth 4 validates). The optimism of
  that is not measured here; the validation curves are flat to within about 0.01 of the summed score around each best epoch.
* The pool is the 2026-11-06 export (14,725 usable Ascension 10 runs outside the exam fifth); `2ebfe6d7` was fitted on 11,961 runs of an earlier
  export. The baseline is a refit on the new data, not the shipped file; the shipped file is scored beside it (the intervals cover zero).
* The rows keep v0's donor maps and the fight block of generation 11's FightNet, as the shipped prior does; S-high's rows would be Ascension 10
  states regardless of their level.
* Runs with unknown content (a card or relic the copy lacks) lose those decisions, counted by the build as in v0. The 11-06 filter's id-map
  screen (which dropped 296 of the 18,421 Ironclad Ascension 10 runs for modded content) was not repeated: the pool keeps those runs and
  the build drops their unknown decisions.

### Rerun

```
python rl/strong_sets.py filter                                        # three working files and filter_stats.json in the data store
python rl/human_prior.py build --corpus <a10_pool> --exam-ids <ids> --out-dir <work>/prior-a10 --procs 3 --clean --reference none --ascensions 10
python rl/human_prior.py build --corpus <s_high>   --exam-ids <ids> --out-dir <work>/prior-high --procs 1 --clean --reference none --ascensions 11-20
python rl/strong_sets.py fit --name baseline|s-win|s-high|s-win+s-high [--threads N]      # CPU, one process each
python rl/strong_sets.py fit --name s-win-ft --init-from <work>/fits/baseline/weights.npz  # the informational arm
python rl/strong_sets.py fit --name s-win-ft --init-from <work>/fits/baseline/weights.npz --val-half A    # and B: the half-split check
python rl/strong_sets.py halves                                        # scores each on the other half, beside the baseline
python rl/strong_sets.py counts ; python rl/strong_sets.py evaluate --incumbent <pi_h-v0/weights.npz>
python rl/strong_sets.py candidate [--set NAME]                        # only when the rule picked a fit (or --set)
```

`donors.npz` is copied beside each build's `decisions` folder. `tests/python/test_strong_sets.py` (35 tests, under a second) pins the filter's
rules and counts, the scrubbing, the build's level option, the loader's parts and the disjoint held-out fifth, the per-decision measures
against `metrics_from_scores`, the cell suppression, the bootstrap, the selection rule's cases, the halves and the candidate copy.

## v2: the map grid (2026-11-11, D58 addendum 13; branch `feat/map-grid`)

Source of the record: Spire Codex public run export for v0.107.1. Aggregates only: no run is named, no cell under 20 decisions is shown, the exam fifth is never
read (`assert_no_exam` on every shard, fit and agreement); the shards and fits stay in the referee's `human-prior` folder, never in a repository.

**Why.** The prior and the student saw the map ahead of the position as 25 numbers (block C: per room type the fewest, mean and most on the paths to the boss, and the
nearest one; the log of the path count). That loses the order of rooms along a path, which paths merge and whether one path gives an elite and then a rest before the
boss. The whole act map is already in the run view (every node, its type and its children, and the position); v2 gives the net all of it, as strong players read it.
Nothing new is read from the game.

### Block M, the encoding (`features.map_grid`; fixed)

An int8 array of 17 x 7 x 13 = 1,547 numbers, laid out [row][column][channel], one cell per map coordinate (row 0 the start, up to row 16, Act 1's boss row; a later act's
boss is on an earlier row and the rows below it are empty; seven columns). Per cell, in this order:

| channels | what |
|---|---|
| 0 to 7 | the node type, one-hot (monster, elite, rest, shop, treasure, boss, unknown, ancient: `vocab.NODE_*` 1 to 8); an empty cell is all zero |
| 8, 9, 10 | the node has a child at column -1, 0, +1 of the next row |
| 11 | reachable: the cell is on some path from the position to the boss, the position included |
| 12 | here: the position |

The start node's fan (its children are all of row 1) and the last row's fan into the boss are the only edges that reach more than one column over; they carry no route and are
not drawn past column +-1 (the reachable bit says what a position can still reach). Every other edge of a real map is one row on and at most one column over (a test draws the
maps of all three acts on the engine and checks every edge). `BASE_SIZE` stays 1,855 and every row of the run judgement, including its value net's input, is byte for byte
unchanged (a digest of a view's row, taken on main before block M, is pinned in a test): block M is appended only where the prior and the student read it.

### The tower and the format

`PriorNet` gains a map tower (`human_prior.map_tower_module`): block M reshaped to 13 channels over 17 x 7, two 3x3 convolutions (padding 1, ReLU; 16 then 32 channels), flatten, a
linear layer to 64, ReLU, concatenated after the judgement's 454 inputs before `hidden1` (which then reads 518 numbers). The 64 new first-layer columns start at zero, so the net
starts as generation 11's trunk. `export_net` writes the six arrays (`map.conv1.weight`, ..., `map.fc.bias`) beside the nine of v0 and v1 and `map_grid: true` in the meta; a file
without them is a v0 or v1 file and scores as it always did. `human_prior_infer.map_tower` runs the tower in numpy (equal to the torch tower to 1e-5 on random grids, a test);
`PriorScorer` and `StudentScorer` carry it and refuse to score without the grids; `option_rows(nets, view, options, with_grid=True)` returns `(rows, grid)`, the grids as a
second int8 array of each option's afterstate (a map step moves the position; any other option leaves it), never a wider row.

### The shards (`build --truth --map-grid`, `decisions-v2`)

`python rl/human_prior.py build --truth <out>/truth --map-grid --procs 4 --clean` writes `decisions-v2` beside `decisions-v1` (it refuses to write over `decisions-v1`): each shard
holds every array of the v1 shard **byte for byte** (checked on all 300 shards: dec, dense, counts, role, ident, run ids) plus `grid` (the distinct grids, int8) and `grid_ix`
(each row's entry). 1,987,020 rows, 391,134 distinct grids (0.20 a row), 156 MB against v1's 136 MB; 428 s on 4 processes at BelowNormal (the machine shared).

### The fits (`fit --decisions decisions-v2 --map-grid`)

The v1 recipe on the same decisions and fifths as v1 (generation 11's trunk, winners 1x, AdamW 3e-4, batches of 256 decisions of mixed kinds, early stopping at patience 3 on the sum of
the mean validation NLLs, fifth 4 validating; 105,624 card, 14,774 ancient, 37,413 rest, 48,452 shop and 74,988 route decisions trained on, 34,323 / 4,775 / 12,090 / 15,630 / 24,298
validated on), one run each on one RTX 3080:

| | v1 | **v2** (the tower, the v1 recipe) | **v2b** (the tower, dropout 0.5 on its 64 outputs and a left-right mirror of a decision's maps with chance 0.5 in training) |
|---|---|---|---|
| best epoch (of the epochs run) | 13 (16) | 9 (12) | 12 (15) |
| validation score | 4.917 | 4.983 | 4.918 |
| wall time | 171 s | 657 s | 1,511 s (the machine shared with the Python gate) |

The plain tower overfits: the training loss falls (1.34 to 1.11) while the validation score stops at 4.98 and the route top-1 falls from 75.4 (epoch 3) to 73.8. A net that sees the whole map
sees a fingerprint of the run (every run's maps are its own). The two aids (`--map-dropout`, `--map-flip`; the mirror turns column c into 6 - c and swaps the edge bits, a test checks that
it is the grid of the mirrored map) take that away: v2b is as good as v1 on every kind. Neither aid changes the encoding or the numpy side (dropout is off at inference; the mirror is
never applied to validation).

### Agreement with the human choice on the validation fifth (`agreement --decisions decisions-v2 --map-grid`, fifth 4)

Top-1 in 100 and the mean negative log-likelihood, v1 as `docs/rl/HUMAN_PRIOR.md` section "v1" reports it, on the same decisions:

| kind | n | v1 | v2 | **v2b** |
|---|---|---|---|---|
| **route, all** | 24,298 | 76.0 / 0.540 | 73.8 / 0.591 | **75.9 / 0.539** |
| ... Act 1 / Act 2 / Act 3 | 14,135 / 6,707 / 3,456 | 75.4 / 76.7 / 76.6 | 73.6 / 74.7 / 73.3 | 75.3 / 76.8 / 76.5 |
| ... rooms differ in type | 17,371 | 79.1 / 0.473 | 77.5 / 0.515 | 79.0 / 0.472 |
| ... winning runs / other runs | | 78.0 / 75.5 | 75.7 / 73.4 | 77.7 / 75.5 |
| **card reward, all** | 34,323 | 50.7 / 1.128 | 50.0 / 1.142 | **51.0 / 1.128** |
| ... Act 1 / Act 2 / Act 3 | 20,587 / 9,523 / 4,213 | 49.8 / 49.9 / 57.0 | 49.3 / 49.0 / 55.3 | 50.3 / 50.1 / 56.8 |
| ... human took a card / skipped | 26,087 / 8,236 | 48.8 / 56.7 | 48.2 / 55.7 | 49.5 / 55.9 |
| ancient | 4,775 | 54.5 / 0.941 | 54.3 / 0.938 | 54.3 / 0.939 |
| rest kind (heal / smith / other) | 12,090 | 75.8 / 0.467 | 76.4 / 0.477 | 75.5 / 0.471 |
| smith target | 6,751 | 48.7 / 1.437 | 48.3 / 1.429 | 49.0 / 1.434 |
| shop, buy or leave (verified rows) | 15,630 | 83.6 / 0.404 | 83.9 / 0.405 | 83.4 / 0.406 |

Skip rate of the card prior: 27.2 (v1), 27.3 (v2), 26.4 (v2b) against the humans' 24.0.

**What it says.** Seeing the whole map does not beat the 25-number summary on what the humans did: with the aids the tower ties v1 on every kind (route 75.9 against 76.0, the
differences on the other kinds are within 0.6 of a point, the size of the run-to-run noise of these fits, which was not measured) and without them it is worse on routes by 2.2 points. The route
kind is where the order of rooms along a path should show, and it did not. Two readings: the humans' route choice is mostly which kind of room they want next and what they carry (HP, gold, deck, the
nearest elite or rest), which block C already has; or the tower needs more than 8,900 training runs' maps to learn path shapes. This is not a verdict on the student: its inputs are the teacher's
labels on the copy's own play, where the map's effect is measured on what wins, not on what a human picks (`RUN_STUDENT.md`, "The map rows").

### Rerun (v2)

```
python rl/human_prior.py build --truth <out>/truth --map-grid --procs 4 --clean                                   # decisions-v2 (428 s)
python rl/human_prior.py fit --decisions decisions-v2 --map-grid --name pi_h-v2  --variants gen11 --winner-weights 1               # GPU, 657 s
python rl/human_prior.py fit --decisions decisions-v2 --map-grid --name pi_h-v2b --variants gen11 --winner-weights 1 --map-dropout 0.5 --map-flip 0.5   # GPU, 1,511 s
python rl/human_prior.py agreement --decisions decisions-v2 --map-grid --model <out>/pi_h-v2b --out <out>/pi_h-v2b/agreement.json
```

`tests/python/test_features_map_grid.py` (the grid on a hand-built map and on real maps), `tests/python/test_human_prior_map.py` (numpy tower equal to the torch tower, the v2 shard equal to the v1
shard plus the grid, a route option's grid, the zero columns, a fit step reaching the tower, the refusals, the mirror and the dropout) and `tests/python/test_human_prior_infer.py` (`option_rows` with
and without the grid, the tower scorers and the file format, the blended chooser) pin it.


## v3: the state gaps (2026-11-12, D58 addenda 14 and 16; branch `feat/state-gaps`)

Source of the record: Spire Codex public run export for v0.107.1. Aggregates only: no run is named, no cell under 20 decisions is shown, the exam fifth is never
read (`assert_no_exam` on every shard, fit and agreement); the shards and fits stay in the referee's `human-prior` folder, never in a repository.

**Why.** The audit of the state the student scores a choice on (D58 addendum 14) found three things a player tracks that the base vector leaves out: the **counters of
relics** (a spent flag, elites killed, a book's count mod 5, a candle's kindle, a box's fights; the state had the relic ids only), the game's **shifting odds for the "?" rooms**
(a fight is 10 in 100 at the start of an act and climbs by 10 with every "?" that was not a fight; a human tracks them from what each earlier "?" turned out to be), and the
**removals bought** at the merchant (they set its quoted price, 75 + 25 a removal, 100 + 50 from Inflation; off the shop screen the state had neither the count nor the price).
The fourth item of the audit, the next act's region, is dropped: every act after the first has one region in this version, so the act index names it and the state has it.

### Block G, the encoding (`features.gaps_block`; fixed)

36 float32 numbers, appended to the first layer's input (after the map tower's 64 when the net has one), in this order:

| numbers | what |
|---|---|
| 27 | one per counter-bearing relic of `vocab.COUNTER_RELICS`, in that order: its counter over its scale (a count to a limit, at most 1; a tally that wraps at its period, `(counter mod period) / period`); 0 when the relic is not held |
| 4 | the "?" rooms visited this act over 10, then the odds that the next one is a fight, a treasure room and a shop, as fractions (0.10, 0.02, 0.03 at the start of an act) |
| 2 | the merchant's removal quote over 100 (from the removals bought and the ascension) and the removals owed over 3 |
| 3 | the **known** bits: counters, "?" rooms, removals (0 where the source cannot know that part; its numbers are then all 0) |

The counter table is read off the engine's sources (`RunNonCombatEffects`, `RunEngine`, `RunMapGenerator`, `RunRewardGenerator` and `RelicEffects.CarryRunCountersBack`): 27 relics, each
with the scale the engine reads its counter over (a test holds every scale to the engine's own constant). The elite odds are not in the block: they start below zero and never become
a chance. `BASE_SIZE` stays 1,855 and `FEATURE_VERSION` 2: every row of the run judgement and its value net's input are byte for byte unchanged (the digest of a view's row, taken on main
before block G, is pinned), because a wider base or a new feature version makes every incumbent checkpoint refuse to load. Block G is an extra block only the prior and the student read
(as block M is), so the brief's "BASE_SIZE grows" was not done: nothing in the base moves.

**Where the engine gives it** (a C# edit in the fork, additive: a library without it answers -3 and the part is then unknown, not an error): state list 25 already gave the relics'
`(id, counter)` pairs (the run's own outside a fight); new list 26 is `(the "?" rooms visited this act, the odds of a fight, a treasure room and a shop in thousandths)`; new list 27
is `(removals bought, removals owed)`. `features.view_from_engine` reads them into `RunView.gaps` (a `StateGaps`; `None` for a view that has no engine state, such as the live coach's, which
then writes zeros and known bits 0). A run record's views carry them (`state.gaps`, written only when known).

**The afterstate** (`features.afterstate_gaps`): a removal at the merchant counts (the next quote is higher); a relic taken arrives with its starting counter (a tea its fights, the Crucible its
upgrades); every other option leaves block G as it is (a step into a "?" does not move the odds: the room is rolled when it is entered, and the pending room carries that).

### What a human record gives of it (`build --truth --state-gaps`)

* **The "?" rooms.** The record names every room by its type; a "?" point is `map_point_type "unknown"` and its first room's `room_type` (monster, elite, treasure, shop or event) is what
  it turned out to be. The state after the "?" rooms of an act entered before a decision is **`features.unknown_room_odds(visits)`**, the one function the engine is tested against: each visit
  resets the odds of the room it was (one of the three) to their base and raises the others the roll could have picked by their base. What the roll could not pick comes from the same
  rule as the engine (`features.unknown_visit_allows`): a shop is out of the roll straight after a shop and when every room the "?" leads to is a shop (read on the true map, at the
  node the human stood on), and a fight is out with Juzu Bracelet. A route decision uses the visits up to the floor before it; every other kind those up to and including the room it stands
  in. The odds restart each act. Where a "?" node's position cannot be read from the route (several paths of the same room types), the "?" state is unknown from there to the end of the
  act (known bit 0; counted as `gaps|visits_unplaced`). Runs with a Golden Compass are already left out of the true-map build.
* **The removals.** The merchant's removals bought before the decision (the record's `cards_removed` at shops, as the build already counted them for the shop price check); the ascension
  is 10, so the quote is 100 + 50 each. Removals owed (a fight's power that grants one and a reward screen not yet left) are not in the record: they are written as 0, which is
  right outside the few runs that held that card.
* **The counters are unknown on a human row** (known bit 0, the 27 numbers 0): the record names the relics a run held, not how many fights each had counted.

### The check against the engine

`features.unknown_room_odds` and `unknown_visit_allows` are tested against the engine on walked runs (`tests/python/test_features_state_gaps.py`): at every "?" visit of 10 runs over three acts the
derived state equals state list 26 (a scratch run of 40 runs: 467 visits, 0 differ; the rooms they turned out to be: 328 events, 86 fights, 35 shops, 18 treasure rooms; 48 visits with Juzu
Bracelet held; every act's reset covered). The removal quote the merchant shows equals `features.removal_price`, and the afterstate of a removal predicts exactly the engine's next block.

### The shards (`build --truth --map-grid --state-gaps`, `decisions-v3`)

`python rl/human_prior.py build --truth <out>/truth --map-grid --state-gaps --procs 4 --clean` writes `decisions-v3` beside `decisions-v1` and `decisions-v2` (it refuses to write over them): each
shard holds every array of the v2 shard **byte for byte** plus `gaps` (the distinct block G vectors, float32) and `gaps_ix` (each row's entry). `--state-gaps` alone (without `--map-grid`)
writes the same folder without the grid.

### The net and the files

`PriorNet(state_gaps=True)` reads block G directly as the last 36 columns of the first layer (no tower of its own: the numbers are few and already scaled); they start at zero (`init_from_judge`,
`init_from_prior`, also from a prior that already has the tower), so a net with them starts as the plain one. `export_net` writes the same arrays as before (the first layer is 36 columns wider) and
`state_gaps: true` and `gaps_version` in the meta; `PriorScorer`, `StudentScorer` and `NumpyPrior` check the first layer's width against the flags and refuse to score a row without its block G. `fit --state-gaps`
(with `--map-grid` for both) reads `decisions-v3`; `agreement --state-gaps` likewise.

### Rerun (v3)

```
python rl/human_prior.py build --truth <out>/truth --map-grid --state-gaps --procs 2 --clean                         # decisions-v3
python rl/human_prior.py fit --decisions decisions-v3 --map-grid --state-gaps --name pi_h-v3 --variants gen11 --winner-weights 1 --map-dropout 0.5 --map-flip 0.5   # GPU
python rl/human_prior.py agreement --decisions decisions-v3 --map-grid --state-gaps --model <out>/pi_h-v3 --out <out>/pi_h-v3/agreement.json
```

`tests/python/test_features_state_gaps.py` (the counter table against the engine's constants, the block's layout and scales, the base vector byte for byte, the rule on hand-made visits and against
the engine at every "?" of walked runs, the merchant's quote and a removal's afterstate, the lists the engine exports, the run record's codec), `tests/python/test_human_prior_gaps.py` (the derivation on a
hand-made run: the shared rule, a shop before a "?", Juzu Bracelet, an ambiguous position, the removals; the v3 shard equal to the v2 shard plus the block; the zero columns; the numpy scorers equal to the
torch net; a fit step reaching the columns; the refusals) and `tests/python/test_human_prior_infer.py` pin it.
