# The human deck prior (`rl/human_prior.py`)

Written 2026-11-05 for milestones 2 and 3 of the strategy memo (sections 3.1 to 3.3 and 3.6). Source of the human record: Spire Codex
public run export for v0.107.1. Aggregates only: no run is named, no cell under 20 runs is shown (every cell below is far over it).
The shards, donor maps and fitted weights are in the referee's `human-prior` folder, not in any repository.

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
