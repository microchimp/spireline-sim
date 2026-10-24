# Divergence by hook family

Basis: divergences. Total agent weight 7043.83; architectural share 54.8%; unassigned 55.2% of the weight (45.5% of the architectural weight); families with weight 5. Evidence by weight: diff 28.0%, reason 1.7%, type 12.8%, shape 2.3%, none 55.2%.

Top five families by architectural weight: F08, F11, F15 hold 54.5% of the architectural weight: between the PLAN 8.3 thresholds (50% to 70%): neither rule fires (at most 100.0% if every unassigned diff belonged to them). Top five by total weight: F08, F16, F15, F11, F10 54.5% of the architectural weight. Only 5 families carry weight at all, so top five is every family the lockstep diff can show.

Other readings of a family boundary:

- B: the draw is F03: F08, F03, F15 hold 54.5%: between the PLAN 8.3 thresholds (50% to 70%): neither rule fires. the dispatch file lists ModifyHandDraw / BeforeHandDraw under F03 (player turn start) and the draw, shuffle and pile moves under F11 (card piles); a hand drawn at turn start can be read either way.
- C: a listed corpse is a death hook: F08, F11, F15, F10 hold 55.3%: between the PLAN 8.3 thresholds (50% to 70%): neither rule fires. if the game keeps a dead enemy in its roster (ShouldCreatureBeRemovedFromCombatAfterDeath refuses) and the emulator removes it, R04 is a death-removal difference in F10; as it stands the comparer's asymmetry alone produces the diff.
- B and C together: F08, F03, F15, F10 hold 55.3%: between the PLAN 8.3 thresholds (50% to 70%): neither rule fires. both boundary readings at once.

The lockstep diff holds hand, enemy hp, hp, deck, gold, potions and relics only, so the families that act through powers, relic counters, energy, orbs or potions in combat cannot show in this table, whatever their share of the real error.

## Families by weight (cumulative curve)

| # | family | name | coarse family | weight | share | cumulative | architectural weight | architectural share in family | share of architectural | groups | divergences | decisions affected | outcome-changing confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F08 | damage / HP | damage and block pipeline | 1939.67 | 27.5% | 27.5% | 1795.94 | 92.6% | 46.6% | 4 | 74 | 871 | 39 |
| 2 | F16 | run shop / rest / map / deck | run flow | 634.08 | 9.0% | 36.5% | 0 | 0.0% | 0.0% | 41 | 99 | 273 | 49 |
| 3 | F15 | run rewards | run flow | 321.95 | 4.6% | 41.1% | 146.52 | 45.5% | 3.8% | 26 | 35 | 138 | 18 |
| 4 | F11 | card piles | RNG order | 161.2 | 2.3% | 43.4% | 161.2 | 100.0% | 4.2% | 6 | 6 | 72 | 4 |
| 5 | F10 | death / victory | death and victory | 100.14 | 1.4% | 44.8% | 0 | 0.0% | 0.0% | 3 | 6 | 47 | 3 |
| - | unassigned | no rule places it | - | 3886.79 | 55.2% | - | 1753.51 | 45.1% | 45.5% | 50 | 124 | 1706 | 59 |

Unassigned by rule (weight): R04 2164.22, R05 6.64, R09 16.23, R12 362.94, R13 1336.76

## Coarse families (PLAN 8.2 names; the last two added)

| coarse family | weight | share | architectural weight | architectural share in family | groups | divergences |
|---|---|---|---|---|---|---|
| damage and block pipeline | 1939.67 | 27.5% | 1795.94 | 92.6% | 4 | 74 |
| run flow | 809.51 | 11.5% | 0 | 0.0% | 46 | 104 |
| RNG order | 276.18 | 3.9% | 276.18 | 100.0% | 26 | 35 |
| death and victory | 100.14 | 1.4% | 0 | 0.0% | 3 | 6 |
| exhaust and discard triggers | 31.54 | 0.4% | 31.54 | 100.0% | 1 | 1 |

## Single-site groups (Lane B's fix queue, ranked)

| rank | group | weight | family | rule | evidence | type: fields | divergences | first floor | outcome-changing confirmed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | G2891c25e0d | 2277.01 | unassigned | R04 | none | select_cards_combat: .combat.enemy_hp | 68 | 2 | 36 |
| 4 | Gbc5227da53 | 108.0 | F16 | R17 | type | treasure: .floor, .gold, .hp, .potions, .relics | 1 | 10 | 1 |
| 5 | G0a127979b6 | 105.76 | F16 | R17 | type | select_cards: .deck | 19 | 3 | 10 |
| 9 | G96df219a44 | 66.42 | F16 | R17 | type | treasure: .hp, .max_hp | 11 | 10 | 4 |
| 19 | G2955289b06 | 48.83 | F16 | R17 | type | select_cards: .deck | 8 | 4 | 5 |
| 21 | G74caa1c59d | 46.17 | F15 | R16 | type | combat: .deck | 1 | 19 | 1 |
| 24 | G8318fb1cf5 | 43.74 | F15 | R16 | type | combat: .deck | 1 | 19 | 1 |
| 25 | G499831a0f5 | 42.78 | F10 | R01 | reason | combat: reason:phase:6 | 1 | 7 | 0 |
| 33 | G2fc811be54 | 35.34 | F16 | R17 | type | treasure: .deck | 6 | 8 | 3 |
| 35 | G88f7b7316f | 34.02 | F15 | R16 | type | combat: .deck | 1 | 19 | 1 |
| 37 | G1b02fc9e96 | 33.44 | F16 | R02 | reason | rest: reason:screen:None!=rest | 1 | 12 | 0 |
| 39 | Gc44f78e430 | 32.52 | F10 | R03 | diff | reward: .hp | 3 | 2 | 1 |
| 42 | G04d74da38a | 31.0 | F16 | R17 | type | select_cards: .deck | 6 | 3 | 3 |
| 44 | Gfb4c7f4acc | 29.9 | F16 | R17 | type | select_cards: .deck | 5 | 4 | 2 |
| 49 | Gb1aa8d4875 | 27.2 | F15 | R16 | type | combat: .deck | 1 | 20 | 0 |
| 52 | G1162c5de18 | 24.84 | F10 | R01 | reason | reward: reason:phase:0!=rewards | 2 | 5 | 2 |
| 55 | G5b5244da1e | 24.3 | F15 | R16 | type | combat: .deck | 1 | 19 | 1 |
| 61 | G5120acf48c | 19.68 | F16 | R17 | type | select_cards: .deck | 2 | 18 | 2 |
| 65 | G6f0314da18 | 18.1 | F16 | R17 | type | select_cards: .deck | 4 | 5 | 1 |
| 72 | Gd703a4790b | 14.08 | F16 | R17 | type | select_cards: .deck | 3 | 6 | 1 |
| 85 | Gac0a0b8a7e | 9.24 | F16 | R17 | type | treasure: .gold | 2 | 6 | 1 |
| 86 | G84828dd79a | 8.28 | F16 | R02 | reason | event: reason:event:WHISPERING_HOLLOW!=JUNGLE_MAZE_ADVENTURE | 1 | 8 | 1 |
| 87 | Gadf072f8ab | 8.01 | F16 | R02 | reason | event: reason:event:JUNGLE_MAZE_ADVENTURE!=TABLET_OF_TRUTH | 1 | 11 | 1 |
| 94 | Gf65e1cf550 | 5.82 | F16 | R17 | type | select_cards: .deck | 1 | 3 | 1 |
| 95 | G3c94e2c2de | 5.7 | F16 | R17 | type | select_cards: .deck | 1 | 5 | 1 |
| 96 | Gbdae48596f | 5.7 | F16 | R17 | type | select_cards: .deck | 1 | 5 | 1 |
| 97 | Gc981582989 | 5.7 | F16 | R17 | type | select_cards: .deck | 1 | 5 | 1 |
| 98 | Gf45701a3de | 5.55 | F16 | R17 | type | map: .floor | 2 | 6 | 2 |
| 99 | G7f549a3298 | 5.34 | F16 | R17 | type | select_cards: .deck | 1 | 11 | 1 |
| 100 | G87f40418b7 | 5.22 | F16 | R17 | type | select_cards: .deck | 1 | 13 | 1 |

## Rules (first match wins)

| rule | family | coarse | evidence | weight | divergences | reason |
|---|---|---|---|---|---|---|
| R01 | F10 | death and victory | reason | 67.62 | 3 | the referee is at the rewards screen while the emulator is still in combat: the combat-end win check at the action boundary (design 2: victory is not checked there); cause unverified |
| R02 | F16 | run flow | reason | 52.13 | 4 | the run layer shows another screen or event page (rest, map, event): flow, not a combat hook |
| R03 | F10 | death and victory | diff | 32.52 | 3 | hp differs at the reward screen: what runs at the victory (AfterCombatVictory heals) differs |
| R04 | unassigned | - | none | 2164.22 | 65 | an enemy with 0 hp is listed on the referee side only: the emulator's lockstep view keeps hp above 0 only, the referee's keeps every enemy the game lists (referee/lockstep.py), so this is a comparer asymmetry unless the corpse matters to play; reading C puts it in F10 (AfterDeath, removal after death) |
| R05 | unassigned | - | none | 6.64 | 1 | an enemy hp stand-in value (999999999 against 32768) is a representation difference of the encounter, not a hook; needs the encounter |
| R06 | F08 | damage and block pipeline | diff | 1939.67 | 74 | only enemy hp and player hp differ: a damage amount (modifier folds, rounding) or an hp change |
| R07 | F09 | damage and block pipeline | diff | 0 | 0 | only block differs: the block gain passes |
| R08 | F11 | exhaust and discard triggers | shape | 0 | 0 | the same cards in another order: pile insertion order (CardPileCmd position), not draw content |
| R09 | unassigned | - | none | 16.23 | 8 | the same cards at another upgrade level (a curse upgraded on one side): an in-combat upgrade effect picked a card the game cannot upgrade; which effect needs the trace |
| R10 | F11 | RNG order | shape | 129.66 | 5 | equal-size hands of other cards: a fresh draw out of a different draw-pile order (shuffle stream and pile order); reading B if read as the turn-start draw |
| R11 | F11 | exhaust and discard triggers | shape | 31.54 | 1 | one card more on one side only: a card drawn, discarded or exhausted on one side (CardPileCmd); reading B if the turn-start draw count |
| R12 | unassigned | - | none | 362.94 | 13 | hands differ by two or more cards or both ways: a turn-boundary flush, a draw count and several plays all fit; needs the trace |
| R13 | unassigned | - | none | 1336.76 | 37 | hand with hp, enemy hp or another field in one diff: each is a consequence of an upstream action or draw the diff does not name; needs the trace |
| R14 | F15 | RNG order | type | 0 | 0 | gold differs at the reward screen: the gold roll and the Rewards stream draws (kit G2877adf2ff) |
| R15 | F15 | RNG order | type | 146.52 | 30 | potion or card reward contents differ at a reward screen: reward generation (rng stream) |
| R16 | F15 | run flow | type | 175.43 | 5 | deck, gold or relics differ at a combat start or reward screen: a card taken from a reward on one side only (kit G05884084d1 lists the reward flow first; an ancient or event card is the other candidate, F16) |
| R17 | F16 | run flow | type | 581.95 | 95 | a run-layer screen (select_cards, treasure, map, event, shop, rest) differs in a carried field |
| R18 | unassigned | - | none | 0 | 0 | no rule places this diff |

Groups whose divergences fall into more than one family: 2 (listed with their split in by_family.json). Weight check against ranked.json: largest per-group difference 0.0.
