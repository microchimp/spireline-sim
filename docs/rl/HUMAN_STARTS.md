# Human fight starts: coverage

Written by `rl/human_starts.py build` (aggregates only; a cell over fewer than 20 runs is not shown).

- runs read (every run of the corpus, any character and Ascension): 653,733
- of those Ironclad, Ascension 10, standard mode, finished, no modifiers: 14,974
- left out as the exam fifth (ids file or hash rule): 2,968
- left out for modded acts: 45
- runs usable: 11,961; runs that gave at least one fight: 11,299
- floors usable (Act 1 and 2 fights whose state could be rebuilt): 122,494; chosen (every elite and boss, hallway capped at 5,000): 46,412
- records written: 46,410 (with an enchanted card: 18,718; potions only partly known: 0)

## Records by act and kind

| act and kind | fights the usable runs entered | records | runs |
|---|---|---|---|
| act1-boss | 8,844 | 8,216 | 8216 |
| act1-elite | 21,552 | 20,169 | 10316 |
| act1-hallway | 59,346 | 2,498 | 2260 |
| act2-boss | 4,563 | 3,948 | 3948 |
| act2-elite | 10,130 | 9,079 | 4933 |
| act2-hallway | 27,644 | 2,500 | 2098 |

## Records by act, kind and enemy (runs per cell)

- act1 boss CEREMONIAL_BEAST_BOSS: 1425
- act1 boss LAGAVULIN_MATRIARCH_BOSS: 1281
- act1 boss SOUL_FYSH_BOSS: 1295
- act1 boss THE_KIN_BOSS: 1442
- act1 boss VANTOM_BOSS: 1507
- act1 boss WATERFALL_GIANT_BOSS: 1266
- act1 elite BYGONE_EFFIGY_ELITE: 3535
- act1 elite BYRDONIS_ELITE: 3519
- act1 elite PHANTASMAL_GARDENERS_ELITE: 3047
- act1 elite PHROG_PARASITE_ELITE: 3442
- act1 elite SKULKING_COLONY_ELITE: 3148
- act1 elite TERROR_EEL_ELITE: 3110
- act1 hallway CORPSE_SLUGS_NORMAL: 34
- act1 hallway CORPSE_SLUGS_WEAK: 166
- act1 hallway CUBEX_CONSTRUCT_NORMAL: 38
- act1 hallway CULTISTS_NORMAL: 51
- act1 hallway FLYCONID_NORMAL: 28
- act1 hallway FOGMOG_NORMAL: 59
- act1 hallway FOSSIL_STALKER_NORMAL: 57
- act1 hallway FUZZY_WURM_CRAWLER_WEAK: 170
- act1 hallway GREMLIN_MERC_NORMAL: 54
- act1 hallway HAUNTED_SHIP_NORMAL: 36
- act1 hallway INKLETS_NORMAL: 41
- act1 hallway LIVING_FOG_NORMAL: 59
- act1 hallway MAWLER_NORMAL: 54
- act1 hallway NIBBITS_NORMAL: 50
- act1 hallway NIBBITS_WEAK: 196
- act1 hallway OVERGROWTH_CRAWLERS: 34
- act1 hallway PUNCH_CONSTRUCT_NORMAL: 48
- act1 hallway RUBY_RAIDERS_NORMAL: 51
- act1 hallway SEAPUNK_NORMAL: 47
- act1 hallway SEAPUNK_WEAK: 184
- act1 hallway SEWER_CLAM_NORMAL: 59
- act1 hallway SHRINKER_BEETLE_WEAK: 194
- act1 hallway SLIMES_NORMAL: 37
- act1 hallway SLIMES_WEAK: 197
- act1 hallway SLITHERING_STRANGLER_NORMAL: 47
- act1 hallway SLUDGE_SPINNER_WEAK: 198
- act1 hallway SNAPPING_JAXFRUIT_NORMAL: 44
- act1 hallway TOADPOLES_WEAK: 180
- act1 hallway TWO_TAILED_RATS_NORMAL: 45
- act1 hallway VINE_SHAMBLER_NORMAL: 40
- act2 boss KAISER_CRAB_BOSS: 1309
- act2 boss KNOWLEDGE_DEMON_BOSS: 1325
- act2 boss THE_INSATIABLE_BOSS: 1314
- act2 elite DECIMILLIPEDE_ELITE: 3059
- act2 elite ENTOMANCER_ELITE: 2996
- act2 elite INFESTED_PRISMS_ELITE: 2946
- act2 hallway BOWLBUGS_NORMAL: 116
- act2 hallway BOWLBUGS_WEAK: 314
- act2 hallway CHOMPERS_NORMAL: 135
- act2 hallway EXOSKELETONS_NORMAL: 127
- act2 hallway EXOSKELETONS_WEAK: 298
- act2 hallway HUNTER_KILLER_NORMAL: 135
- act2 hallway LOUSE_PROGENITOR_NORMAL: 137
- act2 hallway MYTES_NORMAL: 156
- act2 hallway OVICOPTER_NORMAL: 133
- act2 hallway SLUMBERING_BEETLE_NORMAL: 120
- act2 hallway SPINY_TOAD_NORMAL: 132
- act2 hallway THE_OBSCURA_NORMAL: 136
- act2 hallway THIEVING_HOPPER_WEAK: 291
- act2 hallway TUNNELER_WEAK: 270
- 0 further cells are under 20 runs and not shown

## Rebuild failures by reason

- act 3 or later: not built: 24,393
- deck replay disagrees with the run's final deck: run dropped: 7,725
- relic unknown to the copy: 1,273
- an event page before the fight: HP going in not known: 689
- card unknown to the copy: 267
- enchantment unknown to the copy: 160
- boss at an unusual floor: the route is not the copy's: 135
- encounter unknown to the copy: 25
- roster differs: 2

A fight state can fail for one reason only (the first found). Records marked `approx` are rebuilt without the growth of in-play enchantments, relic counters, or potions an event gave.

## Human outcomes

Written by `rl/human_starts.py outcomes` into `starts/human-outcomes.json` (one entry a start id; the start file itself is not rewritten). Source: Spire Codex public run export for v0.107.1, the four training fifths. Aggregates only; a cell over fewer than 20 runs is not shown.

- starts in the set: 46,410; with an outcome: 46,410 (100.00%)
- fields: `human_survived` (the run went on past the floor; a boss floor: the boss died), `human_end_hp` (HP when the fight ended, before the run engine's post-fight heals; 0 for a death), `human_hp_lost` (HP going in less end HP; the whole HP going in for a death), `human_damage_taken` (the floor's raw figure), `human_post_heal`, `human_turns`, `human_potions_used`, `human_potions_held_end`, `human_l0` (the search's currency: end HP plus 10 a potion held, a death -100; what practice regret compares with), `human_l0_unit` (the gym's L0 of 0 to 1, `fight_gym.l0`), `human_flags`.

### The reading of end HP and what post-fight heals do to it

The gym ends a fight when the last enemy dies and reads HP then: it has what the fight healed on the way and not what the run engine heals afterwards (Burning Blood 6, Black Blood 12, Meat on the Bone 12 at half HP or less, Chosen Cheese 1). The record's `current_hp` is after those and `hp_healed` is the floor's total, so the human's end HP is `current_hp` less min(`hp_healed`, the relics' post-fight heal).

- survivor floors checked: 39,548; HP in - damage taken + healed = HP after holds on 99.93%
- mean per survivor floor: HP in - HP after 18.19, damage taken 26.36, post-fight heal 6.27, heal inside the fight 1.90, HP lost as read 24.46
- so `HP in - current HP` would understate the loss by the post-fight heal (6.27 HP on average) and `damage taken` would overstate it by the heal inside the fight (1.90 on average; 18.1% of fights heal beyond the relics)
- a heal inside the fight and a potion drunk on the reward screen are not told apart in the record; the Meat on the Bone reading is a guess (`heal_guess`)

### Coverage by act and kind

| act and kind | starts with an outcome | runs | survived % | HP lost (survivors) | turns | potions used |
|---|---|---|---|---|---|---|
| act1-boss | 8,216 | 8,216 | 72.2 | 35.1 | 8.3 | 0.92 |
| act1-elite | 20,169 | 10,316 | 88.6 | 22.7 | 5.2 | 0.69 |
| act1-hallway | 2,498 | 2,260 | 98.5 | 9.4 | 4.1 | 0.16 |
| act2-boss | 3,948 | 3,948 | 73.9 | 35.8 | 6.9 | 0.94 |
| act2-elite | 9,079 | 4,933 | 87.6 | 24.5 | 4.4 | 0.75 |
| act2-hallway | 2,500 | 2,098 | 96.8 | 12.6 | 4.2 | 0.30 |

### Starts without an outcome, by reason

- none

### Flags and checks

- flag: end_hp_raised_to_1: 13
- flag: heal_guess: 701
