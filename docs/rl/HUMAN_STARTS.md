# Human fight starts: coverage

Written by `rl/human_starts.py build` (aggregates only; a cell over fewer than 20 runs is not shown).

- corpus lines read: 653,733
- Ironclad, Ascension 10, standard, finished: 14,974
- left out as the exam fifth (ids file or hash rule): 2,968
- left out for modded acts: 45
- runs usable: 11,961; runs that gave at least one fight: 11,299
- fight states rebuilt: 122,596; chosen (elites and bosses all, hallway capped): 46,514
- records written: 46,512 (with an enchanted card: 18,786; potions only partly known: 0)

## Records by act and kind

| act and kind | fights the usable runs entered | records | runs |
|---|---|---|---|
| act1-boss | 8,844 | 8,224 | 8223 |
| act1-elite | 21,552 | 20,169 | 10316 |
| act1-hallway | 59,346 | 2,498 | 2260 |
| act2-boss | 4,563 | 4,042 | 4042 |
| act2-elite | 10,130 | 9,079 | 4933 |
| act2-hallway | 27,644 | 2,500 | 2098 |

## Records by act, kind and enemy (runs per cell)

- act1 boss CEREMONIAL_BEAST_BOSS: 1426
- act1 boss LAGAVULIN_MATRIARCH_BOSS: 1281
- act1 boss SOUL_FYSH_BOSS: 1295
- act1 boss THE_KIN_BOSS: 1445
- act1 boss VANTOM_BOSS: 1507
- act1 boss WATERFALL_GIANT_BOSS: 1267
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
- act2 boss KAISER_CRAB_BOSS: 1338
- act2 boss KNOWLEDGE_DEMON_BOSS: 1356
- act2 boss THE_INSATIABLE_BOSS: 1348
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
- 1 further cells are under 20 runs and not shown

## Rebuild failures by reason

- act 3 or later: not built: 24,393
- deck replay disagrees with the run's final deck: run dropped: 7,748
- relic unknown to the copy: 1,275
- an event page before the fight: HP going in not known: 689
- card unknown to the copy: 270
- enchantment unknown to the copy: 165
- encounter unknown to the copy: 25
- roster differs: 2

A fight state can fail for one reason only (the first found). Records marked `approx` are rebuilt without the growth of in-play enchantments, relic counters, or potions an event gave.
