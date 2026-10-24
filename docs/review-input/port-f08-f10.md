# Port list: F08+F10

106 models reached by records override a hook of these families: card 1, monster 22, potion 1, power 64, relic 18.
155 (model, hook) pairs (27 pseudo-hook): not_modelled 5, open 49, registered 101; gate: 150 pairs without a Core2 handler.

## power-1

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| power:STRENGTH_POWER | 2628/0/136835/0 | ModifyDamageAdditive | registered | BuffSystem.Strength_ModifyDamageAdditive:317 |
| power:VULNERABLE_POWER | 2314/0/111793/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:487; EnemyAI.DealAttackDamage:4485 |
| power:WEAK_POWER | 849/0/67339/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:461; BuffSystem.IncomingDamage:507 |
| power:THORNS_POWER | 530/0/47773/0 | BeforeDamageReceived | registered | CardEffects.TriggerEnemyThorns:10625; EnemyAI.ApplyPlayerThorns:4652 |
| power:RAVENOUS_POWER | 464/0/6900/0 | AfterDeath | registered | CombatEngine.Ravenous_AfterDeath:4525 |
| power:SHRINK_POWER | 433/0/6859/0 | AfterDeath | open | elsewhere: BuffSystem.AllowsNegative, BuffSystem.IncomingDamage, BuffSystem.IsDurationDebuff, BuffSystem.TickEndOfTurn |
|  |  | AfterRemoved (pseudo) | open | elsewhere: BuffSystem.AllowsNegative, BuffSystem.IncomingDamage, BuffSystem.IsDurationDebuff, BuffSystem.TickEndOfTurn |
|  |  | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:466 |
| power:INFERNO_POWER | 225/0/7296/0 | AfterDamageReceived | registered | CardEffects.TriggerInfernoAfterPlayerDamage:10317 |
| power:VIGOR_POWER | 149/0/21262/0 | ModifyDamageAdditive | registered | BuffSystem.Vigor_ModifyDamageAdditive:324 |
| power:RUPTURE_POWER | 130/0/6563/0 | AfterDamageReceived | registered | CardEffects.AfterPlayerDamageReceived:4132 |
| power:TANK_POWER | 118/0/1925/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:482 |
| power:KNOCKDOWN_POWER | 93/0/215/0 | ModifyDamageMultiplicative | not_modelled | not modelled: only multiplayer-only cards lead to this power; Spireline plays single player |
| power:COLOSSUS_POWER | 86/0/6800/0 | ModifyDamageMultiplicative | registered | EnemyAI.DealAttackDamage:4484 |
| power:SHRIEK_POWER | 76/0/5760/0 | AfterDamageReceived | registered | EnemyAI.TriggerShriekIfWounded:4802; EnemyAI.TriggerShriekIfWounded:4808 |
| power:MINION_POWER | 74/0/16680/0 | ShouldPowerBeRemovedAfterOwnerDeath (pseudo) | open | elsewhere: CardEffects.TriggersFatal, CombatEngine.IsPrimaryEnemy, CombatFactory.ApplyIntrinsicEnemyPowers, EnemyAI.ApplyBuffIntent |
| power:FLAME_BARRIER_POWER | 60/0/4996/0 | AfterDamageReceived | registered | EnemyAI.ApplyPlayerFlameBarrier:4613 |

## power-2

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| power:CURL_UP_POWER | 50/0/2926/0 | AfterDamageReceived | registered | CardEffects.LandDamageOnEnemy:5477 |
| power:SLIPPERY_POWER | 50/0/1564/0 | AfterDamageReceived | registered | CardEffects.Slippery_AfterModifyingHpLostAfterOsty:5306; CardEffects.LandDamageOnEnemy:5454 |
|  |  | ModifyHpLostAfterOsty | registered | CardEffects.LandDamageOnEnemy:5454 |
| power:BUFFER_POWER | 46/0/3993/0 | AfterModifyingHpLostAfterOsty | registered | CardEffects.Buffer_AfterModifyingHpLostAfterOsty:4005; CardEffects.Buffer_AfterModifyingHpLostAfterOsty:4009; CardEffects.LosePlayerHp:4057 |
|  |  | ModifyHpLostAfterOstyLate | registered | CardEffects.LosePlayerHp:4057 |
| power:THE_GAMBIT_POWER | 46/0/1335/0 | AfterDamageReceived | registered | EnemyAI.TheGambit_AfterDamageReceived:4429; EnemyAI.DealAttackDamage:4528 |
| power:IMBALANCED_POWER | 41/0/30776/0 | AfterDamageGiven | open | elsewhere: CombatFactory.ApplyIntrinsicEnemyPowers |
| power:INTANGIBLE_POWER | 32/0/2438/0 | AfterModifyingDamageAmount | not_modelled | not modelled: the override only gives feedback (Flash, sound, wait); no game state changes |
|  |  | AfterModifyingHpLostAfterOsty | registered | BuffSystem.CapIncomingDamage:557; BuffSystem.CapHpLoss:571 |
|  |  | ModifyDamageCap | registered | BuffSystem.CapIncomingDamage:557 |
|  |  | ModifyHpLostAfterOsty | registered | BuffSystem.CapIncomingDamage:557; BuffSystem.CapHpLoss:571 |
| power:SURPRISE_POWER | 31/0/8481/0 | AfterDeath | registered | CombatEngine.Surprise_AfterDeath:4468; CombatEngine.AnswerDeaths:4628 |
|  |  | ShouldStopCombatFromEnding | registered | CombatEngine.Surprise_ShouldStopCombatFromEnding:4729 |
| power:POSSESS_SPEED_POWER | 28/0/11614/0 | AfterDeath | registered | CombatEngine.PossessSpeed_AfterDeath:4421; CombatEngine.PossessSpeed_AfterDeath:4423; CombatEngine.AnswerDeaths:4587 |
| power:POSSESS_STRENGTH_POWER | 28/0/11614/0 | AfterDeath | registered | CombatEngine.PossessStrength_AfterDeath:4435; CombatEngine.PossessStrength_AfterDeath:4437; CombatEngine.AnswerDeaths:4592 |
| power:PAPER_CUTS_POWER | 27/0/12887/0 | AfterDamageGiven | registered | EnemyAI.DealAttackDamage:4559 |
| power:VITAL_SPARK_POWER | 23/0/4451/0 | AfterRemoved (pseudo) | open | elsewhere: CombatEngine.ApplyAfterCardPlayedPowers, CombatFactory.ApplyIntrinsicEnemyPowers, EnemyAI.ExecuteIntent |
| power:TAINTED_POWER | 23/0/2052/0 | ModifyDamageAdditive | registered | BuffSystem.Tainted_ModifyDamageAdditive:331 |
| power:ILLUSION_POWER | 21/0/2773/0 | AfterDeath | registered | CombatEngine.TryReviveIllusion:5160 |
|  |  | ShouldAllowHitting | registered | CombatEngine.IsPrimaryEnemy:5002; EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath:5375 |
|  |  | ShouldCreatureBeRemovedFromCombatAfterDeath | registered | EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath:5375 |
|  |  | ShouldPowerBeRemovedOnDeath | registered | EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath:5375 |
| power:SWIPE_POWER | 20/0/3797/0 | BeforeDeath | registered | CombatEngine.Swipe_BeforeDeath:4504 |
| power:GIGANTIFICATION_POWER | 19/0/3647/0 | ModifyDamageMultiplicative | registered | CardEffects.DmgFrom:9447 |

## power-3

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| power:SLOW_POWER | 19/0/2096/0 | AfterModifyingDamageAmount | not_modelled | not modelled: the override only gives feedback (Flash, sound, wait); no game state changes |
|  |  | ModifyDamageMultiplicative | open | elsewhere: CardEffects.DealDamageToEnemy, CombatEngine.ApplyAfterCardPlayedPowers, CombatFactory.ApplyIntrinsicEnemyPowers |
| power:FLUTTER_POWER | 19/0/1430/0 | AfterDamageReceived | registered | CardEffects.Flutter_AfterDamageReceived:5324; CardEffects.Flutter_AfterDamageReceived:5325; CardEffects.LandDamageOnEnemy:5531 |
|  |  | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:475 |
| power:REATTACH_POWER | 18/0/14289/0 | AfterDeath | registered | CombatEngine.TryReattachSegment:5207; CombatEngine.TryReattachSegment:5231 |
|  |  | ShouldAllowHitting | registered | EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath:5368 |
|  |  | ShouldCreatureBeRemovedFromCombatAfterDeath | registered | EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath:5368 |
|  |  | ShouldPowerBeRemovedAfterOwnerDeath (pseudo) | open | elsewhere: CardEffects.TriggersFatal, CombatEngine.AnyOtherSegmentAlive, CombatEngine.EndTurn, CombatEngine.TryReattachSegment |
| power:SLUMBER_POWER | 18/0/7937/0 | AfterDamageReceived | registered | CardEffects.Slumber_AfterDamageReceived:5337; CardEffects.Slumber_AfterDamageReceived:5346; CardEffects.LandDamageOnEnemy:5537 |
|  |  | AfterRemoved (pseudo) | open | elsewhere: BuffSystem.TickEndOfTurn, CardEffects.LandDamageOnEnemy, CardEffects.Slumber_AfterDamageReceived, CombatFactory.CreateSlumberingBeetle |
| power:HARD_TO_KILL_POWER | 18/0/6437/0 | AfterModifyingDamageAmount | not_modelled | not modelled: the override only gives feedback (Flash, sound, wait); no game state changes |
|  |  | ModifyDamageCap | open | elsewhere: CardEffects.DealDamageToEnemy, CardEffects.DealUnpoweredDamageToEnemy, CombatFactory.CreateExoskeleton |
| power:HEX_POWER | 17/0/3987/0 | AfterDeath | registered | CombatEngine.Hex_AfterDeath:4446 |
|  |  | AfterRemoved (pseudo) | open | elsewhere: CombatEngine.Hex_AfterDeath, CombatEngine.IsEtherealForPowers, EnemyAI.ApplyDebuffIntent |
| power:PERSONAL_HIVE_POWER | 16/0/4696/0 | AfterDamageReceived | registered | CardEffects.LandDamageOnEnemy:5576 |
| power:DAMPEN_POWER | 15/0/1979/0 | AfterDeath | registered | CombatEngine.AnswerDeaths:4615 |
|  |  | AfterRemoved (pseudo) | open | elsewhere: CardEffects.RemoveDampen, CombatEngine.AnswerDeaths, EnemyAI.ApplyDebuffIntent |
| power:STOCK_POWER | 13/0/13586/0 | AfterDeath | registered | CombatEngine.TryRespawnAxebot:5434; CombatEngine.TryRespawnAxebot:5463 |
|  |  | ShouldStopCombatFromEnding | registered | CombatEngine.Stock_ShouldStopCombatFromEnding:4743 |
| power:ASLEEP_POWER | 12/0/2264/0 | AfterDamageReceived | registered | EnemyAI.WakeMatriarchIfStruck:4874; EnemyAI.WakeMatriarchIfStruck:4881 |
| power:PLOW_POWER | 12/0/577/0 | AfterDamageReceived | registered | EnemyAI.TriggerPlowIfWounded:4832; EnemyAI.TriggerPlowIfWounded:4838 |
| power:CRAB_RAGE_POWER | 11/0/15357/0 | AfterDeath | registered | CombatEngine.TryEnrageCrabPartner:5287; CombatEngine.TryEnrageCrabPartner:5298; CombatEngine.TryEnrageCrabPartner:5311 |
| power:SURROUNDED_POWER | 11/0/15357/0 | AfterDeath | registered | CombatEngine.FacePast:5384; CombatEngine.FacePast:5387; CombatEngine.TurnPlayerToFaceSurvivor:5405 |
|  |  | ModifyDamageMultiplicative | registered | BuffSystem.Surrounded_ModifyDamageMultiplicative:398 |
| power:SANDPIT_POWER | 11/0/1144/0 | AfterOstyRevived | not_modelled | not modelled: only moves the creatures' nodes along the sandpit (a tween, and the track's musi |
|  |  | AfterRemoved (pseudo) | open | elsewhere: CardEffects.ApplyCore, CombatEngine.EndTurn, EnemyAI.ApplyBuffIntent |
| power:SOAR_POWER | 11/0/738/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:528 |

## power-4

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| power:ADAPTABLE_POWER | 9/0/18022/0 | AfterDeath | registered | CombatEngine.TryRespawnTestSubject:5479 |
|  |  | ShouldAllowHitting | registered | EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath:5382 |
|  |  | ShouldCreatureBeRemovedFromCombatAfterDeath | registered | CombatEngine.Adaptable_ShouldStopCombatFromEnding:4722; EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath:5382 |
|  |  | ShouldPowerBeRemovedAfterOwnerDeath (pseudo) | open | elsewhere: CombatEngine.Adaptable_ShouldStopCombatFromEnding, CombatEngine.RespawnTestSubject, CombatEngine.TryRespawnTestSubject, CombatFactory.CreateTestSubject |
|  |  | ShouldStopCombatFromEnding | registered | CombatEngine.Adaptable_ShouldStopCombatFromEnding:4722 |
| power:STEAM_ERUPTION_POWER | 9/0/5129/0 | AfterDeath | registered | EnemyAI.TriggerAboutToBlow:4957 |
|  |  | ShouldCreatureBeRemovedFromCombatAfterDeath | registered | CombatEngine.SteamEruption_ShouldStopCombatFromEnding:4715; EnemyAI.SteamEruption_ShouldCreatureBeRemovedFromCombatAfterDeath:5389 |
|  |  | ShouldPowerBeRemovedAfterOwnerDeath (pseudo) | open | elsewhere: CombatEngine.SteamEruption_ShouldStopCombatFromEnding, EnemyAI.ApplyBuffIntent, EnemyAI.DiedHoldingSteamEruption, EnemyAI.ExecuteIntent |
|  |  | ShouldStopCombatFromEnding | registered | CombatEngine.SteamEruption_ShouldStopCombatFromEnding:4715; EnemyAI.DiedHoldingSteamEruption:4933 |
| power:INFESTED_POWER | 9/0/1155/0 | AfterDeath | registered | CombatEngine.Infested_AfterDeath:4478; CombatEngine.AnswerDeaths:4654 |
|  |  | ShouldStopCombatFromEnding | registered | CombatEngine.Infested_ShouldStopCombatFromEnding:4736 |
| power:HARDENED_SHELL_POWER | 9/0/665/0 | AfterDamageReceived | registered | CombatEngine.RefillHardenedShells:4801; CardEffects.LandDamageOnEnemy:5448 |
|  |  | AfterModifyingHpLostBeforeOsty | registered | CardEffects.LandDamageOnEnemy:5448 |
|  |  | ModifyHpLostBeforeOstyLate | registered | CardEffects.LandDamageOnEnemy:5448 |
| power:CONSTRICT_POWER | 9/0/509/0 | AfterDeath | registered | CombatEngine.Constrict_AfterDeath:4513 |
| power:BURROWED_POWER | 7/0/1322/0 | AfterBlockBroken | registered | EnemyAI.BreakBurrowIfBlockGone:5351; EnemyAI.BreakBurrowIfBlockGone:5356 |
|  |  | AfterRemoved (pseudo) | open | elsewhere: CombatEngine.EndTurn, EnemyAI.ApplyBuffIntent, EnemyAI.BreakBurrowIfBlockGone |
| power:HEIST_POWER | 7/0/666/0 | BeforeDeath | registered | CombatEngine.Heist_BeforeDeath:4493; CombatEngine.SpawnGremlinMercReinforcements:5592 |
| power:TANGLED_POWER | 4/0/267/0 | AfterRemoved (pseudo) | open | elsewhere: CombatEngine.EffectiveCost, CombatEngine.EndTurn, EnemyAI.ExecuteIntent |
| power:RINGING_POWER | 3/0/34/0 | AfterRemoved (pseudo) | open | elsewhere: CombatEngine.EndTurn, CombatEngine.IsBlockedByRinging, EnemyAI.ApplyDebuffIntent |
| power:PAINFUL_STABS_POWER | 2/0/1297/0 | ShouldCreatureBeRemovedFromCombatAfterDeath | registered | EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath:5396 |
|  |  | ShouldPowerBeRemovedAfterOwnerDeath (pseudo) | open | elsewhere: CombatEngine.RespawnTestSubject, EnemyAI.DealAttack, EnemyAI.ExecuteIntent, EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath |
| power:CALCIFY_POWER | 1/0/11/0 | ModifyDamageAdditive | registered | CardEffects.Calcify_ModifyDamageAdditive:9903 |
| power:HANG_POWER | 0/0/59/0 | ModifyDamageMultiplicative | open | elsewhere: CardEffects.ApplyNecrobinderCard |
| power:ACCURACY_POWER | 0/0/24/0 | ModifyDamageAdditive | registered | CardEffects.Accuracy_ModifyDamageAdditive:2793 |
| power:SIC_EM_POWER | 0/0/23/0 | AfterDamageGiven | registered | CardEffects.DealOstyDamage:9969 |
| power:CONQUEROR_POWER | 0/0/11/0 | ModifyDamageMultiplicative | registered | CardEffects.ConqueredDamage:2977 |

## power-5

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| power:PHANTOM_BLADES_POWER | 0/0/7/0 | ModifyDamageAdditive | registered | CardEffects.PhantomBlades_ModifyDamageAdditive:2800 |
| power:LETHALITY_POWER | 0/0/4/0 | ModifyDamageMultiplicative | registered | CardEffects.DealDamageToEnemy:5154 |
| power:TRACKING_POWER | 0/0/2/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:506 |
| power:DOUBLE_DAMAGE_POWER | 0/0/1/0 | ModifyDamageMultiplicative | registered | BuffSystem.IncomingDamage:521 |

## relic-1

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| relic:SELF_FORMING_CLAY | 169/0/40020/1441 | AfterDamageReceived | registered | RelicEffects.ApplyAfterUnblockedDamageReceived:1349 |
| relic:TUNGSTEN_ROD | 152/0/8314/736 | AfterModifyingHpLostAfterOsty | registered | RelicEffects.ModifyHpLost:2652 |
|  |  | ModifyHpLostAfterOsty | registered | RelicEffects.ModifyHpLost:2652; RunNonCombatEffects.LoseHp:3506 |
| relic:STRIKE_DUMMY | 125/1/29288/3237 | ModifyDamageAdditive | registered | RelicEffects.CardDamageBonus:1705 |
| relic:RED_SKULL | 115/0/23624/2754 | AfterCurrentHpChanged | registered | RelicEffects.RedSkull_AfterCurrentHpChanged:1559 |
| relic:CENTENNIAL_PUZZLE | 112/0/43706/3356 | AfterDamageReceived | registered | RelicEffects.ApplyAfterUnblockedDamageReceived:1354 |
| relic:PEN_NIB | 100/1/18421/1762 | ModifyDamageMultiplicative | registered | RelicEffects.CardDamageMultiplier:1753 |
| relic:LIZARD_TAIL | 96/0/15678/798 | AfterPreventingDeath | registered | RelicEffects.ApplyAfterPlayerHpChanged:1581 |
|  |  | ShouldDieLate | registered | RelicEffects.LizardTailUnspent:1537; RelicEffects.ApplyAfterPlayerHpChanged:1581 |
| relic:GREMLIN_HORN | 89/1/16116/1768 | AfterDeath | registered | CombatEngine.AnswerDeaths:4572 |
| relic:MINIATURE_CANNON | 86/0/16031/1746 | ModifyDamageAdditive | registered | RelicEffects.CardDamageBonus:1694 |
| relic:BEATING_REMNANT | 75/0/10936/785 | AfterDamageReceived | registered | CardEffects.LosePlayerHp: its tally, UnblockedDamageThisTurn, counts every HP loss of the player |
|  |  | AfterModifyingHpLostAfterOsty | registered | RelicEffects.ModifyHpLost:2656 |
|  |  | ModifyHpLostAfterOsty | registered | RelicEffects.ModifyHpLost:2656 |
| relic:MYSTIC_LIGHTER | 74/0/4628/1323 | ModifyDamageAdditive | registered | RelicEffects.EnchantedCardDamageBonus:3023 |
| relic:MEAT_ON_THE_BONE | 70/0/10115/842 | AfterCurrentHpChanged | open | elsewhere: RunRewardGenerator.GenerateCombatRewards |
| relic:LAVA_LAMP | 34/0/3234/1349 | AfterDamageReceived | open | elsewhere: RelicEffects.UpgradesCardRewards |
| relic:DEMON_TONGUE | 25/1/6623/646 | AfterDamageReceived | registered | RelicEffects.DemonTongue_AfterDamageReceived:1324; RelicEffects.ApplyAfterUnblockedDamageReceived:1372; RelicEffects.ApplyAfterUnblockedDamageReceived:1373 |
| relic:FAKE_STRIKE_DUMMY | 0/0/0/223 | ModifyDamageAdditive | registered | RelicEffects.CardDamageBonus:1710 |

## relic-2

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| relic:HAND_DRILL | 0/0/0/164 | AfterDamageGiven | registered | RelicEffects.ApplyAfterBlockBroken:1997 |
| relic:THE_BOOT | 0/0/0/142 | AfterModifyingHpLostAfterOsty | registered | RelicEffects.BootDamageFloor:2015 |
|  |  | ModifyHpLostAfterOstyLate | registered | RelicEffects.BootDamageFloor:2015 |
| relic:VITRUVIAN_MINION | 0/0/0/1 | ModifyDamageMultiplicative | registered | RelicEffects.CardDamageMultiplier:1744 |

## card-1

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| card:MELANCHOLY | 0/1/30/52 | AfterDeath | registered | CardEffects.CheapenMelancholyForDeath:10896 |

## monster-1

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| monster:OVICOPTER | 39/0/12683/2230 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:SOUL_NEXUS | 25/0/5505/2395 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
|  |  | Died (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:THIEVING_HOPPER | 20/0/9164/4843 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.ApplyIntrinsicEnemyPowers, CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent |
| monster:CUBEX_CONSTRUCT | 19/0/3436/2585 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateCubexConstruct, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:DECIMILLIPEDE_SEGMENT_BACK | 18/0/14289/4887 | AfterDeath | open |  |
| monster:DECIMILLIPEDE_SEGMENT_FRONT | 18/0/14289/4887 | AfterDeath | open |  |
| monster:DECIMILLIPEDE_SEGMENT_MIDDLE | 18/0/14289/4887 | AfterDeath | open |  |
| monster:SLUMBERING_BEETLE | 18/0/7937/1835 | Died (pseudo) | open | elsewhere: CardEffects.Slumber_AfterDamageReceived, CombatFactory.CreateSlumberingBeetle, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:QUEEN | 17/0/8007/2383 | AfterDeath | registered | CombatEngine.EnrageQueenIfWaitingToBurnBright:5270 |
|  |  | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatEngine.EnrageQueenIfWaitingToBurnBright, CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ApplyDebuffIntent |
| monster:VANTOM | 14/0/958/2320 | AfterDeath | open | elsewhere: CombatFactory.CreateVantom, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:AEONGLASS | 12/0/28459/2473 | AfterDeath | open | elsewhere: CombatFactory.CreateAeonglass, EnemyAI.ApplyDebuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:LAGAVULIN_MATRIARCH | 12/0/2264/2073 | AfterDamageReceived | registered | EnemyAI.WakeMatriarchIfStruck:4873 |
|  |  | AfterDeath | open | elsewhere: CombatFactory.CreateLagavulinMatriarch, EnemyAI.ApplyDebuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:CRUSHER | 11/0/15357/2216 | AfterCurrentHpChanged | open | elsewhere: CombatFactory.CreateKaiserCrab, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
|  |  | BeforeDeath | open | elsewhere: CombatFactory.CreateKaiserCrab, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:ROCKET | 11/0/15357/2216 | AfterCurrentHpChanged | open | elsewhere: CombatFactory.CreateKaiserCrab, EnemyAI.ApplyBuffIntent, EnemyAI.SelectIntent |
|  |  | BeforeDeath | open | elsewhere: CombatFactory.CreateKaiserCrab, EnemyAI.ApplyBuffIntent, EnemyAI.SelectIntent |
|  |  | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateKaiserCrab, EnemyAI.ApplyBuffIntent, EnemyAI.SelectIntent |
| monster:OWL_MAGISTRATE | 11/0/4082/1085 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |

## monster-2

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| monster:THE_INSATIABLE | 11/0/2254/2197 | AfterDeath | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.SelectIntent |
| monster:SNAPPING_JAXFRUIT | 11/0/707/2061 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, CombatFactory.CreateSlitheringStranglerEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent |
| monster:KNOWLEDGE_DEMON | 10/0/2484/2154 | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyDebuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |
| monster:SOUL_FYSH | 10/0/955/2139 | AfterDeath | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ApplyDebuffIntent, EnemyAI.ExecuteIntent |
| monster:TEST_SUBJECT | 9/0/18022/2468 | AfterDeath | registered | CombatEngine.TryRespawnTestSubject:5479 |
|  |  | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatEngine.EndTurn, CombatEngine.TryRespawnTestSubject, CombatFactory.CreateTestSubject, EnemyAI.ApplyDebuffIntent |
| monster:WATERFALL_GIANT | 9/0/10245/2108 | AfterDeath | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.DiedHoldingSteamEruption, EnemyAI.ExecuteIntent |
|  |  | BeforeRemovedFromRoom (pseudo) | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.DiedHoldingSteamEruption, EnemyAI.ExecuteIntent |
| monster:KIN_PRIEST | 8/0/1002/2314 | AfterDeath | open | elsewhere: CombatFactory.CreateEncounter, EnemyAI.ApplyBuffIntent, EnemyAI.ExecuteIntent, EnemyAI.SelectIntent |

## potion-1

| model | reach c/a/l/x | hook | status | old location |
|---|---|---|---|---|
| potion:FAIRY_IN_A_BOTTLE | 77/0/14192/1350 | AfterPreventingDeath | open | elsewhere: CombatEngine.FairyInABottlePotionId, CombatEngine.IsAboutToLose, CombatEngine.PlayerIsDead, PotionEffects.CanBeGeneratedInCombat |
|  |  | ShouldDie | registered | CombatEngine.PlayerIsDead:3231; CombatEngine.IsAboutToLose:5143 |

