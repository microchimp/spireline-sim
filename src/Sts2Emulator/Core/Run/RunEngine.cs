using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;

namespace Sts2Emulator.Core.Run;

public sealed class RunEngine : Core2.ICombatEndHost
{
    public RunState State { get; private set; } = new();

    private Core2.Kernel? _kernel;

    /// <summary>
    /// The Tier 1 kernel's shared part (Core2/Kernel.cs), or null: the old engine. Set, it is bound to the run's state and
    /// to every combat this engine starts (a state keeps its own KernelState, copied with it). With no switch on nothing
    /// is bound.
    /// </summary>
    public Core2.Kernel? Kernel
    {
        get => _kernel;
        set
        {
            _kernel = value;
            State.Kernel = value?.Attach(null, State);
        }
    }

    /// <summary>Start a new run on this seed, standing at Neow.</summary>
    /// <param name="seed">The seed as typed; it is canonicalized before it is hashed.</param>
    /// <param name="ascension">
    /// The run's ascension level, 0 to <see cref="Ascension.MaxRunLevel" />. Every ascension
    /// rule in the run and in the combats it starts reads it. The default is
    /// <see cref="Ascension.DefaultLevel" />, the level the differential captures were taken
    /// at; the native export and the Python wrappers take it explicitly and default to 0.
    /// </param>
    /// <param name="character">
    /// Who plays the run: its starting deck, relic and max HP, and every pool the run reads
    /// "from the character's own" -- the relic bag above all, which is shuffled off UpFront
    /// before the rooms are, so the character moves every draw after it. Ironclad by default.
    /// </param>
    /// <param name="unlocks">
    /// The profile the run is generated for: <see cref="UnlockState.All" /> (every epoch, every
    /// encounter seen, every act discovered) by default. The seed alone does not decide a run;
    /// the act roll, the rooms, the ancients, the bosses and Neow's offer also read this.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">A level the run engine does not model.</exception>
    public void Reset(
        string seed,
        int ascension = Ascension.DefaultLevel,
        PlayerCharacter character = PlayerCharacter.Ironclad,
        UnlockState? unlocks = null
    )
    {
        if (ascension is < 0 or > Ascension.MaxRunLevel)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ascension),
                ascension,
                $"The run engine models ascension 0 to {Ascension.MaxRunLevel}."
            );
        }

        // A fresh state a run: everything a run keeps (encounter and event counters, pending
        // screens, relic bookkeeping) starts from RunState's own defaults, so nothing one run
        // leaves behind reaches the next. Clearing the old object field by field missed any
        // field added after the list was written.
        State = new RunState();
        State.Kernel = _kernel?.Attach(null, State);
        State.AscensionLevel = ascension;
        // Set before anything is generated: the relic bag reads the character's pool.
        State.Character = character;
        // `RunState.UnlockState`, which everything below generates from.
        State.Unlocks = unlocks ?? UnlockState.All;
        // Store the canonical seed, not the typed one: the game canonicalizes before it
        // hashes, so "abcdefo" and "ABCDEF0" are the same run.
        State.Rng = new RunRngSet(seed);
        State.StringSeed = State.Rng.StringSeed;
        State.PlayerRng = new PlayerRngSet(State.Rng);
        // The character's StartingHp, and then Neow: `AncientEventModel.BeforeEventStarted`
        // sets a Neow run's current HP to 0 and heals it like any ancient, so it starts full
        // below WearyTraveler and on 80% of it from it (64 for the Ironclad).
        State.PlayerMaxHp = RunConstants.StartingMaxHpFor(character);
        State.PlayerHp = 0;
        ApplyAncientHeal();
        State.Gold = 99;
        State.Floor = 1;
        State.Phase = RunPhase.Ancient;
        // `AscensionManager.ApplyEffectsTo` puts Ascender's Bane on the end of the deck
        // from AscendersBane up; below it the starting deck is the character's own.
        foreach (int cardId in RunConstants.StartingDeckIds(ascension, character))
        {
            RunNonCombatEffects.AddCardToDeck(State, new CardInstance(cardId, Upgraded: false));
        }

        State.Relics = [new RelicInstance(RunConstants.StartingRelicFor(character))];
        // `Player.CreateForNewRun` starts the belt at three slots, and
        // `AscensionManager.ApplyEffectsTo` takes one away under TightBelt.
        State.MaxPotionSlots = RunConstants.StartingPotionSlots(ascension);
        State.CurrentNodeType = RunConstants.NodeNormal;
        // CardRarityOdds starts at -0.05, not zero: `CardRarityOdds(Rng rng) : base(-0.05f, rng)`,
        // the same value a Rare roll resets it to. Starting at zero made every early
        // reward roll read 0.05 too generous, so cards the game rolled Common came
        // back Uncommon until the offset had grown past the gap.
        State.CardRarityOffset = RunRewardGenerator.CardRarityBaseOffset;
        GenerateNeowOptions();
        RunMapGenerator.SelectActAndGenerateRooms(State);
        RunMapGenerator.GenerateActMap(State);
        // Neow's room is a RoomType.Event (MapPointType.Ancient maps to it), and
        // RunManager marks EVERY room it enters -- so RoomSet.eventsVisited is already 1
        // by the time the run reaches its first real event, without anything having been
        // pulled from the list. NextEvent is events[eventsVisited % count], so the first
        // event a run actually meets is events[1]. Starting the cursor at 0 handed out
        // the entry the game had already stepped past.
        State.EventSequenceIndex = 1;
    }

    public int StartCombat(
        ReadOnlySpan<int> deckIds,
        int encounterId,
        ReadOnlySpan<int> relicIds,
        int playerHp,
        int playerMaxHp,
        ReadOnlySpan<int> potionIds,
        int playerGold
    )
    {
        int[] startingPotions = potionIds.ToArray();
        var runDeck = deckIds
            .ToArray()
            .Select(id => new CardInstance(Math.Abs(id), id < 0))
            .ToList();
        var runRelics = relicIds.ToArray().Select(id => new RelicInstance(id)).ToList();
        return StartCombatWithDeck(
            runDeck,
            encounterId,
            runRelics,
            playerHp,
            playerMaxHp,
            potionIds,
            playerGold
        );
    }

    private int StartCombatWithDeck(
        IReadOnlyList<CardInstance> runDeck,
        int encounterId,
        IReadOnlyList<RelicInstance> runRelics,
        int playerHp,
        int playerMaxHp,
        ReadOnlySpan<int> potionIds,
        int playerGold
    )
    {
        // `DelicateFrond.BeforeCombatStart` fills every open potion slot. Here because this
        // is the one hook every combat entry path in the run shares -- and BEFORE the slots
        // are snapshotted into the combat below, or the fight would be handed the belt as it
        // stood a moment earlier and the potions would exist only on the run.
        RunRewardGenerator.FillPotionSlotsForDelicateFrond(State);

        int[] startingPotions = potionIds.ToArray();
        State.Deck = runDeck.ToList();
        State.Relics = runRelics.ToList();
        State.PlayerHp = Math.Clamp(playerHp, 0, Math.Max(1, playerMaxHp));
        State.PlayerMaxHp = Math.Max(1, playerMaxHp);
        State.Gold = Math.Max(0, playerGold);
        Array.Clear(State.PotionSlots);
        for (int i = 0; i < Math.Min(State.PotionSlots.Length, startingPotions.Length); i++)
        {
            State.PotionSlots[i] = startingPotions[i];
        }

        var combatDeck = runDeck.ToArray();
        State.Rng.Shuffle.Shuffle(combatDeck);

        var shuffleRng = CountingRandom.At(State.Rng.Shuffle);

        // The fight is the run's, and so is its level: every monster's HP band and move
        // table reads it. A bare CombatState is the combat layer's own default of A8.
        var combat = new CombatState
        {
            AscensionLevel = State.AscensionLevel,
            Character = State.Character,
            Unlocks = State.Unlocks,
        };
        var combatRng = new CountingRandom(State.Rng.Niche.RawSeed);
        var nicheHpRng = CountingRandom.At(State.Rng.Niche);

        combat.NicheHpRng = nicheHpRng;

        // Monster move selection reads the run's "monster_ai" stream -- FlutterPower
        // reaches for Monster.RunRng.MonsterAi by name -- so it spans the whole run.
        // Restarting it each combat put every branching monster on the wrong draw:
        // Mawler's third choice came up Rip and Tear where the game rolled Claw.
        var aiRng = CountingRandom.At(State.Rng.MonsterAi);

        // Target choice draws from the run's "combat_targets" stream, so combat picks up
        // where the run left off and hands the call count back when it ends.
        var targetRng = CountingRandom.At(State.Rng.CombatTargets);

        combat.TargetRng = targetRng;

        var cardSelectionRng = CountingRandom.At(State.Rng.CombatCardSelection);

        combat.CardSelectionRng = cardSelectionRng;

        // Slither re-rolls its card's cost off Rng.CombatEnergyCosts every time the card
        // is drawn, and that stream spans the RUN like every other named one. This was the
        // only stream built without fast-forwarding to the run's position and the only one
        // never written back after the combat (see SyncRunRngFromCombat), so it restarted
        // at zero for every fight: two Slither draws in two different combats read the
        // same value, where the game reads consecutive ones. A live capture whose Wood
        // Carvings enchanted a Bash paid 1 energy for it where the emulator paid 2.
        var energyCostRng = CountingRandom.At(State.Rng.CombatEnergyCosts);

        combat.EnergyCostRng = energyCostRng;

        var cardGenerationRng = CountingRandom.At(State.Rng.CombatCardGeneration);

        combat.CardGenerationRng = cardGenerationRng;

        var potionGenerationRng = CountingRandom.At(State.Rng.CombatPotionGeneration);

        combat.PotionGenerationRng = potionGenerationRng;

        // `Rng.CombatOrbGeneration` -- Chaos and Trash to Treasure roll their orb type on
        // it. `RunRngSet` has carried the stream since it was written; nothing had reached
        // for it, because the one card that needed it was rolling on the combat rng.
        var orbGenerationRng = CountingRandom.At(State.Rng.CombatOrbs);

        combat.OrbGenerationRng = orbGenerationRng;

        // The belt's SIZE travels with its contents, and it has to be set BEFORE the reset:
        // `CombatFactory.Reset` runs `RelicEffects.ApplyCombatStart`, and Petrified Toad
        // procures a Shaped Rock into the first FREE slot from in there. `CombatState
        // .MaxPotionSlots` defaults to 3 and was never set from the run, so a fight always
        // believed in three slots and the toad made a third one out of nothing on a run
        // that had two. `TryToProcure` fails silently on a full belt, so the game gives no
        // rock at all. Setting it after the reset is too late by one relic -- which is how
        // this was written first. See E492.
        combat.MaxPotionSlots = State.MaxPotionSlots;

        // `FurCoat.BeforeCombatStart` bites the opening roster, which `CombatFactory.Reset` does
        // before turn one starts; the flag stays on the combat so `AfterCreatureAddedToCombat`
        // can bite reinforcements with the same rule -- see `RelicEffects.Spawned`.
        combat.FurCoatBites = State.FurCoatRooms.Contains(State.CurrentMapCoord);

        // A Combat-layout event's fight reuses the creatures the room built on entry, HP
        // and all (RunNonCombatEffects.BeginEvent): no Niche draw for them here.
        int[]? openingHps =
            State.Phase == RunPhase.Event
            && RunNonCombatEffects.EventCombatEncounter(State.EventId) == encounterId
                ? State.EventCombatHps
                : null;
        State.EventCombatHps = null;

        CombatFactory.Reset(
            combat,
            combatRng,
            combatDeck,
            encounterId,
            runRelics.Select(relic => relic.DefId).ToArray(),
            State.PlayerHp,
            State.PlayerMaxHp,
            startingPotions,
            State.Gold,
            deckPreShuffled: true,
            shuffleRng,
            EncounterRngSeed(encounterId),
            nicheSkipCount: 0,
            aiRng,
            runRelics,
            openingHps
        );
        // Sling of Courage reads the ROOM, not the encounter, so the room type travels
        // with the combat.
        combat.IsEliteRoom = State.LastResolvedRoomType == RunConstants.NodeElite;

        // `LavaLamp.AfterRoomEntered` clears its flag as the fight begins.
        State.TookUnblockedDamageThisCombat = false;

        Effects.RelicEffects.RestoreUsedUpRelics(combat, State.UsedUpRelics);

        // The pre-play phase runs HERE rather than on the first `Step`, because the run
        // hands this combat straight to `CombatEngine.ValidActions` -- an observation of a
        // state that must already include whatever plays itself. After the Fur Coat bites
        // above: the coat is `BeforeCombatStart` and pre-play is inside turn one, so a
        // creature it pins to 1hp must be pinned before an auto-played attack can reach it.
        CombatEngine.ResolvePrePlayPhase(combat, combatRng);

        combat.Kernel = _kernel?.Attach(combat, State);
        State.ActiveCombat = combat;
        State.ActiveCombatRng = combatRng;
        State.LastPlayerWon = false;
        State.Phase = RunPhase.Combat;
        SyncNicheRngFromCombat();
        return 0;
    }

    private int EnterUnknownMapPoint()
    {
        int resolvedNodeType = RollUnknownMapPointNodeType();
        State.UnknownMapPointsVisited++;
        State.CurrentNodeType = resolvedNodeType;
        State.LastResolvedRoomType = resolvedNodeType;
        RunNonCombatEffects.PayMawBank(State);

        // `Planisphere.AfterRoomEntered` asks about `CurrentMapPoint.PointType`, not about
        // the room -- so it pays out on whatever the "?" turned out to be, a fight
        // included. Fired here, before the room is set up, because every branch below
        // either starts a combat or changes phase and there is no single point after them.
        Effects.RelicEffects.ApplyAfterRoomEntered(
            State,
            isRestSite: resolvedNodeType == RunConstants.NodeRest,
            cameFromUnknown: true
        );

        switch (resolvedNodeType)
        {
            case RunConstants.NodeNormal:
                State.NormalEncountersVisited++;
                return StartCombatWithDeck(
                    State.Deck,
                    State.NormalEncounterSequence[
                        (State.NormalEncountersVisited - 1) % State.NormalEncounterSequence.Length
                    ],
                    State.Relics,
                    State.PlayerHp,
                    State.PlayerMaxHp,
                    State.PotionSlots,
                    State.Gold
                );
            case RunConstants.NodeElite:
                State.EliteEncountersVisited++;
                return StartCombatWithDeck(
                    State.Deck,
                    State.EliteEncounterSequence[
                        (State.EliteEncountersVisited - 1) % State.EliteEncounterSequence.Length
                    ],
                    State.Relics,
                    State.PlayerHp,
                    State.PlayerMaxHp,
                    State.PotionSlots,
                    State.Gold
                );
            case RunConstants.NodeShop:
                RunRewardGenerator.EnterShop(State);
                return 0;
            case RunConstants.NodeRelic:
                RunRewardGenerator.EnterTreasureRoom(State);
                return 0;
            case RunConstants.NodeEvent:
                return EnterEventRoom();
            case RunConstants.NodeAncient:
                // The act's ancient, which behaves exactly as Neow does: a screen of
                // blessings, then one more Proceed before the map.
                State.LastResolvedRoomType = RunConstants.NodeEvent;
                ApplyAncientHeal();
                EnterActAncient();
                return 0;
            default:
                EnterMapPhase();
                return 0;
        }
    }

    private int EnterEventRoom()
    {
        // An event room is not a Monster room, and anything a fight-shaped event starts
        // from here inherits that. Left unset, the type stayed at whatever the last map
        // node resolved to, so Punch Off's fight could be counted as an ordinary combat.
        State.LastResolvedRoomType = RunConstants.NodeEvent;
        // An Underdocks event on floor 13 used to be replaced by a fixed Corpse Slugs fight: a
        // survivor of the capture-fitted "answer key" upstream deleted in 8244eee. The game has
        // no such rule (Underdocks, RunManager.EnterMapPointInternal), so every event room is an event.
        RunNonCombatEffects.EnterEvent(State);

        return 0;
    }

    private int RollUnknownMapPointNodeType()
    {
        // `GoldenCompass.ModifyUnknownMapPointRoomTypes` narrows the set to Event alone,
        // for its own act only. A one-element set is not a roll -- the golden path's "?"
        // rooms are events, and the stream is not touched.
        if (
            State.Relics.Any(relic => relic.DefId == RunConstants.RelicGoldenCompass)
            && State.GoldenPathAct == State.CurrentActIndex
        )
        {
            return RunConstants.NodeEvent;
        }

        var allowedRoomTypes = new HashSet<int>
        {
            RunConstants.NodeNormal,
            RunConstants.NodeElite,
            RunConstants.NodeRelic,
            RunConstants.NodeShop,
            RunConstants.NodeEvent,
        };
        if (IsUnknownShopBlacklisted())
        {
            allowedRoomTypes.Remove(RunConstants.NodeShop);
        }

        // `JuzuBracelet.ModifyUnknownMapPointRoomTypes` drops Monster from the SET before
        // the odds are rolled, so its probability mass redistributes across what is left
        // rather than the roll being taken again.
        if (Effects.RelicEffects.ForbidsUnknownMonsterRooms(State))
        {
            allowedRoomTypes.Remove(RunConstants.NodeNormal);
        }

        int roomType = allowedRoomTypes.Contains(RunConstants.NodeEvent)
            ? RunConstants.NodeEvent
            : allowedRoomTypes.Min();
        // UnknownMapPointOdds.Roll draws a FLOAT and accumulates the odds as floats:
        // `float num2 = _rng.NextFloat()` compared against a float running total. The
        // draw is the same value either way, but the comparison is not -- a roll sitting
        // within a float's precision of a threshold falls one side in float arithmetic
        // and the other in double, which resolves a "?" to a different room entirely.
        float roll = State.Rng.UnknownMapPoint.NextFloat();
        float cumulative = 0f;
        foreach (
            var (candidateRoomType, odds) in new (int RoomType, double Odds)[]
            {
                (RunConstants.NodeNormal, State.UnknownMapPointMonsterOdds),
                (RunConstants.NodeElite, State.UnknownMapPointEliteOdds),
                (RunConstants.NodeRelic, State.UnknownMapPointTreasureOdds),
                (RunConstants.NodeShop, State.UnknownMapPointShopOdds),
            }
        )
        {
            if (!allowedRoomTypes.Contains(candidateRoomType) || odds < 0.0)
            {
                continue;
            }

            cumulative += (float)odds;
            if (roll <= cumulative)
            {
                roomType = candidateRoomType;
                break;
            }
        }

        UpdateUnknownMapPointOdds(roomType, allowedRoomTypes);
        return roomType;
    }

    private bool IsUnknownShopBlacklisted()
    {
        if (State.LastResolvedRoomType == RunConstants.NodeShop)
        {
            return true;
        }

        if (
            !State.MapNodes.TryGetValue(State.CurrentMapCoord, out var current)
            || current.Children.Count == 0
        )
        {
            return false;
        }

        return current.Children.All(child =>
            State.MapNodes.TryGetValue(child, out var node)
            && node.NodeType == RunConstants.NodeShop
        );
    }

    private void UpdateUnknownMapPointOdds(int rolledRoomType, IReadOnlySet<int> allowedRoomTypes)
    {
        UpdateUnknownMapPointOdd(
            rolledRoomType,
            allowedRoomTypes,
            RunConstants.NodeNormal,
            0.1,
            odds => State.UnknownMapPointMonsterOdds = odds,
            () => State.UnknownMapPointMonsterOdds
        );
        UpdateUnknownMapPointOdd(
            rolledRoomType,
            allowedRoomTypes,
            RunConstants.NodeElite,
            -1.0,
            odds => State.UnknownMapPointEliteOdds = odds,
            () => State.UnknownMapPointEliteOdds
        );
        UpdateUnknownMapPointOdd(
            rolledRoomType,
            allowedRoomTypes,
            RunConstants.NodeRelic,
            0.02,
            odds => State.UnknownMapPointTreasureOdds = odds,
            () => State.UnknownMapPointTreasureOdds
        );
        UpdateUnknownMapPointOdd(
            rolledRoomType,
            allowedRoomTypes,
            RunConstants.NodeShop,
            0.03,
            odds => State.UnknownMapPointShopOdds = odds,
            () => State.UnknownMapPointShopOdds
        );
    }

    private static void UpdateUnknownMapPointOdd(
        int rolledRoomType,
        IReadOnlySet<int> allowedRoomTypes,
        int roomType,
        double baseOdds,
        Action<double> setOdds,
        Func<double> getOdds
    )
    {
        if (rolledRoomType == roomType)
        {
            setOdds(baseOdds);
        }
        else if (allowedRoomTypes.Contains(roomType))
        {
            setOdds(getOdds() + baseOdds);
        }
    }

    public void WriteActionMask(Span<int> mask)
    {
        mask.Clear();
        WritePhaseActionMask(mask);
        for (int slot = 0; slot < RunConstants.PotionSlotCapacity; slot++)
        {
            if (RunPotionUse.CanUse(State, slot))
            {
                SetMask(mask, RunConstants.UsePotionAction + slot);
            }
        }
    }

    private void WritePhaseActionMask(Span<int> mask)
    {
        switch (State.Phase)
        {
            case RunPhase.Combat:
                if (State.ActiveCombat is not null)
                {
                    foreach (int action in CombatEngine.ValidActions(State.ActiveCombat))
                    {
                        SetMask(mask, action);
                    }
                }

                break;

            case RunPhase.CardReward:
                for (int i = 0; i <= RunConstants.RewardSkipAction; i++)
                {
                    SetMask(mask, i);
                }

                // `PaelsWing.TryModifyCardRewardAlternatives` adds SACRIFICE whenever the
                // relic is held. It is an alternative to taking a card, not to skipping:
                // it ends the screen the same way a claim does.
                if (State.Relics.Any(relic => relic.DefId == RunConstants.RelicPaelsWing))
                {
                    SetMask(mask, RunConstants.RewardSacrificeAction);
                }

                // Lasting Candy's fourth option, offered only on the screens that HAVE one
                // -- every second combat, and never after a reroll.
                if (State.LastingCandyRewardCard != 0)
                {
                    SetMask(mask, RunConstants.RewardFourthCardAction);
                }

                if (State.CardRewardRerollAvailable)
                {
                    SetMask(mask, RunConstants.RewardRerollAction);
                }

                break;

            case RunPhase.Map:
                for (int i = 0; i < State.MapNodeTypes.Length; i++)
                {
                    if (State.MapNodeTypes[i] != RunConstants.NodeNone)
                    {
                        SetMask(mask, i);
                    }
                }

                break;

            case RunPhase.Rest:
                if (!RestOptionSpent(RunConstants.RestHealAction))
                {
                    SetMask(mask, RunConstants.RestHealAction);
                }

                if (
                    State.Deck.Any(RunConstants.IsRunCardUpgradable)
                    && !RestOptionSpent(RunConstants.RestUpgradeAction)
                )
                {
                    SetMask(mask, RunConstants.RestUpgradeAction);
                }

                if (State.Relics.Any(relic => relic.DefId == RunConstants.RelicPaelsGrowth))
                {
                    if (!RestOptionSpent(RunConstants.RestCloneAction))
                    {
                        SetMask(mask, RunConstants.RestCloneAction);
                    }
                }

                // `Girya.TryModifyRestSiteOptions` returns FALSE once three lifts are
                // spent, so the option leaves the screen rather than becoming a no-op.
                // The count is kept on the relic instance, as the game keeps it on the
                // relic model.
                if (GiryaLiftsLeft() > 0)
                {
                    if (!RestOptionSpent(RunConstants.RestLiftAction))
                    {
                        SetMask(mask, RunConstants.RestLiftAction);
                    }
                }

                // `Shovel.TryModifyRestSiteOptions` adds Dig unconditionally while the
                // relic is held; digging with an empty bag falls back the way any relic
                // pull does.
                if (State.Relics.Any(relic => relic.DefId == Effects.RelicEffects.Shovel))
                {
                    if (!RestOptionSpent(RunConstants.RestDigAction))
                    {
                        SetMask(mask, RunConstants.RestDigAction);
                    }
                }

                // `ByrdonisEgg.TryModifyRestSiteOptions`: an egg in the DECK adds HATCH.
                // The only rest option a CARD puts there rather than a relic, so it reads
                // the deck rather than the relic list.
                if (
                    State.Deck.Any(card => card.DefId == RunNonCombatEffects.ByrdonisEggCard)
                    && !RestOptionSpent(RunConstants.RestHatchAction)
                )
                {
                    SetMask(mask, RunConstants.RestHatchAction);
                }

                // `MeatCleaver.TryModifyRestSiteOptions` adds COOK whenever the relic is
                // held; `CookRestSiteOption.IsEnabled` is what wants two removable cards.
                // The emulator has no disabled-but-present option, so the enable check is
                // what gates the move.
                if (
                    State.Relics.Any(relic => relic.DefId == RunConstants.RelicMeatCleaver)
                    && RunNonCombatEffects.CanCook(State)
                    && !RestOptionSpent(RunConstants.RestCookAction)
                )
                {
                    SetMask(mask, RunConstants.RestCookAction);
                }

                // `PumpkinCandle.TryModifyRestSiteOptions` adds KINDLE unconditionally
                // while the relic is held -- there is no cap on the count, so it stays on
                // the screen even with the candle already full.
                if (
                    State.Relics.Any(relic => relic.DefId == RunConstants.RelicPumpkinCandle)
                    && !RestOptionSpent(RunConstants.RestKindleAction)
                )
                {
                    SetMask(mask, RunConstants.RestKindleAction);
                }

                SetMask(mask, RunConstants.RewardSkipAction);
                break;

            case RunPhase.FakeMerchant:
            {
                var stall = RunNonCombatEffects.FakeMerchantStock(State);
                for (int i = 0; i < stall.Count; i++)
                {
                    if (stall[i] != 0 && State.Gold >= RunNonCombatEffects.FakeMerchantPrice(State, i))
                    {
                        SetMask(mask, i);
                    }
                }

                // The potion is thrown from the belt, so the move exists exactly when the
                // run is carrying one -- which is half of the event's own IsAllowed.
                if (State.PotionSlots.Contains(RunNonCombatEffects.NamedPotion("FoulPotion")))
                {
                    SetMask(mask, RunConstants.FakeMerchantThrowAction);
                }

                SetMask(mask, RunConstants.FakeMerchantLeaveAction);
                break;
            }

            case RunPhase.Shop:
                for (int i = 0; i < State.ShopCards.Length; i++)
                {
                    if (State.ShopCards[i] != 0 && State.Gold >= State.ShopCosts[i])
                    {
                        SetMask(mask, i);
                    }
                }

                for (int action = 7; action < 10; action++)
                {
                    if (State.ShopRelics[action - 7] != 0 && State.Gold >= State.ShopCosts[action])
                    {
                        SetMask(mask, action);
                    }
                }

                // USABLE slots only. The array is `PotionSlotCapacity` wide so the
                // observation has a stable shape; the slots past `MaxPotionSlots` are
                // not free, they do not exist yet.
                bool hasPotionSlot =
                    !Effects.RelicEffects.Has(State.Relics, Effects.RelicEffects.Sozu)
                    && State.UsablePotionSlots().Any(potion => potion == 0);
                for (int action = 10; action < 13; action++)
                {
                    if (
                        State.ShopPotions[action - 10] != 0
                        && State.Gold >= State.ShopCosts[action]
                        && hasPotionSlot
                    )
                    {
                        SetMask(mask, action);
                    }
                }

                if (
                    !State.ShopRemovalUsedThisVisit
                    && State.Gold >= State.ShopCosts[RunConstants.ShopRemoveAction]
                    && State.Deck.Count > 1
                )
                {
                    SetMask(mask, RunConstants.ShopRemoveAction);
                }

                SetMask(mask, RunConstants.ShopSkipAction);
                break;

            case RunPhase.RelicReward:
                WriteRewardActionMask(mask);
                SetMask(mask, RunConstants.RewardSkipAction);
                break;

            case RunPhase.Event:
                WriteEventActionMask(mask);
                break;

            case RunPhase.Ancient when State.NeowAwaitingProceed:
                SetMask(mask, 0);
                break;

            case RunPhase.Ancient:
                for (int i = 0; i < State.NeowOptions.Length; i++)
                {
                    if (State.NeowOptions[i] != 0)
                    {
                        SetMask(mask, i);
                    }
                }

                break;

            case RunPhase.BundleSelect:
                for (int i = 0; i < BundleCount; i++)
                {
                    SetMask(mask, i);
                }

                // Confirm is only offered once something is highlighted, which is what
                // the screen does -- its button is dead until a bundle is picked.
                if (State.SelectedBundle >= 0)
                {
                    SetMask(mask, RunConstants.BundleConfirmAction);
                }

                break;

            case RunPhase.TransformSelect:
                if (State.PendingOfferCards.Length > 0)
                {
                    // The grid is the offer, so the action indexes the offer, not the
                    // deck.
                    for (int i = 0; i < State.PendingOfferCards.Length; i++)
                    {
                        SetMask(mask, i);
                    }

                    // One past the last card is DONE, for a grid whose MinSelect is zero.
                    // Without it the only way off a Sea Glass screen is to take all
                    // fifteen, which is a different deck from the one a player who wanted
                    // three would have.
                    if (State.PendingOfferOptional)
                    {
                        SetMask(mask, State.PendingOfferCards.Length);
                    }
                }
                else if (State.PendingRestUpgrade)
                {
                    for (int i = 0; i < State.Deck.Count; i++)
                    {
                        if (RunConstants.IsRunCardUpgradable(State.Deck[i]))
                        {
                            SetMask(mask, i);
                        }
                    }
                }
                else if (State.PendingSelectionKind != DeckSelection.None)
                {
                    for (int i = 0; i < State.Deck.Count; i++)
                    {
                        if (RunNonCombatEffects.CanSelectCard(State, i))
                        {
                            SetMask(mask, i);
                        }
                    }

                    // One past the last card is DONE, for a selection whose minimum is zero.
                    if (State.PendingSelectionOptional)
                    {
                        SetMask(mask, State.Deck.Count);
                    }
                }
                else
                {
                    for (int i = 0; i < State.Deck.Count; i++)
                    {
                        SetMask(mask, i);
                    }
                }
                break;

            case RunPhase.Treasure:
                // The chest's relic while it is still in the chest; leaving is always there.
                if (State.RelicReward != 0)
                {
                    SetMask(mask, RunConstants.TreasureTakeRelicAction);
                }

                SetMask(mask, RunConstants.RewardSkipAction);
                break;

            case RunPhase.CrystalSphere when State.CrystalSphere is not null:
                // Cells that would uncover nothing are left out. The game lets a
                // divination be spent on clear ground and simply wastes it; nothing an
                // agent could learn makes that worth offering.
                for (int cell = 0; cell < RunConstants.CrystalSphereCells; cell++)
                {
                    int x = cell / RunConstants.CrystalSphereSize;
                    int y = cell % RunConstants.CrystalSphereSize;
                    if (State.CrystalSphere.WouldUncover(x, y, CrystalSphereTool.Big))
                    {
                        SetMask(mask, cell);
                    }

                    if (State.CrystalSphere.WouldUncover(x, y, CrystalSphereTool.Small))
                    {
                        SetMask(mask, RunConstants.CrystalSphereSmallToolAction + cell);
                    }
                }

                break;
        }
    }

    /// <summary>
    /// A deep copy of this run, for a search that wants to fork a position instead of
    /// replaying to reach it.
    /// </summary>
    /// <param name="resampleSeed">
    /// When given, everything the run has not yet paid out is resampled off this seed:
    /// future rewards, shop stock, encounter composition, shuffles, and the part of the
    /// draw pile the player has not been shown. What has already happened is untouched.
    /// Omit it for a faithful copy -- which is an oracle, and must not be searched with.
    /// See docs/agent-interface.md.
    /// </param>
    public RunEngine Clone(int? resampleSeed = null)
    {
        var copy = new RunEngine
        {
            State = State.Clone(
                resampleSeed is null ? null : ResampledSeedString(resampleSeed.Value)
            ),
        };
        if (resampleSeed is not null)
        {
            copy.ResampleHiddenState(new Random(resampleSeed.Value));
        }

        return copy;
    }

    /// <summary>The resampled run's seed, distinct from any seed a player could type.</summary>
    private static string ResampledSeedString(int resampleSeed) =>
        $"RESAMPLE{unchecked((uint)resampleSeed):X8}";

    /// <summary>
    /// Point the in-combat streams at the resampled run's streams, reshuffle the region of
    /// the draw pile the player has not seen, roll the enemies' current moves again behind
    /// the intents they show (<see cref="IntentParticles" />), redraw the fights not yet
    /// entered, and move whatever the Crystal Sphere's fog is still hiding.
    /// </summary>
    /// <remarks>
    /// The combat streams keep their call counts: SyncAfterCombat hands those counts
    /// back to the run streams when the fight ends, so the two have to stay in step
    /// even though the values behind them have changed.
    /// </remarks>
    private void ResampleHiddenState(Random rng)
    {
        State.CrystalSphere?.ResampleUnseenItems(rng);
        RunMapGenerator.ResampleUnenteredEncounters(
            State,
            new GameRng(State.Rng.Seed, "resample_encounters")
        );
        RunMapGenerator.ResampleUnseenActContents(
            State,
            new GameRng(State.Rng.Seed, "resample_acts")
        );

        var combat = State.ActiveCombat;
        if (combat is null)
        {
            return;
        }

        State.ActiveCombatRng = RestreamCombat(combat, State.ActiveCombatRng, State.Rng);
        ReshuffleUnknownDrawPile(combat, rng);
        IntentParticles.Resample(combat, rng);
    }

    /// <summary>
    /// A copy of the fight this run stands in, with the stream its steps draw from, for a
    /// search that forks combat handles rather than whole runs (DESIGN.md 6.5).
    /// </summary>
    /// <param name="resampleSeed">
    /// When given, the copy is exactly the fight of <see cref="Clone" /> with this seed: its
    /// streams rebuilt off the resampled run's seed at the same call counts, and the same
    /// reshuffle of the draw pile the player has not seen, drawn after whatever the Crystal
    /// Sphere's resample draws first. Only the fight is copied, never the run around it.
    /// Omit it for a faithful copy, which is an oracle.
    /// </param>
    /// <returns>The fight and its combat stream, or null when the run is not in a fight.</returns>
    public (CombatState Combat, CountingRandom Rng)? ForkActiveCombat(int? resampleSeed = null)
    {
        if (
            State.Phase != RunPhase.Combat
            || State.ActiveCombat is null
            || State.ActiveCombatRng is null
        )
        {
            return null;
        }

        var combat = State.ActiveCombat.Clone();
        var combatRng = State.ActiveCombatRng.Clone();
        if (resampleSeed is int seed)
        {
            var random = new Random(seed);
            State.CrystalSphere?.Clone().ResampleUnseenItems(random);
            var streams = new RunRngSet(ResampledSeedString(seed));
            combatRng = RestreamCombat(combat, combatRng, streams)!;
            ReshuffleUnknownDrawPile(combat, random);
            IntentParticles.Resample(combat, random);
        }

        return (combat, combatRng);
    }

    /// <summary>
    /// Point a fight's streams at <paramref name="streams" />, each at its own call count,
    /// and return its combat stream pointed at theirs the same way.
    /// </summary>
    private static CountingRandom? RestreamCombat(
        CombatState combat,
        CountingRandom? combatRng,
        RunRngSet streams
    )
    {
        combat.ShuffleRng = Restream(combat.ShuffleRng, streams.Shuffle);
        combat.NicheHpRng = Restream(combat.NicheHpRng, streams.Niche);
        combat.TargetRng = Restream(combat.TargetRng, streams.CombatTargets);
        combat.CardSelectionRng = Restream(combat.CardSelectionRng, streams.CombatCardSelection);
        combat.EnergyCostRng = Restream(combat.EnergyCostRng, streams.CombatEnergyCosts);
        combat.CardGenerationRng = Restream(combat.CardGenerationRng, streams.CombatCardGeneration);
        combat.PotionGenerationRng = Restream(
            combat.PotionGenerationRng,
            streams.CombatPotionGeneration
        );
        combat.OrbGenerationRng = Restream(combat.OrbGenerationRng, streams.CombatOrbs);
        combat.AiRng = Restream(combat.AiRng as CountingRandom, streams.MonsterAi);
        return Restream(combatRng, streams.Niche);
    }

    private static CountingRandom? Restream(CountingRandom? current, GameRng stream)
    {
        if (current is null)
        {
            return null;
        }

        var replacement = new CountingRandom(stream.RawSeed);
        for (int i = 0; i < current.CallCount; i++)
        {
            replacement.Next();
        }

        return replacement;
    }

    /// <summary>
    /// Shuffle the part of the draw pile between the known top and the known bottom.
    /// Deliberately a plain uniform shuffle, not the game's StableShuffle: the point is
    /// to sample a world the player cannot rule out, not to reproduce the game's order.
    /// </summary>
    private static void ReshuffleUnknownDrawPile(CombatState combat, Random rng)
    {
        int start = combat.KnownTopCount;
        int end = combat.DrawPile.Count - combat.KnownBottomCount;
        for (int i = end - 1; i > start; i--)
        {
            int j = start + rng.Next(i - start + 1);
            (combat.DrawPile[i], combat.DrawPile[j]) = (combat.DrawPile[j], combat.DrawPile[i]);
        }
    }

    public void WriteObservation(Span<int> obs)
    {
        if (obs.Length < RunConstants.RunObsSize)
        {
            throw new ArgumentException("Run observation buffer is too small.", nameof(obs));
        }

        obs[..RunConstants.RunObsSize].Clear();
        CombatState? activeCombat = State.Phase == RunPhase.Combat ? State.ActiveCombat : null;
        if (activeCombat is not null)
        {
            CombatObservation.Write(activeCombat, obs);
        }

        int offset = RunConstants.CombatObsSize;
        int playerHp = activeCombat?.PlayerHp ?? State.PlayerHp;
        int playerMaxHp = activeCombat?.PlayerMaxHp ?? State.PlayerMaxHp;
        int gold = activeCombat?.PlayerGold ?? State.Gold;
        int[] potionSlots = activeCombat?.PotionSlots ?? State.PotionSlots;
        obs[offset + 0] = (int)State.Phase;
        obs[offset + 1] = State.Floor;
        obs[offset + 2] = State.Act;
        obs[offset + 3] = State.Deck.Count;
        obs[offset + 4] = gold;
        obs[offset + 5] = playerHp;
        obs[offset + 6] = playerMaxHp;
        obs[offset + 7] = State.Relics.Count;
        obs[offset + 8] = State.CurrentNodeType;
        obs[offset + 9] = State.RewardCards[0];
        obs[offset + 10] = State.RewardCards[1];
        obs[offset + 11] = State.RewardCards[2];
        // The FOURTH option lives at the end of the observation rather than here, so that
        // adding it moved nothing. See `RunConstants.FourthRewardObsOffset`.
        obs[offset + RunConstants.FourthRewardObsOffset] = State.LastingCandyRewardCard;
        obs[offset + RunConstants.FourthRewardObsOffset + 1] = State.LastingCandyRewardUpgraded
            ? 1
            : 0;
        for (int i = 0; i < RunConstants.MapChoices; i++)
        {
            // Types only. What is BEHIND a node is not on the game's map -- see
            // RunConstants, where the encounter block used to be.
            obs[offset + RunConstants.MapNodeTypeObsOffset + i] = State.MapNodeTypes[i];
        }

        obs[offset + RunConstants.RelicRewardObsOffset] = State.RelicReward;
        obs[offset + RunConstants.CurrentEventObsOffset] = State.EventId;
        for (int i = 0; i < RunConstants.PotionObsSlots; i++)
        {
            obs[offset + RunConstants.PotionObsOffset + i] = potionSlots[i];
        }

        WriteDeckObservation(obs[(offset + RunConstants.DeckObsOffset)..]);
        WriteRelicObservation(obs[(offset + RunConstants.RelicObsOffset)..]);
        WriteShopObservation(obs[(offset + RunConstants.ShopObsOffset)..]);
    }

    /// <summary>
    /// The deck, card by card. Everything before this reported <c>Deck.Count</c> and
    /// nothing else, which left every decision the run is actually about -- which card
    /// reward to take, what to buy, what to upgrade at a fire, what to transform --
    /// unlearnable: the agent chose between three cards without being told what the other
    /// twenty in its deck were.
    ///
    /// Persistent identity only. <c>CostForCombat</c>, <c>BonusDamage</c>, <c>Retain</c>
    /// and <c>FreeThisTurn</c> are combat-local and belong to the combat observation; what
    /// survives a fight is the card, whether it is upgraded, and its one enchantment with
    /// the amount it was applied at.
    /// </summary>
    private void WriteDeckObservation(Span<int> obs)
    {
        int slots = Math.Min(State.Deck.Count, RunConstants.MaxObservedDeck);
        for (int i = 0; i < slots; i++)
        {
            var card = State.Deck[i];
            int at = i * RunConstants.DeckSlotSize;
            obs[at] = card.DefId;
            obs[at + 1] = card.Upgraded ? 1 : 0;
            obs[at + 2] = (int)card.Enchantment;
            obs[at + 3] = card.EnchantAmount;
        }
    }

    /// <summary>
    /// The relics, in the order they were picked up. The counter is on the slot because
    /// half of what a relic is worth is how much of it is left -- a Silver Crucible with no
    /// charges is a different relic from one with three -- and a used-up relic is reported
    /// as such, since the run keeps carrying it after it stops doing anything.
    /// </summary>
    /// <summary>
    /// The merchant's board, slot by slot, priced. The scalars used to carry three of the
    /// seven cards and none of the prices at all -- so an agent could buy shop slot 5
    /// without ever being shown what was on it, and could not tell a 50-gold card from a
    /// 300-gold one on any slot. Every purchase decision a shop is for turns on the price.
    ///
    /// Indexed by the action that buys the slot, so this reads straight against the shop's
    /// action mask. The removal service is slot 13 and has no item, so its id stays 0.
    /// </summary>
    private void WriteShopObservation(Span<int> obs)
    {
        for (int action = 0; action < RunConstants.ShopSlots; action++)
        {
            int at = action * RunConstants.ShopSlotSize;
            obs[at] = action switch
            {
                < 7 => State.ShopCards[action],
                < 10 => State.ShopRelics[action - 7],
                < 13 => State.ShopPotions[action - 10],
                _ => 0,
            };
            obs[at + 1] = State.ShopCosts[action];
        }
    }

    private void WriteRelicObservation(Span<int> obs)
    {
        int slots = Math.Min(State.Relics.Count, RunConstants.MaxObservedRelics);
        for (int i = 0; i < slots; i++)
        {
            var relic = State.Relics[i];
            int at = i * RunConstants.RelicSlotSize;
            obs[at] = relic.DefId;
            obs[at + 1] = relic.Counter;
            obs[at + 2] = State.UsedUpRelics.Contains(relic.DefId) ? 1 : 0;
        }
    }

    public void WriteInfo(Span<int> info)
    {
        if (info.Length < RunConstants.RunInfoSize)
        {
            throw new ArgumentException("Run info buffer is too small.", nameof(info));
        }

        info[..RunConstants.RunInfoSize].Clear();
        CombatState? activeCombat = State.Phase == RunPhase.Combat ? State.ActiveCombat : null;
        int playerHp = activeCombat?.PlayerHp ?? State.PlayerHp;
        int playerMaxHp = activeCombat?.PlayerMaxHp ?? State.PlayerMaxHp;
        int gold = activeCombat?.PlayerGold ?? State.Gold;
        info[0] = (int)State.Phase;
        info[1] = State.Floor;
        info[2] = State.Act;
        info[3] = State.Deck.Count;
        info[4] = gold;
        info[5] = playerHp;
        info[6] = playerMaxHp;
        info[7] = State.Relics.Count;
        info[8] = State.CurrentNodeType;
        info[9] = State.EventId;
        info[10] = State.RelicReward;
    }

    public int Step(
        int action,
        int targetEnemyIndex,
        out float reward,
        out bool terminal,
        out bool truncated
    )
    {
        int status;
        if (_kernel is { AnyOn: true } && State.Phase == RunPhase.Combat && State.ActiveCombat is { } fight)
        {
            // The kernel's entry gate (TIER1_KERNEL.md section 5): with a switch on, every combat stepped carries its
            // KernelState, bound where the combat was built; a combat without one would run its sites old, silently.
            var k = fight.Kernel ?? throw new InvalidOperationException(
                "kernel guard: a switch is on and the combat about to be stepped has no KernelState (bind it where the combat was built)");
            status = QueueStep(fight, k, action, targetEnemyIndex, out reward, out terminal, out truncated);
        }
        else
        {
            status = StepInner(action, targetEnemyIndex, out reward, out terminal, out truncated);
        }

        // A run ends the moment the player runs out of HP, wherever that happens. Combat
        // had its own path; nothing outside combat did, so an event that took the last of
        // the player's HP left the run walking around at 0 and an agent learned that a
        // lethal option was free. Guarding here rather than at each site means no new
        // effect can forget it.
        if (status == 0 && !terminal && State.PlayerHp <= 0)
        {
            State.PlayerHp = 0;
            State.LastPlayerWon = false;
            State.Phase = RunPhase.Complete;
            terminal = true;
        }

        return status;
    }

    /// <summary>
    /// The developers' console <c>fight</c> as the referee's setup fights drive it
    /// (<c>RunManager.EnterMapCoordDebug</c> with an encounter, the encounter's own Rng seeded as in
    /// any run): walk onto map option <paramref name="option"/> exactly as the map would, but meet
    /// <paramref name="encounterId"/> in a room of <paramref name="nodeType"/> (normal, elite or
    /// boss) instead of the act's next encounter. The id alone names the encounter: a weak fight and
    /// its normal twin are two ids (Corpse Slugs 9 and 88, Seapunk 12 and 22), as they are two
    /// models in the game, so nothing about the run's rooms so far is consulted. Returns -1 off the
    /// map or for another node type.
    /// </summary>
    public int EnterEncounterDebug(int option, int encounterId, int nodeType)
    {
        if (
            State.Phase != RunPhase.Map
            || (uint)option >= RunConstants.MapChoices
            || State.MapNodeTypes[option] == RunConstants.NodeNone
            || nodeType is not (RunConstants.NodeNormal or RunConstants.NodeElite or RunConstants.NodeBoss)
        )
        {
            return -1;
        }

        State.MapNodeTypes[option] = nodeType;
        State.MapChoices[option] = encounterId;
        return Step(option, -1, out _, out _, out _);
    }

    /// <summary>
    /// Binds the kernel's state to the active combat when a switch is on and the combat holds none: a combat built
    /// outside this engine (a snapshot restore, a re-anchor) before it is stepped. A no-op on the old engine.
    /// </summary>
    public void BindActiveCombat()
    {
        if (State.ActiveCombat is { } c)
        {
            BindCombat(c);
        }
    }

    /// <summary>
    /// <see cref="BindActiveCombat" /> for a combat that is not the run's active one (a re-anchored fight is built and
    /// stepped on its own): binds the kernel's state when a switch is on and the combat holds none.
    /// </summary>
    public void BindCombat(Sts2Emulator.Core.CombatState combat)
    {
        if (_kernel is { AnyOn: true } && combat.Kernel is null)
        {
            _kernel.Bind(combat, State);
        }
    }

    /// <summary>
    /// The entry gate of a combat stepped on its own, outside <see cref="State" />'s active combat (EmuDump's re-anchor
    /// steps the fight it anchored, one clone per step). With no switch on it is <see cref="CombatEngine.Step" />. With one
    /// on, the combat must carry its KernelState (throws otherwise, as <see cref="Step" /> does) and the step goes through
    /// the queue as a run's combat step does: the action is enqueued and drained, its body the combat engine's step.
    /// </summary>
    public StepResult StepCombat(Sts2Emulator.Core.CombatState combat, int action, Random rng, int targetEnemyIndex = -1)
    {
        if (_kernel is not { AnyOn: true })
        {
            return Sts2Emulator.Core.CombatEngine.Step(combat, action, rng, targetEnemyIndex);
        }

        var k = combat.Kernel ?? throw new InvalidOperationException(
            "kernel guard: a switch is on and the combat about to be stepped has no KernelState (bind it where the combat was built)");
        var host = RunQueue(combat, k, new QueueHost(this, combat, rng), action, targetEnemyIndex);
        return host.Result ?? StepResult.Invalid;
    }

    /// <summary>
    /// A combat step through the kernel's queue (TIER1_KERNEL.md section 5; only with a switch on): a player action is
    /// enqueued and the executor drains the queue, the action's body being the old engine's step. An answer to an old
    /// screen resumes the action gathering it (a new id; its body replays the step); an answer to a kernel screen between
    /// actions runs that hook action's resume record.
    /// </summary>
    private int QueueStep(Sts2Emulator.Core.CombatState combat, Core2.KernelState k, int action, int target,
        out float reward, out bool terminal, out bool truncated)
    {
        var host = RunQueue(combat, k, new QueueHost(this, combat, null), action, target);
        reward = host.Reward;
        terminal = host.Terminal;
        truncated = host.Truncated;
        return host.Status;
    }

    private QueueHost RunQueue(Sts2Emulator.Core.CombatState combat, Core2.KernelState k, QueueHost host, int action, int target)
    {
        var q = k.Queue!;
        Core2.GameActionRec? fresh = null;
        if (combat.PendingSelection is { KernelAction: > 0 } screen)
        {
            combat.PendingSelection = null;
            k.Answer(screen.KernelAction, action);
        }
        else if (combat.PendingSelection is not null)
        {
            if (q.Front(0, Core2.GameActionState.GatheringPlayerChoice) is { Kind: not Core2.GameActionKind.Hook } gathering)
            {
                q.Resume(gathering.Id!.Value, action);
            }
            else
            {
                // A screen outside any action (the turn's end, the combat's start): its answer is the old step's.
                host.Body(action, target);
            }
        }
        else
        {
            var kind = action < combat.Hand.Count ? Core2.GameActionKind.PlayCard
                : action == combat.Hand.Count ? Core2.GameActionKind.EndTurn
                : Core2.GameActionKind.UsePotion;
            fresh = q.Enqueue(new Core2.GameActionRec(kind, Core2.GameActionRec.TypeOf(kind)) { Action = action, Target = target });
        }

        Core2.ActionExecutor.Drain(k, host);
        // game: a canceled action does nothing. When the queue ran no body (the action was canceled: its queue cancels play
        // cards or player-driven actions, or the combat is over; or the executor is paused and the action waits; or the step
        // was the answer to a kernel screen and the hook action's resume record ran) the old step does not run in its place.
        // A canceled action is refused: status -1, nothing changed.
        if (!host.Ran && fresh is { State: Core2.GameActionState.Canceled })
        {
            host.Status = -1;
        }

        if (!host.InProgress)
        {
            k.Queue!.CombatEnded();  // not `q`: a replayed step restored the kernel state and its queue (see ActionExecutor.Drain)
        }

        return host;
    }

    /// <summary>
    /// The executor's host for one step: a player action's body is <see cref="StepInner" />, or, for a combat stepped on
    /// its own (<see cref="StepCombat" />), the combat engine's step on the caller's stream.
    /// </summary>
    private sealed class QueueHost(RunEngine engine, Sts2Emulator.Core.CombatState combat, Random? bare)
        : Core2.IActionHost, Core2.ICombatEndHost
    {
        public bool Ran;
        public int Status;
        public float Reward;
        public bool Terminal;
        public bool Truncated;
        public StepResult? Result;

        public void Body(int action, int target)
        {
            Ran = true;
            if (bare is not null)
            {
                var r = Sts2Emulator.Core.CombatEngine.Step(combat, action, bare, target);
                Result = r;
                (Reward, Terminal) = (r.Reward, r.Terminal);
                if (r.Terminal && combat.Kernel is { } k && k.On(Core2.Switch.F10))
                {
                    Core2.CombatEnd.OldStepEnded(k, r.PlayerWon, this);
                }
            }
            else
            {
                Status = engine.StepInner(action, target, out Reward, out Terminal, out Truncated);
            }
        }

        public bool Run(Core2.KernelState k, Core2.GameActionRec a)
        {
            if (a.Answer is { } answer)
            {
                a.Answer = null;
                Body(answer, -1);
            }
            else
            {
                Body(a.Action, a.Target);
            }

            return combat.PendingSelection is { Replay: not null, KernelAction: 0 };
        }

        // game: CombatManager.IsInProgress: Core2's with F10 on (CombatEnd), else the old gate.
        public bool InProgress =>
            (bare is not null || ReferenceEquals(engine.State.ActiveCombat, combat))
            && (combat.Kernel is { } k && k.On(Core2.Switch.F10)
                ? Core2.CombatEnd.IsInProgress(k)
                : !Sts2Emulator.Core.CombatEngine.IsOverOrEnding(combat));

        // game: CheckWinCondition at the executor's boundary (called with F10 on only). A combat that ends here hands off
        // to the run engine's FinishCombat (a combat stepped on its own: this host), whose result is the step's.
        public void CheckWin(Core2.KernelState k)
        {
            if (bare is not null)
            {
                Core2.CombatEnd.AfterAction(k, this);
                return;
            }

            Core2.CombatEnd.AfterAction(k, engine);
            Status = engine.TakeHandOff(Status, ref Terminal);
        }

        // A combat stepped on its own has no run to finish: its end is the step's terminal result.
        public void EndCombat(Core2.KernelState k, bool playerWon)
        {
            Terminal = true;
        }
    }

    private int StepInner(
        int action,
        int targetEnemyIndex,
        out float reward,
        out bool terminal,
        out bool truncated
    )
    {
        reward = 0.0f;
        terminal = false;
        truncated = false;

        // A potion used from the belt, away from a fight: whatever screen the run is on. A
        // fight's own action ids are the combat engine's.
        if (State.Phase != RunPhase.Combat && RunPotionUse.SlotOf(action) is var usedSlot and >= 0)
        {
            return RunPotionUse.Use(State, usedSlot) ? 0 : -1;
        }

        if (State.Phase == RunPhase.Ancient)
        {
            if (State.NeowAwaitingProceed)
            {
                if (action != 0)
                {
                    return -1;
                }

                State.NeowAwaitingProceed = false;
                EnterMapPhase();
                return 0;
            }

            if (action is < 0 or >= 3 || State.NeowOptions[action] == 0)
            {
                return -1;
            }

            ApplyAncientChoice(State.NeowOptions[action]);
            if (State.Phase == RunPhase.Ancient)
            {
                // Every blessing ends in AncientEventModel.Done(), which is a
                // SetEventFinished -- a page whose only option is Proceed. Only
                // Kaleidoscope was given one here, because it was the one whose reward
                // screen made the missing page visible; the rest went straight to the
                // map and swallowed an action the run really takes.
                State.NeowAwaitingProceed = true;
            }

            return 0;
        }

        if (State.Phase == RunPhase.Map)
        {
            if (
                !RunMapGenerator.ChooseMapNode(State, action, out int nodeType, out int encounterId)
            )
            {
                return -1;
            }

            // `MawBank.AfterRoomEntered(RunState.BaseRoom == room)` -- the room the player
            // just walked into, whatever kind it is. Fired here rather than inside the
            // switch because every branch below either starts a combat or changes phase,
            // and the bank does not care which. The unknown-node path pays through
            // `EnterUnknownMapPoint`, which resolves to a room of its own.
            RunNonCombatEffects.PayMawBank(State);
            // `LavaLamp.AfterRoomEntered` clears its flag on entering ANY room, not only a
            // fight's: a card reward outside combat (an event's, a shop's) is upgraded unless
            // damage was taken since the last room. Fights entered otherwise clear it again
            // as they start.
            State.TookUnblockedDamageThisCombat = false;

            switch (nodeType)
            {
                case RunConstants.NodeNormal:
                case RunConstants.NodeElite:
                case RunConstants.NodeBoss:
                    State.LastResolvedRoomType = nodeType;
                    // `Pantograph.BeforeCombatStart` reads the ROOM type, so the heal is a
                    // boss-room thing rather than a boss-encounter one.
                    if (nodeType == RunConstants.NodeBoss)
                    {
                        Effects.RelicEffects.ApplyBeforeBossCombat(State);
                    }

                    State.Phase = RunPhase.Combat;
                    return StartCombatWithDeck(
                        State.Deck,
                        encounterId,
                        State.Relics,
                        State.PlayerHp,
                        State.PlayerMaxHp,
                        State.PotionSlots,
                        State.Gold
                    );
                case RunConstants.NodeRest:
                    State.LastResolvedRoomType = RunConstants.NodeRest;
                    Effects.RelicEffects.ApplyAfterRoomEntered(
                        State,
                        isRestSite: true,
                        cameFromUnknown: false
                    );
                    State.RestOptionsTaken = 0;
                    State.RestOptionsTaken = 0;
                    State.Phase = RunPhase.Rest;
                    break;
                case RunConstants.NodeShop:
                    State.LastResolvedRoomType = RunConstants.NodeShop;
                    RunRewardGenerator.EnterShop(State);
                    break;
                case RunConstants.NodeRelic:
                    State.LastResolvedRoomType = RunConstants.NodeRelic;
                    RunRewardGenerator.EnterTreasureRoom(State);
                    break;
                case RunConstants.NodeEvent:

                    return EnterUnknownMapPoint();
                case RunConstants.NodeAncient:
                    // The act's ancient, which behaves as Neow does: a screen of
                    // blessings, then one more Proceed before the map. This is the
                    // dispatch the map TRAVEL goes through -- there is a second one for
                    // rooms entered another way, and the Ancient needs both.
                    State.LastResolvedRoomType = RunConstants.NodeEvent;
                    ApplyAncientHeal();
                    EnterActAncient();
                    break;
                default:
                    EnterMapPhase();
                    break;
            }
            return 0;
        }

        if (State.Phase == RunPhase.Combat)
        {
            if (State.ActiveCombat is null || State.ActiveCombatRng is null)
            {
                return -1;
            }

            var result =
                targetEnemyIndex >= 0
                    ? CombatEngine.Step(
                        State.ActiveCombat,
                        action,
                        State.ActiveCombatRng,
                        targetEnemyIndex
                    )
                    : CombatEngine.Step(State.ActiveCombat, action, State.ActiveCombatRng);
            reward = result.Reward;
            terminal = result.Terminal;
            State.LastPlayerWon = result.Terminal && result.PlayerWon;
            // `LavaLamp.AfterDamageReceived` latches on the first UNBLOCKED, blockable hit.
            // Read off the combat's own tally, which counts exactly that.
            if (State.ActiveCombat is { } fight && fight.TookUnblockedDamage)
            {
                State.TookUnblockedDamageThisCombat = true;
            }
            if (result.Terminal)
            {
                // With F10 on the combat's end is Core2's (CombatEnd.CheckWinCondition), which hands it back to
                // FinishCombat once (the F17 hand-off). Until R4 reroutes the old in-step check, its point stands for the
                // game's (CombatEnd.OldStepEnded).
                if (State.ActiveCombat?.Kernel is { } k && k.On(Core2.Switch.F10))
                {
                    terminal = false;
                    Core2.CombatEnd.OldStepEnded(k, result.PlayerWon, this);
                    return TakeHandOff(0, ref terminal);
                }
                return FinishCombat(result.PlayerWon, out terminal);
            }
            return 0;
        }

        if (State.Phase == RunPhase.CardReward)
        {
            return StepCardReward(action, out terminal);
        }

        if (State.Phase == RunPhase.RelicReward)
        {
            return StepRelicReward(action, out terminal);
        }

        if (State.Phase == RunPhase.Shop)
        {
            return StepShop(action, out terminal);
        }

        if (State.Phase == RunPhase.FakeMerchant)
        {
            return StepFakeMerchant(action, out terminal);
        }

        if (State.Phase == RunPhase.Rest)
        {
            return StepRest(action, out terminal);
        }

        if (State.Phase == RunPhase.Event)
        {
            return StepEvent(action, out terminal);
        }

        if (State.Phase == RunPhase.TransformSelect)
        {
            return StepTransformSelect(action, out terminal);
        }

        if (State.Phase == RunPhase.CrystalSphere)
        {
            return StepCrystalSphere(action, out terminal);
        }

        if (State.Phase == RunPhase.BundleSelect)
        {
            return StepBundleSelect(action, out terminal);
        }

        if (State.Phase == RunPhase.Treasure)
        {
            return StepTreasure(action, out terminal);
        }

        if (State.Phase == RunPhase.Complete)
        {
            terminal = true;
            return 0;
        }

        return -1;
    }

    /// <summary>
    /// The end of a fight, won or lost: every after-combat hook, the sync back to the run,
    /// and on a win the room's rewards. The fight that ends on a card and the fight a
    /// replay resolves (<see cref="DebugResolveCombat" />) leave through this one door, so
    /// the two cannot drift apart.
    /// </summary>
    /// <summary>What the F17 hand-off's FinishCombat returned, until the step that caused it takes it.</summary>
    private (int Status, bool Terminal)? _handOff;

    /// <summary>
    /// The F17 hand-off (Core2.ICombatEndHost; only with F10 on): Core2 decided the combat ended, after IsInProgress went
    /// false, at the game's EndCombatInternal point (or ProcessPendingLoss): the old end of the fight runs once, with the
    /// arguments the old in-step caller passed.
    /// </summary>
    void Core2.ICombatEndHost.EndCombat(Core2.KernelState k, bool playerWon)
    {
        State.LastPlayerWon = playerWon;
        int status = FinishCombat(playerWon, out bool terminal);
        _handOff = (status, terminal);
    }

    /// <summary>The hand-off's status and terminal flag when one ran since the last take, else <paramref name="status" />
    /// and <paramref name="terminal" /> unchanged.</summary>
    private int TakeHandOff(int status, ref bool terminal)
    {
        if (_handOff is not { } h)
            return status;
        _handOff = null;
        terminal = h.Terminal;
        return h.Status;
    }

    /// <summary>How many times <see cref="FinishCombat" /> ran (the F17 hand-off's test: once per combat).</summary>
    internal int FinishCombatCalls { get; private set; }

    private int FinishCombat(bool playerWon, out bool terminal)
    {
        FinishCombatCalls++;
        terminal = true;
        // `RoyaltiesPower.AfterCombatEnd` and
        // `ForbiddenGrimoirePower.AfterCombatEnd` each hang an extra reward on the
        // room. Read off the COMBAT here, before the sync drops it -- both powers
        // were applied by their cards and asked about by nothing, so the gold and
        // the removals never arrived.
        if (State.ActiveCombat is { } finished)
        {
            // First, because the deck the after-combat hooks read had already lost them.
            TakeStolenCardsFromTheDeck(finished);
            State.RoyaltiesGoldOwed += BuffSystem.Get(finished.PlayerBuffs, BuffId.Royalties);
            State.CardRemovalsOwed += BuffSystem.Get(
                finished.PlayerBuffs,
                BuffId.ForbiddenGrimoire
            );
        }
        // `PumpkinCandle.AfterCombatEnd` burns a point of Kindle whether the fight
        // was won or not, so it sits here rather than behind the victory check.
        RunNonCombatEffects.BurnPumpkinCandle(State);

        // `ToyBox.AfterCombatEnd` counts the fight whether or not it was won, the
        // same as the candle -- and every third one melts a wax relic.
        RunNonCombatEffects.MeltAToyBoxRelic(State);
        SyncAfterCombat();
        // In the game, Improvement's AfterCombatEnd reads the same run stream combat
        // used. Copy combat's position back before drawing its deck upgrades.
        RunNonCombatEffects.ApplyImprovementUpgrades(State);
        if (playerWon)
        {
            RunNonCombatEffects.TriggerFishingRod(State);
            RunNonCombatEffects.ServeGuiltySentences(State);
            // `SwordOfStone.AfterCombatVictory(room.RoomType == Elite)`: five
            // elites and it becomes Sword of Jade. Counted on the VICTORY rather
            // than on entering the room, so fleeing or dying does not count.
            // `LastingCandy.AfterCombatEnd` counts the fight; the reward screen
            // that follows is what reads the count.
            Effects.RelicEffects.CountCombatForLastingCandy(State);
            Effects.RelicEffects.CountCombatForWongosTicket(State);
            RunNonCombatEffects.CountEliteVictoryForSwordOfStone(
                State,
                State.LastResolvedRoomType == RunConstants.NodeElite
            );

            RunNonCombatEffects.ApplyWarHammerOnVictory(
                State,
                State.LastResolvedRoomType == RunConstants.NodeElite
            );

            // `EnterCombatWithoutExitingEvent(..., shouldResumeAfterCombat: true)`:
            // the room was never the combat's, so the event pays instead of the
            // fight. `ShouldGiveRewards => false` on the encounter is the same
            // statement from the other side.
            if (State.ResumeEventId != 0)
            {
                return ResumeEventAfterCombat(out terminal);
            }

            RunRewardGenerator.GenerateCombatRewards(State);
            terminal = false;
        }
        else
        {
            State.Phase = RunPhase.Complete;
        }
        return 0;
    }

    /// <summary>
    /// Win the fight in progress without playing it. A DEBUG hook, not a game rule: it
    /// exists so scripts/replay_real_run.py can replay a real run's RUN LAYER, whose fights
    /// are measured separately by the live combat checks.
    /// </summary>
    /// <remarks>
    /// Only the fight's own turns are skipped. Everything the game does as a fight is set
    /// up (the roster, its HP rolls, the opening shuffle) has already happened when the
    /// combat was built, and the victory runs through <see cref="FinishCombat" /> exactly as
    /// a real one does. What a fight would have drawn mid-turn -- a card generated, a
    /// potion rolled -- is not drawn, which is the one way a resolved fight can leave a
    /// stream behind the game's.
    /// The three outcome flags are the fight's own results that its REWARDS read, which a
    /// resolved fight cannot produce by itself: gold a Heist hands back
    /// (<see cref="CombatState.StolenBackGold" />), and how a Gremlin Merc fight ended
    /// (<see cref="CombatState.FatGremlinEscaped" />, <see cref="CombatState.MercGoldWasStolen" />),
    /// which decides whether the fight's gold is rolled at all. So are the creatures that
    /// escaped (<see cref="CombatState.EscapedEnemies" />, which cut the gold), the cards a
    /// Thieving Hopper took (<see cref="CombatState.StolenCards" />, gone from the deck) and
    /// those its death handed back (<see cref="CombatState.StolenBackCards" />, offered as
    /// rewards).
    /// Returns -1, changing nothing, when no fight is in progress.
    /// </remarks>
    public int DebugResolveCombat(
        int stolenBackGold = 0,
        bool fatGremlinEscaped = false,
        bool mercGoldWasStolen = false,
        int escapedEnemies = 0,
        IReadOnlyList<CardInstance>? stolenCards = null,
        IReadOnlyList<CardInstance>? stolenBackCards = null
    )
    {
        if (State.Phase != RunPhase.Combat || State.ActiveCombat is not { } combat)
        {
            return -1;
        }

        combat.StolenBackGold = Math.Max(0, stolenBackGold);
        combat.FatGremlinEscaped = fatGremlinEscaped;
        combat.MercGoldWasStolen = mercGoldWasStolen;
        combat.EscapedEnemies = Math.Max(0, escapedEnemies);
        combat.StolenCards = [.. stolenCards ?? []];
        combat.StolenBackCards = [.. stolenBackCards ?? []];
        State.LastPlayerWon = true;
        return FinishCombat(playerWon: true, out _);
    }

    /// <summary>
    /// <c>SwipePower.Steal</c> removed each stolen card's <c>DeckVersion</c> from the deck as
    /// it was taken, and <c>BeforeDeath</c> offers that deck version back. The combat holds
    /// only its own copy, so the deck version is the deck card it was dealt from: the same
    /// card if nothing in the fight changed it, else the same card id (an Armaments upgrade
    /// is the fight's, not the deck's).
    /// </summary>
    private void TakeStolenCardsFromTheDeck(CombatState combat)
    {
        if (combat.StolenCards.Count == 0)
        {
            return;
        }

        var returned = new List<CardInstance>();
        foreach (var stolen in combat.StolenCards)
        {
            int at = State.Deck.IndexOf(stolen);
            if (at < 0)
            {
                at = State.Deck.FindIndex(card =>
                    card.DefId == stolen.DefId
                    && card.Enchantment == stolen.Enchantment
                    && card.TinkerType == stolen.TinkerType
                );
            }

            if (at < 0)
            {
                at = State.Deck.FindIndex(card => card.DefId == stolen.DefId);
            }

            var deckVersion = at >= 0 ? State.Deck[at] : stolen;
            if (at >= 0)
            {
                State.Deck.RemoveAt(at);
            }

            int back = combat.StolenBackCards.IndexOf(stolen);
            if (back >= 0)
            {
                combat.StolenBackCards.RemoveAt(back);
                returned.Add(deckVersion);
            }
        }

        combat.StolenCards.Clear();
        combat.StolenBackCards.InsertRange(0, returned);
    }

    /// <summary>
    /// Put the player's holdings where a real run had them. A DEBUG hook for the run
    /// replay, which uses it after a resolved fight and wherever the real run's holdings
    /// part company with the emulator's, so the offerings after it stay comparable.
    /// </summary>
    /// <remarks>
    /// The holdings are SET, not earned: no pickup effect runs, because the real run's
    /// totals already include every one. A relic the run already holds keeps its
    /// <see cref="RelicInstance.Counter" />. A negative scalar or a null list leaves that
    /// holding as it is. <paramref name="cards" /> is four numbers a card: id, upgraded,
    /// enchantment, enchantment amount.
    /// </remarks>
    public void DebugSetPlayer(
        int hp,
        int maxHp,
        int gold,
        int[]? cards,
        int[]? relics,
        int[]? potions,
        int maxPotionSlots = -1
    )
    {
        if (maxHp >= 0)
        {
            State.PlayerMaxHp = Math.Max(1, maxHp);
        }

        if (hp >= 0)
        {
            State.PlayerHp = Math.Clamp(hp, 0, State.PlayerMaxHp);
        }

        if (gold >= 0)
        {
            State.Gold = gold;
        }

        if (cards is not null)
        {
            State.Deck.Clear();
            for (int i = 0; i + 3 < cards.Length; i += 4)
            {
                State.Deck.Add(
                    new CardInstance(
                        cards[i],
                        cards[i + 1] != 0,
                        Enchantment: (Enchantment)cards[i + 2],
                        EnchantAmount: cards[i + 3]
                    )
                );
            }
        }

        if (relics is not null)
        {
            var held = State.Relics.ToList();
            State.Relics.Clear();
            foreach (int relicId in relics)
            {
                int at = held.FindIndex(relic => relic.DefId == relicId);
                if (at >= 0)
                {
                    State.Relics.Add(held[at]);
                    held.RemoveAt(at);
                }
                else
                {
                    State.Relics.Add(new RelicInstance(relicId));
                }
            }
        }

        if (maxPotionSlots >= 0)
        {
            State.MaxPotionSlots = Math.Min(maxPotionSlots, State.PotionSlots.Length);
        }

        if (potions is not null)
        {
            Array.Clear(State.PotionSlots);
            for (int i = 0; i < potions.Length && i < State.PotionSlots.Length; i++)
            {
                State.PotionSlots[i] = potions[i];
            }
        }
    }

    public int ActiveEncounterId => State.ActiveCombat?.EncounterId ?? -1;

    public int ActiveShuffleRngCallCount => State.ActiveCombat?.ShuffleRng?.CallCount ?? 0;

    public int ActiveNicheRngCallCount => State.ActiveCombat?.NicheHpRng?.CallCount ?? 0;

    private void GenerateNeowOptions()
    {
        var rng = State.Rng.NeowRng();
        // `Neow.GenerateInitialOptions` drops every option `IsAllowedAtNeow` refuses twice: from
        // the curses before their NextItem, and from the positives after the pairs are added and
        // before the shuffle -- so a refused blessing changes which two the shuffle deals.
        int[] curseOptions =
        [
            .. RunConstants
                .NeowCurseOptions.ToArray()
                .Where(relic => RunNonCombatEffects.IsAllowedAtNeow(State, relic)),
        ];
        int cursed = curseOptions[rng.NextInt(curseOptions.Length)];

        List<int> positive = RunConstants.NeowPositiveOptions.ToArray().ToList();
        if (cursed == RunConstants.RelicCursedPearl)
        {
            positive.Remove(RunConstants.RelicGoldenPearl);
        }
        else if (cursed == RunConstants.RelicHeftyTablet)
        {
            positive.Remove(RunConstants.RelicArcaneScroll);
        }
        else if (cursed == RunConstants.RelicLeafyPoultice)
        {
            positive.Remove(RunConstants.RelicNewLeaf);
        }
        else if (cursed == RunConstants.RelicPrecariousShears)
        {
            positive.Remove(RunConstants.RelicPreciseScissors);
        }

        if (cursed != RunConstants.RelicLargeCapsule)
        {
            positive.Add(
                rng.NextBool() ? RunConstants.RelicLavaRock : RunConstants.RelicSmallCapsule
            );
        }

        positive.Add(
            rng.NextBool() ? RunConstants.RelicNutritiousOyster : RunConstants.RelicStoneHumidifier
        );
        positive.Add(rng.NextBool() ? RunConstants.RelicNeowsTalisman : RunConstants.RelicPomander);
        positive.RemoveAll(r => !RunNonCombatEffects.IsAllowedAtNeow(State, r));
        rng.Shuffle(positive);

        State.NeowOptions[0] = positive[0];
        State.NeowOptions[1] = positive[1];
        State.NeowOptions[2] = cursed;
    }

    /// <summary>
    /// Put the act's own ancient on screen, with the three blessings it offers.
    /// </summary>
    /// <remarks>
    /// Neow keeps the options generated at run start — it has its own generator and its
    /// own shape. Every act after draws one of Orobas, Pael or Tezcatara (or Darv, the
    /// one shared ancient) and offers three, one from each of its pools.
    /// </remarks>
    private void EnterActAncient()
    {
        if (State.Ancient != RunConstants.AncientNeow)
        {
            Array.Clear(State.NeowOptions);
            int[] options = RunNonCombatEffects.GenerateAncientOptions(State, State.Ancient);
            for (int i = 0; i < options.Length && i < State.NeowOptions.Length; i++)
            {
                State.NeowOptions[i] = options[i];
            }
        }

        State.Phase = RunPhase.Ancient;
    }

    /// <summary>
    /// <c>AncientEventModel.BeforeEventStarted</c>: entering an ancient heals the player
    /// toward full -- and under <c>WearyTraveler</c> only 80% of what they are missing.
    /// </summary>
    /// <remarks>
    /// This is the rule behind the run's own opening HP. Neow calls
    /// <c>SetCurrentHpInternal(0)</c> first, so the heal is of the whole 80 max: 64 from
    /// WearyTraveler, which the emulator once hardcoded in <c>Reset</c> as a starting
    /// constant without knowing where it came from, and a full 80 below it. Every act after
    /// the first walks onto its own ancient at whatever HP it finished the last one with,
    /// and is topped up the same way: a capture crosses into act 2 on 264/280 and stands on
    /// Pael at 276.
    /// </remarks>
    private void ApplyAncientHeal() =>
        State.PlayerHp = HpAfterAncientHeal(
            State.PlayerHp,
            State.PlayerMaxHp,
            State.AscensionLevel
        );

    /// <summary>
    /// The HP a run stands on once Neow has healed it: <see cref="HpAfterAncientHeal" />
    /// from zero, of <see cref="RunConstants.StartingMaxHp" />. 80 below WearyTraveler and
    /// 64 from it.
    /// </summary>
    /// <remarks>
    /// The combat API's synthetic start opens on this too, so a fight built without a run
    /// stands where the run's first fight would.
    /// </remarks>
    public static int StartingHp(int ascension) =>
        HpAfterAncientHeal(0, RunConstants.StartingMaxHp, ascension);

    /// <summary>The rule <see cref="ApplyAncientHeal" /> applies, on plain numbers.</summary>
    private static int HpAfterAncientHeal(int hp, int maxHp, int ascension)
    {
        int missing = maxHp - hp;
        if (missing <= 0)
        {
            return hp;
        }

        return hp
            + (
                Ascension.Has(ascension, Ascension.WearyTraveler)
                    ? (int)(missing * WearyTravelerHealFraction)
                    : missing
            );
    }

    /// <summary>What WearyTraveler leaves of an ancient's heal.</summary>
    private const double WearyTravelerHealFraction = 0.8;

    private void EnterMapPhase()
    {
        State.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(State);
    }

    private void ApplyAncientChoice(int relicId)
    {
        Array.Clear(State.NeowOptions);
        RunFollowUp followUp = RunNonCombatEffects.ApplyRelicPickup(State, relicId);
        if (relicId == RunConstants.RelicLostCoffer)
        {
            // LostCoffer.AfterObtained is a RewardsCmd.OfferCustom of TWO rewards -- a
            // three-card CardReward and a PotionReward -- so both sit on a rewards SCREEN
            // for the player to claim, the same shape as Kaleidoscope. Handing the card
            // reward straight over skipped the screen, and with it the claim the run
            // really makes.
            // Its card reward is CardCreationSource.Other: base odds, the rare offset left
            // alone (run referee, seed UK2BJLWEV1: the game's offset unchanged after it).
            RunRewardGenerator.ClearRewardScreen(State);
            RunRewardGenerator.OfferLostCofferRewards(State);
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.RelicReward;
            return;
        }
        if (followUp == RunFollowUp.PreRolledCardReward)
        {
            // Orrery already opened its card-reward screen. The other pickups leave a
            // pre-rolled grid, which must be answered before Neow's Proceed.
            if (State.PendingOfferCards.Length > 0)
            {
                State.Phase = RunPhase.TransformSelect;
            }
            State.NeowAwaitingProceed = true;
            return;
        }
        if (followUp == RunFollowUp.BonusRelicRewards)
        {
            // Neow's Bones: two relic rewards on one screen, claimed one at a time, and
            // the curse only once both are gone. Neow itself stays on screen afterwards
            // the way it does for every other blessing.
            RunRewardGenerator.OfferNextBonusRelic(State);
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.RelicReward;
            return;
        }
        if (followUp == RunFollowUp.BundleSelect)
        {
            // Neow stays on screen for its Proceed once the bundle is taken, the way it
            // does after every other blessing.
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.BundleSelect;
            return;
        }
        if (followUp == RunFollowUp.BonusCardOffers)
        {
            // The first offer goes onto the screen and the rest queue behind it; Neow
            // stays up for its Proceed once they are all answered, the way it does after
            // every other blessing.
            RunRewardGenerator.OfferFirstPendingCardOffer(State);
            State.RewardCardPending = true;
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.RelicReward;
            return;
        }
        if (followUp == RunFollowUp.TransformSelect)
        {
            // Neow stays on screen for its Proceed once the selection is answered, the way
            // it does after every other blessing.
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.TransformSelect;
            return;
        }
        if (followUp == RunFollowUp.CardReward)
        {
            RunRewardGenerator.EnterCardReward(State);
            return;
        }
        if (relicId == RunConstants.RelicKaleidoscope)
        {
            // "Obtain 2 card rewards from other characters". RewardsCmd.OfferCustom puts
            // both on the rewards screen at once; the player claims one, picks a card,
            // lands back on the screen and claims the other. Both are built as the relic is
            // obtained (OfferKaleidoscopeRewards).
            RunRewardGenerator.ClearRewardScreen(State);
            RunRewardGenerator.OfferKaleidoscopeRewards(State);
            State.NeowAwaitingProceed = true;
            State.Phase = RunPhase.RelicReward;
            return;
        }
        AdvanceRewardRngForNeowRelic(relicId);
    }

    /// <summary>
    /// Rewards draws a Neow relic's pickup makes in the game that the emulator's own
    /// model of it does not.
    /// </summary>
    /// <remarks>
    /// A standing debt, not a design. Hefty Tablet and Lead Paperweight both build their
    /// offer with <c>CardFactory.CreateForReward</c>, which draws from the Rewards stream;
    /// the emulator rolls their cards off <c>Rng.UpFront</c> instead, so the count here is
    /// what keeps everything downstream aligned. Modelling either properly means rolling
    /// its cards off Rewards and deleting its row -- and neither has a live capture yet,
    /// so neither has a way to be checked.
    ///
    /// <para>
    /// Phial Holster used to have a row of 4 and no longer does: its potions come off
    /// <c>Rng.CombatPotionGeneration</c> and its slot grant is free, so the game makes NO
    /// Rewards draws for it. Those four put the next combat's gold reward at stream
    /// position 5 where the game had it at 1 -- 7 gold against the capture's 15 -- and
    /// every card reward after it read from the wrong place too.
    /// </para>
    /// </remarks>
    private void AdvanceRewardRngForNeowRelic(int relicId)
    {
        int advances = relicId switch
        {
            _ => 0,
        };
        for (int i = 0; i < advances; i++)
        {
            State.PlayerRng.Rewards.NextDouble();
        }
    }

    /// <summary>
    /// Seed for the encounter's own stream, the one EncounterModel builds from the run
    /// seed, TotalFloor and a hash of the encounter's entry id. Every encounter that
    /// rolls its own roster reads it, so handing back a seed for Slimes alone left the
    /// rest -- Slithering Strangler's secondary enemy, Flyconid's slime, the raiders --
    /// rolling off a constant.
    /// </summary>
    private int? EncounterRngSeed(int encounterId) =>
        EncounterRng.SeedFor((int)State.Rng.Seed, State.Floor, encounterId);

    /// <summary>
    /// Goopy grows on play and the game bumps the DECK version's amount as well as the
    /// combat card's — so the block it gains is permanent for the rest of the run.
    /// </summary>
    /// <remarks>
    /// The emulator has no link from a combat card back to the deck card it was copied
    /// from, so the amounts are matched by card identity: for each (id, upgraded) the
    /// combat's Goopy amounts are written onto the run deck's, largest first. The combat
    /// deck IS the run deck copied, so the two multisets agree unless a card was added or
    /// removed mid-fight — and a card added mid-fight is not in the run deck to grow.
    /// </remarks>
    private void CarryGoopyGrowthToTheDeck(CombatState combat)
    {
        var amounts = combat
            .AllCards()
            .Where(card => card.Enchantment == Enchantment.Goopy)
            .GroupBy(card => (card.DefId, card.Upgraded))
            .ToDictionary(
                group => group.Key,
                group => new Queue<int>(
                    group.Select(card => card.EnchantAmount).OrderByDescending(a => a)
                )
            );
        if (amounts.Count == 0)
        {
            return;
        }

        foreach (
            int index in Enumerable
                .Range(0, State.Deck.Count)
                .Where(i => State.Deck[i].Enchantment == Enchantment.Goopy)
                .OrderByDescending(i => State.Deck[i].EnchantAmount)
        )
        {
            var card = State.Deck[index];
            if (amounts.TryGetValue((card.DefId, card.Upgraded), out var queue) && queue.Count > 0)
            {
                State.Deck[index] = card with { EnchantAmount = queue.Dequeue() };
            }
        }
    }

    /// <summary>The post-combat sync, exposed so a test can drive it without a fight.</summary>
    internal void SyncAfterCombatForTest() => SyncAfterCombat();

    private void SyncAfterCombat()
    {
        if (State.ActiveCombat is null)
        {
            return;
        }

        // Carried out of the combat before it is dropped: the event that resumes reads the
        // ENCOUNTER's `RanOutOfTime`, and by then there is no encounter left to read.
        State.ResumeEventDummyEscaped = State.ActiveCombat.BattlewornDummyRanOutOfTime;

        Effects.RelicEffects.CollectUsedUpRelics(State.ActiveCombat, State.UsedUpRelics);
        Effects.RelicEffects.CarryRunCountersBack(State.ActiveCombat, State.Relics);
        CarryGoopyGrowthToTheDeck(State.ActiveCombat);
        State.PlayerHp = Math.Max(0, State.ActiveCombat.PlayerHp);
        State.PlayerMaxHp = Math.Max(1, State.ActiveCombat.PlayerMaxHp);
        State.Gold = Math.Max(0, State.ActiveCombat.PlayerGold);
        for (int i = 0; i < State.PotionSlots.Length; i++)
        {
            State.PotionSlots[i] = State.ActiveCombat.PotionSlots[i];
        }

        if (State.ActiveCombat.ShuffleRng is not null)
        {
            State.Rng.Shuffle.AdvanceToCallCount(State.ActiveCombat.ShuffleRng.CallCount);
        }

        if (State.ActiveCombat.TargetRng is not null)
        {
            State.Rng.CombatTargets.AdvanceToCallCount(State.ActiveCombat.TargetRng.CallCount);
        }

        if (State.ActiveCombat.AiRng is CountingRandom aiRng)
        {
            State.Rng.MonsterAi.AdvanceToCallCount(aiRng.CallCount);
        }

        if (State.ActiveCombat.CardSelectionRng is not null)
        {
            State.Rng.CombatCardSelection.AdvanceToCallCount(
                State.ActiveCombat.CardSelectionRng.CallCount
            );
        }

        if (State.ActiveCombat.CardGenerationRng is not null)
        {
            State.Rng.CombatCardGeneration.AdvanceToCallCount(
                State.ActiveCombat.CardGenerationRng.CallCount
            );
        }

        if (State.ActiveCombat.PotionGenerationRng is not null)
        {
            State.Rng.CombatPotionGeneration.AdvanceToCallCount(
                State.ActiveCombat.PotionGenerationRng.CallCount
            );
        }

        if (State.ActiveCombat.EnergyCostRng is not null)
        {
            State.Rng.CombatEnergyCosts.AdvanceToCallCount(
                State.ActiveCombat.EnergyCostRng.CallCount
            );
        }

        SyncNicheRngFromCombat();
    }

    private void SyncNicheRngFromCombat()
    {
        if (State.ActiveCombat?.NicheHpRng is not null)
        {
            State.Rng.Niche.AdvanceToCallCount(State.ActiveCombat.NicheHpRng.CallCount);
        }
    }

    private int StepCardReward(int action, out bool terminal)
    {
        terminal = false;

        // `Driftwood`'s reroll answers the screen WITHOUT closing it: the options are
        // thrown away, three more are rolled, and the player chooses again.
        if (action == RunConstants.RewardRerollAction && State.CardRewardRerollAvailable)
        {
            RunRewardGenerator.PopulateCardReward(State, isReroll: true);
            return 0;
        }

        // The three ordinary options are actions 0-2; Lasting Candy's fourth is claimed at
        // `RewardFourthCardAction` and reads slot 3. Two ids for one array, because three
        // is the SKIP and always has been -- see the constant.
        // Lasting Candy's option is claimed by its own action and comes from its own field,
        // so it takes none of the three the roll produced.
        if (action == RunConstants.RewardFourthCardAction && State.LastingCandyRewardCard != 0)
        {
            RunNonCombatEffects.AddCardToDeck(
                State,
                new CardInstance(State.LastingCandyRewardCard, State.LastingCandyRewardUpgraded)
            );
        }
        else if (0 <= action && action < State.RewardCards.Length)
        {
            int cardId = State.RewardCards[action];
            if (cardId == 0)
            {
                return -1;
            }

            RunNonCombatEffects.AddCardToDeck(
                State,
                new CardInstance(
                    cardId,
                    State.RewardUpgraded[action],
                    // Wing Charm enchants the OPTION on the screen, so the enchantment
                    // travels with the card the player picks -- and only with that one.
                    // `Glitter.TryModifyCardRewardOptionsLate` enchants EVERY eligible
                    // option instead, so it is asked per-card rather than by index.
                    Enchantment: action == State.RewardEnchantIndex
                        ? State.RewardEnchantment
                        : Effects.RelicEffects.GlitterEnchantmentFor(
                            State,
                            new CardInstance(cardId, State.RewardUpgraded[action])
                        ),
                    EnchantAmount: action == State.RewardEnchantIndex ? 1
                        : Effects.RelicEffects.GlitterEnchantmentFor(
                            State,
                            new CardInstance(cardId, State.RewardUpgraded[action])
                        ) == Enchantment.None
                            ? 0
                        : 1
                )
            );
        }
        else if (
            action == RunConstants.RewardSacrificeAction
            && State.Relics.Any(relic => relic.DefId == RunConstants.RelicPaelsWing)
        )
        {
            // `EndSelectionAndCompleteReward`: the screen closes exactly as it does for a
            // claim, so this falls through to the same clear-and-advance below rather than
            // being a skip. What it costs is the card; what it buys is every second relic.
            RunNonCombatEffects.SacrificeACardReward(State);
        }
        else if (action != RunConstants.RewardSkipAction)
        {
            return -1;
        }

        Array.Clear(State.RewardCards);
        Array.Clear(State.RewardUpgraded);
        State.LastingCandyRewardCard = 0;
        State.LastingCandyRewardUpgraded = false;
        State.CardRewardRerollAvailable = false;
        Effects.RelicEffects.SpendSilkenTress(State);

        // A standalone chain of offers (Orrery's five): they were all populated when the set
        // was offered, so the next is only shown. On a rewards screen the player comes back
        // to the screen instead and claims the next item from it.
        if (!State.ReturnToRewardScreenAfterCardReward && State.PendingCardOffers.Count > 0)
        {
            RunRewardGenerator.OfferFirstPendingCardOffer(State);
            return 0;
        }

        if (State.ReturnToRewardScreenAfterCardReward)
        {
            State.ReturnToRewardScreenAfterCardReward = false;
            if (State.NeowAwaitingProceed && !RunRewardGenerator.HasPendingRewards(State))
            {
                State.Phase = RunPhase.Ancient;
                return 0;
            }

            // Brain Leech's colourless reward belongs to the EVENT, the same way Neow's
            // does: once the screen is empty the run goes back to the event's finished
            // page, not to an empty rewards screen. The check for this sat below, after
            // the return here, so it never ran.
            if (
                State.EventId == RunConstants.EventBrainLeech
                && !RunRewardGenerator.HasPendingRewards(State)
            )
            {
                State.EventId = RunConstants.EventResultPending;
                State.Phase = RunPhase.Event;
                return 0;
            }

            // Back to the rewards screen even when nothing is left on it. The game keeps
            // it open and waits to be dismissed — the player still has to proceed — so
            // advancing straight to the map skips an action the run actually takes.
            State.Phase = RunPhase.RelicReward;
            return 0;
        }

        if (State.EventId == RunConstants.EventBrainLeech)
        {
            State.EventId = RunConstants.EventResultPending;
            State.Phase = RunPhase.Event;
            return 0;
        }

        if (State.PendingRelicReward)
        {
            State.PendingRelicReward = false;
            RunRewardGenerator.EnterRelicReward(State);
            return 0;
        }
        return AdvanceAfterNode(out terminal);
    }

    /// <summary>
    /// The open chest. Taking the relic is `PickRelicLocally` and the award is
    /// `RelicCmd.Obtain`, so its pickup effect runs; the room then waits for its Proceed,
    /// as the game's does. Proceeding with the relic still in the chest is
    /// `SkipRelicLocally` and hands over nothing.
    /// </summary>
    /// <remarks>
    /// The pickup's follow-up is not routed anywhere, the same as a relic claimed off an
    /// elite's reward screen: a chest holds a Common, Uncommon or Rare, and none of those
    /// opens a screen of its own on pickup -- every relic that does is Ancient or Shop.
    /// </remarks>
    private int StepTreasure(int action, out bool terminal)
    {
        terminal = false;
        if (action == RunConstants.TreasureTakeRelicAction && State.RelicReward != 0)
        {
            int claimed = State.RelicReward;
            State.RelicReward = 0;
            RunNonCombatEffects.ApplyRelicPickup(State, claimed);
            return 0;
        }

        if (action != RunConstants.RewardSkipAction)
        {
            return -1;
        }

        State.RelicReward = 0;
        return AdvanceAfterNode(out terminal);
    }

    private int StepRelicReward(int action, out bool terminal)
    {
        terminal = false;
        if (State.RewardPotion != 0 && action is >= 4 and <= 6)
        {
            int slot = action - 4;
            if ((uint)slot >= State.PotionSlots.Length || State.PotionSlots[slot] == 0)
            {
                return -1;
            }

            State.PotionSlots[slot] = 0;
            return 0;
        }

        if (action == RunConstants.RewardSkipAction)
        {
            // Skipping the screen abandons everything still on it, queued potions included.
            // It still LEAVES the screen the same way answering it does: this used to go
            // straight to AdvanceAfterRelicReward and so walked past the returns below,
            // which is invisible for Neow (its rewards are WithSkippingDisallowed) and
            // wrong for every event, whose second queued potion is declined far more often
            // than it is taken.
            //
            // Everything else on it goes too: a card reward left unclaimed (RewardCardPending,
            // queued offers), gold, a relic. Only the potions were dropped, so a combat's
            // skipped card reward stayed pending and reappeared on the NEXT rewards screen
            // the run opened -- an event's potions, in the run referee's 6BJ7MYTYCK. Each of
            // the game's RewardsSets is its own screen; leaving one drops what it held.
            State.PendingPotionRewards.Clear();
            RunRewardGenerator.ClearRewardScreen(State);
            State.PendingBonusRelicRewards.Clear();
            State.PendingOtherCharacterCardRewards = 0;
            return LeaveRewardScreen(out terminal);
        }

        if (!RunRewardGenerator.HasPendingRewards(State))
        {
            if (action is 0 or RunConstants.RewardSkipAction)
            {
                // Neow stays on screen once its rewards are answered, waiting for one
                // more Proceed -- the same return the card-reward path already made, and
                // the only one a relic-only screen (Neow's Bones) could not. Advancing
                // straight to the map from here skips a decision the run really makes.
                return LeaveRewardScreen(out terminal);
            }

            return -1;
        }

        if (!RunRewardGenerator.ClaimRewardAtIndex(State, action))
        {
            return -1;
        }

        OfferNextRestPotion();
        // The claim that empties the screen returns to Neow by itself -- a live capture's
        // second relic lands and the very same snapshot is already back on the ancient,
        // with no separate action between them. Only when the claim did not open a screen
        // of its own: a card reward sets its own phase and owns the return.
        if (
            State.Phase == RunPhase.RelicReward
            && State.NeowAwaitingProceed
            && !RunRewardGenerator.HasPendingRewards(State)
        )
        {
            State.Phase = RunPhase.Ancient;
        }
        else if (
            State.Phase == RunPhase.RelicReward
            && State.EventAwaitingProceed
            && !RunRewardGenerator.HasPendingRewards(State)
        )
        {
            ReturnFromEventRewards();
        }

        return 0;
    }

    /// <summary>
    /// The event's rewards screen is answered: back to its result page, or -- for the one event
    /// that keeps going after <c>RewardsCmd.OfferCustom</c> (the Endless Conveyor's potion) -- to
    /// its options with the belt turned, since the dish is rolled once the await returns.
    /// </summary>
    private void ReturnFromEventRewards()
    {
        State.EventAwaitingProceed = false;
        if (State.EventResumesAfterRewards)
        {
            State.EventResumesAfterRewards = false;
            RunNonCombatEffects.RollNextConveyorDish(State);
        }
        else
        {
            State.EventId = RunConstants.EventResultPending;
        }

        State.Phase = RunPhase.Event;
    }

    /// <summary>
    /// RewardsCmd.OfferCustom with potions: the run goes to the reward screen and the
    /// player may decline, or drop a held potion to make room. Anything past the first is
    /// queued, because the screen carries one at a time. A 0 is rolled when it reaches
    /// the screen.
    /// </summary>
    /// <summary>
    /// The <c>AfterRestSiteHeal</c> hook. Stone Humidifier is its only holder here:
    /// <c>MaxHpVar(5)</c> through <c>CreatureCmd.GainMaxHp</c>, and it ignores the hook's
    /// <c>isMimicked</c> flag, so an event that heals you like a rest pays it too.
    /// </summary>
    /// <summary>
    /// Leave the rewards screen for wherever opened it.
    /// </summary>
    /// <remarks>
    /// Neow stays on screen once its rewards are answered, waiting for one more Proceed.
    /// An event does the same for its own reason: every event that hands out rewards
    /// awaits <c>RewardsCmd.OfferCustom</c> and calls <c>SetEventFinished</c> on the line
    /// below, so the result page is shown once the screen is done with. Only a screen
    /// nobody is waiting on advances to the map.
    /// </remarks>
    private int LeaveRewardScreen(out bool terminal)
    {
        terminal = false;

        // `CardRemovalReward` is claimed FROM the reward screen, so it is answered as the
        // screen is left rather than before it. One selection at a time: the screen reopens
        // for the next while any are still owed.
        if (
            State.CardRemovalsOwed > 0
            && State.Deck.Count > 1
            && State.PendingSelectionKind == DeckSelection.None
        )
        {
            State.CardRemovalsOwed--;
            if (
                RunNonCombatEffects.BeginDeckSelection(
                    State,
                    DeckSelection.Remove,
                    0,
                    count: 1,
                    returnTo: SelectionReturn.Map
                )
            )
            {
                State.Phase = RunPhase.TransformSelect;
                return 0;
            }
        }

        if (State.NeowAwaitingProceed)
        {
            State.Phase = RunPhase.Ancient;
            return 0;
        }

        if (State.EventAwaitingProceed)
        {
            ReturnFromEventRewards();
            return 0;
        }

        return AdvanceAfterRelicReward(out terminal);
    }

    private void ApplyAfterRestSiteHeal()
    {
        if (State.Relics.Any(relic => relic.DefId == RunConstants.RelicStoneHumidifier))
        {
            RunNonCombatEffects.GainMaxHp(State, 5);
        }

        // `DreamCatcher.TryModifyRestSiteHealRewards`: resting also offers a card, from
        // the MONSTER room's creation options -- `CardCreationOptions.ForRoom(player,
        // RoomType.Monster)`, an ordinary combat reward three wide, bolted onto the sleep.
        // Encounter-sourced, so its rarity rolls read and grow the rare offset and each
        // card rolls its upgrade: the same reward, and the same three draws a card, as a
        // monster room's. The rest site reads as a monster room to `RollRewardCardRarity`.
        if (Effects.RelicEffects.Has(State.Relics, Effects.RelicEffects.DreamCatcher))
        {
            RunRewardGenerator.PopulateCardReward(State);
            State.RewardCardPending = true;
            State.Phase = RunPhase.CardReward;
        }
    }

    private void OfferPotionRewards(params int[] potionIds)
    {
        // Every caller is an event branch, and every one of those events awaits the offer
        // and calls SetEventFinished on the next line -- so the screen owes the event a
        // return. Read off the phase rather than passed in, so a new event that offers
        // rewards gets the return without having to remember to ask for it.
        State.EventAwaitingProceed = State.Phase == RunPhase.Event;
        State.RewardGold = 0;
        State.RelicReward = 0;
        State.RewardPotion = 0;
        State.PendingPotionRewards.Clear();
        QueuePotionRewards(potionIds);
        OfferNextRestPotion();
        State.Phase = RunPhase.RelicReward;
    }

    /// <summary>
    /// Queue potion rewards the way one RewardsSet holds them: every reward is populated
    /// when the set is offered (<c>RewardsSet.GenerateWithoutOffering</c> calls each
    /// <c>Populate</c>, and <c>PotionReward.Populate</c> rolls its potion there), so a
    /// queued 0 is rolled now, in list order -- not when it reaches the one-potion screen,
    /// which drew the second potion of a pair only once the first was answered, and never
    /// when the screen was left.
    /// </summary>
    private void QueuePotionRewards(IEnumerable<int> potionIds)
    {
        foreach (int potionId in potionIds)
        {
            State.PendingPotionRewards.Add(
                potionId != 0 ? potionId : RunRewardGenerator.NextPotion(State, State.PlayerRng.Rewards)
            );
        }
    }

    /// <summary>
    /// Moves the next queued potion onto the reward screen, if the screen is free. Queued
    /// potions are rolled when their set is offered (<see cref="QueuePotionRewards"/>); a 0
    /// still in the queue is rolled here as a fallback.
    /// </summary>
    private void OfferNextRestPotion()
    {
        if (State.PendingPotionRewards.Count == 0 || State.RewardPotion != 0)
        {
            return;
        }

        int potionId = State.PendingPotionRewards[0];
        State.PendingPotionRewards.RemoveAt(0);
        State.RewardPotion =
            potionId != 0
                ? potionId
                : RunRewardGenerator.NextPotion(State, State.PlayerRng.Rewards);
    }

    private int StepShop(int action, out bool terminal)
    {
        terminal = false;
        bool shouldAdvance = action == RunConstants.ShopSkipAction;
        if (0 <= action && action < State.ShopCards.Length)
        {
            int cardId = State.ShopCards[action];
            int cost = State.ShopCosts[action];
            if (cardId == 0 || State.Gold < cost)
            {
                return -1;
            }

            State.Gold -= cost;
            // `MawBank.AfterItemPurchased`: any merchant purchase with
            // `goldSpent > 0` closes the bank for the rest of the run.
            RunNonCombatEffects.CloseMawBank(State, cost);
            RunNonCombatEffects.AddCardToDeck(
                State,
                new CardInstance(cardId, Upgraded: State.ShopCardUpgraded[action])
            );
            State.ShopCards[action] = 0;
            State.ShopCardUpgraded[action] = false;
        }
        else if (7 <= action && action < 10)
        {
            int index = action - 7;
            int relicId = State.ShopRelics[index];
            int cost = State.ShopCosts[action];
            if (relicId == 0 || State.Gold < cost)
            {
                return -1;
            }

            State.Gold -= cost;
            RunNonCombatEffects.CloseMawBank(State, cost);
            State.ShopRelics[index] = 0;

            // A bought relic is `RelicCmd.Obtain` like any other, so its `AfterObtained`
            // runs. Adding it to the list bare made every pickup effect free for the shop
            // to swallow: a Distinguished Cape bought here cost no max HP and paid no
            // Apparitions, and a Preserved Fog removed nothing.
            var followUp = RunNonCombatEffects.ApplyRelicPickup(
                State,
                relicId,
                SelectionReturn.Shop
            );
            // `MerchantCardEntry.UpdateEntry` re-runs the merchant hook on what is still
            // stocked, so an egg bought here upgrades the shelf it was bought from.
            RunRewardGenerator.UpgradeShopCardsByEggs(State);
            if (followUp == RunFollowUp.TransformSelect)
            {
                State.Phase = RunPhase.TransformSelect;
                return 0;
            }
        }
        else if (10 <= action && action < 13)
        {
            int index = action - 10;
            int potionId = State.ShopPotions[index];
            int cost = State.ShopCosts[action];
            if (
                potionId == 0
                || State.Gold < cost
                || !RunRewardGenerator.AddPotion(State, potionId)
            )
            {
                return -1;
            }

            State.Gold -= cost;
            RunNonCombatEffects.CloseMawBank(State, cost);
            State.ShopPotions[index] = 0;
        }
        else if (action == RunConstants.ShopRemoveAction)
        {
            int cost = State.ShopCosts[RunConstants.ShopRemoveAction];
            if (State.ShopRemovalUsedThisVisit || State.Gold < cost || State.Deck.Count <= 1)
            {
                return -1;
            }

            // The merchant's service opens a removal SCREEN — which card goes is the
            // player's decision, and the whole of what the gold buys. This used to call
            // RemoveLowestPriorityCard, an emulator-invented preference order that starts
            // with Ascender's Bane, a card the game will not even offer.
            //
            // Open BEFORE charging: a deck with nothing the screen would take is not a
            // sale, and taking the gold for a screen that never appears is worse than
            // refusing the click.
            if (
                !RunNonCombatEffects.BeginDeckSelection(
                    State,
                    DeckSelection.Remove,
                    0,
                    count: 1,
                    returnTo: SelectionReturn.Shop
                )
            )
            {
                return -1;
            }

            State.Gold -= cost;
            RunNonCombatEffects.CloseMawBank(State, cost);
            State.ShopRemovalsUsed++;
            State.ShopRemovalUsedThisVisit = true;
        }
        else if (action != RunConstants.ShopSkipAction)
        {
            return -1;
        }

        return shouldAdvance ? AdvanceAfterNode(out terminal) : 0;
    }

    private int AdvanceAfterRelicReward(out bool terminal)
    {
        if (State.CurrentNodeType == RunConstants.NodeBoss)
        {
            // `NRewardsScreen.OnProceedButtonPressed`: with a second boss on the map and the
            // run standing on the first, Proceed goes back to the MAP
            // (`ProceedFromTerminalRewardsScreen`) instead of changing act. A10's last act.
            if (
                State.MapNodes.TryGetValue(State.CurrentMapCoord, out var bossNode)
                && bossNode.Children.Any(child =>
                    State.MapNodes.TryGetValue(child, out var next)
                    && RunMapGenerator.IsSecondBoss(State, next)
                )
            )
            {
                EnterMapPhase();
                terminal = false;
                return 0;
            }

            // RunManager.EnterNextAct: a boss is the end of the RUN only in the last act.
            // Anywhere else it is the end of an ACT, and the run walks onto the next act's
            // map — a live capture proceeds from the act 1 boss reward straight onto act
            // 2's map, still on floor 17. Treating every boss as terminal is what made act
            // 2 unreachable.
            if (EnterNextAct())
            {
                terminal = false;
                return 0;
            }

            EnterTheArchitect();
            terminal = false;
            return 0;
        }
        return AdvanceAfterNode(out terminal);
    }

    /// <summary>
    /// The room a won run ends in: <c>RunManager.EnterNextAct</c> in the last act enters an
    /// <c>EventRoom</c> holding <c>TheArchitect</c> (<c>AbstractRoom.IsVictoryRoom</c>), and
    /// only its PROCEED ends the run.
    /// </summary>
    /// <remarks>
    /// Its dialogue lines come first, but how many there are is picked from the profile's
    /// win count (<c>LoadDialogue</c>), which a run does not carry, and none of them does
    /// anything -- so the emulator's event is the PROCEED alone.
    /// </remarks>
    private void EnterTheArchitect()
    {
        State.EventId = RunConstants.EventTheArchitect;
        State.EventPage = 0;
        State.Phase = RunPhase.Event;
    }

    /// <summary>
    /// Move to the next act, if there is one, and land on its map.
    /// </summary>
    /// <remarks>
    /// Shared by the real boss path and the debug hook, deliberately: reaching act 2 the
    /// honest way costs a buffed run that wins a boss fight, several minutes a go, so
    /// every act-2 test would otherwise be gated behind completing act 1. The debug hook
    /// makes it a couple of seconds — and because both routes are this one function, the
    /// shortcut cannot drift from the thing it is standing in for.
    /// </remarks>
    public bool EnterNextAct()
    {
        if (!RunMapGenerator.AdvanceToNextAct(State))
        {
            return false;
        }

        RunRewardGenerator.ClearRewardScreen(State);
        // Through EnterMapPhase, not by setting the phase: the options have to be
        // refreshed too, and on a new act the only one is the act's starting ancient.
        EnterMapPhase();
        return true;
    }

    private int AdvanceAfterNode(out bool terminal)
    {
        terminal = false;
        // The BACKSTOP, not the real ending: a run finishes by beating the last act's boss,
        // which `AdvanceAfterRelicReward` handles. This only catches a run that has walked
        // past where any map could take it -- so it reads the boss row off the map the run
        // is on rather than assuming the standard one. Golden Compass lays a map a row
        // taller, and a constant here cut its last room off. It reads the map POSITION,
        // not the floor: the floor carries across acts, so act 2's rooms are floors 19-34,
        // and comparing them with one act's boss row ended every act-2 run on floor 19.
        int bossRow = RunMapGenerator.BossRow(State);
        if (!State.AwaitingActStartNode && State.CurrentMapCoord.Row >= bossRow)
        {
            if (EnterNextAct())
            {
                return 0;
            }

            State.Phase = RunPhase.Complete;
            terminal = true;
            return 0;
        }
        EnterMapPhase();
        return 0;
    }

    /// <summary>
    /// Girya's remaining lifts. `maxLifts` is 3 and `TimesLifted` lives on the relic; the
    /// emulator keeps it in the instance's Counter.
    /// </summary>
    /// <summary>
    /// Whether this rest visit has already taken that option. Only reachable with
    /// Miniature Tent, which is the one relic that lets a visit take more than one.
    /// </summary>
    private bool RestOptionSpent(int action) => (State.RestOptionsTaken & (1 << action)) != 0;

    private int GiryaLiftsLeft()
    {
        int index = State.Relics.FindIndex(relic => relic.DefId == Effects.RelicEffects.Girya);
        return index < 0 ? 0 : Math.Max(0, 3 - State.Relics[index].Counter);
    }

    /// <summary>
    /// Ends the rest visit after an option is taken — unless Miniature Tent is held, whose
    /// `ShouldDisableRemainingRestSiteOptions` returns FALSE and leaves the rest of the
    /// screen available. The option just taken is spent either way, so the Tent buys
    /// another DIFFERENT option rather than the same one twice.
    /// </summary>
    private void FinishRestOption(int action)
    {
        State.RestOptionsTaken |= 1 << action;
        if (!Effects.RelicEffects.KeepsRestSiteOpen(State))
        {
            State.RestResultPending = true;
        }
    }

    private int StepRest(int action, out bool terminal)
    {
        terminal = false;
        if (State.RestResultPending)
        {
            State.RestResultPending = false;
            return AdvanceAfterNode(out terminal);
        }

        if (action == RunConstants.RestHealAction)
        {
            State.PlayerHp = Math.Min(State.PlayerMaxHp, State.PlayerHp + RestHealAmount());
            ApplyAfterRestSiteHeal();
            // Tiny Mailbox's TryModifyRestSiteHealRewards adds two PotionRewards to the
            // rest, and a reward is offered rather than given: the player chooses whether
            // to take it, and may drop a held potion to make room.
            if (Effects.RelicEffects.Has(State.Relics, Effects.RelicEffects.TinyMailbox))
            {
                QueuePotionRewards([0, 0]);
                OfferNextRestPotion();
                State.Phase = RunPhase.RelicReward;
                return 0;
            }

            FinishRestOption(action);
            return 0;
        }
        if (action == RunConstants.RestLiftAction && GiryaLiftsLeft() > 0)
        {
            int index = State.Relics.FindIndex(relic => relic.DefId == Effects.RelicEffects.Girya);
            State.Relics[index] = State.Relics[index] with
            {
                Counter = State.Relics[index].Counter + 1,
            };
            FinishRestOption(action);
            return 0;
        }

        if (
            action == RunConstants.RestDigAction
            && State.Relics.Any(relic => relic.DefId == Effects.RelicEffects.Shovel)
        )
        {
            // `RelicCmd.Obtain(RelicFactory.PullNextRelicFromFront(owner))` -- the same
            // queue and the same END an elite reward pulls from, so a dug relic is one the
            // run will not offer again.
            RunNonCombatEffects.ApplyRelicPickup(State, RunRewardGenerator.NextRelic(State));
            FinishRestOption(action);
            return 0;
        }

        if (
            action == RunConstants.RestHatchAction
            && State.Deck.Any(card => card.DefId == RunNonCombatEffects.ByrdonisEggCard)
        )
        {
            // `HatchRestSiteOption.OnSelect` is `RelicCmd.Obtain<Byrdpip>` and nothing
            // else -- the egg's transformation is the RELIC's `AfterObtained`, so it lands
            // however the relic is obtained rather than only through the rest site.
            RunNonCombatEffects.ApplyRelicPickup(State, Effects.RelicEffects.Byrdpip);
            FinishRestOption(action);
            return 0;
        }

        if (
            action == RunConstants.RestKindleAction
            && State.Relics.Any(relic => relic.DefId == RunConstants.RelicPumpkinCandle)
        )
        {
            // `KindleRestSiteOption.OnSelect` is `Rekindle()` and nothing else, which is
            // exactly what the pickup does -- so it goes through the same arm.
            RunNonCombatEffects.ApplyRelicPickup(State, RunConstants.RelicPumpkinCandle);
            FinishRestOption(action);
            return 0;
        }

        if (
            action == RunConstants.RestCookAction
            && State.Relics.Any(relic => relic.DefId == RunConstants.RelicMeatCleaver)
            && RunNonCombatEffects.CanCook(State)
        )
        {
            // Remove TWO and then gain nine max HP. The HP rides the follow-up because the
            // game gains it after the removals resolve, and because a selection the player
            // cancels out of pays nothing -- `OnSelect` returns false on an empty pick.
            if (
                RunNonCombatEffects.BeginDeckSelection(
                    State,
                    DeckSelection.Remove,
                    0,
                    count: 2,
                    followUpMaxHpGain: 9,
                    returnTo: SelectionReturn.Rest
                )
            )
            {
                State.Phase = RunPhase.TransformSelect;
                return 0;
            }

            FinishRestOption(action);
            return 0;
        }

        if (
            action == RunConstants.RestCloneAction
            && State.Relics.Any(relic => relic.DefId == RunConstants.RelicPaelsGrowth)
        )
        {
            // CloneRestSiteOption: one copy of every Clone-enchanted card, added to the
            // deck. This is the whole of what the Clone enchantment does — it overrides
            // nothing in a fight — so a run holding Pael's Growth and never resting gets
            // nothing from it at all.
            var copies = State.Deck.Where(card => card.Enchantment == Enchantment.Clone).ToList();
            foreach (var card in copies)
            {
                RunNonCombatEffects.AddCardToDeck(State, card);
            }

            FinishRestOption(action);
            return 0;
        }

        if (action == RunConstants.RestUpgradeAction)
        {
            if (!State.Deck.Any(RunConstants.IsRunCardUpgradable))
            {
                return -1;
            }

            State.PendingRestUpgrade = true;
            State.Phase = RunPhase.TransformSelect;
            return 0;
        }
        if (action == RunConstants.RewardSkipAction)
        {
            return AdvanceAfterNode(out terminal);
        }

        return -1;
    }

    private int StepEvent(int action, out bool terminal)
    {
        terminal = false;
        if (State.EventId == RunConstants.EventResultPending)
        {
            State.EventId = 0;
            return AdvanceAfterNode(out terminal);
        }

        switch (State.EventId)
        {
            case RunConstants.EventTheArchitect:
                // `WinRun`: the run ends, won. PROCEED is its one option and the only way
                // out of the room, so leaving is the same step.
                if (action is not (0 or RunConstants.EventSkipAction))
                {
                    return -1;
                }

                State.EventId = 0;
                State.Phase = RunPhase.Complete;
                State.LastPlayerWon = true;
                terminal = true;
                return 0;
            case RunConstants.EventUnrestSite:
                if (action == 0)
                {
                    State.PlayerHp = State.PlayerMaxHp;
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(
                            RunNonCombatEffects.NamedCard("PoorSleep"),
                            Upgraded: false
                        )
                    );
                }
                else if (action == 1)
                {
                    State.PlayerMaxHp = Math.Max(1, State.PlayerMaxHp - 8);
                    State.PlayerHp = Math.Min(State.PlayerHp, State.PlayerMaxHp);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }
                break;
            case RunConstants.EventAromaOfChaos:
                // Both options are a choice: which card is transformed, which upgraded.
                if (action is 0 or 1)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        action == 0 ? DeckSelection.TransformToRandom : DeckSelection.Upgrade,
                        0,
                        eventEntry: action == 0 ? "AROMA_OF_CHAOS" : null
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventJungleMazeAdventure:
                // Both purses are rolled with NextFloat, not fixed: 150 and 50, each
                // shifted by NextFloat(-15, 15) off the event's own stream.
                if (action == 0)
                {
                    RunNonCombatEffects.LoseHp(State, 18);
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.JungleMazeSoloGold(State)
                    );
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.JungleMazeJoinForcesGold(State)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventMorphicGrove:
                // Group transforms TWO cards the player picks.
                if (action == 0)
                {
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.TransformToRandom,
                            0,
                            count: 2,
                            eventEntry: "MORPHIC_GROVE"
                        )
                    )
                    {
                        return -1;
                    }

                    State.Gold = 0;
                    return 0;
                }

                if (action == 1)
                {
                    RunNonCombatEffects.GainMaxHp(State, 5);
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSelfHelpBook:
                if (action is >= 0 and <= 2)
                {
                    var page = RunNonCombatEffects.SelfHelpBookEnchantment(action);
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)page,
                        cardType: RunNonCombatEffects.SelfHelpBookCardType(page)
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventDoorsOfLightAndDark:
                if (action == 0)
                {
                    RunNonCombatEffects.UpgradeTwoRandomCardsForLightDoor(State);
                    State.EventId = RunConstants.EventResultPending;
                    return 0;
                }
                if (action == 1)
                {
                    // The Dark door removes one card the player picks. Light stays a
                    // roll: it StableShuffles the upgradable cards and takes two.
                    return RunNonCombatEffects.BeginDeckSelection(State, DeckSelection.Remove, 0)
                        ? 0
                        : -1;
                }
                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSunkenTreasury:
                if (action == 0)
                {
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.SunkenTreasurySmallChestGold(State)
                    );
                }
                else if (action == 1)
                {
                    // The big chest is paid for with Greed.
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.SunkenTreasuryLargeChestGold(State)
                    );
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Greed"), Upgraded: false)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventBrainLeech:
                if (action == 0)
                {
                    // Share Knowledge rolls five cards through the reward machinery and
                    // opens a grid to keep one. The emulator used to pick a card itself,
                    // off the map-generation stream at that, which is neither the right
                    // choice nor the right stream.
                    // `CardFactory.CreateForReward` with `ForNonCombatWithDefaultOdds`: the
                    // NoUpgradeRoll flag and nothing else. No IsCardReward, so Silver
                    // Crucible stays out of it; the eggs, Wing Charm and Glitter do not.
                    RunRewardGenerator.OpenRewardGrid(
                        State,
                        RunRewardGenerator.CreateForReward(
                            State,
                            RunConstants.BrainLeechCardChoices,
                            RunRewardGenerator.OwnCardPool(State),
                            CardCreationFlags.NoUpgradeRoll
                        ),
                        picks: 1,
                        hooked: true
                    );
                    return 0;
                }
                else if (action == 1)
                {
                    // Rip offers a COLORLESS card reward -- three cards rolled from the
                    // colourless pool at the usual odds. Three card ids stood here,
                    // hard-written, which is the same reward whatever the seed.
                    RunNonCombatEffects.LoseHp(State, 5);
                    State.RewardGold = 0;
                    State.RewardPotion = 0;
                    State.RelicReward = 0;
                    State.PendingPotionRewards.Clear();
                    // A `CardReward` over the colourless pool: NoUpgradeRoll,
                    // NoRarityModification and NoCardPoolModifications from the event and
                    // IsCardReward from the reward itself -- so Silver Crucible upgrades the
                    // three and the eggs and Wing Charm take their turn (real runs, e.g.
                    // ABB6PCF77A: the Crucible's three arrived plain).
                    RunRewardGenerator
                        .CreateForReward(
                            State,
                            State.RewardCards.Length,
                            GeneratedData.CardPools.Colorless,
                            CardCreationFlags.NoUpgradeRoll
                                | CardCreationFlags.NoRarityModification
                                | CardCreationFlags.NoCardPoolModifications
                                | CardCreationFlags.IsCardReward
                        )
                        .ShowOn(State);
                    State.RewardCardPending = true;
                    State.Phase = RunPhase.RelicReward;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventByrdonisNest:
                if (action == 0)
                {
                    RunNonCombatEffects.GainMaxHp(State, 7);
                    State.EventId = RunConstants.EventResultPending;
                    return 0;
                }

                // Take the Egg adds the Byrdonis Egg quest card, which had no id until
                // the negative-cost cards were extracted -- so the option was refused.
                if (action == 1)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(
                            RunNonCombatEffects.NamedCard("ByrdonisEgg"),
                            Upgraded: false
                        )
                    );
                    State.EventId = RunConstants.EventResultPending;
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTheLegendsWereTrue:
                if (action == 0)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunConstants.SpoilsMapCard, Upgraded: false)
                    );
                }
                else if (action == 1)
                {
                    // `SlowlyFindAnExit`: the damage (lethal is allowed; Step ends the run at 0),
                    // then ONE `PlayerRng.Rewards.NextItem` over the character's potion pool
                    // followed by the shared one -- a uniform pick, not a reward's rarity roll.
                    RunNonCombatEffects.LoseHp(State, 8);
                    OfferPotionRewards(RunRewardGenerator.NextPoolPotion(State, State.PlayerRng.Rewards));
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventDenseVegetation when State.EventPage == 1:
                // The page the Rest answers with: one option, and it is the fight.
                if (action != 0)
                {
                    return -1;
                }

                return StartCombatWithDeck(
                    State.Deck,
                    RunConstants.DenseVegetationEncounterId,
                    State.Relics,
                    State.PlayerHp,
                    State.PlayerMaxHp,
                    State.PotionSlots,
                    State.Gold
                );

            case RunConstants.EventDenseVegetation:
                if (action == 0)
                {
                    // The gold is rolled -- NextInt(61, 100) on the event's own stream --
                    // and RunNonCombatEffects already rolled it. The handler was paying a
                    // flat 80, which is inside the range and so looked right in every
                    // capture that never checked the number.
                    RunNonCombatEffects.LoseHp(State, 8);
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.DenseVegetationGold(State)
                    );
                }
                else if (action == 1)
                {
                    // Rest mimics a rest site and then wakes something up: the page it
                    // answers with has one option, and that option is the fight.
                    HealPlayer(RestHealAmount());
                    ApplyAfterRestSiteHeal();
                    State.EventPage = 1;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventLuminousChoir:
                // The player picks the two cards that go; the Spore Mind arrives after
                // they are gone. RemoveLowestPriorityCard was the emulator choosing for
                // them, which is the whole cost of the option.
                if (action == 0)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Remove,
                        0,
                        count: 2,
                        followUpCard: RunNonCombatEffects.NamedCard("SporeMind"),
                        followUpCount: 1
                    )
                        ? 0
                        : -1;
                }

                if (action == 1)
                {
                    // The tribute is the ROLLED price -- 149 minus NextInt(0, 50) -- which
                    // is what the action mask already read. Charging the flat 149 here
                    // meant a run holding, say, 120 gold was offered the tribute and then
                    // refused when it took it: the same mask/step disagreement the Tea
                    // Master had.
                    int tribute = RunNonCombatEffects.LuminousChoirTributeCost(State);
                    if (State.Gold < tribute)
                    {
                        return -1;
                    }

                    State.Gold -= tribute;
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSapphireSeed:
                // Consume heals and then asks WHICH card to upgrade; Plant enchants with
                // Sown. The emulator healed and upgraded whatever came first, and had no
                // Plant at all, so the game's second option was refused.
                if (action == 0)
                {
                    HealPlayer(9);
                    RunNonCombatEffects.BeginDeckSelection(State, DeckSelection.Upgrade, 0);
                    return 0;
                }

                if (action == 1)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Sown
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSunkenStatue:
                // GrabSword obtains one named relic -- RelicCmd.Obtain<SwordOfStone> --
                // not a roll from the reward pool.
                if (action == 0)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.SwordOfStoneRelic
                    );
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.GainGold(
                        State,
                        RunNonCombatEffects.SunkenStatueGold(State)
                    );
                    RunNonCombatEffects.LoseHp(State, 7);
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTabletOfTruth:
                // The tablet is five secrets, not one, and each costs more than the last:
                // 3, 6, 12, 24, and then everything but a single point of Max HP. The
                // emulator read exactly one of them for a flat 3 and upgraded whichever
                // card came first, so the decision the event is built around -- how far to
                // read before it takes the run -- did not exist.
                if (action == 0)
                {
                    RunNonCombatEffects.Decipher(State);
                    if (RunNonCombatEffects.TabletHasMoreToSay(State))
                    {
                        return 0;
                    }
                }
                else if (action == 1)
                {
                    // Smash heals 20 on the first page; on a later one the second option
                    // is Give Up, which does nothing at all.
                    if (State.EventPage == 0)
                    {
                        HealPlayer(20);
                    }
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWellspring:
                if (action == 0)
                {
                    OfferPotionRewards(RunRewardGenerator.NextPoolPotion(State, State.PlayerRng.Rewards));
                    return 0;
                }

                if (action == 1)
                {
                    // Bathe removes one card the PLAYER picks, then adds BatheCurses (1)
                    // Guilty.
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Remove,
                        0,
                        count: 1,
                        followUpCard: RunNonCombatEffects.NamedCard("Guilty"),
                        followUpCount: 1
                    )
                        ? 0
                        : -1;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWhisperingHollow:
                // Exchange Gold SPENDS the rolled amount for two potions on a reward
                // screen -- the emulator was paying it out instead. Hug transforms a card
                // the player picks and only then charges the 9 HP, which is why a capture
                // taken at the selector shows full health.
                if (action == 0)
                {
                    int cost = RunNonCombatEffects.WhisperingHollowGold(State);
                    if (State.Gold < cost)
                    {
                        return -1;
                    }

                    State.Gold -= cost;
                    OfferPotionRewards(0, 0);
                    return 0;
                }

                if (action == 1)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.TransformToRandom,
                        0,
                        count: 1,
                        followUpHpLoss: 9,
                        eventEntry: "WHISPERING_HOLLOW"
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWoodCarvings:
                // Three carvings, three different things: a Basic card becomes a Peck, or
                // is enchanted with Slither, or becomes a Toric Toughness. All three used
                // to transform the first card in the deck into a rolled one.
                if (action is 0 or 2)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.TransformTo,
                        action == 0
                            ? RunNonCombatEffects.NamedCard("Peck")
                            : RunNonCombatEffects.NamedCard("ToricToughness")
                    )
                        ? 0
                        : -1;
                }

                if (action == 1)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Slither
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventAbyssalBaths:
                // Immersing is not one choice but the first of a series. OnImmerse gains
                // MaxHpVar(2), takes DamageVar and then raises that damage by one, and
                // the page it answers with offers Linger -- the same immersion again, a
                // point dearer every time. The emulator modelled a single dip for a flat
                // 1 damage, which is the NET of the first one rather than the hit, and
                // never offered the page at all: the whole decision the event is about,
                // how deep to go before the escalation catches you, did not exist.
                //
                // EventPage counts immersions, which is what sets the price of the next.
                if (action == 0)
                {
                    RunNonCombatEffects.Immerse(State);
                    return 0;
                }

                // On the second page the options are Linger and Exit; on the first they
                // are Immerse and Abstain, and only Abstain heals.
                if (action == 1)
                {
                    if (State.EventPage > 0)
                    {
                        State.EventId = RunConstants.EventResultPending;
                        return 0;
                    }

                    HealPlayer(10);
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventDrowningBeacon:
                // Bottle offers the Glowwater Potion on a reward screen
                // (RewardsCmd.OfferCustom), so the player can decline it or make room.
                // Climb is the option that costs Max HP -- the emulator had the cost on
                // Bottle and no Climb at all.
                if (action == 0)
                {
                    State.RewardGold = 0;
                    State.RelicReward = 0;
                    State.RewardPotion = RunNonCombatEffects.GlowwaterPotion;
                    State.Phase = RunPhase.RelicReward;
                    return 0;
                }

                if (action == 1)
                {
                    RunNonCombatEffects.LoseMaxHp(State, 13);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.FresnelLensRelic
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventEndlessConveyor:
                // The belt keeps turning: grabbing pays for whatever dish RollDish
                // landed on and then offers the next one, so the event is a loop the
                // player leaves rather than a single choice.
                if (action == 0)
                {
                    // Grabbing pays GoldVar(40) for whatever the belt is carrying, then
                    // the belt turns. Jelly Liver is the one dish that needs the player:
                    // it transforms a card they pick.
                    // GenerateGrabSomethingOffTheBeltOption locks the grab below the
                    // price, so a run that cannot pay is not offered it and must not be
                    // allowed to take it.
                    if (State.Gold < RunConstants.ConveyorGrabCost)
                    {
                        return -1;
                    }

                    int dish = RunNonCombatEffects.EndlessConveyorDish(State);
                    // `GrabSomethingOffTheBelt` charges the 40 unless the dish is the Golden
                    // Fysh, which is free and pays 75.
                    if (dish != RunNonCombatEffects.DishGoldenFysh)
                    {
                        State.Gold -= RunConstants.ConveyorGrabCost;
                    }

                    State.EventPage++;
                    // The dish's action runs to its end -- the transform or the potion screen
                    // included, both awaited -- and only then does RollDish draw the next one,
                    // off the same event stream. Rolling first put the transform's draw after
                    // the roll's (real runs: the Jelly Liver turned a Strike into another card).
                    if (dish == RunNonCombatEffects.DishJellyLiver)
                    {
                        if (
                            RunNonCombatEffects.BeginDeckSelection(
                                State,
                                DeckSelection.TransformToRandom,
                                arg: 0,
                                eventEntry: "ENDLESS_CONVEYOR"
                            )
                        )
                        {
                            State.PendingSelectionReturnsToEvent = true;
                            State.Phase = RunPhase.TransformSelect;
                            return 0;
                        }

                        RunNonCombatEffects.RollNextConveyorDish(State);
                        return 0;
                    }

                    if (dish == RunNonCombatEffects.DishSuspiciousCondiment)
                    {
                        // `RewardsCmd.OfferCustom(PotionReward)`: the potion is on a rewards
                        // screen, and the belt turns when it is answered.
                        RunNonCombatEffects.ApplyEndlessConveyorDish(State, dish);
                        int potion = State.PendingPotionRewards[^1];
                        State.PendingPotionRewards.RemoveAt(State.PendingPotionRewards.Count - 1);
                        OfferPotionRewards(potion);
                        State.EventResumesAfterRewards = true;
                        return 0;
                    }

                    RunNonCombatEffects.ApplyEndlessConveyorDish(State, dish);
                    RunNonCombatEffects.RollNextConveyorDish(State);
                    return 0;
                }

                if (action == 1)
                {
                    // Only the FIRST page offers Observe the Chef. Once the player has
                    // grabbed, the belt's second option is Leave, which does nothing --
                    // upgrading a card for it paid them to walk away.
                    if (State.EventPage == 0)
                    {
                        RunNonCombatEffects.UpgradeRandomCard(State, "ENDLESS_CONVEYOR");
                    }
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventPunchOff:
                // Nab takes an Injury and offers a relic on a reward screen -- it pays no
                // gold and no potion. "I Can Take Them" does not fight: it answers with a
                // second page whose only option does, which is the shape the capture
                // shows -- the run stays on the event and nothing changes.
                if (State.EventPage == 1)
                {
                    if (action != 0)
                    {
                        return -1;
                    }

                    return StartCombatWithDeck(
                        State.Deck,
                        RunConstants.PunchOffEncounterId,
                        State.Relics,
                        State.PlayerHp,
                        State.PlayerMaxHp,
                        State.PotionSlots,
                        State.Gold
                    );
                }

                if (action == 1)
                {
                    State.EventPage = 1;
                    return 0;
                }

                if (action == 0)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Injury"), Upgraded: false)
                    );
                    State.RewardGold = 0;
                    State.RewardPotion = 0;
                    State.PendingPotionRewards.Clear();
                    State.RelicReward = RunRewardGenerator.NextRelic(State);
                    State.Phase = RunPhase.RelicReward;
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSpiralingWhirlpool:
                // Observe enchants; Drink heals a third of Max HP. The emulator had the
                // healing on Observe, at the rest-site amount, and made Drink transform.
                if (action == 0)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Spiral
                    )
                        ? 0
                        : -1;
                }

                if (action == 1)
                {
                    HealPlayer(RunNonCombatEffects.SpiralingWhirlpoolHeal(State));
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTrashHeap:
                // DiveIn pays HP for a relic; Grab pays nothing for gold and a card.
                // Both prizes come off the event's own fixed tables, not the reward
                // pools -- the emulator had the two options' costs and prizes crossed.
                if (action == 0)
                {
                    RunNonCombatEffects.LoseHp(State, 8);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.TrashHeapRelic(State)
                    );
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.GainGold(State, 100);
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.TrashHeapCard(State), Upgraded: false)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWaterloggedScriptorium:
                // Bloody Ink is the free option and comes FIRST; the two paid options
                // both enchant with Steady, at 55 gold for one card and 99 for two. The
                // emulator had Bloody Ink second, upgraded instead of enchanting, and
                // turned the 99-gold option into a card reward.
                if (action == 0)
                {
                    RunNonCombatEffects.GainMaxHp(State, 6);
                }
                else if (action is 1 or 2)
                {
                    int cost =
                        action == 1
                            ? RunConstants.ScriptoriumQuillCost
                            : RunConstants.ScriptoriumSpongeCost;
                    if (State.Gold < cost)
                    {
                        return -1;
                    }

                    // The gold is spent before the selector opens, and the game does not
                    // refund it when the player picks nothing -- so a refused selection
                    // must not have taken the gold either.
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Enchant,
                            (int)Enchantment.Steady,
                            action == 1 ? 1 : 2
                        )
                    )
                    {
                        return -1;
                    }

                    State.Gold -= cost;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventCrystalSphere:
                // Both options end in the same minigame, differing only in what they cost
                // and how many divinations they buy. The price is not a flat 100: it is
                // rolled in CalculateVars, off the first draw of the event's own stream.
                if (action == 0)
                {
                    int cost = RunNonCombatEffects.CrystalSphereCost(State);
                    if (State.Gold < cost)
                    {
                        return -1;
                    }

                    State.Gold -= cost;
                    RunNonCombatEffects.OpenCrystalSphere(State, 3);
                    return 0;
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Debt"), Upgraded: false)
                    );
                    RunNonCombatEffects.OpenCrystalSphere(State, 6);
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventDollRoom when State.EventPage > 0:
            {
                // The second page: pick one of the dolls the wait bought.
                var offer = RunNonCombatEffects.DollRoomOffer(State, State.EventPage == 1 ? 2 : 3);
                if ((uint)action >= (uint)offer.Count)
                {
                    return action == RunConstants.EventSkipAction ? 0 : -1;
                }

                RunNonCombatEffects.ApplyRelicPickup(State, offer[action]);
                State.EventId = RunConstants.EventResultPending;
                return 0;
            }

            case RunConstants.EventDollRoom:
                // The dolls are three named relics, rolled from the event's own stream --
                // not from the reward pool. Only "Pick at Random" hands one over here:
                // the two paid options buy a CHOICE of doll on a second page, so they
                // cost the HP and grant nothing yet.
                if (action == 0)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.DollRoomRandomDoll(State)
                    );
                }
                else if (action is 1 or 2)
                {
                    // What the HP buys is the CHOICE, on a second page: five for two
                    // dolls, fifteen for all three. Costing the HP and finishing there
                    // charged the player for a decision they never got to make.
                    RunNonCombatEffects.LoseHp(State, action == 1 ? 5 : 15);
                    State.EventPage = action;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventFakeMerchant:
                // `GenerateInitialOptions` returns Array.Empty: the stall has no options
                // at all, only its own screen. Anything but the proceed that opens it is
                // not a move the event has. Taking one relic off the pool queue -- which
                // is what was here -- was neither the right relic nor the right screen.
                if (action == 0)
                {
                    State.Phase = RunPhase.FakeMerchant;
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventPotionCourier:
                // Both options offer potions on a reward screen -- three named Foul
                // Potions, or one rolled Uncommon. Neither forces anything into the belt.
                if (action == 0)
                {
                    int foul = RunNonCombatEffects.NamedPotion("FoulPotion");
                    OfferPotionRewards(foul, foul, foul);
                    return 0;
                }

                if (action == 1)
                {
                    OfferPotionRewards(
                        RunRewardGenerator.NextUncommonPotion(State, State.PlayerRng.Rewards)
                    );
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventRanwidTheElder:
                if (action == 0)
                {
                    // The potion the elder named when the options were built, not the first held.
                    int slot = RunNonCombatEffects.EventPotionSlot(State);
                    if (slot < 0)
                    {
                        return -1;
                    }

                    State.PotionSlots[slot] = 0;
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action == 1)
                {
                    // GiveGold is offered unconditionally and simply loses GoldVar(100);
                    // LoseGold floors at zero, so a player with 99 hands over 99 and
                    // still gets the relic. Refusing below 100 made the option look
                    // unaffordable when the game never priced it that way.
                    State.Gold = Math.Max(0, State.Gold - 100);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action == 2)
                {
                    // The elder names the relic he wants -- rolled off the event's own
                    // stream from the ones he will take -- and hands back TWO. The
                    // emulator gave up whatever sat in slot 1, which is a positional
                    // guess that would surrender an untradable relic, and returned one.
                    int traded = RunNonCombatEffects.RanwidTradeIndex(State);
                    if (traded < 0)
                    {
                        return -1;
                    }

                    State.Relics.RemoveAt(traded);
                    for (int i = 0; i < 2; i++)
                    {
                        RunNonCombatEffects.ApplyRelicPickup(
                            State,
                            RunRewardGenerator.NextRelic(State)
                        );
                    }
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventRelicTrader:
                // The trader offers up to three of the player's OWN tradable relics, in
                // the order it shuffled them -- so the option index reads that list, not
                // state.Relics. The step used to index state.Relics directly (skipping
                // slot 0 to dodge Burning Blood) and to accept every option when nothing
                // was tradable, where the game offers a lone Proceed.
                if (action is >= 0 and <= 2)
                {
                    var stock = RunNonCombatEffects.RelicTraderStock(State);
                    if (stock.Count == 0)
                    {
                        // Nothing to trade: index 0 is the Proceed and the rest are not
                        // options at all.
                        if (action != 0)
                        {
                            return -1;
                        }

                        break;
                    }

                    if (action >= stock.Count)
                    {
                        return -1;
                    }

                    State.Relics.RemoveAt(stock[action]);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventRoomFullOfCheese:
                if (action == 0)
                {
                    // Gorge rolls eight Commons and opens a grid to keep two. Uniform
                    // odds with the pool filtered to Commons means no rarity roll at all,
                    // so each card is a single draw.
                    // NoUpgradeRoll and NoRarityModification, no IsCardReward: the eggs,
                    // Wing Charm and Glitter reach the grid, Silver Crucible does not.
                    RunRewardGenerator.OpenRewardGrid(
                        State,
                        RunRewardGenerator.CreateForReward(
                            State,
                            RunConstants.GorgeCardChoices,
                            RunRewardGenerator.OwnCardPool(State),
                            CardCreationFlags.NoUpgradeRoll | CardCreationFlags.NoRarityModification,
                            CardRarity.Common
                        ),
                        picks: RunConstants.GorgeCardsKept,
                        hooked: true
                    );
                    return 0;
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.LoseHp(State, 14);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("ChosenCheese")
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSlipperyBridge:
                // Overcome loses the ONE card the bridge is holding -- rolled off the event's
                // stream, non-Basic on the first roll -- and Hold On costs 3 + NumberOfHoldOns
                // HP, which is 3 the first time, not a flat 10.
                if (action == 0)
                {
                    int index = RunNonCombatEffects.SlipperyBridgeCardIndex(State);
                    if (index < 0)
                    {
                        return -1;
                    }

                    State.Deck.RemoveAt(index);
                }
                else if (action == 1)
                {
                    // CurrentHpLoss is 3 + NumberOfHoldOns, charged BEFORE the counter
                    // moves -- so 3, then 4, then 5 -- and each hold re-rolls which card
                    // the bridge is threatening. Holding on was a flat 3 that never
                    // escalated and never changed the card, which makes it free to repeat.
                    RunNonCombatEffects.LoseHp(
                        State,
                        RunNonCombatEffects.SlipperyBridgeHpLoss(State)
                    );
                    State.EventPage++;
                    // GetNewRandomCard() again -- the re-roll is an explicit step now that
                    // the named card is remembered rather than recomputed per read.
                    RunNonCombatEffects.RollSlipperyBridgeCard(State);
                    // The bridge answers with the same two options, so the event stays
                    // open rather than falling through to the end-of-event path.
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventStoneOfAllTime:
                // Lift drinks a potion for 10 Max HP, not 5. Push costs 6 HP, not 10, and
                // enchants an Attack with Vigorous 8 rather than upgrading whatever came
                // first -- and the HP is taken BEFORE the selector opens.
                if (action == 0)
                {
                    // `DrinkAndLiftPotion`: the potion picked when the options were built.
                    int slot = RunNonCombatEffects.EventPotionSlot(State);
                    if (slot < 0)
                    {
                        return -1;
                    }

                    State.PotionSlots[slot] = 0;
                    RunNonCombatEffects.GainMaxHp(State, 10);
                }
                else if (action == 1)
                {
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Enchant,
                            (int)Enchantment.Vigorous
                        )
                    )
                    {
                        return -1;
                    }

                    RunNonCombatEffects.LoseHp(State, 6);
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSymbiote:
                // Approach enchants an Attack with Corrupted -- it does not upgrade --
                // and Kill with Fire lets the player CHOOSE what burns.
                if (action == 0)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Corrupted
                    )
                        ? 0
                        : -1;
                }

                if (action == 1)
                {
                    return RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.TransformToRandom,
                        0,
                        eventEntry: "SYMBIOTE"
                    )
                        ? 0
                        : -1;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTeaMaster:
                // Three named teas at three prices. Bone Tea is 50, not 150 -- the
                // action mask already read BoneTeaCost while the step charged the
                // Ember price for both, and all three handed over a pool relic.
                if (action == 0)
                {
                    if (State.Gold < RunConstants.BoneTeaCost)
                    {
                        return -1;
                    }

                    State.Gold -= RunConstants.BoneTeaCost;
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("BoneTea")
                    );
                }
                else if (action == 1)
                {
                    if (State.Gold < RunConstants.EmberTeaCost)
                    {
                        return -1;
                    }

                    State.Gold -= RunConstants.EmberTeaCost;
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("EmberTea")
                    );
                }
                else if (action == 2)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("TeaOfDiscourtesy")
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTheFutureOfPotions:
                if (action is >= 0 and <= 2)
                {
                    int seen = -1;
                    int slot = -1;
                    for (int i = 0; i < State.PotionSlots.Length; i++)
                    {
                        if (State.PotionSlots[i] == 0)
                        {
                            continue;
                        }

                        seen++;
                        if (seen == action)
                        {
                            slot = i;
                            break;
                        }
                    }

                    if (slot < 0)
                    {
                        // With nothing to trade the event offers a lone Proceed at index
                        // 0, which is what the mask offers -- so refusing it here left
                        // the two disagreeing about the same action.
                        if (action == 0 && !State.PotionSlots.Any(potion => potion != 0))
                        {
                            break;
                        }

                        return -1;
                    }

                    // Three upgraded cards on a reward screen, all of the rarity the
                    // potion buys and the type the event rolled for it -- not one card
                    // taken off the map-generation stream and pushed into the deck.
                    int traded = State.PotionSlots[slot];
                    var type = (uint)action < (uint)State.EventRandomOffer.Length
                        ? (CardType)State.EventRandomOffer[action]
                        : RunNonCombatEffects.EventStream(State, "THE_FUTURE_OF_POTIONS")
                            .NextItem(RunRewardGenerator.FutureOfPotionsCardTypes(traded));
                    State.PotionSlots[slot] = 0;
                    RunRewardGenerator.EnterFutureOfPotionsReward(State, traded, type);
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventThisOrThat:
                // The plain chest costs 6 HP, not 7, and its purse is NextInt(41, 69) off
                // the event's own stream rather than a generic 55 +/- 15 off UpFront.
                if (action == 0)
                {
                    RunNonCombatEffects.LoseHp(State, 6);
                    RunNonCombatEffects.GainGold(State, RunNonCombatEffects.ThisOrThatGold(State));
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Clumsy"), Upgraded: false)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWarHistorianRepy:
                // Unlock the Cage frees Repy for the named History Course; Unlock the
                // Chest opens a reward screen carrying two potions and two relics. The
                // emulator had the potion on the cage option and a pool relic on both.
                // Both doors spend a Lantern Key, and a run that arrived holding two
                // gets to open the other one as well -- ShouldGetSecondReward. The key
                // has a card id now that the negative-cost cards are extracted, so the
                // deck is where that is read from rather than being assumed away.
                if (action == 0)
                {
                    RunNonCombatEffects.SpendLanternKey(State);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("HistoryCourse")
                    );
                    // A run that arrived with two keys still holds one, and Repy offers
                    // the other door.
                    if (State.EventPage == 0 && RunNonCombatEffects.RepyOwesASecondReward(State))
                    {
                        State.EventPage = 1;
                        return 0;
                    }
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.SpendLanternKey(State);
                    // `UnlockChest`: RewardsCmd.OfferCustom of [PotionReward, PotionReward,
                    // RelicReward, RelicReward], populated in that order -- two potion rolls,
                    // then two relics off the front of the queue. The screen carries one potion
                    // at a time, so the second waits behind the first.
                    OfferPotionRewards(0, 0);
                    State.RelicReward = RunRewardGenerator.NextRelic(State);
                    State.PendingBonusRelicRewards.Add(RunRewardGenerator.NextRelic(State));
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventWelcomeToWongos:
                // Three price tags, not one: 100, 200 and 300, and each option is locked
                // below its OWN price. Charging a flat 100 for all three sold the mystery
                // box at a third of its price, and only the mask knew the real numbers.
                if (action is >= 0 and <= 2)
                {
                    int price = action switch
                    {
                        0 => RunConstants.WongosBargainBinCost,
                        1 => RunConstants.WongosFeaturedItemCost,
                        _ => RunConstants.WongosMysteryBoxCost,
                    };
                    if (State.Gold < price)
                    {
                        return -1;
                    }

                    State.Gold -= price;

                    // Three tiers, three different things -- not one rolled relic three
                    // times. The bin is a shop-legal COMMON, the featured item a shop-legal
                    // RARE named on the option when the event opened, and the mystery box
                    // is Wongo's Mystery Ticket by name.
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        action switch
                        {
                            0 => RunRewardGenerator.NextShopRelicOfRarity(
                                State,
                                RelicRarity.Common
                            ),
                            1 => RunNonCombatEffects.WongosFeaturedItem(State),
                            _ => RunNonCombatEffects.NamedRelic("WongosMysteryTicket"),
                        }
                    );

                    // Buying also earns Wongo points toward a customer badge, which
                    // accumulate in the PROFILE across runs (SaveManager.Progress) rather
                    // than in the run. A run cannot know that total, so the badge is not
                    // modelled -- see the fixed-profile note in HANDOFF.md.
                }
                else if (action == RunConstants.EventSkipAction)
                {
                    // `Leave` is a real option here, and it is not free: it downgrades one
                    // upgraded card, picked off the event's own stream.
                    RunNonCombatEffects.DowngradeRandomUpgradedCard(State, "WELCOME_TO_WONGOS");
                }
                else
                {
                    return -1;
                }

                break;
            case RunConstants.EventAmalgamator:
            {
                // TWO cards the player picks, removed, and ONE named card put in their
                // place: `FromDeckForRemoval(count: 2, filter: IsValid(tag, c))` then
                // `CardPileCmd.Add(UltimateStrike)`. `IsValid` is the TAG plus BASIC rarity
                // plus removable, so the filter is narrower than a plain removal in two
                // directions -- Ultimate Strike is Strike-tagged, and without the rarity
                // clause the event could eat its own reward.
                //
                // The emulator TRANSFORMED two Ironclad Strikes into random cards off the
                // Transformations stream. Wrong effect, wrong stream, wrong card, no
                // choice -- and it matched Ironclad's Strike id rather than the tag, so for
                // every other character it found nothing and silently did nothing.
                if (action is not (0 or 1))
                {
                    if (action != RunConstants.EventSkipAction)
                    {
                        return -1;
                    }

                    break;
                }

                bool strikes = action == 0;
                if (
                    !RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.RemoveTaggedBasic,
                        strikes ? 0 : 1,
                        count: 2,
                        followUpCard: RunNonCombatEffects.NamedCard(
                            strikes ? "UltimateStrike" : "UltimateDefend"
                        ),
                        followUpCount: 1
                    )
                )
                {
                    // Nothing to offer: the game still awaits an empty selection and adds
                    // the card below it, the way Field Of Man-Sized Holes does.
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(
                            RunNonCombatEffects.NamedCard(
                                strikes ? "UltimateStrike" : "UltimateDefend"
                            ),
                            false
                        )
                    );
                    break;
                }

                State.Phase = RunPhase.TransformSelect;
                return 0;
            }
            case RunConstants.EventBugslayer:
                // Both options ADD A NAMED CARD to the deck and nothing else --
                // `AddAndPreview<Exterminate>` and `AddAndPreview<Squash>`. There is no
                // gold, no HP cost and no card reward anywhere in the model; the emulator
                // had 75 gold and an upgraded random card, which is a different event.
                if (action == 0)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Exterminate"), false)
                    );
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Squash"), false)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventColorfulPhilosophers:
                // Each option is one of the OTHER characters' card pools, and taking it
                // opens three card-reward screens over that pool: one Common, one
                // Uncommon, one Rare, three cards each. What was here added a single
                // random Ironclad card to the deck and upgraded it when the player picked
                // the third option -- the player's own pool, no screen, no rarity, and an
                // upgrade nothing in the event grants.
                if (action is >= 0 and <= 2)
                {
                    var pools = ColorfulPhilosopherPools();
                    if (action >= pools.Count)
                    {
                        return -1;
                    }

                    OfferColorfulPhilosopherRewards(pools[action]);
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventColossalFlower:
                // A ladder, not a pair of one-shot options: three prizes at 35/75/135
                // gold, each dig deeper costing 5/6/7 unblockable damage, and the third
                // rung swapping the gold for Pollinous Core. `EventPage` is the dig
                // count, which is exactly the event's own `NumberOfDigs`.
                //
                // What was here -- 125 gold or a rolled potion -- was a placeholder: no
                // ladder, no damage, a number the event never pays and a potion it never
                // offers.
                if (action == 0)
                {
                    // Extract: the prize for the rung the player is standing on. On the
                    // last rung this is `ExtractInstead`, which pays the same 135.
                    // The FLAT prize, not a jittered one. `GainGold(_prizeCosts[digs])`
                    // takes the raw array entry; the `GoldVar`s of the same numbers exist
                    // only to write the prices into the option text. Lost Wisp's jitter is
                    // real because Lost Wisp gains `DynamicVars.Gold` -- this one does not.
                    RunNonCombatEffects.GainGold(
                        State,
                        RunConstants.ColossalFlowerPrizes[State.EventPage]
                    );
                }
                else if (action == 1)
                {
                    // Reach deeper takes the damage for the rung BEING LEFT -- the game
                    // deals `_prizeDamage[NumberOfDigs]` and increments afterwards, so the
                    // first dig costs 5 and the last costs 7. On the third the option is
                    // Pollinous Core instead, and it pays the same 7 to get it.
                    ColossalFlowerDig();
                    if (State.EventPage == 2)
                    {
                        RunNonCombatEffects.ApplyRelicPickup(
                            State,
                            RunNonCombatEffects.NamedRelic("PollinousCore")
                        );
                    }
                    else
                    {
                        State.EventPage++;
                        // Still standing at the flower: the next page offers the same two.
                        return 0;
                    }
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventFieldOfManSizedHoles:
                // Resist is CardSelectCmd.FromDeckForRemoval at CardsVar(2) followed by
                // AddCursesToDeck(Normality) -- TWO cards the player picks, and a curse for
                // them. The emulator removed ONE card of its own choosing and charged
                // nothing for it.
                if (action == 0)
                {
                    // A deck with nothing removable does not REFUSE the option: the game
                    // awaits an empty selection and runs AddCursesToDeck below it either
                    // way, so the curse still lands.
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Remove,
                            0,
                            count: 2,
                            followUpCard: RunNonCombatEffects.NamedCard("Normality"),
                            followUpCount: 1
                        )
                    )
                    {
                        RunNonCombatEffects.AddCardToDeck(
                            State,
                            new CardInstance(RunNonCombatEffects.NamedCard("Normality"), false)
                        );
                        State.EventId = RunConstants.EventResultPending;
                    }

                    return 0;
                }
                else if (action == 1)
                {
                    // Enter Your Hole is `FromDeckForEnchantment(PerfectFit, 1)` and
                    // nothing else -- one card the player picks, enchanted. It costs no HP
                    // and pays no relic; the 12 HP and a relic that used to stand here were
                    // invented, admitted as such in a TODO, and belonged to no option of
                    // this event. A live capture caught them: 64 HP in the game, 52 here.
                    if (
                        RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Enchant,
                            (int)Enchantment.PerfectFit,
                            count: 1
                        )
                    )
                    {
                        State.Phase = RunPhase.TransformSelect;
                        return 0;
                    }
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventInfestedAutomaton:
                // Both options add ONE card straight to the deck, rolled from the
                // CHARACTER'S OWN pool at non-combat odds and filtered: Study takes a
                // POWER, Touch Core takes anything printed at zero energy that is not an X
                // card. Neither offers a choice and neither costs anything. The emulator
                // opened a card-reward screen, or took 10 HP for a random relic.
                if (action == 0)
                {
                    AddFilteredPoolCardToDeck(def => def.Type == CardType.Power);
                }
                else if (action == 1)
                {
                    AddFilteredPoolCardToDeck(def => def.Cost == 0 && !def.HasEnergyCostX);
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventLostWisp:
                // Claim takes the LOST WISP relic and a DECAY curse with it; Search is
                // `GoldVar(60)` and nothing else. The emulator healed as if at a rest site
                // and handed out an upgraded random card -- neither option, and no curse.
                if (action == 0)
                {
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Decay"), false)
                    );
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("LostWisp")
                    );
                }
                else if (action == 1)
                {
                    // `GoldVar(60)` with `Gold.BaseValue += Rng.NextInt(-15, 16)`, rolled
                    // in the model when the options are generated -- so the option TEXT
                    // already names the jittered figure, and a live capture paid 70.
                    //
                    // Off the EVENT'S OWN stream, the way Sunken Statue and Luminous Choir
                    // roll theirs. `EventGoldAmount` applies the same +/-15 from
                    // `Rng.UpFront`, which is a different sequence: it paid 72 against the
                    // game's 70. That helper is shared by several other events and is worth
                    // re-reading for the same reason.
                    RunNonCombatEffects.GainGold(
                        State,
                        60 + RunNonCombatEffects.EventStream(State, "LOST_WISP").NextInt(-15, 16)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventSpiritGrafter:
                // Neither option was this event. LetItIn heals HealVar(25) and adds a
                // Metamorphosis to the deck; Rejection is FromDeckForUpgrade -- the player
                // upgrades a card of their choosing -- and then takes HpLossVar(10)
                // UNBLOCKABLE damage. The emulator removed a card and granted 3 max HP for
                // the first, and transformed-then-upgraded the deck's first card for the
                // second, charging nothing either way.
                if (action == 0)
                {
                    HealPlayer(25);
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Metamorphosis"), false)
                    );
                }
                else if (action == 1)
                {
                    // FromDeckForUpgrade on a fully upgraded deck returns nothing and
                    // Rejection carries on to the damage regardless -- so the option is
                    // still takeable, and refusing it made the event's mask offer a move
                    // its own step would not accept.
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Upgrade,
                            0,
                            followUpHpLoss: 10
                        )
                    )
                    {
                        RunNonCombatEffects.LoseHp(State, 10);
                        State.EventId = RunConstants.EventResultPending;
                    }

                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventTheLanternKey when State.EventPage == 1:
                // Keep The Key answers with a page carrying one option, and that option is
                // the fight -- `EnterCombatWithoutExitingEvent<MysteriousKnightEventEncounter>`
                // with a LanternKey card as its reward. Dense Vegetation's shape.
                if (action != 0)
                {
                    return -1;
                }

                State.PendingSpecialCardReward = RunNonCombatEffects.NamedCard("LanternKey");
                return StartCombatWithDeck(
                    State.Deck,
                    RunConstants.MysteriousKnightEncounterId,
                    State.Relics,
                    State.PlayerHp,
                    State.PlayerMaxHp,
                    State.PotionSlots,
                    State.Gold
                );

            case RunConstants.EventTheLanternKey:
                // Return The Key is `GoldVar(100)` flat -- `Gold.BaseValue`, with none of
                // the jitter Lost Wisp rolls into its own var. Keep The Key opens the fight
                // page above. The emulator had the two the wrong way round AND wrong: it
                // handed out the key card and a relic for FREE where the game makes you
                // beat a knight for the card, and paid a jittered 99 for the other.
                if (action == 0)
                {
                    RunNonCombatEffects.GainGold(State, 100);
                }
                else if (action == 1)
                {
                    State.EventPage = 1;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventZenWeaver:
                // The weaver sells removals. What was here -- heal 12 / upgrade a card /
                // +5 max HP -- matched none of its three options; the model is
                // BreathingTechniques, EmotionalAwareness and ArachnidAcupuncture.
                if (action == 0)
                {
                    // BreathingTechniquesCost 50, then two Enlightenments into the deck.
                    State.Gold -= RunConstants.ZenWeaverBreathingCost;
                    for (int i = 0; i < 2; i++)
                    {
                        RunNonCombatEffects.AddCardToDeck(
                            State,
                            new CardInstance(RunConstants.CardEnlightenment, Upgraded: false)
                        );
                    }
                }
                else if (action is 1 or 2)
                {
                    // EmotionalAwareness removes one for 125, ArachnidAcupuncture two for
                    // 250, and both are CreateLockedOption when the gold is not there --
                    // a locked option is shown but does nothing, so refuse the action.
                    int cost =
                        action == 1
                            ? RunConstants.ZenWeaverEmotionalCost
                            : RunConstants.ZenWeaverAcupunctureCost;
                    if (State.Gold < cost)
                    {
                        return -1;
                    }

                    // RemoveCardsAndProceed loses the gold AFTER the selector, but nothing
                    // between the two can change it, and charging here keeps the cost from
                    // outliving a selection that could not open.
                    if (
                        !RunNonCombatEffects.BeginDeckSelection(
                            State,
                            DeckSelection.Remove,
                            0,
                            count: action == 1 ? 1 : 2,
                            eventEntry: "ZEN_WEAVER"
                        )
                    )
                    {
                        return -1;
                    }

                    State.Gold -= cost;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventBattlewornDummy:
                // Three settings, and each one is a FIGHT: a Battle Friend of 75, 150 or
                // 300 HP that never attacks and escapes after three of its own turns.
                // Beat the clock and the event pays -- a potion, two upgrades, or a relic;
                // miss it and the dummy walks off with the reward.
                //
                // The emulator paid 40/60/80 gold and a card, with no fight at all, which
                // is the whole event: it is a damage check against a timer.
                if (action is >= 0 and <= 2)
                {
                    State.ResumeEventId = RunConstants.EventBattlewornDummy;
                    State.ResumeEventPage = action;
                    return StartCombatWithDeck(
                        State.Deck,
                        RunConstants.BattlewornDummyEncounterIds0 + action,
                        State.Relics,
                        State.PlayerHp,
                        State.PlayerMaxHp,
                        State.PotionSlots,
                        State.Gold
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventGraveOfTheForgotten:
                // Confront pays a Decay AND asks for a card to enchant with Souls Power;
                // Accept just takes Forgotten Soul. The placeholder had 13 damage and a
                // pool relic on one side and a Decay plus 150 gold on the other -- the
                // curse was the only piece of it the event actually has, and it was on
                // the wrong option.
                if (action == 0)
                {
                    // CONFRONT_LOCKED is an EventOption with a NULL action, so a deck
                    // with nothing Souls Power can take does not eat the curse and get
                    // nothing -- the option does not run at all.
                    if (
                        !RunNonCombatEffects.CanBeginDeckSelection(
                            State,
                            DeckSelection.Enchant,
                            (int)Enchantment.SoulsPower
                        )
                    )
                    {
                        return -1;
                    }

                    // The curse lands FIRST: `AddCurseToDeck` is awaited before the
                    // selection screen opens, so it is paid up front and the fresh Decay
                    // is itself in the deck the screen then offers -- not that it can
                    // take the enchantment.
                    RunNonCombatEffects.AddCardToDeck(
                        State,
                        new CardInstance(RunNonCombatEffects.NamedCard("Decay"), false)
                    );
                    RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.SoulsPower
                    );
                    return 0;
                }

                if (action == 1)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("ForgottenSoul")
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventHungryForMushrooms:
                // Two RelicOptions: BIG MUSHROOM outright, or FRAGRANT MUSHROOM with
                // `.ThatDoesDamage(15m)` on the option. The emulator gave max HP and a
                // potion. Granting the right relic puts the run's relic list where the game
                // puts it, which is what an event owes; both relics are modelled, and Big
                // Mushroom's opening-hand penalty lives in `RelicEffects`.
                // The HP swing belongs to the RELICS, not to the event: both are
                // `HasUponPickupEffect`, so it lands through `ApplyRelicPickup` and would
                // land the same way if the relic arrived by any other route. The option's
                // `.ThatDoesDamage(15m)` is the label, not the effect.
                if (action == 0)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("BigMushroom")
                    );
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("FragrantMushroom")
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventReflections:
                // Neither option asks the player anything -- Touch a Mirror rolls its
                // cards off the event's own Rng and Shatter takes the whole deck. What
                // was here (transform the first card / upgrade the first card) was a
                // placeholder standing in for both.
                if (action == 0)
                {
                    RunNonCombatEffects.ReflectionsTouchAMirror(State);
                }
                else if (action == 1)
                {
                    RunNonCombatEffects.ReflectionsShatter(State);
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            // Picking the fight is a two-page option: the first page only warns, and
            // CONTINUE_FIGHT on the second is what charges the 11 and pays the relic.
            // `.ThatWontSaveToChoiceHistory()` marks it as the continuation of a choice
            // already made rather than a choice of its own.
            case RunConstants.EventRoundTeaParty when State.EventPage == 1:
                if (action == 0)
                {
                    // Unblockable and Unpowered, so no block and no Vulnerable apply --
                    // and at 11 it can end a run that walked in under 12, which is what
                    // the event's `CurrentHp >= 12` IsAllowed is there to prevent.
                    RunNonCombatEffects.LoseHp(State, 11);
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            case RunConstants.EventRoundTeaParty:
                // Royal Poison and a FULL heal, or 11 unblockable for a pool relic. The
                // 18-heal and 80 gold here were a placeholder: neither number, neither
                // relic, and no second page.
                if (action == 0)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunNonCombatEffects.NamedRelic("RoyalPoison")
                    );
                    // `Heal(MaxHp - CurrentHp)` -- to full, and AFTER the relic, so a
                    // relic that raises max HP is healed up to its new cap.
                    HealPlayer(State.PlayerMaxHp - State.PlayerHp);
                }
                else if (action == 1)
                {
                    State.EventPage = 1;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            // The Trial. Accept draws a DEFENDANT off the event's own stream -- merchant,
            // noble or nondescript -- and each is a page of Guilty / Innocent with its own
            // pair of outcomes. Reject answers with a page offering Accept again or Double
            // Down, and Double Down abandons the run outright.
            //
            // The emulator had two options: 10 HP for an upgraded random card, or 100 gold.
            // Neither is any of the eight things this event can do.
            //
            // EventPage carries the defendant: 1 merchant, 2 noble, 3 nondescript, and 4
            // the Reject page. EventValue0 is not used -- the page IS the state.
            case RunConstants.EventTrial when State.EventPage is 1 or 2 or 3:
                return StepTrialVerdict(action, out terminal);

            case RunConstants.EventTrial when State.EventPage == 4:
                // Reject's page: Accept after all, or Double Down and end the run.
                if (action == 0)
                {
                    State.EventPage = TrialDefendantPage();
                    return 0;
                }

                if (action == 1)
                {
                    // `.ThatWillKillPlayerIf(_ => true)`, and the option opens
                    // `NAbandonRunConfirmPopup`: Double Down is the run. Taking the HP to
                    // zero rather than setting `terminal` outright lets the shared
                    // death guard below run it -- which is what marks the run LOST rather
                    // than merely over.
                    State.PlayerHp = 0;
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;

            case RunConstants.EventTrial:
                if (action == 0)
                {
                    State.EventPage = TrialDefendantPage();
                    return 0;
                }
                else if (action == 1)
                {
                    State.EventPage = 4;
                    return 0;
                }
                else if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            // Tinker Time builds a card over three pages: pick a type, pick a rider, take
            // the Mad Science it made. What was here -- upgrade a card / transform a card
            // / gain a potion -- was three options the event does not have, on a page it
            // does not have (the model declares exactly ONE initial option).
            case RunConstants.EventTinkerTime when State.EventPage == 2:
                // The rider page. Both offered riders are legal; which is which came out
                // of the shuffle, so the action indexes the offer.
                if ((uint)action >= (uint)State.EventRandomOffer.Length)
                {
                    return -1;
                }

                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(
                        RunConstants.CardMadScience,
                        Upgraded: false,
                        TinkerType: State.TinkerCardType,
                        TinkerRider: (TinkerRider)State.EventRandomOffer[action]
                    )
                );
                break;

            case RunConstants.EventTinkerTime when State.EventPage == 1:
                // The card-type page, two of Attack/Skill/Power. Choosing one settles the
                // type and rolls the riders that go with it.
                if ((uint)action >= (uint)State.EventRandomOffer.Length)
                {
                    return -1;
                }

                State.TinkerCardType = (CardType)State.EventRandomOffer[action];
                RunNonCombatEffects.BeginTinkerRiderPage(State);
                return 0;

            case RunConstants.EventTinkerTime:
                // One option, and it only turns the page.
                if (action == 0)
                {
                    RunNonCombatEffects.BeginTinkerCardTypePage(State);
                    return 0;
                }

                if (action != RunConstants.EventSkipAction)
                {
                    return -1;
                }

                break;
            default:
                // Every event the game can deal has a case above. This used to be a
                // generic stand-in -- gold and a potion, a heal, or a card off Rng.UpFront
                // -- for an invented "simple reward" event only the old fallback pool
                // could open; the fallback follows RoomSet.EnsureNextEventIsValid now.
                return -1;
        }

        State.EventId = RunConstants.EventResultPending;
        return 0;
    }

    /// <summary>
    /// Take one card off an offered grid. The card joins the deck and leaves the grid;
    /// the screen stays up while more picks are owed and there is still something to
    /// pick, which is where the game's own selector stops too.
    /// </summary>
    private int StepOfferSelect(int action)
    {
        if (State.PendingOfferOptional && action == State.PendingOfferCards.Length)
        {
            // The confirm. `CardSelectCmd.FromSimpleGridForRewards` hands back whatever
            // was chosen, which may be nothing at all.
            return CloseOfferGrid();
        }

        if ((uint)action >= (uint)State.PendingOfferCards.Length)
        {
            return -1;
        }

        bool upgraded =
            action < State.PendingOfferUpgraded.Length && State.PendingOfferUpgraded[action];
        RunNonCombatEffects.AddCardToDeck(
            State,
            State.PendingOfferHooked
                ? RunRewardGenerator.TakeFromRewardSet(
                    State,
                    State.PendingOfferCards[action],
                    upgraded,
                    State.PendingOfferSwiftIndexes.Contains(action)
                )
                : new CardInstance(State.PendingOfferCards[action], upgraded)
        );
        State.PendingOfferSwiftIndexes =
        [
            .. State
                .PendingOfferSwiftIndexes.Where(index => index != action)
                .Select(index => index > action ? index - 1 : index),
        ];

        State.PendingOfferCards = [.. State.PendingOfferCards.Where((_, index) => index != action)];
        State.PendingOfferUpgraded =
        [
            .. State.PendingOfferUpgraded.Where((_, index) => index != action),
        ];
        State.PendingOfferPicks--;

        if (State.PendingOfferPicks > 0 && State.PendingOfferCards.Length > 0)
        {
            return 0;
        }

        return CloseOfferGrid();
    }

    /// <summary>
    /// Where a screen opened from Neow (or from a reward claimed on Neow's behalf) goes back
    /// to. Neow's Bones offers two relics on ONE rewards screen, and a relic claimed there
    /// can open a grid of its own (Lead Paperweight): the grid closes onto the rewards screen
    /// while the other relic is still owed, and only the claim that empties it returns to
    /// Neow -- as the game's `RewardsSet` does. Landing on Neow early let the next action
    /// choose a SECOND blessing and left the curse unadded.
    /// </summary>
    private RunPhase NeowReturnPhase() =>
        RunRewardGenerator.HasPendingRewards(State) ? RunPhase.RelicReward : RunPhase.Ancient;

    /// <summary>Shuts the offer grid and runs whatever was waiting behind it.</summary>
    private int CloseOfferGrid()
    {
        State.PendingOfferCards = [];
        State.PendingOfferUpgraded = [];
        State.PendingOfferSwiftIndexes = [];
        State.PendingOfferHooked = false;
        State.PendingOfferPicks = 0;
        State.PendingOfferOptional = false;

        // Hefty Tablet's Injury comes WITH the card the grid hands over, not before it:
        // CardPileCmd.Add takes ONE list holding the curse with the chosen card inserted
        // at its front, so a live capture shows both land in the same snapshot.
        if (State.PendingHeftyTabletCurse)
        {
            State.PendingHeftyTabletCurse = false;
            RunNonCombatEffects.AddCardToDeck(
                State,
                new CardInstance(RunNonCombatEffects.NamedCard("Injury"), Upgraded: false)
            );
        }

        // A grid Neow opened returns to Neow, which stays up for one more Proceed. One an
        // EVENT opened returns to the event's finished page.
        if (State.NeowAwaitingProceed)
        {
            State.Phase = NeowReturnPhase();
            return 0;
        }

        State.EventId = RunConstants.EventResultPending;
        State.Phase = RunPhase.Event;
        return 0;
    }

    /// <summary>
    /// One divination. The action carries both the cell and the tool -- 0..120 is the big
    /// <summary>
    /// The Fake Merchant's stall: six relics at a flat 50 apiece, a Foul Potion to throw,
    /// and the door.
    /// </summary>
    /// <remarks>
    /// Throwing the potion is the interesting move. `FoulPotionThrown` gives the player
    /// Fake Merchant's Rug AND every relic still on the shelf as combat rewards, so the
    /// stall is worth more robbed than bought -- buying one at 50 is buying a relic out
    /// of your own loot pile. The fight is entered with `shouldResumeAfterCombat: false`,
    /// so the event does not come back afterwards.
    /// </remarks>
    private int StepFakeMerchant(int action, out bool terminal)
    {
        terminal = false;
        var stall = RunNonCombatEffects.FakeMerchantStock(State);

        if (action == RunConstants.FakeMerchantLeaveAction)
        {
            return AdvanceAfterNode(out terminal);
        }

        if (action == RunConstants.FakeMerchantThrowAction)
        {
            int foul = RunNonCombatEffects.NamedPotion("FoulPotion");
            int slot = Array.IndexOf(State.PotionSlots, foul);
            if (slot < 0)
            {
                return -1;
            }

            State.PotionSlots[slot] = 0;
            // The rug first, then the shelf in slot order -- the order the rewards list
            // is built in, which is the order the screen offers them.
            State.PendingBonusRelicRewards.Add(RunNonCombatEffects.NamedRelic("FakeMerchantsRug"));
            State.PendingBonusRelicRewards.AddRange(stall.Where(relicId => relicId != 0));
            return StartCombatWithDeck(
                State.Deck,
                RunConstants.FakeMerchantEncounterId,
                State.Relics,
                State.PlayerHp,
                State.PlayerMaxHp,
                State.PotionSlots,
                State.Gold
            );
        }

        if (action < 0 || action >= stall.Count)
        {
            return -1;
        }

        int price = RunNonCombatEffects.FakeMerchantPrice(State, action);
        if (stall[action] == 0 || State.Gold < price)
        {
            return -1;
        }

        State.Gold -= price;
        // Sold out rather than removed: the slot indices have to hold still for the rest
        // of the visit, and an emptied slot is also one fewer relic the fight would pay.
        int bought = stall[action];
        stall[action] = 0;
        RunNonCombatEffects.ApplyRelicPickup(State, bought);
        return 0;
    }

    /// tool, 121..241 the small one -- and the last one spent settles up.
    /// </summary>
    private int StepCrystalSphere(int action, out bool terminal)
    {
        terminal = false;
        var game = State.CrystalSphere;
        if (game is null || game.IsFinished)
        {
            return -1;
        }

        if (action < 0 || action >= 2 * RunConstants.CrystalSphereCells)
        {
            return -1;
        }

        var tool =
            action >= RunConstants.CrystalSphereSmallToolAction
                ? CrystalSphereTool.Small
                : CrystalSphereTool.Big;
        int cell = action % RunConstants.CrystalSphereSmallToolAction;
        int x = cell / RunConstants.CrystalSphereSize;
        int y = cell % RunConstants.CrystalSphereSize;
        if (!game.WouldUncover(x, y, tool))
        {
            return -1;
        }

        game.Tool = tool;
        foreach (int index in game.Click(x, y))
        {
            RunNonCombatEffects.RevealCrystalSphereItem(State, game.Items[index]);
        }

        if (!game.IsFinished)
        {
            return 0;
        }

        return CompleteCrystalSphere(out terminal);
    }

    /// <summary>
    /// Settles the sphere once the divinations run out: every fully-uncovered thing turns
    /// into a reward, and the run goes to the reward screen -- or straight on, if the fog
    /// kept everything.
    ///
    /// The draw order is the game's, and it is two passes rather than one. Every reward is
    /// built first (<c>ToReward</c>, which is where a potion picks itself), then every
    /// reward is populated (<c>RewardsSet.GenerateWithoutOffering</c>, which is where the
    /// relic is pulled and the card offers rolled). Both passes walk the items in the order
    /// they were uncovered.
    /// </summary>
    private int CompleteCrystalSphere(out bool terminal)
    {
        var game = State.CrystalSphere!;
        var rng = State.CrystalSphereRng!;
        var revealed = game
            .Revealed.Select(index => game.Items[index])
            // The curse has no reward: it was taken at the moment it was uncovered.
            .Where(item => item.Kind != CrystalSphereItemKind.Curse)
            .ToList();

        // Pass one, ToReward. Only a potion draws here -- it picks itself as the reward is
        // built, where everything else waits to be populated.
        var potions = new List<int>();
        foreach (var item in revealed)
        {
            if (item.Kind == CrystalSphereItemKind.Potion)
            {
                potions.Add(
                    RunRewardGenerator.NextPotionOfRarity(
                        State.Character,
                        rng,
                        (PotionRarity)item.Rarity
                    )
                );
            }
        }

        // Pass two, Populate, in the same order. A gold pile draws even though its amount
        // is fixed -- GoldReward.Populate is Rng.NextInt(amount, amount + 1) whether or not
        // there is a range -- and that draw sits between the piles and the card offers.
        var golds = new List<int>();
        int relic = 0;
        var offers = new List<CardOffer>();
        foreach (var item in revealed)
        {
            switch (item.Kind)
            {
                case CrystalSphereItemKind.Gold:
                    int amount = item.IsBigGold ? 30 : 10;
                    golds.Add(rng.NextInt(amount, amount + 1));
                    break;
                case CrystalSphereItemKind.Relic:
                    relic = RunRewardGenerator.NextRelic(State, rng);
                    break;
                case CrystalSphereItemKind.CardReward:
                    // `CrystalSphereCardReward.ToReward`: Uniform odds over the pool
                    // filtered to the rarity, the event's Rng as the override, no other flag
                    // but the CardReward's own IsCardReward (so the pool may be widened).
                    offers.Add(
                        RunRewardGenerator.CreateForReward(
                            State,
                            State.RewardCards.Length,
                            RunRewardGenerator.OwnRewardPool(State, CardCreationFlags.IsCardReward),
                            CardCreationFlags.IsCardReward,
                            (CardRarity)item.Rarity,
                            rng
                        )
                    );
                    break;
            }
        }

        State.CrystalSphere = null;
        State.CrystalSphereRng = null;
        RunRewardGenerator.ClearRewardScreen(State);
        State.PendingPotionRewards.Clear();

        if (golds.Count == 0 && potions.Count == 0 && relic == 0 && offers.Count == 0)
        {
            return AdvanceAfterNode(out terminal);
        }

        State.PendingGoldRewards.AddRange(golds);
        RunRewardGenerator.OfferNextGold(State);
        State.PendingPotionRewards.AddRange(potions);
        OfferNextRestPotion();
        State.RelicReward = relic;
        if (offers.Count > 0)
        {
            offers[0].ShowOn(State);
            State.RewardCardPending = true;
            State.PendingCardOffers.AddRange(offers.Skip(1));
        }

        State.Phase = RunPhase.RelicReward;
        terminal = false;
        return 0;
    }

    /// <summary>How many bundles Scroll Boxes puts on the screen.</summary>
    private const int BundleCount = 2;

    /// <summary>Cards per bundle: two Commons and an Uncommon.</summary>
    private const int BundleSize = 3;

    /// <summary>
    /// The choose-a-bundle screen, answered the way the game's is: highlight a bundle,
    /// then confirm it. A capture spends one action on each, and re-selecting before
    /// confirming just moves the highlight.
    /// </summary>
    private int StepBundleSelect(int action, out bool terminal)
    {
        terminal = false;
        if (
            action >= 0
            && action < BundleCount
            && State.BundleOffer.Length >= BundleSize * BundleCount
        )
        {
            State.SelectedBundle = action;
            return 0;
        }

        if (action != RunConstants.BundleConfirmAction || State.SelectedBundle < 0)
        {
            return -1;
        }

        for (int i = 0; i < BundleSize; i++)
        {
            RunNonCombatEffects.AddCardToDeck(
                State,
                new CardInstance(State.BundleOffer[State.SelectedBundle * BundleSize + i], false)
            );
        }

        State.BundleOffer = [];
        State.SelectedBundle = -1;

        // A screen Neow opened returns to Neow, which stays up for one more Proceed.
        if (State.NeowAwaitingProceed)
        {
            State.Phase = RunPhase.Ancient;
            return 0;
        }

        return AdvanceAfterNode(out terminal);
    }

    private int StepTransformSelect(int action, out bool terminal)
    {
        terminal = false;
        if (State.PendingOfferCards.Length > 0)
        {
            return StepOfferSelect(action);
        }

        if (State.PendingRestUpgrade)
        {
            if (
                (uint)action >= (uint)State.Deck.Count
                || !RunConstants.IsRunCardUpgradable(State.Deck[action])
            )
            {
                return -1;
            }

            State.Deck[action] = State.Deck[action] with { Upgraded = true };
            State.PendingRestUpgrade = false;
            State.RestResultPending = true;
            State.RestOptionsTaken = 0;
            State.Phase = RunPhase.Rest;
            return 0;
        }

        if (State.PendingSelectionKind != DeckSelection.None)
        {
            // DONE closes a selection whose minimum is zero with the picks made so far.
            bool done = State.PendingSelectionOptional && action == State.Deck.Count;
            if (!done && !RunNonCombatEffects.ApplyDeckSelection(State, action))
            {
                return -1;
            }

            // A selection for two cards stays on the screen for the second -- unless the
            // deck has run out of cards it would take, which is where the game's own
            // selector stops too.
            if (
                !done
                && State.PendingSelectionCount > 0
                && Enumerable
                    .Range(0, State.Deck.Count)
                    .Any(i => RunNonCombatEffects.CanSelectCard(State, i))
            )
            {
                return 0;
            }

            RunNonCombatEffects.ResolveDeckSelectionFollowUp(State);
            bool returnsToEvent = State.PendingSelectionReturnsToEvent;
            var returnTo = State.PendingSelectionReturn;
            RunNonCombatEffects.ClearDeckSelection(State);

            // A selection NEOW opened returns to Neow, which stays up for one more
            // Proceed. Leaving the flag set instead is worse than landing on the wrong
            // screen once: it is read again by the next card reward the run claims, which
            // sends a combat's reward screen back to the ancient floors later.
            if (State.NeowAwaitingProceed)
            {
                State.Phase = NeowReturnPhase();
                return 0;
            }

            // A selection goes back to whatever opened it. Most are events, which is why
            // that is the default — but a shop's removal service and a relic's pickup are
            // not events, and assuming they were is what kept those two choosing a card
            // for the player rather than asking.
            switch (returnTo)
            {
                case SelectionReturn.Shop:
                    State.Phase = RunPhase.Shop;
                    // A parasol sweep that stopped to ask something picks up where it
                    // left off -- the shelves are the cursor, so this finishes the rest
                    // of them and then asks for the removal it still owes.
                    RunRewardGenerator.ContinueLordsParasolSweep(State);
                    return 0;
                case SelectionReturn.Map:
                    return AdvanceAfterNode(out terminal);
                case SelectionReturn.Rest:
                    // The option was spent on the way in; whether the screen stays up is
                    // the Tent's call, which is what FinishRestOption decides.
                    FinishRestOption(RunConstants.RestCookAction);
                    State.Phase = RunPhase.Rest;
                    return 0;
                default:
                    // The belt's Jelly Liver keeps its options up, because the belt turns
                    // and offers the next dish; everything else shows its result page.
                    if (!returnsToEvent && returnTo != SelectionReturn.EventOptions)
                    {
                        State.EventId = RunConstants.EventResultPending;
                    }
                    else if (State.EventId == RunConstants.EventEndlessConveyor)
                    {
                        // The Jelly Liver's transform is done; now the belt turns.
                        RunNonCombatEffects.RollNextConveyorDish(State);
                    }

                    State.Phase = RunPhase.Event;
                    return 0;
            }
        }

        if (State.TransformSelectedDeckIndex < 0)
        {
            int count = Math.Abs(State.TransformSelectedDeckIndex.Value);
            int relicId = State.Relics.LastOrDefault().DefId;
            if (relicId == RunConstants.RelicAstrolabe)
            {
                for (int i = 0; i < count && State.Deck.Count > 0; i++)
                {
                    RunNonCombatEffects.TransformCardAt(State, 0, State.Rng.Niche);
                    State.Deck[^1] = State.Deck[^1] with { Upgraded = true };
                }
            }
            else if (relicId == RunConstants.RelicEmptyCage)
            {
                // FromDeckForRemoval, like every other removal: the player picks. The
                // relic is taken outside an event, so the selection goes back to the MAP
                // rather than to an event page that is not there.
                State.TransformSelectedDeckIndex = null;
                terminal = false;
                return RunNonCombatEffects.BeginDeckSelection(
                    State,
                    DeckSelection.Remove,
                    0,
                    count: count,
                    returnTo: SelectionReturn.Map
                )
                    ? 0
                    : AdvanceAfterNode(out terminal);
            }

            State.TransformSelectedDeckIndex = null;
            return AdvanceAfterNode(out terminal);
        }

        if (State.TransformSelectedDeckIndex is null)
        {
            if ((uint)action >= (uint)State.Deck.Count)
            {
                return -1;
            }

            State.TransformSelectedDeckIndex = action;
            return 0;
        }

        RunNonCombatEffects.TransformCardAt(
            State,
            State.TransformSelectedDeckIndex.Value,
            State.Rng.Niche
        );
        State.TransformSelectedDeckIndex = null;
        return AdvanceAfterNode(out terminal);
    }

    /// <summary>
    /// One card rolled from the CHARACTER'S OWN pool at non-combat odds, filtered, and put
    /// straight into the deck -- Infested Automaton's two options, which are
    /// `CardFactory.CreateForReward(owner, 1, ForNonCombatWithDefaultOdds(pool, filter))`
    /// followed by `CardPileCmd.Add(card, PileType.Deck)`.
    /// </summary>
    /// <remarks>
    /// The filter narrows the POOL before the roll, not the result after it: rolling from
    /// the whole pool and rejecting misses would be a different number of draws on the
    /// Rewards stream and slide every later roll in the run.
    /// </remarks>
    private void AddFilteredPoolCardToDeck(Func<CardDef, bool> allowed)
    {
        var pool = new List<int>();
        foreach (int cardId in RunRewardGenerator.OwnCardPool(State))
        {
            if (allowed(GeneratedData.Cards.Get(cardId)))
            {
                pool.Add(cardId);
            }
        }

        if (pool.Count == 0)
        {
            return;
        }

        RunRewardGenerator.AddRewardCardToDeck(
            State,
            RunRewardGenerator.CreateForReward(
                State,
                1,
                pool.ToArray(),
                CardCreationFlags.NoUpgradeRoll
            ),
            0
        );
    }

    /// <summary>
    /// Which defendant Accept draws: `base.Rng.NextInt(3)` off the event's own stream,
    /// mapped to pages 1 (merchant), 2 (noble) and 3 (nondescript).
    /// </summary>
    private int TrialDefendantPage() =>
        1 + RunNonCombatEffects.EventStream(State, "TRIAL").NextInt(3);

    /// <summary>
    /// A verdict on whichever defendant Accept drew. Six outcomes, and no two of them are
    /// the same shape: two hand out relics or gold outright, two raise a card selection,
    /// and one opens a pair of card-reward offers. Every one of them starts with a CURSE
    /// except the noble's Guilty.
    /// </summary>
    private int StepTrialVerdict(int action, out bool terminal)
    {
        terminal = false;
        if (action is not (0 or 1))
        {
            return action == RunConstants.EventSkipAction ? AdvanceAfterNode(out terminal) : -1;
        }

        bool guilty = action == 0;
        switch (State.EventPage)
        {
            case 1 when guilty:
                // Merchant, Guilty: a Regret, and TWO relics off the front of the pool.
                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(RunNonCombatEffects.NamedCard("Regret"), false)
                );
                for (int i = 0; i < 2; i++)
                {
                    RunNonCombatEffects.ApplyRelicPickup(
                        State,
                        RunRewardGenerator.NextRelic(State)
                    );
                }

                break;

            case 1:
                // Merchant, Innocent: a Shame, then TWO cards the player upgrades.
                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(RunNonCombatEffects.NamedCard("Shame"), false)
                );
                if (
                    RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.Upgrade,
                        0,
                        count: 2
                    )
                )
                {
                    State.Phase = RunPhase.TransformSelect;
                    return 0;
                }

                break;

            case 2 when guilty:
                // Noble, Guilty: heal 10, and it is the only verdict with no curse.
                HealPlayer(10);
                break;

            case 2:
                // Noble, Innocent: a Regret and 300 gold, flat.
                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(RunNonCombatEffects.NamedCard("Regret"), false)
                );
                RunNonCombatEffects.GainGold(State, 300);
                break;

            case 3 when guilty:
            {
                // Nondescript, Guilty: a Doubt, then TWO card-reward offers of three from
                // the character's own pool. The emulator's reward screen carries one offer,
                // so the second waits behind it the way a second relic does.
                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(RunNonCombatEffects.NamedCard("Doubt"), false)
                );
                // Both offers are rolled HERE, in order, because that is when the game
                // rolls them: `RewardsCmd.OfferCustom` populates every reward as the screen
                // opens and the draws all come off the same stream.
                // `new CardReward(ForNonCombatWithDefaultOdds(pool), 3, owner)`: NoUpgradeRoll
                // plus the reward's own IsCardReward, so Silver Crucible, the eggs and Wing
                // Charm act on each screen and Prismatic Gem / Dingy Rug widen its pool.
                const CardCreationFlags verdictFlags =
                    CardCreationFlags.NoUpgradeRoll | CardCreationFlags.IsCardReward;
                var first = RunRewardGenerator.CreateForReward(
                    State,
                    State.RewardCards.Length,
                    RunRewardGenerator.OwnRewardPool(State, verdictFlags),
                    verdictFlags
                );
                var second = RunRewardGenerator.CreateForReward(
                    State,
                    State.RewardCards.Length,
                    RunRewardGenerator.OwnRewardPool(State, verdictFlags),
                    verdictFlags
                );
                // One rewards screen holding both, as Colorful Philosophers' three
                // (OfferColorfulPhilosopherRewards): straight into the first card screen with no
                // way back lost the second as soon as the first was answered.
                State.EventAwaitingProceed = State.Phase == RunPhase.Event;
                State.RewardGold = 0;
                State.RewardPotion = 0;
                State.RelicReward = 0;
                first.ShowOn(State);
                State.RewardCardPending = true;
                State.PendingCardOffers.Add(second);
                State.Phase = RunPhase.RelicReward;
                return 0;
            }

            default:
                // Nondescript, Innocent: a Doubt, then TWO cards the player picks are
                // transformed at RANDOM -- `TransformToRandom` off the event's own stream,
                // not the Transformations one.
                RunNonCombatEffects.AddCardToDeck(
                    State,
                    new CardInstance(RunNonCombatEffects.NamedCard("Doubt"), false)
                );
                if (
                    RunNonCombatEffects.BeginDeckSelection(
                        State,
                        DeckSelection.TransformToRandom,
                        0,
                        count: 2,
                        eventEntry: "TRIAL"
                    )
                )
                {
                    State.Phase = RunPhase.TransformSelect;
                    return 0;
                }

                break;
        }

        return AdvanceAfterNode(out terminal);
    }

    /// <summary>
    /// The event picks back up where the fight interrupted it -- `EventModel.Resume(room)`.
    /// </summary>
    /// <remarks>
    /// Battleworn Dummy is the only event that does this so far, and its Resume is a
    /// switch on which setting was fought, skipped entirely if the dummy ran out of time.
    /// The rewards are not the fight's: `ShouldGiveRewards => false` on the encounter, so
    /// nothing goes through `GenerateCombatRewards`.
    /// </remarks>
    private int ResumeEventAfterCombat(out bool terminal)
    {
        int eventId = State.ResumeEventId;
        int setting = State.ResumeEventPage;
        bool escaped = State.ResumeEventDummyEscaped;
        State.ResumeEventId = 0;
        State.ResumeEventPage = 0;
        State.ResumeEventDummyEscaped = false;

        if (eventId != RunConstants.EventBattlewornDummy || escaped)
        {
            // Ran out of time: the dummy walks off and the event finishes with nothing.
            return AdvanceAfterNode(out terminal);
        }

        switch (setting)
        {
            case 0:
                // Setting 1: a POTION, rolled off the Rewards stream from the character's
                // own pool concatenated with the shared one -- and OFFERED, not procured:
                // `RewardsCmd.OfferCustom` of a PotionReward, so a full belt makes the player
                // drop something or decline it.
                OfferPotionRewards(RunRewardGenerator.NextPoolPotion(State, State.PlayerRng.Rewards));
                terminal = false;
                return 0;

            case 1:
                // Setting 2: two upgradable deck cards upgraded, off the EVENT's stream.
                RunNonCombatEffects.UpgradeRandomDeckCardsForEvent(
                    State,
                    2,
                    RunNonCombatEffects.EventStream(State, "BATTLEWORN_DUMMY")
                );
                break;

            default:
                // Setting 3: a relic off the front of the pool.
                RunNonCombatEffects.ApplyRelicPickup(State, RunRewardGenerator.NextRelic(State));
                break;
        }

        return AdvanceAfterNode(out terminal);
    }

    private int RestHealAmount() =>
        Math.Max(1, (int)(State.PlayerMaxHp * 0.3))
        // Regal Pillow's ModifyRestSiteHealAmount adds its HealVar(15m) to whatever the
        // site was going to heal.
        + (
            Effects.RelicEffects.Has(State.Relics, Effects.RelicEffects.RegalPillow)
                ? Effects.RelicEffects.RegalPillowRestHeal
                : 0
        );

    private void HealPlayer(int amount)
    {
        State.PlayerHp = Math.Min(State.PlayerMaxHp, State.PlayerHp + amount);
    }

    /// <summary>The surviving pools, in option order. Rolled on entry, never here.</summary>
    private List<int[]> ColorfulPhilosopherPools()
    {
        var order = RunNonCombatEffects.ColorfulPhilosopherPoolOrder(State);
        return RunNonCombatEffects
            .ColorfulPhilosopherPools(State)
            .Select(index => order[index])
            .ToList();
    }

    /// <summary>
    /// `OfferRewards`: three `CardReward`s over the chosen pool at `CardsVar(3)` cards
    /// each, one per rarity, offered together by `RewardsCmd.OfferCustom`.
    /// </summary>
    /// <remarks>
    /// `CardRarityOddsType.Uniform` with a rarity filter means `CreateForReward` skips
    /// `RollForRarity` entirely, so each card is two draws -- itself, then its upgrade
    /// roll -- rather than three. All nine come off the player's Rewards stream in
    /// Common, Uncommon, Rare order, which is the order the rewards are constructed in.
    /// </remarks>
    private void OfferColorfulPhilosopherRewards(int[] pool)
    {
        Array.Clear(State.RewardCards);
        Array.Clear(State.RewardUpgraded);
        var offers = new[] { CardRarity.Common, CardRarity.Uncommon, CardRarity.Rare }
            .Select(rarity =>
                RunRewardGenerator.CreateForReward(
                    State,
                    RunConstants.ColorfulPhilosophersCards,
                    pool,
                    CardCreationFlags.NoRarityModification
                        | CardCreationFlags.NoCardPoolModifications
                        | CardCreationFlags.IsCardReward,
                    rarity
                )
            )
            .ToList();
        // `RewardsCmd.OfferCustom` with the three CardRewards: ONE rewards screen holding all
        // three, each claimed (or left) from it, and the event's finished page after it. Going
        // straight into the first card screen with no way back dropped the other two the moment
        // the first was answered (run referee, seed 6HMCK2BXTJ: the game took two of three).
        State.EventAwaitingProceed = State.Phase == RunPhase.Event;
        State.RewardGold = 0;
        State.RewardPotion = 0;
        State.RelicReward = 0;
        offers[0].ShowOn(State);
        State.RewardCardPending = true;
        State.PendingCardOffers.AddRange(offers.Skip(1));
        State.Phase = RunPhase.RelicReward;
    }

    /// <summary>
    /// `ColossalFlower.DealReachDeeperDamage`: `_prizeDamage[NumberOfDigs]`, Unblockable
    /// and Unpowered, read BEFORE the dig count moves.
    /// </summary>
    private void ColossalFlowerDig()
    {
        RunNonCombatEffects.LoseHp(State, RunConstants.ColossalFlowerDigDamage[State.EventPage]);
    }

    private void TransformTwoMatchingCards(int cardId)
    {
        for (int count = 0; count < 2; count++)
        {
            int index = State.Deck.FindIndex(card => Math.Abs(card.DefId) == cardId);
            if (index < 0)
            {
                return;
            }

            RunNonCombatEffects.TransformCardAt(State, index, State.PlayerRng.Transformations);
        }
    }

    private static void SetMask(Span<int> mask, int action)
    {
        if ((uint)action < (uint)mask.Length)
        {
            mask[action] = 1;
        }
    }

    private void WriteEventActionMask(Span<int> mask)
    {
        SetMask(mask, RunConstants.EventSkipAction);
        switch (State.EventId)
        {
            case RunConstants.EventUnrestSite:
                // Neither option is ever locked. Resting at full health still costs a
                // Poor Sleep, and Kill's LoseMaxHp floors the cap at 1 rather than
                // refusing, so gating these hid two legal moves.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventAromaOfChaos:
                if (State.Deck.Count > 0)
                {
                    SetMask(mask, 0);
                }

                if (State.Deck.Any(RunConstants.IsRunCardUpgradable))
                {
                    SetMask(mask, 1);
                }

                break;
            case RunConstants.EventJungleMazeAdventure:
                // The solo quest is offered at any health; its 18 damage can end the run.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventMorphicGrove:
                // Group is LoseGold(Owner.Gold) -- it takes whatever the run holds, with
                // no minimum -- so the only thing that can stop it is having nothing to
                // transform. The gold >= 100 rule is IsAllowed, which decides whether the
                // EVENT appears, not whether the option is takeable once it has.
                if (
                    RunNonCombatEffects.CanBeginDeckSelection(
                        State,
                        DeckSelection.TransformToRandom,
                        0
                    )
                )
                {
                    SetMask(mask, 0);
                }

                SetMask(mask, 1);
                break;
            case RunConstants.EventTinkerTime:
                // Tinker Time builds its options into a List, so EventOptions.g.cs has no
                // count for it and the fallback offered a flat 0..3 on every page. Page 0
                // declares exactly one option; the two after it show two of three, and
                // which two came out of the shuffle -- so the mask follows the offer.
                if (State.EventPage == 0)
                {
                    SetMask(mask, 0);
                    break;
                }

                for (int i = 0; i < State.EventRandomOffer.Length; i++)
                {
                    SetMask(mask, i);
                }

                break;
            case RunConstants.EventZenWeaver:
                // Breathing Techniques is never locked; the two removals are
                // CreateLockedOption below their price, which shows the row greyed out
                // and does nothing when it is clicked.
                SetMask(mask, 0);
                if (State.Gold >= RunConstants.ZenWeaverEmotionalCost)
                {
                    SetMask(mask, 1);
                }

                if (State.Gold >= RunConstants.ZenWeaverAcupunctureCost)
                {
                    SetMask(mask, 2);
                }

                break;
            case RunConstants.EventSelfHelpBook:
                SetSelfHelpBookMask(mask, 0);
                SetSelfHelpBookMask(mask, 1);
                SetSelfHelpBookMask(mask, 2);
                break;
            case RunConstants.EventBrainLeech:
                // Rip is offered at any health -- the game lets it kill you, and the run
                // ends when it does.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventTheLegendsWereTrue:
                // Both options are always there: Slowly Find an Exit is only flagged as lethal
                // (`ThatDoesDamage`), never locked, and its potion is OFFERED, so a full belt
                // does not close it (run referee, seed 0NT10MDKY3).
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventDoorsOfLightAndDark:
                // Both doors are always open. Light takes up to two upgradable cards and
                // is content with none; Dark opens a removal the player may leave empty.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventSunkenTreasury:
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventSymbiote:
                // Approach's locked variant is `!pile.Cards.Any(CanEnchant)`, and
                // `CanEnchant` is the whole enchantment rule, not just the card type: an
                // Attack that already carries an enchantment cannot take Corrupted
                // either, and neither can an Unplayable one. Testing only
                // `Type == Attack` offered the option to a deck of enchanted Strikes and
                // then refused the step.
                if (
                    RunNonCombatEffects.CanBeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Corrupted
                    )
                )
                {
                    SetMask(mask, 0);
                }

                // Kill with Fire has no locked variant at all -- the game builds it
                // unconditionally -- so it stays on the table even for a deck nothing can
                // be transformed out of.
                SetMask(mask, 1);
                break;
            case RunConstants.EventColorfulPhilosophers:
                // The option count is run state -- one per unlocked pool the player is
                // not, capped at three -- so it is not in the extracted count table. No
                // option is ever locked; the event only ever shows what it can offer.
                for (int i = 0; i < ColorfulPhilosopherPools().Count; i++)
                {
                    SetMask(mask, i);
                }

                break;
            case RunConstants.EventGraveOfTheForgotten:
                // CONFRONT_LOCKED when no card in the deck can take Souls Power, which
                // means no card in the deck exhausts. Accept is never locked.
                if (
                    RunNonCombatEffects.CanBeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.SoulsPower
                    )
                )
                {
                    SetMask(mask, 0);
                }

                SetMask(mask, 1);
                break;
            case RunConstants.EventRoundTeaParty when State.EventPage == 1:
                // The warning page carries a single CONTINUE_FIGHT option. Falling
                // through to the count table offered two, and the second was a step that
                // could only refuse.
                SetMask(mask, 0);
                break;
            case RunConstants.EventTeaMaster:
                if (State.Gold >= RunConstants.BoneTeaCost)
                {
                    SetMask(mask, 0);
                }

                if (State.Gold >= RunConstants.EmberTeaCost)
                {
                    SetMask(mask, 1);
                }

                // Tea of Discourtesy is free.
                SetMask(mask, 2);
                break;
            case RunConstants.EventRelicTrader:
                SetTradeMask(
                    mask,
                    State.Relics.Count(relic => RunNonCombatEffects.IsTradableRelic(State, relic))
                );
                break;
            case RunConstants.EventTheFutureOfPotions:
                SetTradeMask(mask, State.PotionSlots.Count(potion => potion != 0));
                break;
            case RunConstants.EventLuminousChoir:
                SetMask(mask, 0);
                if (State.Gold >= RunNonCombatEffects.LuminousChoirTributeCost(State))
                {
                    SetMask(mask, 1);
                }

                break;
            case RunConstants.EventDollRoom when State.EventPage > 0:
                // The second page offers the dolls the wait bought, and nothing else.
                for (int i = 0; i < (State.EventPage == 1 ? 2 : 3); i++)
                {
                    SetMask(mask, i);
                }

                break;
            case RunConstants.EventWarHistorianRepy when State.EventPage == 1:
                // Repy's second page offers only the door the player has not opened.
                SetMask(mask, 1);
                break;
            case RunConstants.EventAbyssalBaths when State.EventPage > 0:
                // Linger, or climb out.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventEndlessConveyor:
                // Grabbing is locked below its price; the second option is Observe the
                // Chef on the first page and Leave on every later one, and neither costs
                // anything.
                if (State.Gold >= RunConstants.ConveyorGrabCost)
                {
                    SetMask(mask, 0);
                }

                SetMask(mask, 1);
                break;
            case RunConstants.EventTabletOfTruth when State.EventPage > 0:
                // A later page offers Decipher again, or Give Up.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventDenseVegetation when State.EventPage == 1:
                // The Rest's page offers only the fight.
                SetMask(mask, 0);
                break;
            case RunConstants.EventPunchOff when State.EventPage == 1:
                // "I Can Take Them" answers with a page whose only option is the fight.
                SetMask(mask, 0);
                break;
            case RunConstants.EventTheLanternKey when State.EventPage == 1:
                // Keep The Key's page offers only the fight, the same shape as those two.
                SetMask(mask, 0);
                break;
            case RunConstants.EventTrial when State.EventPage != 0:
                // Every page past the first offers exactly two: Guilty / Innocent on a
                // defendant's page, Accept / Double Down on Reject's.
                SetMask(mask, 0);
                SetMask(mask, 1);
                break;
            case RunConstants.EventStoneOfAllTime:
                // Lift needs a potion to drink; Push needs a card Vigorous can enchant,
                // which Vigorous.CanEnchantCardType limits to Attacks. The emulator does
                // not model Vigorous itself, so "already enchanted" is not checked --
                // pinned approximation, and it only ever over-offers a card the game
                // would have skipped past.
                if (State.PotionSlots.Any(potion => potion != 0))
                {
                    SetMask(mask, 0);
                }

                if (
                    State.Deck.Any(card =>
                        GeneratedData.Cards.Get(card.DefId).Type == CardType.Attack
                    )
                )
                {
                    SetMask(mask, 1);
                }

                break;
            case RunConstants.EventRanwidTheElder:
                // Give a potion, give gold, give a relic. Gold is always on the table;
                // the other two need something to give, and the relic has to be one the
                // game will take -- Burning Blood is a Starter relic, so a fresh run can
                // only ever pick the gold.
                if (State.PotionSlots.Any(potion => potion != 0))
                {
                    SetMask(mask, 0);
                }

                SetMask(mask, 1);
                if (State.Relics.Any(relic => RunNonCombatEffects.IsTradableRelic(State, relic)))
                {
                    SetMask(mask, 2);
                }

                break;
            case RunConstants.EventWelcomeToWongos:
                // A shop with three price tags and a door. The costs are the event's own
                // DynamicVars, not the shop's pricing.
                if (State.Gold >= RunConstants.WongosBargainBinCost)
                {
                    SetMask(mask, 0);
                }

                if (State.Gold >= RunConstants.WongosFeaturedItemCost)
                {
                    SetMask(mask, 1);
                }

                if (State.Gold >= RunConstants.WongosMysteryBoxCost)
                {
                    SetMask(mask, 2);
                }

                // Leave, which is action 3 and therefore already set above.
                SetMask(mask, 3);
                break;
            case RunConstants.EventResultPending:
                SetMask(mask, 0);
                break;
            case RunConstants.EventTheArchitect:
                SetMask(mask, 0);
                break;
            case RunConstants.EventCrystalSphere:
                // Paying for three divinations needs the rolled price; taking the Debt
                // instead is free.
                if (State.Gold >= RunNonCombatEffects.CrystalSphereCost(State))
                {
                    SetMask(mask, 0);
                }

                SetMask(mask, 1);
                break;
            case RunConstants.EventWhisperingHollow:
                if (State.Gold >= RunNonCombatEffects.WhisperingHollowGold(State))
                {
                    SetMask(mask, 0);
                }

                if (
                    RunNonCombatEffects.CanBeginDeckSelection(
                        State,
                        DeckSelection.TransformToRandom,
                        0
                    )
                )
                {
                    SetMask(mask, 1);
                }

                break;
            case RunConstants.EventWaterloggedScriptorium:
                // Bloody Ink is free; the quill and the sponge are locked below their own
                // prices, and both need a card Steady can take.
                SetMask(mask, 0);
                if (
                    RunNonCombatEffects.CanBeginDeckSelection(
                        State,
                        DeckSelection.Enchant,
                        (int)Enchantment.Steady
                    )
                )
                {
                    if (State.Gold >= RunConstants.ScriptoriumQuillCost)
                    {
                        SetMask(mask, 1);
                    }

                    if (State.Gold >= RunConstants.ScriptoriumSpongeCost)
                    {
                        SetMask(mask, 2);
                    }
                }

                break;
            case RunConstants.EventFakeMerchant:
                // The event page carries nothing to choose: option 0 walks up to the
                // stall, which is its own screen.
                SetMask(mask, 0);
                break;
            default:
            {
                // An event with no bespoke case above has no option gating either, so
                // every option it offers is takeable. What it must not do is offer
                // options the event does not have: the old fallback set a flat 0..3,
                // which let an agent choose a third option at the many Act 1 events
                // that only have two, and a fourth at every event that is not Welcome
                // to Wongos.
                int optionCount = GeneratedData.EventOptions.CountFor(State.EventId);
                for (
                    int i = 0;
                    i < (optionCount > 0 ? optionCount : RunConstants.EventSkipAction);
                    i++
                )
                {
                    SetMask(mask, i);
                }

                break;
            }
        }
    }

    /// <summary>
    /// The game's <c>RelicModel.IsTradable</c> for a relic actually in the run: the
    /// definition's own answer, plus the two run-state exclusions it also applies.
    /// </summary>
    /// <summary>
    /// Both trading events offer one option per thing they can take, capped at three,
    /// and fall back to a single Proceed when there is nothing to trade.
    /// </summary>
    private static void SetTradeMask(Span<int> mask, int tradableCount)
    {
        int offered = Math.Min(3, tradableCount);
        if (offered == 0)
        {
            SetMask(mask, 0);
            return;
        }

        for (int i = 0; i < offered; i++)
        {
            SetMask(mask, i);
        }
    }

    /// <summary>
    /// One page of the book is offered when the deck holds a card that page would take
    /// -- Sharp on an Attack for action 0, Nimble on a Skill for 1, Swift on a Power for
    /// 2 (`PlayerHasCardsAvailable&lt;T&gt;(owner, type)`). The page index IS the action,
    /// so mask and step read the same number.
    /// </summary>
    private void SetSelfHelpBookMask(Span<int> mask, int action)
    {
        var enchantment = RunNonCombatEffects.SelfHelpBookEnchantment(action);
        if (State.Deck.Any(card => RunNonCombatEffects.SelfHelpBookPageTakes(card, enchantment)))
        {
            SetMask(mask, action);
        }
    }

    private void WriteRewardActionMask(Span<int> mask)
    {
        int action = 0;
        if (State.RewardGold != 0)
        {
            SetMask(mask, action++);
        }

        if (State.RewardPotion != 0)
        {
            SetMask(mask, action++);
        }

        if (State.RelicReward != 0)
        {
            SetMask(mask, action++);
        }

        // The same count the state list reports (`RunRewardGenerator.CardItemCount`).
        for (int i = 0; i < RunRewardGenerator.CardItemCount(State); i++)
        {
            SetMask(mask, action++);
        }

        for (int i = 0; i < RunRewardGenerator.SpecialCardItemCount(State); i++)
        {
            SetMask(mask, action++);
        }
    }
}
