using Sts2Emulator.Core.Rng;

namespace Sts2Emulator.Core.Run;

public enum RunFollowUp
{
    None,
    CardReward,
    TransformSelect,

    /// <summary>A rewards screen holding relics the player must claim one at a time.</summary>
    BonusRelicRewards,

    /// <summary>
    /// A choose-a-card screen whose cards are ALREADY rolled, so the caller must not roll
    /// another offer over the top of them.
    /// </summary>
    PreRolledCardReward,

    /// <summary>Scroll Boxes' choose-a-bundle screen.</summary>
    BundleSelect,

    /// <summary>
    /// A queue of card-reward screens claimed one after another, as Glass Eye's five are.
    /// </summary>
    BonusCardOffers,
}

public static class RunNonCombatEffects
{
    /// <summary>
    /// The game's Hook.TryModifyCardBeingAddedToDeck, which the egg relics answer: a card
    /// of their type joins the deck already upgraded. Every deck addition goes through
    /// here so an event, a shop and a card reward all agree.
    /// </summary>
    /// <summary>
    /// `MawBank.AfterRoomEntered`: GoldVar(12) on entering a room, every room, until the
    /// player buys ANYTHING from a merchant -- `AfterItemPurchased` with `goldSpent > 0`
    /// sets `HasItemBeenBought` and the bank is used up for good.
    /// </summary>
    /// <remarks>
    /// The counter is the spent flag, so it survives combats the way the teas' does. The
    /// gold goes through `GainGold`, which is the run's chokepoint -- Bloody Idol and the
    /// rest read a gain however it arrives.
    /// </remarks>
    public static void PayMawBank(RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == Effects.RelicEffects.MawBank);
        if (index >= 0 && state.Relics[index].Counter == 0)
        {
            GainGold(state, 12);
        }
    }

    /// <summary>Any purchase closes the bank, whatever it cost.</summary>
    public static void CloseMawBank(RunState state, int goldSpent)
    {
        if (goldSpent <= 0)
        {
            return;
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == Effects.RelicEffects.MawBank);
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with { Counter = 1 };
        }
    }

    /// <summary>
    /// `SwordOfStone.AfterCombatVictory`: five ELITE wins and `RelicCmd.Replace` swaps it
    /// for Sword of Jade, which is Strength 3 at the top of every fight thereafter.
    /// </summary>
    /// <remarks>
    /// Five, not three -- `DynamicVar("Elites", 5m)`. A REPLACEMENT rather than a second
    /// relic: the stone is gone, and the relic list keeps its length.
    /// </remarks>
    /// <summary>
    /// `ImprovementPower.AfterCombatEnd`: upgrade `Amount` upgradable cards in the DECK,
    /// each drawn off `Rng.CombatCardSelection` and removed from the pool so one card is
    /// never upgraded twice by the same power.
    /// </summary>
    /// <remarks>
    /// Mad Science's Improvement rider. The power lives on the player IN COMBAT and pays
    /// out against the RUN's deck once the fight is over, which is why it sat unmodelled
    /// while the other eight riders were done -- there was nowhere in combat to put it.
    /// It fires whether the fight was won or lost; `AfterCombatEnd` is not
    /// `AfterCombatVictory`.
    /// </remarks>
    public static void ApplyImprovementUpgrades(RunState state)
    {
        int amount = state.ActiveCombat is null
            ? 0
            : BuffSystem.Get(state.ActiveCombat.PlayerBuffs, BuffId.Improvement);
        if (amount <= 0)
        {
            return;
        }

        var upgradable = Enumerable
            .Range(0, state.Deck.Count)
            .Where(index => RunConstants.IsRunCardUpgradable(state.Deck[index]))
            .ToList();
        for (int i = 0; i < amount && upgradable.Count > 0; i++)
        {
            int pick = state.Rng.CombatCardSelection.NextInt(upgradable.Count);
            int deckIndex = upgradable[pick];
            upgradable.RemoveAt(pick);
            state.Deck[deckIndex] = state.Deck[deckIndex] with { Upgraded = true };
        }
    }

    public static void CountEliteVictoryForSwordOfStone(RunState state, bool wasElite)
    {
        if (!wasElite)
        {
            return;
        }

        int index = state.Relics.FindIndex(relic =>
            relic.DefId == Effects.RelicEffects.SwordOfStone
        );
        if (index < 0)
        {
            return;
        }

        int elites = state.Relics[index].Counter + 1;
        state.Relics[index] = state.Relics[index] with { Counter = elites };
        if (elites >= Effects.RelicEffects.SwordOfStoneElites)
        {
            state.Relics[index] = new RelicInstance(Effects.RelicEffects.SwordOfJade);
        }
    }

    public static void AddCardToDeck(RunState state, CardInstance card)
    {
        // `DarkstonePeriapt.AfterCardChangedPiles` fires on a CURSE entering the DECK, so
        // it pays for every curse however it arrives -- an event's Decay, a Neow drawback,
        // a Normality. Read before the add so the max HP lands with the card.
        if (
            GeneratedData.Cards.Get(card.DefId).Type == CardType.Curse
            && Effects.RelicEffects.Has(state.Relics, Effects.RelicEffects.DarkstonePeriapt)
        )
        {
            GainMaxHp(state, 6);
        }

        var landed = EnchantedByFresnelLens(state, UpgradedByEggs(state, card));
        state.Deck.Add(landed);

        // `BingBong.AfterCardChangedPiles`: every card entering the DECK is CLONED to the
        // bottom of it -- `clonedBy == null` and a `CardsToSkip` set, which together mean
        // the clone does not clone itself. The doll's gift doubles a deck it likes and
        // doubles a curse just as happily.
        if (
            Effects.RelicEffects.Has(state.Relics, Effects.RelicEffects.BingBong)
            && !state.AddingBingBongClone
        )
        {
            state.AddingBingBongClone = true;
            AddCardToDeck(state, landed);
            state.AddingBingBongClone = false;
        }

        // `AfterCardChangedPiles` with the destination pile being the DECK. Two relics
        // read it, and both are per-card rather than per-reward: taking three cards in one
        // Neow blessing pays three times.
        if (Effects.RelicEffects.Has(state.Relics, Effects.RelicEffects.LuckyFysh))
        {
            GainGold(state, 15);
        }

        int book = state.Relics.FindIndex(relic =>
            relic.DefId == Effects.RelicEffects.BookOfFiveRings
        );
        if (book >= 0)
        {
            // `CardsAddedSinceLastTrigger` is `CardsAdded % 5`, so the heal lands on every
            // fifth card rather than once at five.
            int added = state.Relics[book].Counter + 1;
            state.Relics[book] = state.Relics[book] with { Counter = added % 5 };
            if (added % 5 == 0)
            {
                state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 20);
            }
        }
    }

    /// <summary>
    /// `FresnelLens.TryModifyCardBeingAddedToDeck`: every card entering the deck that
    /// Nimble can enchant arrives already carrying Nimble at
    /// `DynamicVar("NimbleAmount", 2m)`.
    /// </summary>
    /// <remarks>
    /// The lens has three hooks and they are the same rule at three doors --
    /// `TryModifyCardRewardOptionsLate`, `ModifyMerchantCardCreationResults` and this one.
    /// The first two only change what the SCREEN shows; this one is what actually lands,
    /// and it catches every route into the deck including the ones the other two miss.
    /// Modelling only the reward screen would have enchanted a bought card and not an
    /// event's gift.
    ///
    /// `CanEnchant` is the whole rule, so a card that already carries an enchantment keeps
    /// it -- the lens does not overwrite.
    /// </remarks>
    public static CardInstance EnchantedByFresnelLens(RunState state, CardInstance card)
    {
        if (
            !Effects.RelicEffects.Has(state.Relics, Effects.RelicEffects.FresnelLens)
            || !Enchantments.CanEnchant(card, Enchantment.Nimble)
        )
        {
            return card;
        }

        return card with
        {
            Enchantment = Enchantment.Nimble,
            EnchantAmount = 2,
        };
    }

    public static CardInstance UpgradedByEggs(RunState state, CardInstance card)
    {
        if (!RunConstants.IsRunCardUpgradable(card))
        {
            return card;
        }

        int eggForType = GeneratedData.Cards.Get(card.DefId).Type switch
        {
            CardType.Power => RunConstants.RelicFrozenEgg,
            CardType.Attack => RunConstants.RelicMoltenEgg,
            CardType.Skill => RunConstants.RelicToxicEgg,
            _ => 0,
        };

        return eggForType != 0 && HasRelic(state, eggForType)
            ? card with
            {
                Upgraded = true,
            }
            : card;
    }

    /// <summary>
    /// War Paint and Whetstone: CardsVar(2) upgraded off the deck on pickup, taken from
    /// Deck.Where(type and IsUpgradable).StableShuffle(Rng.Niche).Take(2).
    /// </summary>
    /// <param name="type">
    /// The card type to filter to, or null for ANY upgradable card -- Fragrant Mushroom's
    /// `Deck.Where(c => c.IsUpgradable)` has no type filter where War Paint's does.
    /// </param>
    /// <summary>
    /// The same StableShuffle-and-take as the pickup relics', off a stream the CALLER
    /// names -- Battleworn Dummy upgrades off the event's own Rng, not `Rng.Niche`.
    /// </summary>
    public static void UpgradeRandomDeckCardsForEvent(RunState state, int count, GameRng rng)
    {
        var candidates = Enumerable
            .Range(0, state.Deck.Count)
            .Where(index => RunConstants.IsRunCardUpgradable(state.Deck[index]))
            .ToList();

        StableShuffleDeck(state, candidates, rng);

        foreach (int index in candidates.Take(count))
        {
            state.Deck[index] = state.Deck[index] with { Upgraded = true };
        }
    }

    /// <summary>
    /// <c>ListExtensions.StableShuffle</c> over deck positions: <c>List.Sort()</c> on the cards --
    /// <c>CardModel.CompareTo</c> is the model id (ordinal, the entry) and then the upgrade level --
    /// and then the Fisher-Yates on <paramref name="rng" />.
    /// </summary>
    /// <remarks>
    /// The sort is the framework's introsort, which is not stable for more than sixteen items, so
    /// which of several equal cards (five Strikes) lands where is the sort's doing, and with it
    /// which copy an upgrade picks. A LINQ <c>OrderBy</c> (stable) or a comparison on the table id
    /// (the ids are not in entry order: DARK_EMBRACE is 119 and DARKNESS 120) puts them elsewhere.
    /// </remarks>
    internal static void StableShuffleDeck(RunState state, List<int> positions, GameRng rng)
    {
        positions.Sort(
            (left, right) =>
            {
                var a = state.Deck[left];
                var b = state.Deck[right];
                int byEntry = string.CompareOrdinal(
                    GeneratedData.Cards.Get(a.DefId).Entry,
                    GeneratedData.Cards.Get(b.DefId).Entry
                );
                return byEntry != 0 ? byEntry : a.Upgraded.CompareTo(b.Upgraded);
            }
        );
        rng.Shuffle(positions);
    }

    private static void UpgradeRandomDeckCards(RunState state, CardType? type, int count)
    {
        var candidates = Enumerable
            .Range(0, state.Deck.Count)
            .Where(index =>
                (type is null || GeneratedData.Cards.Get(state.Deck[index].DefId).Type == type)
                && RunConstants.IsRunCardUpgradable(state.Deck[index])
            )
            .ToList();

        StableShuffleDeck(state, candidates, state.Rng.Niche);

        foreach (int index in candidates.Take(count))
        {
            state.Deck[index] = state.Deck[index] with { Upgraded = true };
        }
    }

    private static bool HasRelic(RunState state, int relicId) =>
        state.Relics.Any(relic => relic.DefId == relicId);

    public static RunFollowUp ApplyRelicPickup(
        RunState state,
        int relicId,
        SelectionReturn returnTo = SelectionReturn.Map
    )
    {
        if (state.Relics.All(relic => relic.DefId != relicId))
        {
            state.Relics.Add(new RelicInstance(relicId, StartingRelicCounter(relicId)));

            // `Player.AddRelicInternal` raises `RelicObtained` before `AfterObtained` runs, and
            // every CardReward still waiting listens: an egg taken off the same rewards screen
            // upgrades the card reward that has not been opened yet.
            RunRewardGenerator.ApplyObtainedRelicToWaitingCardRewards(state, relicId);

            // `PotionBelt.AfterObtained` -- GainMaxPotionCount(2). Inside the not-already-
            // held branch so a duplicate pickup cannot pay twice, which is the same reason
            // the strike-from-the-grab-bag below sits here.
            if (relicId == Effects.RelicEffects.PotionBelt)
            {
                state.MaxPotionSlots += 2;
            }

            // `SignetRing.AfterObtained`: GoldVar(999). Through `GainGold` rather than a
            // bare add, because that is the run's chokepoint and Bloody Idol is listening.
            if (relicId == Effects.RelicEffects.SignetRing)
            {
                GainGold(state, 999);
            }

            // `BloodSoakedRose.AfterObtained`: an Enthralled into the deck. That curse is
            // the whole price of the relic's +1 max energy -- the energy half lives in
            // `RelicEffects.MaxEnergyRelics`, and modelling only that made it free.
            //
            // Through `AddCardToDeck` so Darkstone Periapt and Bing Bong see the curse,
            // which a direct `state.Deck.Add` would have hidden from both.
            if (relicId == Effects.RelicEffects.BloodSoakedRose)
            {
                AddCardToDeck(state, new CardInstance(166, Upgraded: false));
            }

            // `TanxsWhistle.AfterObtained` and `JewelryBox.AfterObtained` are the same
            // one line with a different card: `CreateCard<T>` then `CardPileCmd.Add` to
            // the DECK. Through `AddCardToDeck` for the reason Blood Soaked Rose's curse
            // is -- Bing Bong clones whatever enters the deck, and Fresnel Lens enchants
            // it, and a direct `Deck.Add` is invisible to both.
            if (relicId == Effects.RelicEffects.TanxsWhistle)
            {
                AddCardToDeck(state, new CardInstance(539, Upgraded: false));
            }

            if (relicId == Effects.RelicEffects.JewelryBox)
            {
                AddCardToDeck(state, new CardInstance(16, Upgraded: false));
            }
        }

        // RelicCmd.Obtain strikes the relic from both grab bags unless it is stackable,
        // and Circlet -- the fallback -- is the only stackable relic there is. Without
        // this a relic handed over by name (the Sword of Stone, a Neow pick) stays in the
        // queue and can be offered a second time.
        //
        // Guarded rather than a plain Get: RunConstants carries five Neow-only relic ids
        // in the 1300-1500 range (Astrolabe = 1332, where Relics.g.cs says Astrolabe is
        // id 8) that resolve to nothing. Those are ancient relics and are not in any grab
        // bag anyway, so skipping them is correct here -- but the mismatch is real and is
        // its own bug.
        if (relicId != CircletRelic && GeneratedData.Relics.TryGet(relicId, out _))
        {
            state.RelicBag.Remove(relicId);
            state.SharedRelicBag.Remove(relicId);
        }

        switch (relicId)
        {
            case RunConstants.RelicWarPaint:
                UpgradeRandomDeckCards(state, CardType.Skill, 2);
                break;
            case RunConstants.RelicWhetstone:
                UpgradeRandomDeckCards(state, CardType.Attack, 2);
                break;
            case RunConstants.RelicGoldenPearl:
                GainGold(state, 150);
                break;
            case RunConstants.RelicNeowsTorment:
                AddCardToDeck(state, new CardInstance(RunConstants.NeowsFuryCard, Upgraded: false));
                break;
            case RunConstants.RelicNeowsBones:
                // NeowsBones.AfterObtained: shuffle every relic Neow could offer except
                // itself on PlayerRng.Rewards, take two, and OFFER them -- a RewardsSet
                // with WithSkippingDisallowed, so the player claims both -- and only then
                // add the curse.
                //
                // This used to be two Rng.UpFront.NextItem draws applied on the spot: the
                // wrong stream, a draw that can hand out the SAME relic twice where a
                // shuffle-and-take cannot, no screen at all, and a candidate list of only
                // the positives. A live capture answers a rewards screen twice here and
                // comes away with Winged Boots and Silken Tress.
                state.PendingBonusRelicRewards.Clear();
                state.PendingBonusRelicRewards.AddRange(NeowsBonesRelicOffer(state));
                state.PendingNeowsBonesCurse = true;
                // Put the first one ON the screen here rather than leaving that to the
                // caller: the pickup should leave a state that makes sense however it was
                // reached, and a queue with nothing offered from it is a screen that
                // reports rewards it will not hand over.
                RunRewardGenerator.OfferNextBonusRelic(state);
                return RunFollowUp.BonusRelicRewards;
            case RunConstants.RelicNutritiousOyster:
                GainMaxHp(state, 11);
                break;
            case RunConstants.RelicStrawberry:
                GainMaxHp(state, 7);
                break;
            case RunConstants.RelicPear:
                GainMaxHp(state, 10);
                break;
            case RunConstants.RelicMango:
                GainMaxHp(state, 14);
                break;
            // Both mushrooms are `HasUponPickupEffect`, and the Hungry For Mushrooms event
            // is only the door they come through: the HP swing is the RELIC's, so it lands
            // however the relic is obtained. The emulator had the numbers on the event
            // instead, and had them wrong -- +7 max HP and -9.
            // `LoomingFruit.AfterObtained`: MaxHpVar(31m), an Ancient relic. The
            // cornucopia its icon may or may not show is decided by the last byte of the
            // PROFILE's unique id and is purely cosmetic -- it changes `IconBaseName` and
            // nothing else, which is a joke about multiplayer rather than a mechanic.
            // `Byrdpip.AfterObtained`: every BYRDONIS EGG in the deck becomes a Byrd
            // Swoop. That is the relic's whole mechanic -- the pet it also summons is
            // 9999 HP with an invisible health bar and a `NOTHING_MOVE` state machine
            // that returns a completed task, an animation anchor for the card and nothing
            // the rules can see. Its `Skin` roll is off `new Rng(Owner, Id)`, a stream of
            // its own, so it spends none of the run's.
            case Effects.RelicEffects.Byrdpip:
            {
                // The eggs are listed first, then each one is transformed in deck order; a
                // Swoop lands at the bottom of the deck (TransformInDeck), and is no egg.
                int eggs = state.Deck.Count(card => card.DefId == ByrdonisEggCard);
                for (int i = 0; i < eggs; i++)
                {
                    TransformInDeck(
                        state,
                        state.Deck.FindIndex(card => card.DefId == ByrdonisEggCard),
                        new CardInstance(ByrdSwoopCard, Upgraded: false)
                    );
                }

                break;
            }
            // `FakeMango.AfterObtained`: MaxHpVar(3m) against the real Mango's 14.
            case Effects.RelicEffects.FakeMango:
                GainMaxHp(state, 3);
                break;
            // `FakeLeesWaffle.AfterObtained`: heal `MaxHp * (HealVar(10) / 100)` -- a
            // PERCENTAGE, where the real waffle grants 7 flat max HP. Integer-truncated,
            // the way every decimal-to-int in the damage path is.
            case Effects.RelicEffects.FakeLeesWaffle:
                state.PlayerHp = Math.Min(
                    state.PlayerMaxHp,
                    state.PlayerHp + state.PlayerMaxHp * 10 / 100
                );
                break;
            case RunConstants.RelicLoomingFruit:
                GainMaxHp(state, 31);
                break;
            case RunConstants.RelicBigMushroom:
                // `MaxHpVar(20)`. GainMaxHp heals by the same amount, which the live
                // capture shows: 64/80 became 84/100.
                GainMaxHp(state, 20);
                break;
            case RunConstants.RelicFragrantMushroom:
                // `HpLossVar(15)` Unblockable | Unpowered, then `CardsVar(2)` upgradable
                // deck cards upgraded off `Rng.Niche` -- no type filter, unlike War Paint.
                state.PlayerHp = Math.Max(0, state.PlayerHp - 15);
                UpgradeRandomDeckCards(state, null, 2);
                break;
            case RunConstants.RelicLeesWaffle:
                GainMaxHp(state, 7);
                state.PlayerHp = state.PlayerMaxHp;
                break;
            case RunConstants.RelicOldCoin:
                GainGold(state, 300);
                break;
            case RunConstants.RelicGoldenCompass:
                // `AfterObtained` latches the act and then calls `GenerateMap()` outright:
                // the map the run is standing on is REPLACED, not the next one.
                state.GoldenPathAct = state.CurrentActIndex;
                RunMapGenerator.GenerateActMap(state);
                break;

            case RunConstants.RelicFurCoat:
                // `AfterObtained` latches the CURRENT act and marks its map at once. Seven
                // fights in this act only -- the coat does not follow the run upstairs.
                RunMapGenerator.MarkRoomsWhenFurCoatArrives(state);
                break;

            case RunConstants.RelicToyBox:
                // FOUR relics off the FRONT of the grab bag, offered on a screen the way
                // Small Capsule's one is. Each is marked WAX, and every third combat melts
                // one -- so this is four relics on loan, not four relics.
                state.PendingBonusRelicRewards.Clear();
                for (int i = 0; i < RunConstants.ToyBoxRelics; i++)
                {
                    int wax = RunRewardGenerator.NextRelic(state);
                    state.PendingBonusRelicRewards.Add(wax);
                    state.WaxRelics.Add(wax);
                }

                RunRewardGenerator.OfferNextBonusRelic(state);
                return RunFollowUp.BonusRelicRewards;
            case RunConstants.RelicSmallCapsule:
                // RewardsCmd.OfferCustom with a single RelicReward: the relic goes on a
                // SCREEN the player claims from, not straight into the run. Granting it
                // outright also ran its pickup effect a step early, so a capture
                // (`P14DQ9GNPW`) is on `rewards` holding two relics where the emulator
                // was back on the ancient holding three.
                state.PendingBonusRelicRewards.Clear();
                state.PendingBonusRelicRewards.Add(RunRewardGenerator.NextRelic(state));
                RunRewardGenerator.OfferNextBonusRelic(state);
                return RunFollowUp.BonusRelicRewards;
            case RunConstants.RelicLargeCapsule:
                ApplyRelicPickup(state, RunRewardGenerator.NextRelic(state));
                ApplyRelicPickup(state, RunRewardGenerator.NextRelic(state));
                AddCardToDeck(state, new CardInstance(472, Upgraded: false));
                AddCardToDeck(state, new CardInstance(131, Upgraded: false));
                break;
            case RunConstants.RelicPomander:
                // CardsVar(1) through CardSelectCmd.FromDeckForUpgrade: the PLAYER picks
                // which card is upgraded. UpgradeFirstCard upgraded the deck's first
                // upgradable card, which in a starting deck is a Strike -- and a live
                // capture (`RRRR6WR3C4`) shows the game opening a card_select screen at
                // step 1 where the emulator had already silently upgraded one.
                return BeginDeckSelection(state, DeckSelection.Upgrade, 0, count: 1)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicNeowsTalisman:
                // The LAST BASIC card tagged Strike and the last tagged Defend -- the real
                // tags, as Leafy Poultice's are. Ironclad's two ids found nothing for any
                // other character.
                UpgradeLastTaggedBasic(state, strike: true);
                UpgradeLastTaggedBasic(state, strike: false);
                break;
            case RunConstants.RelicCursedPearl:
                AddCardToDeck(state, new CardInstance(NamedCard("Greed"), Upgraded: false));
                GainGold(state, 333);
                break;
            case RunConstants.RelicHeftyTablet:
                // AfterObtained offers CardsVar(3) RARE cards from the owner's own pool --
                // Uniform odds, NoUpgradeRoll -- on a choose-a-card screen, and adds its
                // Injury together with whichever the player takes. It used to hand over
                // one card rolled off Rng.UpFront and drop the Injury immediately: no
                // screen, no choice, the wrong stream and the curse a decision early.
                RunRewardGenerator.OfferPreRolledCards(
                    state,
                    RunRewardGenerator.GenerateRareOnlyCardOffer(state, 3)
                );
                state.PendingHeftyTabletCurse = true;
                return RunFollowUp.PreRolledCardReward;
            case RunConstants.RelicKaleidoscope:
                // Handled as two card rewards in RunEngine.ApplyAncientChoice; the relic
                // offers screens to choose from, it does not put cards in the deck.
                break;
            case RunConstants.RelicArcaneScroll:
                AddCardToDeck(
                    state,
                    new CardInstance(
                        state.PlayerRng.Rewards.NextItem(
                            RunRewardGenerator.RareCardPool(state)
                        ),
                        Upgraded: false
                    )
                );
                break;
            case RunConstants.RelicLeadPaperweight:
                // Two COLOURLESS cards on a choose-a-card screen, RegularEncounter odds
                // through CardCreationSource.Other. Same story as Hefty Tablet: one card
                // off the wrong stream, granted rather than offered.
                RunRewardGenerator.OfferPreRolledCards(
                    state,
                    RunRewardGenerator.GenerateOtherSourceCardOffer(
                        state,
                        RunRewardGenerator.ColorlessRewardPool,
                        2
                    )
                );
                return RunFollowUp.PreRolledCardReward;
            case RunConstants.RelicPhialHolster:
                // PhialHolster.AfterObtained: GainMaxPotionCount(PotionSlots=1) FIRST --
                // so both of the potions it then rolls have somewhere to go -- and
                // CreateRandomPotionsOutOfCombat(Potions=2, Rng.CombatPotionGeneration).
                //
                // The stream is the whole point. These rolled off PlayerRng.Rewards, which
                // is the stream every card reward, shop and transformation in the run also
                // draws from: two draws the game never makes there put the very next
                // combat's gold reward and card offer at the wrong position, and every one
                // after it. A live capture of a run that took this from Neow paid 15 gold
                // where the emulator paid 9.
                // One CreateRandomPotionsOutOfCombat call: the second cannot repeat the first.
                state.MaxPotionSlots++;
                foreach (int potion in RunRewardGenerator.NextPotions(state, state.Rng.CombatPotionGeneration, 2))
                {
                    RunRewardGenerator.AddPotion(state, potion);
                }

                break;
            // ---- the act ancients' blessings -------------------------------------
            case RunConstants.RelicPaelsHorn:
                // Two Relax into the deck. The one ancient blessing a live capture pins:
                // `ACT2TEST01` takes it and comes away with two.
                AddCardToDeck(state, new CardInstance(NamedCard("Relax"), Upgraded: false));
                AddCardToDeck(state, new CardInstance(NamedCard("Relax"), Upgraded: false));
                break;
            case RunConstants.RelicStorybook:
                AddCardToDeck(
                    state,
                    new CardInstance(NamedCard("BrightestFlame"), Upgraded: false)
                );
                break;
            case RunConstants.RelicSandCastle:
                // CardsVar(6), StableShuffled on Rng.NICHE -- not the event's own stream
                // and not UpFront. Same shape as Doors of Light and Dark's light door,
                // which is worth knowing because that one taught the lesson twice: sort
                // by ModelId before shuffling, and only upgradable cards are candidates.
                UpgradeRandomCards(state, 6, state.Rng.Niche);
                break;
            case RunConstants.RelicAlchemicalCoffer:
                // PotionSlots(4): GainMaxPotionCount FIRST so all four potions have
                // somewhere to go, then four rolled off CombatPotionGeneration. Phial
                // Holster is the same shape, and the stream is the point -- rolling these
                // off PlayerRng.Rewards would move every card reward after it.
                //
                // One CreateRandomPotionsOutOfCombat call, so no repeats; and each goes to
                // `originalSlotCount + i` -- the NEW slots -- leaving an old empty slot empty
                // (run referee, seed RXVFG7FZ86: the game's slot 1 stayed empty).
                int originalSlots = state.MaxPotionSlots;
                state.MaxPotionSlots += 4;
                int[] coffer = RunRewardGenerator.NextPotions(state, state.Rng.CombatPotionGeneration, 4);
                for (int i = 0; i < coffer.Length; i++)
                {
                    RunRewardGenerator.AddPotionAt(state, coffer[i], originalSlots + i);
                }

                break;
            case RunConstants.RelicPaelsClaw:
                // Goopy onto EVERY card in the deck that can take it — no screen, no
                // choice. `CanEnchant` is the filter, which for Goopy is the Defend tag.
                EnchantEveryCard(state, Enchantment.Goopy, 1);
                break;
            case RunConstants.RelicNutritiousSoup:
                // Tezcatara's Ember onto every BASIC Strike. The rarity and tag checks
                // are the relic's own, on top of the enchantment's CanEnchant.
                EnchantEveryCard(
                    state,
                    Enchantment.TezcatarasEmber,
                    1,
                    card => IsBasicStrike(card)
                );
                break;
            case RunConstants.RelicArchaicTooth:
            {
                // `GetTranscendenceStarterCard` is `Deck.Cards.FirstOrDefault(...)` -- the
                // FIRST match in deck order, not a choice and not a roll.
                // `GetTranscendenceTransformedCard` builds the ancient card fresh and then
                // carries the starter's upgrade (`CardCmd.Upgrade`) and its enchantment, at its
                // amount, across (run referee, seed CNV3YDEJXD: an upgraded Bash became an
                // upgraded Break in the game).
                int index = state.Deck.FindIndex(card =>
                    RunConstants.TranscendenceUpgrades.Any(pair => pair.Starter == card.DefId)
                );
                if (index >= 0)
                {
                    int ancient = RunConstants
                        .TranscendenceUpgrades.First(pair =>
                            pair.Starter == state.Deck[index].DefId
                        )
                        .Ancient;
                    var starterCard = state.Deck[index];
                    var made = new CardInstance(ancient, Upgraded: false);
                    if (starterCard.Upgraded && RunConstants.IsRunCardUpgradable(made))
                    {
                        made = made with { Upgraded = true };
                    }

                    if (starterCard.Enchantment != Enchantment.None)
                    {
                        made = made with
                        {
                            Enchantment = starterCard.Enchantment,
                            EnchantAmount = starterCard.EnchantAmount,
                        };
                    }

                    TransformInDeck(state, index, made);
                }

                break;
            }

            case RunConstants.RelicTouchOfOrobas:
            {
                // `RelicCmd.Replace(starter, refined)` -- the starter relic is GONE, not
                // kept alongside. A starter with no entry in the table becomes a Circlet,
                // which is the game's fallback rather than a no-op.
                // `GetStarterRelic` is `Relics.FirstOrDefault(r => r.Rarity == Starter)`,
                // so it finds the relic by RARITY rather than by table membership -- which
                // is what makes the Circlet fallback reachable at all.
                int index = state.Relics.FindIndex(relic =>
                    GeneratedData.Relics.Get(relic.DefId).Rarity == RelicRarity.Starter
                );
                if (index >= 0)
                {
                    int starter = state.Relics[index].DefId;
                    var match = RunConstants.RefinementUpgrades.FirstOrDefault(pair =>
                        pair.Starter == starter
                    );
                    state.Relics[index] = new RelicInstance(
                        match.Starter == starter ? match.Upgraded : RunConstants.RelicCirclet
                    );
                }

                break;
            }

            case RunConstants.RelicTriBoomerang:
                // `FromDeckForEnchantment` with Instinct at 1, on three cards. Instinct
                // takes Attacks only, so a deck of nothing but Skills is offered nothing.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Instinct,
                    count: 3,
                    enchantAmount: 1
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case RunConstants.RelicSereTalon:
            {
                // TWO curses off `Rng.Niche`, drawn WITHOUT replacement -- the game removes
                // each pick from its own list -- and then THREE Wishes. The curse list is
                // ordered by Id, which `RollGeneratableCurse` matches, but the removal is
                // the reason this cannot just call it twice.
                var curses = GeneratedData
                    .CardPools.Curse.ToArray()
                    .Where(id => GeneratedData.Cards.Get(id).CanBeGeneratedByModifiers)
                    .OrderBy(id => GeneratedData.Cards.Get(id).Entry, StringComparer.Ordinal)
                    .ToList();
                for (int i = 0; i < 2 && curses.Count > 0; i++)
                {
                    int curse = state.Rng.Niche.NextItem(curses);
                    curses.Remove(curse);
                    AddCardToDeck(state, new CardInstance(curse, Upgraded: false));
                }

                for (int i = 0; i < 3; i++)
                {
                    AddCardToDeck(state, new CardInstance(541, Upgraded: false));
                }

                break;
            }

            case RunConstants.RelicPumpkinCandle:
                // `AfterObtained` is `Rekindle()`, which is `KindleCount += 5`. The count
                // rides the relic's Counter because it is RUN state -- it survives the
                // fight that spends it.
                {
                    int index = state.Relics.FindIndex(relic =>
                        relic.DefId == RunConstants.RelicPumpkinCandle
                    );
                    if (index >= 0)
                    {
                        state.Relics[index] = state.Relics[index] with
                        {
                            Counter = state.Relics[index].Counter + 5,
                        };
                    }
                }

                break;

            case RunConstants.RelicDistinguishedCape:
                // `LoseMaxHp(9)` and then THREE Apparitions into the deck. No selection --
                // it takes the HP and hands over the cards, which is why it is the only one
                // of this batch that resolves in place.
                state.PlayerMaxHp = Math.Max(1, state.PlayerMaxHp - 9);
                state.PlayerHp = Math.Min(state.PlayerHp, state.PlayerMaxHp);
                for (int i = 0; i < 3; i++)
                {
                    AddCardToDeck(state, new CardInstance(17, Upgraded: false));
                }

                break;

            case RunConstants.RelicPreservedFog:
                // `FromDeckForRemoval` for THREE, and then a Folly whatever was removed.
                // The curse rides the follow-up rather than being added here, so it arrives
                // after the removals the way the game orders it.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Remove,
                    0,
                    count: 3,
                    followUpCard: 10028,
                    followUpCount: 1
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case RunConstants.RelicBeautifulBracelet:
                // `FromDeckForEnchantment` with Swift at THREE, on three cards, and no filter
                // of its own: any card Swift can take, which is any enchantable card. The
                // count is what the screen asks for, not what it is guaranteed to get.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Swift,
                    count: 3,
                    enchantAmount: 3
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case RunConstants.RelicClaws:
                // `FromDeckForTransformation` into Maul, SIX cards. The game preserves the
                // original's upgrade on the Maul when the Maul is upgradable; the emulator's
                // TransformTo does not carry that across yet, which is the one thing this
                // arm approximates.
                return BeginDeckSelection(state, DeckSelection.TransformToCarrying, 299, count: 6)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case RunConstants.RelicElectricShrymp:
                // FromDeckForEnchantment with Imbued: ONE card the player picks, and
                // Imbued takes skills only, so a deck with no skill left is offered
                // nothing.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Imbued,
                    count: 1
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicPaelsGrowth:
                // Likewise, with Clone at amount FOUR rather than one.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Clone,
                    count: 1,
                    enchantAmount: 4
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicSeaGlass:
                // CardsVar(15) / 3 = five cards of each rarity from the OTHER character's
                // pool, all fifteen offered on one grid. `ForNonCombatWithUniformOdds`
                // (NoUpgradeRoll) with NoRarityModification and NoCardPoolModifications:
                // each card is the pick alone, fifteen draws, one factory call per rarity
                // and so one pass of the hooks per call.
                //
                // The screen is a 0-to-15 multi-select with a confirm
                // (`CardSelectorPrefs(prompt, 0, list.Count)`): MinSelect ZERO, so the
                // player may take any number of the fifteen, including none. That is what
                // `PendingOfferOptional` is -- every other grid names one number and
                // closes itself when the picks run out. Closes O16.
                var seaGlass = new[] { CardRarity.Common, CardRarity.Uncommon, CardRarity.Rare }
                    .Select(rarity =>
                        RunRewardGenerator.CreateForReward(
                            state,
                            5,
                            RunConstants.OtherCharacterPoolFor(
                                state.Character,
                                Math.Max(0, state.SeaGlassCharacter)
                            ),
                            CardCreationFlags.NoUpgradeRoll
                                | CardCreationFlags.NoRarityModification
                                | CardCreationFlags.NoCardPoolModifications,
                            rarity
                        )
                    )
                    .ToList();
                RunRewardGenerator.FillRewardGrid(
                    state,
                    seaGlass,
                    seaGlass.Sum(set => set.Cards.Length),
                    hooked: true
                );
                state.PendingOfferOptional = true;
                return RunFollowUp.PreRolledCardReward;
            case RunConstants.RelicGlassEye:
                // FIVE card rewards on one screen — Common, Common, Uncommon, Uncommon,
                // Rare — each offering three of that rarity. `ForNonCombatWithUniformOdds`
                // carries NoUpgradeRoll and the relic adds NoRarityModification, and the
                // CardReward its IsCardReward: no rarity roll and no upgrade roll, so each
                // card is the pick alone -- fifteen draws, not thirty (real runs: every
                // Glass Eye run was a screen out from the first of the five). The hooks
                // act on each screen. No RngOverride: PlayerRng.Rewards.
                const CardCreationFlags glassEyeFlags =
                    CardCreationFlags.NoUpgradeRoll
                    | CardCreationFlags.NoRarityModification
                    | CardCreationFlags.IsCardReward;
                state.PendingCardOffers.Clear();
                foreach (
                    CardRarity rarity in new[]
                    {
                        CardRarity.Common,
                        CardRarity.Common,
                        CardRarity.Uncommon,
                        CardRarity.Uncommon,
                        CardRarity.Rare,
                    }
                )
                {
                    state.PendingCardOffers.Add(
                        RunRewardGenerator.CreateForReward(
                            state,
                            3,
                            RunRewardGenerator.OwnRewardPool(state, glassEyeFlags),
                            glassEyeFlags,
                            rarity
                        )
                    );
                }

                return RunFollowUp.BonusCardOffers;
            case RunConstants.RelicPaelsTooth:
                // CardsVar(5) through FromDeckForRemoval with a filter of `IsUpgradable`
                // -- five cards the player picks, and only upgradable ones are offered.
                // The relic keeps copies of what it took for its own combat effect, which
                // the run layer does not need.
                return BeginDeckSelection(state, DeckSelection.RemoveUpgradable, 0, count: 5)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicYummyCookie:
                // CardsVar(4) through FromDeckForUpgrade: four cards the PLAYER picks.
                return BeginDeckSelection(state, DeckSelection.Upgrade, 0, count: 4)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicBiiigHug:
                // CardsVar(4) through FromDeckForRemoval.
                return BeginDeckSelection(state, DeckSelection.Remove, 0, count: 4)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicPreciseScissors:
                // CardsVar(1) through CardSelectCmd.FromDeckForRemoval -- the PLAYER picks
                // which card goes, and picking is the whole of the blessing.
                // RemoveLowestPriorityCard was the emulator choosing instead, and choosing
                // badly: its priority list starts with the curse placeholder, which is
                // Ascender's Bane, a card the game will not even offer for removal.
                return BeginDeckSelection(state, DeckSelection.Remove, 0, count: 1)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicScrollBoxes:
                // GenerateRandomBundles on PlayerRng.Rewards, then
                // CardSelectCmd.FromChooseABundleScreen: the player takes ONE bundle of
                // three whole, and never sees the other three cards again. This handed
                // over three cards rolled off Rng.UpFront -- the wrong stream, the wrong
                // count of draws, six cards' worth of choice collapsed into none, and no
                // screen at all.
                state.BundleOffer =
                [
                    .. RunRewardGenerator.GenerateScrollBoxBundles(state).SelectMany(b => b),
                ];
                state.SelectedBundle = -1;
                return RunFollowUp.BundleSelect;
            case RunConstants.RelicLeafyPoultice:
                state.PlayerMaxHp = Math.Max(1, state.PlayerMaxHp - 12);
                state.PlayerHp = Math.Min(state.PlayerHp, state.PlayerMaxHp);
                // The first BASIC card tagged Strike and the first tagged Defend -- the
                // real tags, not Ironclad's two ids, which found nothing for any other
                // character. Both transformed together off `PlayerRng.Transformations`.
                TransformFirstTaggedBasic(state, strike: true);
                TransformFirstTaggedBasic(state, strike: false);
                break;
            case RunConstants.RelicPrecariousShears:
                // CardsVar(2) through the same removal screen, then DamageVar(16). The
                // damage is owed only once the cards are gone, which BeginDeckSelection's
                // follow-up already models -- the same shape Luminous Choir uses.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Remove,
                    0,
                    count: 2,
                    followUpHpLoss: 16
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicSilkenTress:
                state.Gold = 0;
                break;
            case RunConstants.RelicPandorasBox:
                // `Cards.Where(c.IsBasicStrikeOrDefend && c.IsRemovable)`, transformed off
                // `Rng.Niche`. This matched Ironclad's two card IDS on the Transformations
                // stream -- wrong filter, wrong stream, and for any other character it
                // found nothing at all. The TAG is what the game reads, and Ascender's
                // Bane fails `IsRemovable` rather than the tag.
                PandorasBoxTransform(state);
                break;
            case RunConstants.RelicCallingBell:
                // A Curse of the Bell and THREE relics on a rewards SCREEN, one of each
                // rarity: `RelicReward(Common)`, `RelicReward(Uncommon)`, `RelicReward(Rare)`.
                // The emulator handed over three ROLLED pool relics directly -- no screen,
                // no rarities, and three draws off the wrong end of the queue.
                AddCardToDeck(
                    state,
                    new CardInstance(NamedCard("CurseOfTheBell"), Upgraded: false)
                );
                state.PendingBonusRelicRewards.Add(
                    RunRewardGenerator.NextShopRelicOfRarity(state, RelicRarity.Common)
                );
                state.PendingBonusRelicRewards.Add(
                    RunRewardGenerator.NextShopRelicOfRarity(state, RelicRarity.Uncommon)
                );
                state.PendingBonusRelicRewards.Add(
                    RunRewardGenerator.NextShopRelicOfRarity(state, RelicRarity.Rare)
                );
                return RunFollowUp.BonusRelicRewards;
            case RunConstants.RelicDustyTome:
                // An UPGRADED copy of the ANCIENT-rarity card the relic was set up with --
                // `SetupForPlayer` picks it off `PlayerRng.Rewards` from the character's
                // own pool, filtered to Ancient rarity minus Archaic Tooth's transcendence
                // cards. For the Ironclad that pool is Break and Corruption, and Break is
                // a transcendence card -- so it is always CORRUPTION, off one draw.
                //
                // The emulator added a random card off the whole reward pool on the UpFront
                // stream: wrong card, wrong pool, wrong stream.
                AddCardToDeck(
                    state,
                    new CardInstance(state.EventValue0 ?? DustyTomeCard(state), Upgraded: true)
                );
                state.EventValue0 = null;
                break;
            case RunConstants.RelicNewLeaf:
                // CardsVar(1) through CardSelectCmd.FromDeckForTransformation, then
                // CardCmd.TransformToRandom on Rng.Niche. This used the older
                // TransformSelectedDeckIndex path, which answers the screen a step later
                // than the game does -- the `N11HWGCNUN` capture is back at the event on
                // step 3 with the card already transformed while the emulator is still
                // holding the selection open.
                return BeginDeckSelection(state, DeckSelection.TransformToRandom, 0, count: 1)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicAstrolabe:
                // THREE cards the player picks, each transformed off `Rng.Niche` and then
                // UPGRADED. The legacy path transformed the first three cards in the deck
                // and upgraded whatever landed last, which is neither a choice nor the
                // right cards.
                return BeginDeckSelection(
                    state,
                    DeckSelection.TransformToRandomUpgraded,
                    0,
                    count: 3,
                    returnTo: SelectionReturn.Map
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
            case RunConstants.RelicEmptyCage:
                // TWO cards the player picks, removed.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Remove,
                    0,
                    count: 2,
                    returnTo: SelectionReturn.Map
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            // `Orrery.AfterObtained` offers FIVE CardRewards at once -- a whole screen each,
            // not five cards on one. The count field Prayer Wheel added carries them.
            case Effects.RelicEffects.Orrery:
                state.ExtraCardRewardsOwed += 5;
                RunRewardGenerator.PopulateCardReward(state);
                return RunFollowUp.PreRolledCardReward;

            // `Cauldron.AfterObtained` offers five POTION rewards, through the same pending
            // queue Tiny Mailbox's two go through. One RewardsSet, so all five are rolled as
            // it is offered (PotionReward.Populate), not one by one as the screen frees up.
            case Effects.RelicEffects.Cauldron:
                for (int i = 0; i < 5; i++)
                {
                    state.PendingPotionRewards.Add(
                        RunRewardGenerator.NextPotion(state, state.PlayerRng.Rewards)
                    );
                }

                return RunFollowUp.BonusRelicRewards;

            // The five shop relics whose AfterObtained raises a DECK-SELECTION screen.
            // All five go through the same machinery Empty Cage and the events use; what
            // differs is the kind, the count and the enchantment they apply.
            case Effects.RelicEffects.DollysMirror:
                return BeginDeckSelection(state, DeckSelection.Duplicate, 0, returnTo: returnTo)
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case Effects.RelicEffects.GnarledHammer:
                // `CardSelectorPrefs(prompt, 0, 3)` -- up to THREE cards, and Sharp at 3
                // rather than the 2 Self-Help Book applies.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Sharp,
                    count: 3,
                    enchantAmount: 3,
                    returnTo: returnTo
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case Effects.RelicEffects.Kifuda:
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Adroit,
                    count: 3,
                    enchantAmount: 3,
                    returnTo: returnTo
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case Effects.RelicEffects.PunchDagger:
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.Momentum,
                    enchantAmount: 5,
                    returnTo: returnTo
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;

            case Effects.RelicEffects.RoyalStamp:
                // The game shuffles the eligible cards on Rng.Niche before offering them,
                // which changes the ORDER of the screen and not what is on it. The
                // emulator offers the deck in deck order, so the shuffle is not modelled
                // and no stream draw is spent -- recorded rather than faked, because a
                // wrong draw on Niche desynchronises far more than a screen's ordering.
                return BeginDeckSelection(
                    state,
                    DeckSelection.Enchant,
                    (int)Enchantment.RoyallyApproved,
                    enchantAmount: 1,
                    returnTo: returnTo
                )
                    ? RunFollowUp.TransformSelect
                    : RunFollowUp.None;
        }

        return RunFollowUp.None;
    }

    /// <summary>
    /// <c>NeowsBones.GetValidRelics</c>, shuffled and cut to two.
    /// </summary>
    /// <remarks>
    /// The list is <c>Neow.AllPossibleOptions</c> in DECLARATION order -- curse options,
    /// then positive ones, then the six that are offered as one of a pair -- minus Neow's
    /// Bones itself and anything <c>IsAllowedAtNeow</c> refuses. Order is load-bearing:
    /// the shuffle that follows is over this exact sequence, so a list assembled any other
    /// way draws different relics from the same stream position. MassiveScroll is absent
    /// for the same reason it is absent from Neow's own offer -- its IsAllowed is
    /// <c>Players.Count > 1</c>.
    /// </remarks>
    private static List<int> NeowsBonesRelicOffer(RunState state)
    {
        var candidates = new List<int>();
        foreach (int relicId in RunConstants.NeowCurseOptions)
        {
            if (relicId != RunConstants.RelicNeowsBones)
            {
                candidates.Add(relicId);
            }
        }

        candidates.AddRange(RunConstants.NeowPositiveOptions);
        candidates.AddRange(RunConstants.NeowPairedOptions);
        state.PlayerRng.Rewards.Shuffle(candidates);
        return candidates.Take(2).ToList();
    }

    /// <summary>
    /// The curse Neow's Bones adds once its relics are claimed:
    /// <c>Rng.Niche.NextItem</c> over the generatable curses, ordered by id.
    /// </summary>
    public static void AddNeowsBonesCurse(RunState state)
    {
        if (!state.PendingNeowsBonesCurse)
        {
            return;
        }

        state.PendingNeowsBonesCurse = false;
        AddCardToDeck(state, new CardInstance(RollGeneratableCurse(state), Upgraded: false));
    }

    /// <summary>
    /// `ToyBox.AfterCombatEnd`: count the fight, and on every third one melt the first wax
    /// relic still held. `RelicCmd.Melt` takes it away -- the box lends four relics for
    /// twelve fights, it does not give four.
    /// </summary>
    public static void MeltAToyBoxRelic(RunState state)
    {
        int box = state.Relics.FindIndex(relic => relic.DefId == RunConstants.RelicToyBox);
        if (box < 0)
        {
            return;
        }

        int seen = state.Relics[box].Counter + 1;
        state.Relics[box] = state.Relics[box] with { Counter = seen };
        if (seen % RunConstants.ToyBoxCombats != 0)
        {
            return;
        }

        // `FirstOrDefault(r => r.IsWax && !r.IsMelted)` -- the first still HELD, which is
        // why a wax relic the player never claimed is skipped rather than counted.
        int melting = state.WaxRelics.FindIndex(id => state.Relics.Any(relic => relic.DefId == id));
        if (melting >= 0)
        {
            int relicId = state.WaxRelics[melting];
            state.WaxRelics.RemoveAt(melting);
            state.Relics.RemoveAll(relic => relic.DefId == relicId);
        }
    }

    /// <summary>
    /// `PaelsWing.OnSacrifice`: the card reward is given up instead of claimed, and every
    /// SECOND sacrifice pulls a relic from the FRONT of the grab bag -- the same end an
    /// elite reward and Shovel's dig pull from.
    /// </summary>
    public static void SacrificeACardReward(RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == RunConstants.RelicPaelsWing);
        if (index < 0)
        {
            return;
        }

        int sacrificed = state.Relics[index].Counter + 1;
        state.Relics[index] = state.Relics[index] with { Counter = sacrificed };
        if (sacrificed % RunConstants.PaelsWingSacrifices == 0)
        {
            ApplyRelicPickup(state, RunRewardGenerator.NextRelic(state));
        }
    }

    /// <summary>
    /// `PumpkinCandle.AfterCombatEnd`: <c>KindleCount = Math.Max(KindleCount - 1, 0)</c>.
    /// Fires whether or not the fight was won, so it sits with the other unconditional
    /// end-of-combat work rather than behind the victory check.
    /// </summary>
    public static void BurnPumpkinCandle(RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == RunConstants.RelicPumpkinCandle);
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with
            {
                Counter = Math.Max(state.Relics[index].Counter - 1, 0),
            };
        }
    }

    /// <summary>
    /// `CookRestSiteOption.IsEnabled`: two REMOVABLE cards in the deck. Removability is
    /// `!Keywords.Contains(Eternal)`, the same rule a plain removal screen uses.
    /// </summary>
    public static bool CanCook(RunState state) =>
        state.Deck.Count(card => !GeneratedData.Cards.Get(card.DefId).Eternal) >= 2;

    private static int StartingRelicCounter(int relicId)
    {
        // The three teas count DOWN the combats they have left, and that count is run
        // state: the relic is one object for the whole run, so it has to arrive charged
        // rather than be charged by the first fight it sees.
        return relicId switch
        {
            RunConstants.RelicSilverCrucible => 3,
            Effects.RelicEffects.BoneTea => Effects.RelicEffects.BoneTeaCombats,
            Effects.RelicEffects.EmberTea => Effects.RelicEffects.EmberTeaCombats,
            Effects.RelicEffects.TeaOfDiscourtesy => Effects.RelicEffects.TeaOfDiscourtesyCombats,
            _ => 0,
        };
    }

    /// <summary>
    /// Put the run into <paramref name="eventId"/> the way the game does.
    /// </summary>
    /// <remarks>
    /// Entering an event is not just setting its id: the game calls the event's
    /// <c>CalculateVars</c> as it generates the options, and several events DRAW from
    /// their own Rng stream there. Skip it and every later draw in that event lands one
    /// position early -- which is not a visible failure, it is a different outcome. The
    /// Endless Conveyor rolls its dish in CalculateVars and nothing else touches it, so
    /// Observe the Chef upgraded the wrong card against a live capture.
    /// </remarks>
    public static void BeginEvent(RunState state, int eventId)
    {
        state.EventValue0 = null;
        state.EventValue1 = null;
        state.EventPage = 0;
        state.EventRandomOffer = [];
        state.CrystalSphere = null;
        state.CrystalSphereRng = null;
        state.EventRngStream = null;
        state.EventRngName = null;
        state.EventRelicStock = null;
        state.EventSkippedCards = [];
        state.EventId = eventId;
        CalculateEventVars(state);
        state.Phase = RunPhase.Event;
    }

    public static void EnterEvent(RunState state)
    {
        state.EventValue0 = null;
        state.EventValue1 = null;
        state.EventPage = 0;
        state.EventRandomOffer = [];
        state.CrystalSphere = null;
        state.CrystalSphereRng = null;
        state.EventRngStream = null;
        state.EventRngName = null;
        state.EventRelicStock = null;
        state.EventSkippedCards = [];
        // The game has no "New Leaf forces The Legends Were True" rule: NewLeaf is a plain
        // transform-on-pickup relic with no event hook, and the only ModifyNextEvent
        // override in the game is LanternKey's (below). That invented shortcut bypassed
        // the ring -- and its act gate -- so it is gone.

        // ActModel.PullNextEvent: RoomSet.EnsureNextEventIsValid, then ModifyNextEvent,
        // then RunState.AddVisitedEvent of whatever comes out the far end.
        // EnsureNextEventIsValid: NextEvent is events[visited % count], and the cursor is
        // walked forward while the candidate is disallowed OR already seen. The sequence
        // WRAPS -- it is a ring, not a list that runs out -- and the game only gives up
        // after a full lap, logging "all unique events exhausted". No RNG is drawn.
        if (state.EventSequence.Length > 0)
        {
            int chosen = -1;
            for (int i = 0; i < state.EventSequence.Length && chosen < 0; i++)
            {
                int eventId = state.EventSequence[
                    state.EventSequenceIndex % state.EventSequence.Length
                ];
                // The pull is `RunManager.CreateRoom`, which EnterMapPointInternal calls BEFORE
                // `AppendToMapPointHistory`: an event's IsAllowed reads `TotalFloor` (the history's
                // length) without the room being entered, one less than the emulator's Floor,
                // which the map move has already counted.
                if (IsEventAllowed(state, eventId, totalFloor: state.Floor - 1)
                    && !state.VisitedEventIds.Contains(eventId))
                {
                    chosen = eventId;
                }
                else
                {
                    state.EventSequenceIndex++;
                }
            }

            // A whole lap found nothing allowed and unseen: "All unique events exhausted,
            // allowing repetition". The game then hands out `NextEvent` as it stands --
            // the event under the cursor, allowed or not. The lap above walked the cursor
            // all the way round, so it is back where it started.
            if (chosen < 0)
            {
                chosen = state.EventSequence[
                    state.EventSequenceIndex % state.EventSequence.Length
                ];
            }

            // LanternKey.ModifyNextEvent: a Lantern Key card in the deck turns the third
            // act's next event into War Historian Repy. The ring still advances past the
            // event it displaced, and it is Repy that goes on the visited list.
            if (state.CurrentActIndex == 2 && state.Deck.Any(card => card.DefId == LanternKeyCard))
            {
                chosen = RunConstants.EventWarHistorianRepy;
            }

            // PullNextEvent records it; MarkRoomVisited moves the cursor past it.
            if (!state.VisitedEventIds.Contains(chosen))
            {
                state.VisitedEventIds.Add(chosen);
            }

            state.EventSequenceIndex++;
            BeginEvent(state, chosen);
        }
    }

    /// <summary>Test seam for <see cref="IsEventAllowed"/>.</summary>
    public static bool IsEventAllowedForTests(RunState state, int eventId) =>
        IsEventAllowed(state, eventId);

    /// <summary>
    /// <c>EventModel.IsAllowed</c> for the run, as <c>RoomSet.EnsureNextEventIsValid</c>
    /// calls it. Every act's ring is dealt from its own pool plus the shared one
    /// (<c>ActModel.GenerateRooms</c>), so which events CAN be reached is the pool's
    /// business; this is only each event's own gate, act index included.
    /// </summary>
    /// <remarks>
    /// This used to refuse outright the nine shared events whose gate reads
    /// <c>CurrentActIndex</c> and the four later-act events, on the grounds that the
    /// emulator models Act 1 only. It plays Acts 2 and 3 now
    /// (<see cref="RunMapGenerator.AdvanceToNextAct"/>), where those are exactly the
    /// events the game offers. All thirteen have handlers in the run engine, so none is
    /// kept refused as unimplemented.
    /// </remarks>
    internal static bool IsEventAllowed(RunState state, int eventId, int? totalFloor = null) =>
        IsEventAllowedForRun(state, eventId, totalFloor);

    /// <summary>
    /// Each event's own <c>IsAllowed(IRunState)</c>, transcribed -- whether THIS RUN may
    /// be shown the event, which is a separate question from whether the act's pool
    /// contains it. The base <c>EventModel.IsAllowed</c> is <c>true</c>, so an event
    /// missing below really has no rule (DrowningBeacon, SelfHelpBook, SpiritGrafter and
    /// InfestedAutomaton once carried invented ones).
    /// </summary>
    /// <remarks>
    /// A solo run, so each <c>Players.All(...)</c> is a test of the one player.
    /// </remarks>
    /// <param name="totalFloor">
    /// The game's <c>TotalFloor</c> as the gate reads it; the run's Floor when not given.
    /// </param>
    public static bool IsEventAllowedForRun(RunState state, int eventId, int? totalFloor = null)
    {
        int floor = totalFloor ?? state.Floor;
        bool HoldsPotion() => state.PotionSlots.Any(potion => potion != 0);
        return eventId switch
        {
            // `Gold >= 100 && count(IsTransformable) >= 2`. A deck card is transformable
            // exactly when it is removable (CardModel.IsTransformable).
            RunConstants.EventMorphicGrove => state.Gold >= 100
                && state.Deck.Count(IsRemovable) >= 2,
            // `Gold >= GoldVar(149)` -- the CANONICAL value, before CalculateVars shaves
            // NextInt(0, 50) off the instance's price -- and
            // RelicGrabBag.HasAvailableRelics, which only fails once the whole bag is
            // drained and is not modelled. It was `Gold >= 100 || Deck.Count >= 3`.
            RunConstants.EventLuminousChoir => state.Gold >= 149,
            // Basic rarity and IsRemovable.
            RunConstants.EventWoodCarvings => state.Deck.Any(card =>
                GeneratedData.Cards.Get(card.DefId).Rarity == CardRarity.Basic
                && IsRemovable(card)
            ),
            // Gold >= 120, not 40 -- the belt is a rich man's event.
            RunConstants.EventEndlessConveyor => state.Gold >= 120,
            // Gold >= 55 and nothing else. The `|| Deck.Count > 0` made this always true,
            // because a run always has a deck.
            RunConstants.EventWaterloggedScriptorium => state.Gold >= 55,
            RunConstants.EventWhisperingHollow => state.Gold >= 44,
            RunConstants.EventTrashHeap => state.PlayerHp > 5,
            RunConstants.EventSpiralingWhirlpool => state.Deck.Any(card =>
                Enchantments.CanEnchant(card, Enchantment.Spiral)
            ),
            // TotalFloor, which is floors across the whole run. `Floor` does not reset
            // between acts (see AdvanceToNextAct), so it is the same number.
            RunConstants.EventPunchOff => floor >= 6,
            RunConstants.EventSlipperyBridge => floor > 6 && state.Deck.Any(IsRemovable),
            // WarHistorianRepy.IsAllowed is `return false` outright: it is never drawn
            // from the sequence, only reached by its own means. Falling through to the
            // default handed it out as an ordinary event.
            RunConstants.EventWarHistorianRepy => false,
            // CurrentActIndex == 0, a non-empty deck and CurrentHp >= 10. It had no act
            // test, so once Acts 2 and 3 were dealt it could turn up in them.
            RunConstants.EventTheLegendsWereTrue => state.CurrentActIndex == 0
                && state.PlayerHp >= 10
                && state.Deck.Count > 0,
            // CurrentActIndex < 2 AND gold: the stall only stands in the first two acts.
            RunConstants.EventTeaMaster => state.CurrentActIndex < 2 && state.Gold >= 150,
            RunConstants.EventTheFutureOfPotions => state.PotionSlots.Count(potion => potion != 0)
                >= 2,
            // At least two removable Basic Strikes AND two removable Basic Defends
            // (Amalgamator.IsValid, counted per tag).
            RunConstants.EventAmalgamator => state.Deck.Count(card =>
                    IsRemovableTaggedBasic(card, 0)
                ) >= 2
                && state.Deck.Count(card => IsRemovableTaggedBasic(card, 1)) >= 2,
            // Any card PerfectFit will take.
            RunConstants.EventFieldOfManSizedHoles => state.Deck.Any(card =>
                Enchantments.CanEnchant(card, Enchantment.PerfectFit)
            ),
            // Reflections and TinkerTime declare no IsAllowed at all: both were gated on
            // a non-empty deck, which is a rule neither model has. The deck-size gates
            // came in with placeholder options that no longer stand.
            RunConstants.EventReflections => true,
            RunConstants.EventTinkerTime => true,
            // The weaver only turns up for a run that can afford EmotionalAwarenessCost:
            // `Players.All(p => p.Gold >= 125)`. The upgradable-or-hurt test belonged to
            // the placeholder options.
            RunConstants.EventZenWeaver => state.Gold >= RunConstants.ZenWeaverEmotionalCost,
            // The site only turns up on a run that is actually hurt: CurrentHp <= 70% of
            // max. It was grouped with the unconditional events.
            RunConstants.EventUnrestSite => state.PlayerHp <= state.PlayerMaxHp * 0.70m,
            // ByrdonisNest is `!HasEventPet()`. Pets are not modelled, and the sequence
            // offers each event at most once per act, so within an act it cannot matter.
            // BrainLeech and RoomFullOfCheese are `CurrentActIndex < 2`: shared events,
            // dealt into every act's list, that the last act passes over.
            RunConstants.EventBrainLeech or RunConstants.EventRoomFullOfCheese =>
                state.CurrentActIndex < 2,
            // Dense Vegetation only refuses in multiplayer, and the run is solo --
            // transcribed rather than left to the default so the gate says so.
            RunConstants.EventDenseVegetation
            or RunConstants.EventByrdonisNest
            or RunConstants.EventAromaOfChaos => true,

            // ---- Gated on the act INDEX. Acts 2 and 3 are dealt, so these are live.
            // CurrentActIndex > 0 and Gold >= 100.
            RunConstants.EventCrystalSphere => state.CurrentActIndex > 0 && state.Gold >= 100,
            // CurrentActIndex == 1 alone.
            RunConstants.EventDollRoom => state.CurrentActIndex == 1,
            // CurrentActIndex > 0 alone.
            RunConstants.EventPotionCourier or RunConstants.EventSymbiote =>
                state.CurrentActIndex > 0,
            // Act index aside, the stall wants a customer: `Gold >= 100 || holds a Foul
            // Potion`. The potion half is the interesting one -- it is the price of the
            // ROBBERY, not of a purchase, and a broke player carrying one still qualifies.
            // Solo-only too, which this run is.
            RunConstants.EventFakeMerchant => state.CurrentActIndex >= 1
                && (state.Gold >= 100 || state.PotionSlots.Contains(NamedPotion("FoulPotion"))),
            // ALL of: act index > 0, a tradable relic to give, Gold >= 100, and a potion
            // to give. It was `||` across gold, potions and relic count.
            RunConstants.EventRanwidTheElder => state.CurrentActIndex > 0
                && state.Relics.Any(relic => IsTradableRelic(state, relic))
                && state.Gold >= 100
                && HoldsPotion(),
            // Act index > 0 and five TRADABLE relics (RelicModel.IsTradable), not five
            // relics: starter, event and ancient relics do not count.
            RunConstants.EventRelicTrader => state.CurrentActIndex > 0
                && state.Relics.Count(relic => IsTradableRelic(state, relic)) >= 5,
            // CurrentActIndex == 1 and a potion in hand. The enchantable-card half was
            // the placeholder option's.
            RunConstants.EventStoneOfAllTime => state.CurrentActIndex == 1 && HoldsPotion(),
            // CurrentActIndex == 1 and Gold >= 100.
            RunConstants.EventWelcomeToWongos => state.CurrentActIndex == 1
                && state.Gold >= 100,

            // ---- Later-act pool events: no act test of their own. The pool decides where
            // they appear (Hive: Colorful Philosophers, Colossal Flower; Glory: Grave of
            // the Forgotten, Round Tea Party); these are the run-state gates.
            // Colossal Flower's ladder costs 5 + 6 + 7 = 18 unblockable to walk to the
            // bottom of, and its gate is `CurrentHp >= 19` -- one more than the whole
            // climb, so the event can never be the thing that kills you.
            RunConstants.EventColossalFlower => state.PlayerHp >= 19,
            // The tea party's fight is 11 unblockable and its gate is `CurrentHp >= 12`,
            // the same one-more-than-the-damage rule.
            RunConstants.EventRoundTeaParty => state.PlayerHp >= 12,
            // `All(HasEnchantableCards)`, where enchantable means Souls Power will take
            // it -- so a deck with nothing that exhausts never sees the grave at all.
            // Note this is the WHOLE event's gate, not just Confront's lock: the game
            // refuses to show it rather than showing it with one option greyed out.
            RunConstants.EventGraveOfTheForgotten => state.Deck.Any(card =>
                Enchantments.CanEnchant(card, Enchantment.SoulsPower)
            ),
            // `All(p => p.UnlockState.CharacterCardPools.Count() > 1)` -- a profile that
            // has unlocked a second character. The emulator runs a fixed profile with all
            // five pools, so this is true. See the fixed-profile note in HANDOFF.md.
            RunConstants.EventColorfulPhilosophers => true,
            _ => true,
        };
    }

    /// <summary>
    /// The book offers Sharp to Attacks, Nimble to Skills and Swift to Powers, so a card
    /// is a candidate for SOME page of it when its own type's page would take it.
    /// </summary>
    public static bool CardCanReceiveSelfHelpBookEnchantment(CardInstance card) =>
        SelfHelpBookPageTakes(card, Enchantment.Sharp)
        || SelfHelpBookPageTakes(card, Enchantment.Nimble)
        || SelfHelpBookPageTakes(card, Enchantment.Swift);

    /// <summary>
    /// The card type each page restricts its enchantment to: the book's own
    /// `SelectAndEnchant&lt;T&gt;(2, CardType.X, ...)`, whose `DeckFilter` checks the type
    /// before `CanEnchant`. For Sharp and Nimble it repeats the enchantment's own rule;
    /// for Swift, which takes any card, it is the whole restriction.
    /// </summary>
    public static CardType SelfHelpBookCardType(Enchantment enchantment) =>
        enchantment switch
        {
            Enchantment.Sharp => CardType.Attack,
            Enchantment.Nimble => CardType.Skill,
            Enchantment.Swift => CardType.Power,
            _ => throw new ArgumentOutOfRangeException(nameof(enchantment), enchantment, null),
        };

    /// <summary>Whether the book's page for <paramref name="enchantment"/> would take this card.</summary>
    public static bool SelfHelpBookPageTakes(CardInstance card, Enchantment enchantment) =>
        GeneratedData.Cards.Get(card.DefId).Type == SelfHelpBookCardType(enchantment)
        && Enchantments.CanEnchant(card, enchantment);

    public static Enchantment SelfHelpBookEnchantment(int enchantType) =>
        enchantType switch
        {
            0 => Enchantment.Sharp,
            1 => Enchantment.Nimble,
            2 => Enchantment.Swift,
            _ => Enchantment.None,
        };

    /// <summary>
    /// Open a deck selection: the run moves to the card-select screen and stays there
    /// until <paramref name="count"/> cards have been chosen. Refuses -- and leaves the
    /// run where it was -- when no card in the deck is eligible, which is what an event
    /// with a locked option is telling the player.
    /// </summary>
    /// <summary>
    /// Whether <see cref="BeginDeckSelection"/> would open rather than refuse. The action
    /// mask has to ask the same question the step will, or the two disagree about an
    /// option and the agent is offered a move it cannot make -- which is the single most
    /// common defect this layer has had.
    /// </summary>
    public static bool CanBeginDeckSelection(
        RunState state,
        DeckSelection kind,
        int arg,
        CardType? cardType = null
    )
    {
        var savedKind = state.PendingSelectionKind;
        int savedArg = state.PendingSelectionArg;
        var savedCardType = state.PendingSelectionCardType;
        state.PendingSelectionKind = kind;
        state.PendingSelectionArg = arg;
        state.PendingSelectionCardType = cardType;
        bool any = Enumerable.Range(0, state.Deck.Count).Any(i => CanSelectCard(state, i));
        state.PendingSelectionKind = savedKind;
        state.PendingSelectionArg = savedArg;
        state.PendingSelectionCardType = savedCardType;
        return any;
    }

    public static bool BeginDeckSelection(
        RunState state,
        DeckSelection kind,
        int arg,
        int count = 1,
        int followUpCard = 0,
        int followUpCount = 0,
        int followUpHpLoss = 0,
        int followUpMaxHpGain = 0,
        string? eventEntry = null,
        int enchantAmount = 0,
        SelectionReturn returnTo = SelectionReturn.EventResult,
        CardType? cardType = null
    )
    {
        state.PendingSelectionEventEntry = eventEntry;
        state.PendingSelectionCardType = cardType;
        state.PendingSelectionKind = kind;
        state.PendingSelectionArg = arg;
        state.PendingSelectionCount = count;
        state.PendingSelectionFollowUpCard = followUpCard;
        state.PendingSelectionFollowUpCount = followUpCount;
        state.PendingSelectionFollowUpHpLoss = followUpHpLoss;
        state.PendingSelectionFollowUpMaxHpGain = followUpMaxHpGain;
        state.PendingSelectionEnchantAmount = enchantAmount;
        state.PendingSelectionReturn = returnTo;
        if (!Enumerable.Range(0, state.Deck.Count).Any(i => CanSelectCard(state, i)))
        {
            ClearDeckSelection(state);
            return false;
        }

        state.Phase = RunPhase.TransformSelect;
        return true;
    }

    public static void ClearDeckSelection(RunState state)
    {
        state.PendingSelectionKind = DeckSelection.None;
        state.PendingSelectionArg = 0;
        state.PendingSelectionCount = 0;
        state.PendingSelectionFollowUpCard = 0;
        state.PendingSelectionFollowUpCount = 0;
        state.PendingSelectionFollowUpHpLoss = 0;
        state.PendingSelectionFollowUpMaxHpGain = 0;
        state.PendingSelectionEnchantAmount = 0;
        state.PendingSelectionCardType = null;
        state.PendingSelectionReturn = SelectionReturn.EventResult;
        state.PendingSelectionReturnsToEvent = false;
    }

    /// <summary>
    /// Pay out whatever the event owes once the selection is done -- the curse Luminous
    /// Choir and the Wellspring hand over in exchange for the removal.
    /// </summary>
    public static void ResolveDeckSelectionFollowUp(RunState state)
    {
        for (int i = 0; i < state.PendingSelectionFollowUpCount; i++)
        {
            AddCardToDeck(
                state,
                new CardInstance(state.PendingSelectionFollowUpCard, Upgraded: false)
            );
        }

        state.PlayerHp = Math.Max(0, state.PlayerHp - state.PendingSelectionFollowUpHpLoss);

        // `CreatureCmd.GainMaxHp` HEALS by the same amount it raises the maximum, which is
        // why the cook is worth taking at full HP as well as hurt.
        if (state.PendingSelectionFollowUpMaxHpGain > 0)
        {
            state.PlayerMaxHp += state.PendingSelectionFollowUpMaxHpGain;
            state.PlayerHp = Math.Min(
                state.PlayerMaxHp,
                state.PlayerHp + state.PendingSelectionFollowUpMaxHpGain
            );
        }
    }

    /// <summary>Whether the pending selection would take the card at this deck index.</summary>
    public static bool CanSelectCard(RunState state, int deckIndex)
    {
        if ((uint)deckIndex >= (uint)state.Deck.Count)
        {
            return false;
        }

        var card = state.Deck[deckIndex];
        return state.PendingSelectionKind switch
        {
            DeckSelection.Enchant => Enchantments.CanEnchant(
                card,
                (Enchantment)state.PendingSelectionArg
            )
                && (
                    state.PendingSelectionCardType is not CardType type
                    || GeneratedData.Cards.Get(card.DefId).Type == type
                ),
            // CardSelectCmd.FromDeckGeneric(c => c.IsTransformable && c.Rarity == Basic)
            // -- Wood Carvings carves a Basic card into a Peck or a Toric Toughness.
            DeckSelection.TransformTo => GeneratedData.Cards.Get(card.DefId).Rarity
                == CardRarity.Basic,
            // Claws passes no predicate of its own, but FromDeckForTransformation does: no Quest
            // card, and nothing Eternal (IsTransformable) -- Ascender's Bane stays put.
            DeckSelection.TransformToCarrying => IsOfferedForTransformation(card),
            DeckSelection.Upgrade => RunConstants.IsRunCardUpgradable(card),
            // Dolly's Mirror excludes only Quest cards; a curse is a legal copy.
            DeckSelection.Duplicate => GeneratedData.Cards.Get(card.DefId).Type != CardType.Quest,
            // `FromDeckForRemoval` filters on `c.IsRemovable && filter(c)`, so the Eternal
            // check applies to this one too.
            DeckSelection.RemoveUpgradable => RunConstants.IsRunCardUpgradable(card)
                && !GeneratedData.Cards.Get(card.DefId).Eternal,
            // `CardSelectCmd.FromDeckForRemoval` filters on `c.IsRemovable`, which is
            // `!Keywords.Contains(CardKeyword.Eternal)` -- so the game will not so much as
            // OFFER an Eternal card for removal. Seven curses carry it, Ascender's Bane
            // among them, and the emulator let a run delete every one.
            DeckSelection.Remove => !GeneratedData.Cards.Get(card.DefId).Eternal,
            DeckSelection.RemoveTaggedBasic => IsRemovableTaggedBasic(
                card,
                state.PendingSelectionArg
            ),
            DeckSelection.TransformToRandom => IsOfferedForTransformation(card),
            DeckSelection.TransformToRandomUpgraded => IsOfferedForTransformation(card),
            _ => false,
        };
    }

    /// <summary>
    /// `CardSelectCmd.FromDeckForTransformation`: <c>c.Type != Quest &amp;&amp; c.IsTransformable</c>, and a
    /// deck card is transformable exactly when it is removable (not Eternal).
    /// </summary>
    private static bool IsOfferedForTransformation(CardInstance card)
    {
        var def = GeneratedData.Cards.Get(card.DefId);
        return def.Type != CardType.Quest && !def.Eternal;
    }

    /// <summary>
    /// Amalgamator's `IsValid(tag, card)`: the tag, BASIC rarity, and removable.
    /// </summary>
    /// <param name="tag">0 for `CardTag.Strike`, 1 for `CardTag.Defend`.</param>
    private static bool IsRemovableTaggedBasic(CardInstance card, int tag)
    {
        var def = GeneratedData.Cards.Get(card.DefId);
        return (tag == 0 ? def.StrikeTag : def.DefendTag)
            && def.Rarity == CardRarity.Basic
            && !def.Eternal;
    }

    /// <summary>
    /// Take one card for the pending selection. Returns false -- changing nothing -- when
    /// the index is out of range or the card is not one the selection would take.
    /// </summary>
    public static bool ApplyDeckSelection(RunState state, int deckIndex)
    {
        if (!CanSelectCard(state, deckIndex))
        {
            return false;
        }

        var card = state.Deck[deckIndex];
        switch (state.PendingSelectionKind)
        {
            case DeckSelection.Enchant:
                state.Deck[deckIndex] = card with
                {
                    Enchantment = (Enchantment)state.PendingSelectionArg,
                    // Self-Help Book applies at 2 (its Enchantment*Amount vars); every
                    // event enchantment is CardCmd.Enchant<T>(card, 1m).
                    EnchantAmount =
                        state.PendingSelectionEnchantAmount > 0
                            ? state.PendingSelectionEnchantAmount
                            : SelfHelpBookAmount((Enchantment)state.PendingSelectionArg),
                };
                break;
            case DeckSelection.Duplicate:
                AddCardToDeck(state, card);
                break;
            case DeckSelection.TransformTo:
                TransformInDeck(
                    state,
                    deckIndex,
                    new CardInstance(state.PendingSelectionArg, Upgraded: false)
                );
                break;
            case DeckSelection.TransformToCarrying:
            {
                // `CreateMaulFromOriginal`: the upgrade rides across when the NEW card is
                // upgradable, and the enchantment when it can take on the new card. Both
                // are the new card's own rules, not the old one's -- transforming an
                // upgraded Strike into something unupgradable loses the upgrade.
                var made = new CardInstance(state.PendingSelectionArg, Upgraded: false);
                if (card.Upgraded && RunConstants.IsRunCardUpgradable(made))
                {
                    made = made with { Upgraded = true };
                }

                if (
                    card.Enchantment != Enchantment.None
                    && Enchantments.CanEnchant(made, card.Enchantment)
                )
                {
                    made = made with
                    {
                        Enchantment = card.Enchantment,
                        EnchantAmount = card.EnchantAmount,
                    };
                }

                TransformInDeck(state, deckIndex, made);
                break;
            }
            case DeckSelection.Upgrade:
                state.Deck[deckIndex] = card with { Upgraded = true };
                break;
            case DeckSelection.TransformToRandom:
                // Every event hands CardCmd.TransformToRandom its OWN base.Rng; only
                // NewLeaf, which does not come through here, uses Rng.Niche. Rolling the
                // new card off Niche gave the event a card from a stream the game never
                // touched for it.
                TransformCardAt(
                    state,
                    deckIndex,
                    state.PendingSelectionEventEntry is null
                        ? state.Rng.Niche
                        : EventRng(state, state.PendingSelectionEventEntry)
                );
                break;
            case DeckSelection.TransformToRandomUpgraded:
                // Astrolabe: the roll is off `Rng.Niche`, and the UPGRADE lands on the new
                // card -- `CardCmd.Upgrade(cardModel)` happens between the create and the
                // transform, so an unupgradable roll simply arrives unupgraded.
                TransformCardAt(state, deckIndex, state.Rng.Niche);
                if (RunConstants.IsRunCardUpgradable(state.Deck[^1]))
                {
                    state.Deck[^1] = state.Deck[^1] with { Upgraded = true };
                }

                break;
            case DeckSelection.Remove:
            case DeckSelection.RemoveUpgradable:
            case DeckSelection.RemoveTaggedBasic:
                state.Deck.RemoveAt(deckIndex);
                break;
            default:
                return false;
        }

        state.PendingSelectionCount--;
        return true;
    }

    /// <summary>
    /// What amount an enchantment lands at when nothing overrides it.
    /// </summary>
    /// <remarks>
    /// Self-Help Book applies Sharp, Nimble and Swift at 2 (its Enchantment*Amount vars);
    /// every event enchantment is <c>CardCmd.Enchant&lt;T&gt;(card, 1m)</c>. Pael's Growth
    /// is the exception that made this an override rather than a rule — it applies Clone
    /// at FOUR — so a caller that knows its own amount passes it.
    /// </remarks>
    private static int SelfHelpBookAmount(Enchantment enchantment) =>
        enchantment is Enchantment.Sharp or Enchantment.Nimble or Enchantment.Swift ? 2 : 1;

    /// <summary>
    /// What Luminous Choir asks for its tribute. The event starts from a GoldVar of 149
    /// and, on generate, takes off <c>Rng.NextInt(0, 50)</c> from its own stream -- so
    /// the price is somewhere in 100..149 and the option is locked below it.
    /// </summary>
    public static int LuminousChoirTributeCost(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateLuminousChoirVars(state);
        }

        return state.EventValue0!.Value;
    }

    /// <summary>
    /// <c>LuminousChoir.CalculateVars</c> takes GoldVar(149) and subtracts
    /// <c>Rng.NextInt(0, 50)</c> -- once, as the event is generated. Reading it re-rolled
    /// it, which was invisible only because every draw came off a fresh stream.
    /// </summary>
    private static void CalculateLuminousChoirVars(RunState state) =>
        state.EventValue0 = 149 - EventRng(state, "LUMINOUS_CHOIR").NextInt(0, 50);

    /// <summary>
    /// What the sphere charges to Uncover Future. <c>CrystalSphere.CalculateVars</c> adds
    /// <c>Rng.NextInt(1, 50)</c> to a base of 50, so the price is 51..99 -- and since the
    /// event only turns up on a run holding at least 100 gold, it is always affordable and
    /// neither option is ever locked.
    ///
    /// The draw matters beyond the price: it is the first thing taken from the event's
    /// stream, and the board the minigame lays out is drawn from the same stream straight
    /// after. A cost read off a fresh Rng would leave the board one draw early.
    /// </summary>
    public static int CrystalSphereCost(RunState state)
    {
        EnsureCrystalSphereVars(state);
        return state.EventValue0!.Value;
    }

    private static void EnsureCrystalSphereVars(RunState state)
    {
        if (state.CrystalSphereRng is not null)
        {
            return;
        }

        state.CrystalSphereRng = EventRng(state, "CRYSTAL_SPHERE");
        state.EventValue0 = 50 + state.CrystalSphereRng.NextInt(1, 50);
    }

    /// <summary>
    /// Opens the sphere with the given number of divinations: three for Uncover Future,
    /// six for the Payment Plan. Both options roll nothing of their own -- one spends gold,
    /// the other adds a Debt -- so the board is the same either way for a given seed.
    /// </summary>
    public static void OpenCrystalSphere(RunState state, int divinations)
    {
        EnsureCrystalSphereVars(state);
        state.CrystalSphere = CrystalSphereGame.Create(state.CrystalSphereRng!, divinations);
        state.Phase = RunPhase.CrystalSphere;
    }

    /// <summary>
    /// What uncovering a thing does on the spot. Only the curse acts immediately -- its
    /// <c>RevealItem</c> puts a Doubt in the deck there and then; everything else waits for
    /// the last divination and arrives as a reward.
    /// </summary>
    public static void RevealCrystalSphereItem(RunState state, CrystalSphereItem item)
    {
        if (item.Kind == CrystalSphereItemKind.Curse)
        {
            AddCardToDeck(state, new CardInstance(NamedCard("Doubt"), Upgraded: false));
        }
    }

    public static int SunkenTreasurySmallChestGold(RunState state)
    {
        EnsureSunkenTreasuryVars(state);
        return state.EventValue0!.Value;
    }

    public static int SunkenTreasuryLargeChestGold(RunState state)
    {
        EnsureSunkenTreasuryVars(state);
        return state.EventValue1!.Value;
    }

    private static void EnsureSunkenTreasuryVars(RunState state)
    {
        if (state.EventValue0 is null || state.EventValue1 is null)
        {
            CalculateSunkenTreasuryVars(state);
        }
    }

    private static void CalculateEventVars(RunState state)
    {
        switch (state.EventId)
        {
            case RunConstants.EventTheFutureOfPotions:
                // `PotionToCardType`, built as the options are: one card type per potion in the
                // belt, in slot order, off the event's own stream (Power only for a potion
                // better than Common). Held per traded option in EventRandomOffer.
                var futureRng = EventRng(state, "THE_FUTURE_OF_POTIONS");
                state.EventRandomOffer = state.PotionSlots.Where(potion => potion != 0)
                    .Select(potion => (int)futureRng.NextItem(RunRewardGenerator.FutureOfPotionsCardTypes(potion)))
                    .ToArray();
                break;
            case RunConstants.EventSunkenTreasury:
                CalculateSunkenTreasuryVars(state);
                break;
            case RunConstants.EventEndlessConveyor:
                CalculateEndlessConveyorVars(state);
                break;
            case RunConstants.EventSunkenStatue:
                CalculateSunkenStatueVars(state);
                break;
            case RunConstants.EventDenseVegetation:
                CalculateDenseVegetationVars(state);
                break;
            case RunConstants.EventWhisperingHollow:
                CalculateWhisperingHollowVars(state);
                break;
            case RunConstants.EventJungleMazeAdventure:
                CalculateJungleMazeVars(state);
                break;
            case RunConstants.EventThisOrThat:
                CalculateThisOrThatVars(state);
                break;
            case RunConstants.EventLuminousChoir:
                CalculateLuminousChoirVars(state);
                break;
            case RunConstants.EventCrystalSphere:
                // The cost is CalculateVars' only draw, and the board comes off the same
                // stream straight after -- so it has to be spent on entry, not on the
                // first read. Reading the mask would otherwise roll it.
                EnsureCrystalSphereVars(state);
                break;
            case RunConstants.EventSlipperyBridge:
                RollSlipperyBridgeCard(state);
                break;
            case RunConstants.EventRanwidTheElder:
                CalculateRanwidVars(state);
                break;
            case RunConstants.EventStoneOfAllTime:
                CalculateStoneOfAllTimeVars(state);
                break;
            case RunConstants.EventWelcomeToWongos:
                // The featured item is pulled when the options are generated, so it comes
                // off the bag on entry whether or not the player buys it.
                WongosFeaturedItem(state);
                break;
            case RunConstants.EventColorfulPhilosophers:
                // Which three of the four pools survive the cut is decided in
                // `GenerateInitialOptions`, so it is spent on entry. Deriving it lazily
                // rolled the event's stream on a mask READ and could show a different
                // three than the ones the player is looking at -- Tinker Time's bug.
                ColorfulPhilosopherPools(state);
                break;
            case RunConstants.EventFakeMerchant:
                // `BeforeEventStarted` shuffles the nine fakes and takes six, so the
                // shelf exists before the player has done anything -- and prices each of the
                // six as its entry is built, off the Shops stream. Pricing them drew nothing
                // before, so Shops stood six behind the game's after every Fake Merchant (run
                // referee, seed CNV3YDEJXD).
                state.EventRandomOffer = FakeMerchantStock(state)
                    .Select(_ => (int)Math.Round((float)RunConstants.FakeMerchantRelicCost
                        * state.PlayerRng.Shops.NextFloat(0.85f, 1.15f)))
                    .ToArray();
                break;
        }
    }

    /// <summary>
    /// The gold the statue's pool pays. <c>SunkenStatue.CalculateVars</c> takes a
    /// GoldVar of 111 and adds <c>Rng.NextInt(-10, 11)</c> from the event's own stream,
    /// so the amount is 101..121 and is rolled when the event is generated -- before
    /// either option is taken, which is why it cannot be rolled inside the option.
    /// </summary>
    public static int SunkenStatueGold(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateSunkenStatueVars(state);
        }

        return state.EventValue0!.Value;
    }

    private static void CalculateSunkenStatueVars(RunState state)
    {
        state.EventValue0 = 111 + EventRng(state, "SUNKEN_STATUE").NextInt(-10, 11);
    }

    /// <summary>
    /// What trudging through the vegetation pays: <c>Rng.NextInt(61, 100)</c>, rolled in
    /// CalculateVars rather than in the option.
    /// </summary>
    public static int DenseVegetationGold(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateDenseVegetationVars(state);
        }

        return state.EventValue0!.Value;
    }

    private static void CalculateDenseVegetationVars(RunState state)
    {
        state.EventValue0 = EventRng(state, "DENSE_VEGETATION").NextInt(61, 100);
    }

    /// <summary>
    /// Trash Heap's two prize tables, transcribed from the event's own literal arrays.
    /// The game lists them by class, so they are resolved by name here rather than by
    /// number: an id that stops resolving throws instead of silently picking whatever
    /// now sits at that index.
    /// </summary>
    private static readonly string[] TrashHeapRelicNames =
    [
        "DarkstonePeriapt",
        "DreamCatcher",
        "HandDrill",
        "MawBank",
        "TheBoot",
    ];

    private static readonly string[] TrashHeapCardNames =
    [
        "Caltrops",
        "Clash",
        "Distraction",
        "DualWield",
        "Entrench",
        "HelloWorld",
        "Outmaneuver",
        "Rebound",
        "RipAndTear",
        "Stack",
    ];

    public static int TrashHeapRelic(RunState state) =>
        ResolveRelic(EventRng(state, "TRASH_HEAP").NextItem(TrashHeapRelicNames));

    public static int TrashHeapCard(RunState state) =>
        ResolveCard(EventRng(state, "TRASH_HEAP").NextItem(TrashHeapCardNames));

    /// <summary>
    /// A relic an event names outright. Most events that hand over a relic hand over a
    /// SPECIFIC one -- <c>RelicCmd.Obtain&lt;ChosenCheese&gt;</c> -- and the emulator was
    /// rolling one from the reward pool at every such site, which is a different relic
    /// and also burns a draw the game never makes.
    /// </summary>
    public static int NamedRelic(string name) => ResolveRelic(name);

    /// <summary>A card an event names outright -- Wood Carvings' Peck and Toric Toughness.</summary>
    public static int NamedCard(string name) => ResolveCard(name);

    /// <summary>Byrdonis Nest's quest card, which is what puts HATCH on a rest site.</summary>
    public static int ByrdonisEggCard => ResolveCard("ByrdonisEgg");

    /// <summary>What the egg becomes: a 0-cost, 14-damage Attack.</summary>
    public static int ByrdSwoopCard => ResolveCard("ByrdSwoop");

    /// <summary>Circlet, the fallback relic -- the only stackable one in the game.</summary>
    public static int CircletRelic => ResolveRelic("Circlet");

    /// <summary>
    /// The three relics the trader is willing to take, in the order it offers them:
    /// <c>player.Relics.Where(IsTradable).StableShuffle(Rng).Take(3)</c>. StableShuffle
    /// sorts by ModelId -- Category then Entry, as ordinal strings -- before Fisher-Yates,
    /// which is why RelicDef carries an Entry now: our own numeric ids sort differently,
    /// and a different pre-shuffle order is a different shuffle from the same stream.
    ///
    /// Returns indices into <c>state.Relics</c>, because the trade removes the relic the
    /// player owns rather than a copy.
    /// </summary>
    public static List<int> RelicTraderStock(RunState state)
    {
        state.EventRelicStock ??= RollRelicTraderStock(state);
        return state.EventRelicStock;
    }

    /// <summary>
    /// Colorful Philosophers' option list: which of the four candidate pools survived the
    /// cut, as indices into <see cref="ColorfulPhilosopherPoolOrder"/>.
    /// </summary>
    /// <remarks>
    /// `CardPoolColorOrder` is Necrobinder, Ironclad, Regent, Silent, Defect, filtered by
    /// `character.CardPool != cardPool` -- which for the emulator's Ironclad leaves four
    /// -- and then `while (list.Count > 3) list.RemoveAt(Rng.NextInt(list.Count))` drops
    /// one. Removing is not taking three at random: it is ONE draw rather than three, and
    /// the survivors keep the colour order rather than a shuffled one, so the same seed
    /// gives a different set under either reading.
    /// </remarks>
    public static int[] ColorfulPhilosopherPools(RunState state)
    {
        if (state.EventRandomOffer.Length > 0)
        {
            return state.EventRandomOffer;
        }

        var candidates = Enumerable
            .Range(0, ColorfulPhilosopherPoolOrder(state.Character).Length)
            .ToList();
        var rng = EventRng(state, "COLORFUL_PHILOSOPHERS");
        while (candidates.Count > 3)
        {
            candidates.RemoveAt(rng.NextInt(candidates.Count));
        }

        state.EventRandomOffer = candidates.ToArray();
        return state.EventRandomOffer;
    }

    /// <summary>
    /// The four pools the event can offer, in `CardPoolColorOrder` (Necrobinder, Ironclad,
    /// Regent, Silent, Defect) minus the player's own.
    /// </summary>
    public static int[][] ColorfulPhilosopherPoolOrder(PlayerCharacter own) =>
        [
            .. new[]
            {
                PlayerCharacter.Necrobinder,
                PlayerCharacter.Ironclad,
                PlayerCharacter.Regent,
                PlayerCharacter.Silent,
                PlayerCharacter.Defect,
            }
                .Where(character => character != own)
                .Select(character => CharacterPools.Cards(character).ToArray()),
        ];

    /// <summary>
    /// The Fake Merchant's stall: six of the nine fake relics, in the order it shuffled
    /// them. A slot the player has bought is set to 0 rather than removed, so the indices
    /// stay put for the whole visit -- and the still-stocked ones are what the fight pays
    /// out if the player throws a Foul Potion instead.
    /// </summary>
    /// <remarks>
    /// <c>BeforeEventStarted</c> builds this ONCE, so it does not reroll when the player
    /// looks again -- the Relic Trader's shelf rule, and the reason both live on
    /// <c>EventRelicStock</c>. Fake Merchant's Rug is deliberately NOT in the list: the
    /// game's own comment says it is a combat reward, not something on sale.
    /// </remarks>
    public static List<int> FakeMerchantStock(RunState state)
    {
        state.EventRelicStock ??= RollFakeMerchantStock(state);
        return state.EventRelicStock;
    }

    /// <summary>
    /// The stall's price for slot <paramref name="slot"/>: each `MerchantRelicEntry` prices its
    /// relic as it is built (`SetModel` -> `CalcCost`), `round(MerchantCost * Shops.NextFloat(0.85,
    /// 1.15))`, and every fake relic's MerchantCost is 50 (FakeMerchant.relicCost). Rolled with the
    /// stall on entering, in stall order (<see cref="CalculateEventVars"/>).
    /// </summary>
    public static int FakeMerchantPrice(RunState state, int slot) =>
        state.EventId == RunConstants.EventFakeMerchant && (uint)slot < (uint)state.EventRandomOffer.Length
            ? state.EventRandomOffer[slot]
            : RunConstants.FakeMerchantRelicCost;

    private static List<int> RollFakeMerchantStock(RunState state)
    {
        // `_inventoryRelics` is a fixed array in the game's own order, which is
        // alphabetical by class name; UnstableShuffle then permutes it in place and the
        // first six are the stall.
        var fakes = new List<int>
        {
            ResolveRelic("FakeAnchor"),
            ResolveRelic("FakeBloodVial"),
            ResolveRelic("FakeHappyFlower"),
            ResolveRelic("FakeLeesWaffle"),
            ResolveRelic("FakeMango"),
            ResolveRelic("FakeOrichalcum"),
            ResolveRelic("FakeSneckoEye"),
            ResolveRelic("FakeStrikeDummy"),
            ResolveRelic("FakeVenerableTeaSet"),
        };
        EventRng(state, "FAKE_MERCHANT").Shuffle(fakes);
        return fakes.Take(RunConstants.FakeMerchantInventorySize).ToList();
    }

    /// <summary>
    /// <c>RelicTrader._ownedRelics</c> is built once and kept: the shelf does not reshuffle
    /// itself every time the player looks at it.
    /// </summary>
    private static List<int> RollRelicTraderStock(RunState state)
    {
        var tradable = Enumerable
            .Range(0, state.Relics.Count)
            .Where(i => IsTradableRelic(state, state.Relics[i]))
            .OrderBy(
                i => GeneratedData.Relics.Get(state.Relics[i].DefId).Entry,
                StringComparer.Ordinal
            )
            .ToList();
        EventRng(state, "RELIC_TRADER").Shuffle(tradable);
        return tradable.Take(3).ToList();
    }

    /// <summary>
    /// The relic Ranwid asks for: <c>Rng.NextItem(Relics.Where(IsTradable))</c> off the
    /// event's own stream. Returns an index into <c>state.Relics</c>, or -1 when the run
    /// holds nothing the elder will take.
    /// </summary>
    public static int RanwidTradeIndex(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateRanwidVars(state);
        }

        return state.EventValue0!.Value;
    }

    /// <summary>
    /// The belt slot of the potion an event picked with <c>Rng.NextItem(Owner.Potions)</c> while it
    /// built its options: Ranwid the Elder's to give, the Stone of All Time's to drink. Not the first
    /// one held. -1 when the belt was empty.
    /// </summary>
    public static int EventPotionSlot(RunState state)
    {
        if (state.EventValue1 is null)
        {
            if (state.EventId == RunConstants.EventRanwidTheElder)
            {
                CalculateRanwidVars(state);
            }
            else
            {
                CalculateStoneOfAllTimeVars(state);
            }
        }

        int pick = state.EventValue1!.Value;
        for (int slot = 0; slot < state.PotionSlots.Length; slot++)
        {
            if (state.PotionSlots[slot] != 0 && pick-- == 0)
            {
                return slot;
            }
        }

        return -1;
    }

    /// <summary>
    /// <c>StoneOfAllTime.GenerateInitialOptions</c>: <c>DrinkAndLiftPotion = Rng.NextItem(Owner.Potions)</c>,
    /// one draw off the event's own stream, none on an empty belt.
    /// </summary>
    private static void CalculateStoneOfAllTimeVars(RunState state)
    {
        int potions = state.PotionSlots.Count(slot => slot != 0);
        state.EventValue1 = potions > 0 ? EventRng(state, "STONE_OF_ALL_TIME").NextInt(0, potions) : -1;
    }

    /// <summary>
    /// <c>RanwidTheElder.GenerateInitialOptions</c> draws twice: a potion, then a relic.
    /// </summary>
    /// <remarks>
    /// The potion draw was never spent here, so the relic was read one position early
    /// whenever the run held a potion. <c>Rng.NextItem</c> returns default and consumes
    /// NOTHING on an empty sequence, and <c>Player.Potions</c> filters empty slots out --
    /// so a run carrying no potions really does draw the relic first.
    /// </remarks>
    private static void CalculateRanwidVars(RunState state)
    {
        var rng = EventRng(state, "RANWID_THE_ELDER");
        int potions = state.PotionSlots.Count(slot => slot != 0);
        state.EventValue1 = potions > 0 ? rng.NextInt(0, potions) : -1;

        var tradable = Enumerable
            .Range(0, state.Relics.Count)
            .Where(i => IsTradableRelic(state, state.Relics[i]))
            .ToList();
        state.EventValue0 = tradable.Count == 0 ? -1 : rng.NextItem(tradable);
    }

    /// <summary>
    /// <c>RelicModel.IsTradable</c>: Starter, Event and Ancient relics are never traded,
    /// nor is one whose pickup already paid out, nor one with a pet attached.
    /// </summary>
    public static bool IsTradableRelic(RunState state, RelicInstance relic) =>
        GeneratedData.Relics.Get(relic.DefId).IsTradable
        && !state.UsedUpRelics.Contains(relic.DefId);

    /// <summary>
    /// A curse rolled the way anything that hands one out rolls it: the curse pool,
    /// filtered to cards that may be generated -- eight of the eighteen refuse -- ordered
    /// by ModelId, and picked off the Niche stream.
    ///
    /// Ascender's Bane stood here for every such roll, which is not merely the wrong curse:
    /// it is one of the eight the game will never generate.
    /// </summary>
    public static int RollGeneratableCurse(RunState state)
    {
        var curses = GeneratedData
            .CardPools.Curse.ToArray()
            .Where(id => GeneratedData.Cards.Get(id).CanBeGeneratedByModifiers)
            .OrderBy(id => GeneratedData.Cards.Get(id).Entry, StringComparer.Ordinal)
            .ToArray();
        return curses.Length == 0
            ? RunConstants.CursePlaceholderCard
            : state.Rng.Niche.NextItem(curses);
    }

    /// <summary>A potion an event names outright -- the Potion Courier's Foul Potions.</summary>
    public static int NamedPotion(string name) =>
        GeneratedData.Potions.FindId(name)
        ?? throw new InvalidOperationException($"No potion named {name}");

    /// <summary>
    /// What Whispering Hollow charges for its two potions: a GoldVar of 35 shifted by
    /// <c>Rng.NextInt(-9, 10)</c> in CalculateVars, so 26..44 -- and it is SPENT, not paid
    /// out. The event only appears at all when the run holds 44 gold.
    /// </summary>
    public static int WhisperingHollowGold(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateWhisperingHollowVars(state);
        }

        return state.EventValue0!.Value;
    }

    private static void CalculateWhisperingHollowVars(RunState state)
    {
        state.EventValue0 = 35 + EventRng(state, "WHISPERING_HOLLOW").NextInt(-9, 10);
    }

    /// <summary>
    /// Jungle Maze's two purses. CalculateVars shifts each by <c>Rng.NextFloat(-15f, 15f)</c>
    /// off the same stream, solo first, and DynamicVar.IntValue truncates the decimal.
    /// </summary>
    public static int JungleMazeSoloGold(RunState state)
    {
        EnsureJungleMazeVars(state);
        return state.EventValue0!.Value;
    }

    public static int JungleMazeJoinForcesGold(RunState state)
    {
        EnsureJungleMazeVars(state);
        return state.EventValue1!.Value;
    }

    private static void EnsureJungleMazeVars(RunState state)
    {
        if (state.EventValue0 is null || state.EventValue1 is null)
        {
            CalculateJungleMazeVars(state);
        }
    }

    /// <summary>
    /// This or That's purse: <c>Rng.NextInt(41, 69)</c>, rolled in CalculateVars.
    /// </summary>
    public static int ThisOrThatGold(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateThisOrThatVars(state);
        }

        return state.EventValue0!.Value;
    }

    private static void CalculateThisOrThatVars(RunState state)
    {
        state.EventValue0 = EventRng(state, "THIS_OR_THAT").NextInt(41, 69);
    }

    /// <summary>
    /// What one immersion in the Abyssal Baths costs. <c>OnImmerse</c> gains MaxHpVar(2),
    /// takes DamageVar and then raises that damage by one, so the sequence is 3, 4, 5 and
    /// on -- and the Max HP arrives first, carrying current HP up with it, which is why
    /// the first dip is a net loss of only one.
    /// </summary>
    public static void Immerse(RunState state)
    {
        GainMaxHp(state, 2);
        state.PlayerHp = Math.Max(0, state.PlayerHp - AbyssalBathsDamage(state));
        state.EventPage++;
    }

    /// <summary>
    /// What the next Decipher costs at the Tablet of Truth. The tablet is not one choice
    /// but five, and the price doubles each time before the last one asks for everything:
    /// 3, 6, 12, 24, then MaxHp - 1. <c>GetDecipherCost</c> reads the count AFTER it moves,
    /// which is why the first is the DynamicVar's own 3 rather than anything the switch
    /// returns.
    /// </summary>
    public static int TabletOfTruthCost(RunState state) =>
        state.EventPage switch
        {
            0 => 3,
            1 => 6,
            2 => 12,
            3 => 24,
            _ => state.PlayerMaxHp - 1,
        };

    /// <summary>
    /// One Decipher: pay the Max HP and upgrade. Every stage but the last upgrades ONE
    /// card rolled off the event's stream; the fifth upgrades every upgradable card in the
    /// deck, which is the whole reason to have kept paying.
    ///
    /// <c>LoseMaxHpAndUpgrade</c> also kills outright when the price is not less than Max
    /// HP -- it takes MaxHp - 1 and then calls Kill -- so a tablet the run cannot afford
    /// ends it.
    /// </summary>
    public static void Decipher(RunState state)
    {
        int cost = TabletOfTruthCost(state);
        bool lethal = cost >= state.PlayerMaxHp;
        LoseMaxHp(state, lethal ? state.PlayerMaxHp - 1 : cost);
        if (lethal)
        {
            state.PlayerHp = 0;
            return;
        }

        if (state.EventPage == 4)
        {
            for (int i = 0; i < state.Deck.Count; i++)
            {
                if (RunConstants.IsRunCardUpgradable(state.Deck[i]))
                {
                    state.Deck[i] = state.Deck[i] with { Upgraded = true };
                }
            }
        }
        else
        {
            UpgradeRandomCard(state, "TABLET_OF_TRUTH");
        }

        state.EventPage++;
    }

    /// <summary>The tablet has five secrets and no more.</summary>
    public static bool TabletHasMoreToSay(RunState state) => state.EventPage < 5;

    /// <summary>The damage the NEXT immersion will do: DamageVar(3), plus one per dip.</summary>
    public static int AbyssalBathsDamage(RunState state) => 3 + state.EventPage;

    /// <summary>
    /// What the next Hold On costs: <c>CurrentHpLoss => 3 + NumberOfHoldOns</c>, read
    /// before the counter moves.
    /// </summary>
    public static int SlipperyBridgeHpLoss(RunState state) => 3 + state.EventPage;

    /// <summary>
    /// The card the Slippery Bridge is holding over the player. The first roll prefers a
    /// non-Basic card; every re-roll after a Hold On works from the whole removable deck, so
    /// it is a roll, not a fixed pick.
    /// </summary>
    public static int SlipperyBridgeCardIndex(RunState state)
    {
        if (state.EventValue0 is null)
        {
            RollSlipperyBridgeCard(state);
        }

        return state.EventValue0!.Value;
    }

    /// <summary>
    /// <c>SlipperyBridge.GetNewRandomCard</c>: called once from GenerateInitialOptions and
    /// again from every Hold On, storing the result in <c>RandomCardToLose</c>. Recomputing
    /// it per read drew a new card each time the mask or a test looked at it.
    /// </summary>
    public static void RollSlipperyBridgeCard(RunState state) =>
        state.EventValue0 = PickSlipperyBridgeCard(state);

    private static int PickSlipperyBridgeCard(RunState state)
    {
        // `SlipperyBridge.GetNewRandomCard`. The first roll takes the deck's non-Basic cards.
        // Every later one (after a Hold On) first notes the card it is leaving as skipped, then
        // takes every card of a DIFFERENT card (the game compares model types, i.e. ids) that has
        // not been skipped -- Basic cards included. Either way it drops the cards that cannot be
        // removed, falls back to every removable card when nothing is left, and draws ONCE.
        bool first = state.EventValue0 is null or < 0;
        var skipped = state.EventSkippedCards;
        int heldId = -1;
        if (!first)
        {
            skipped.Add(state.EventValue0!.Value);
            heldId = state.Deck[state.EventValue0.Value].DefId;
        }

        var candidates = Enumerable
            .Range(0, state.Deck.Count)
            .Where(i =>
                IsRemovable(state.Deck[i])
                && !skipped.Contains(i)
                && (
                    first
                        ? GeneratedData.Cards.Get(state.Deck[i].DefId).Rarity != CardRarity.Basic
                        : state.Deck[i].DefId != heldId
                )
            )
            .ToList();
        if (candidates.Count == 0)
        {
            candidates = Enumerable
                .Range(0, state.Deck.Count)
                .Where(i => IsRemovable(state.Deck[i]))
                .ToList();
        }

        if (candidates.Count == 0)
        {
            return -1;
        }

        return EventRng(state, "SLIPPERY_BRIDGE").NextItem(candidates);
    }

    /// <summary>
    /// <c>CardModel.IsRemovable</c> is <c>!Keywords.Contains(Eternal)</c> and nothing else:
    /// every Eternal card stays in the deck whatever removes from it (Ascender's Bane and
    /// the other Eternal curses), while quest cards are ordinary removable cards.
    /// </summary>
    private static bool IsRemovable(CardInstance card) =>
        !GeneratedData.Cards.Get(card.DefId).Eternal;

    /// <summary>
    /// <c>FishingRod.AfterCombatEnd</c>: every third MONSTER room upgrades a card.
    /// </summary>
    /// <remarks>
    /// Elites, bosses and the fights an event starts are all skipped -- the relic returns
    /// early on anything that is not <c>RoomType.Monster</c>, so they do not even advance
    /// the counter. The card is a plain <c>Rng.Niche.NextItem</c> over the upgradable
    /// ones, and the counter lives on the relic so it survives a save.
    /// </remarks>
    /// <summary>
    /// `Guilty.AfterCombatEnd`: a Guilty sitting in the DECK counts the combat, and at
    /// FIVE it removes itself. The only card in the game that leaves on its own.
    /// </summary>
    /// <remarks>
    /// Counted per COPY -- the count is `[SavedProperty]` on the card model, so two
    /// Guilties taken at different times serve their sentences separately. `Pile.Type ==
    /// Deck` is checked twice in the source, before and after the increment, because the
    /// removal is awaited; here the list is rebuilt against the deck each time, which
    /// says the same thing.
    /// </remarks>
    public static void ServeGuiltySentences(RunState state)
    {
        int guilty = ResolveCard("Guilty");
        int copies = state.Deck.Count(card => card.DefId == guilty);
        if (copies == 0)
        {
            state.GuiltyCombatsServed.Clear();
            return;
        }

        while (state.GuiltyCombatsServed.Count < copies)
        {
            state.GuiltyCombatsServed.Add(0);
        }

        while (state.GuiltyCombatsServed.Count > copies)
        {
            state.GuiltyCombatsServed.RemoveAt(state.GuiltyCombatsServed.Count - 1);
        }

        for (int i = 0; i < state.GuiltyCombatsServed.Count; i++)
        {
            state.GuiltyCombatsServed[i]++;
        }

        int served = state.GuiltyCombatsServed.Count(count => count >= GuiltyCombats);
        for (int i = 0; i < served; i++)
        {
            int index = state.Deck.FindIndex(card => card.DefId == guilty);
            if (index >= 0)
            {
                state.Deck.RemoveAt(index);
            }
        }

        state.GuiltyCombatsServed.RemoveAll(count => count >= GuiltyCombats);
    }

    /// <summary>`Guilty.maxCombats`.</summary>
    private const int GuiltyCombats = 5;

    public static void TriggerFishingRod(RunState state)
    {
        if (state.LastResolvedRoomType != RunConstants.NodeNormal)
        {
            return;
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == RunConstants.RelicFishingRod);
        if (index < 0)
        {
            return;
        }

        int seen = state.Relics[index].Counter + 1;
        state.Relics[index] = state.Relics[index] with { Counter = seen };
        if (seen % RunConstants.FishingRodCombats != 0)
        {
            return;
        }

        var upgradable = Enumerable
            .Range(0, state.Deck.Count)
            .Where(i => RunConstants.IsRunCardUpgradable(state.Deck[i]))
            .ToList();
        if (upgradable.Count == 0)
        {
            return;
        }

        int pick = state.Rng.Niche.NextItem(upgradable);
        state.Deck[pick] = state.Deck[pick] with { Upgraded = true };
    }

    /// <summary>
    /// <c>WelcomeToWongos.Leave</c>: <c>base.Rng.NextItem(Deck.Where(IsUpgraded))</c> and
    /// <c>CardCmd.Downgrade</c>. No upgraded card, no pick (an empty list is no draw).
    /// </summary>
    public static bool DowngradeRandomUpgradedCard(RunState state, string eventEntry)
    {
        var candidates = Enumerable
            .Range(0, state.Deck.Count)
            .Where(i => state.Deck[i].Upgraded)
            .ToList();
        if (candidates.Count == 0)
        {
            return false;
        }

        int index = EventRng(state, eventEntry).NextItem(candidates);
        state.Deck[index] = state.Deck[index] with { Upgraded = false };
        return true;
    }

    /// <summary>Upgrade one upgradable card, chosen off the event's own stream.</summary>
    public static bool UpgradeRandomCard(RunState state, string eventEntry)
    {
        var candidates = Enumerable
            .Range(0, state.Deck.Count)
            .Where(i => RunConstants.IsRunCardUpgradable(state.Deck[i]))
            .ToList();
        if (candidates.Count == 0)
        {
            return false;
        }

        int index = EventRng(state, eventEntry).NextItem(candidates);
        state.Deck[index] = state.Deck[index] with { Upgraded = true };
        return true;
    }

    private static void CalculateJungleMazeVars(RunState state)
    {
        var rng = EventRng(state, "JUNGLE_MAZE_ADVENTURE");
        state.EventValue0 = (int)(150m + (decimal)rng.NextFloat(-15f, 15f));
        state.EventValue1 = (int)(50m + (decimal)rng.NextFloat(-15f, 15f));
    }

    /// <summary>Doll Room's three dolls, in the event's own declaration order.</summary>
    private static readonly string[] DollRoomDolls =
    [
        "DaughterOfTheWind",
        "MrStruggles",
        "BingBong",
    ];

    public static int DollRoomRandomDoll(RunState state) =>
        ResolveRelic(EventRng(state, "DOLL_ROOM").NextItem(DollRoomDolls));

    /// <summary>
    /// The dolls the second page puts in front of the player: the three shuffled, then
    /// two of them for Take Some Time and all three for Examine. StableShuffle sorts by
    /// ModelId -- Category then Entry as ordinal strings -- before Fisher-Yates, which is
    /// what RelicDef.Entry is for.
    /// </summary>
    public static List<int> DollRoomOffer(RunState state, int count)
    {
        var dolls = DollRoomDolls
            .Select(ResolveRelic)
            .OrderBy(id => GeneratedData.Relics.Get(id).Entry, StringComparer.Ordinal)
            .ToList();
        EventRng(state, "DOLL_ROOM").Shuffle(dolls);
        return dolls.Take(count).ToList();
    }

    /// <summary>
    /// The Rare relic Wongo has on display. It is pulled in GenerateInitialOptions -- when
    /// the event OPENS, not when it is bought -- so the option can name it, and it leaves
    /// the bag either way. Cached in EventValue1 so reading it twice does not pull twice.
    /// </summary>
    public static int WongosFeaturedItem(RunState state)
    {
        state.EventValue1 ??= RunRewardGenerator.NextShopRelicOfRarity(state, RelicRarity.Rare);
        return state.EventValue1.Value;
    }

    /// <summary>The Lantern Key card, which War Historian Repy spends to open anything.</summary>
    public static int LanternKeyCard => ResolveCard("LanternKey");

    /// <summary>
    /// Repy gives a SECOND reward when the run still holds a Lantern Key after the first
    /// is spent -- <c>ShouldGetSecondReward</c> -- which for a solo run means it arrived
    /// holding two. The first choice spends one key; the page that follows offers only
    /// the door the player did not take.
    /// </summary>
    public static bool RepyOwesASecondReward(RunState state) =>
        state.Deck.Any(card => card.DefId == LanternKeyCard);

    /// <summary>Spends one Lantern Key, if the run has one to spend.</summary>
    public static void SpendLanternKey(RunState state)
    {
        int index = state.Deck.FindIndex(card => card.DefId == LanternKeyCard);
        if (index >= 0)
        {
            state.Deck.RemoveAt(index);
        }
    }

    private static int ResolveRelic(string name) =>
        GeneratedData.Relics.FindId(name)
        ?? throw new InvalidOperationException($"No relic named {name}");

    private static int ResolveCard(string name) =>
        GeneratedData.Cards.FindId(name)
        ?? throw new InvalidOperationException($"No card named {name}");

    /// <summary>
    /// What Spiraling Whirlpool's Drink heals: a HealVar whose BaseValue is
    /// <c>MaxHp * 0.33m</c>, read through <c>DynamicVar.IntValue</c>, which is a plain
    /// <c>(int)</c> cast and therefore truncates.
    /// </summary>
    public static int SpiralingWhirlpoolHeal(RunState state) => (int)(state.PlayerMaxHp * 0.33m);

    public static int FresnelLensRelic => ResolveRelic("FresnelLens");

    public static int SwordOfStoneRelic => ResolveRelic("SwordOfStone");

    public static int GlowwaterPotion =>
        GeneratedData.Potions.FindId("GlowwaterPotion")
        ?? throw new InvalidOperationException("No potion named GlowwaterPotion");

    // ── Endless Conveyor's belt ───────────────────────────────────────────────
    // The dish is rolled before the option is even shown -- the option reads "Grab
    // Fried Eel off the Belt" -- and rolled again after every grab, so the belt is a
    // small state machine rather than a one-off. EventValue0 carries the dish now on
    // the belt and EventValue1 how many grabs have happened.

    public const int DishCaviar = 1;
    public const int DishSpicySnappy = 2;
    public const int DishJellyLiver = 3;
    public const int DishFriedEel = 4;
    public const int DishSuspiciousCondiment = 5;
    public const int DishClamRoll = 6;
    public const int DishGoldenFysh = 7;
    public const int DishSeapunkSalad = 8;

    /// <summary>The dish currently on the belt.</summary>
    public static int EndlessConveyorDish(RunState state)
    {
        if (state.EventValue0 is null)
        {
            CalculateEndlessConveyorVars(state);
        }

        return state.EventValue0!.Value;
    }

    private static void CalculateEndlessConveyorVars(RunState state)
    {
        state.EventValue1 = 0;
        RollDish(state);
    }

    /// <summary>
    /// EndlessConveyor.RollDish: every fifth grab is a Seapunk Salad outright, and
    /// otherwise a weighted pick over the dishes the run currently qualifies for, minus
    /// whatever was on the belt a moment ago.
    /// </summary>
    /// <remarks>
    /// The forced fifth returns before the roll, so it consumes nothing -- which is why
    /// the stream position is the grab count less the fifths, not the grab count.
    /// </remarks>
    /// <summary>Turn the belt after a grab.</summary>
    public static void RollNextConveyorDish(RunState state) => RollDish(state);

    private static void RollDish(RunState state)
    {
        int grabs = (state.EventValue1 ?? 0) + 1;
        state.EventValue1 = grabs;
        if (grabs % 5 == 0)
        {
            state.EventValue0 = DishSeapunkSalad;
            return;
        }

        int lastDish = state.EventValue0 ?? 0;
        var dishes = new List<(int Dish, float Weight)>
        {
            (DishCaviar, 6f),
            (DishSpicySnappy, 3f),
            (DishJellyLiver, 3f),
            (DishFriedEel, 3f),
        };
        if (state.UsablePotionSlots().Any(potion => potion == 0))
        {
            dishes.Add((DishSuspiciousCondiment, 3f));
        }

        if (state.PlayerHp != state.PlayerMaxHp)
        {
            dishes.Add((DishClamRoll, 6f));
        }

        if (grabs > 1)
        {
            dishes.Add((DishGoldenFysh, 1f));
        }

        dishes.RemoveAll(dish => dish.Dish == lastDish);

        // The event's stream is persistent now, so it is already where this roll belongs.
        // This used to fast-forward a freshly built stream to grabs-minus-the-fifths --
        // the fifth grab returns above without drawing -- which became a REWIND as soon
        // as anything else drew from the event, and threw.
        var rng = EventRng(state, "ENDLESS_CONVEYOR");

        float total = dishes.Sum(dish => dish.Weight);
        float roll = (float)rng.NextDouble() * total;
        float running = 0f;
        foreach (var (dish, weight) in dishes)
        {
            running += weight;
            if (roll < running)
            {
                state.EventValue0 = dish;
                return;
            }
        }

        state.EventValue0 = dishes[^1].Dish;
    }

    /// <summary>
    /// Eat what was grabbed, then let the belt turn. Returns false when the dish needs a
    /// screen the caller has to open.
    /// </summary>
    public static void ApplyEndlessConveyorDish(RunState state, int dish)
    {
        switch (dish)
        {
            case DishClamRoll:
                state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 10);
                break;
            case DishCaviar:
                GainMaxHp(state, 4);
                break;
            case DishGoldenFysh:
                GainGold(state, 75);
                break;
            case DishSeapunkSalad:
                AddCardToDeck(state, new CardInstance(NamedCard("FeedingFrenzy"), Upgraded: false));
                break;
            case DishFriedEel:
                // `CardFactory.CreateForReward(owner, 1, ForNonCombatWithDefaultOdds(Colorless))`
                // and the card straight into the deck.
                RunRewardGenerator.AddRewardCardToDeck(
                    state,
                    RunRewardGenerator.CreateForReward(
                        state,
                        1,
                        RunRewardGenerator.ColorlessRewardPool,
                        CardCreationFlags.NoUpgradeRoll
                    ),
                    0
                );
                break;
            case DishSpicySnappy:
                UpgradeRandomCard(state, "ENDLESS_CONVEYOR");
                break;
            case DishSuspiciousCondiment:
                state.PendingPotionRewards.Add(
                    RunRewardGenerator.NextPoolPotion(state, state.PlayerRng.Rewards)
                );
                break;
            default:
                break;
        }
    }

    private static void CalculateSunkenTreasuryVars(RunState state)
    {
        GameRng rng = EventRng(state, "SUNKEN_TREASURY");
        state.EventValue0 = 60 + rng.NextInt(16) - 8;
        state.EventValue1 = 333 + rng.NextInt(61) - 30;
    }

    /// <summary>
    /// An event's own stream. EventModel seeds it with
    /// <c>Rng.Seed + (IsShared ? 0 : GetPlayerSlotIndex(Owner)) + hash(Id.Entry)</c>,
    /// and a solo run's only player is slot 0 either way -- so the term is zero, not
    /// one. It was one here, which is the same off-by-one that had Neow offering the
    /// wrong relics and the player rng set reading the wrong stream. Sunken Treasury's
    /// chests paid 67 and 343 where the game pays 63 and 340.
    /// </summary>
    /// <summary>An event's own stream, for callers outside this class.</summary>
    public static GameRng EventStream(RunState state, string eventEntry) =>
        EventRng(state, eventEntry);

    /// <summary>
    /// The event's own Rng stream -- one per event, not one per draw.
    /// </summary>
    /// <remarks>
    /// An event holds a single <c>base.Rng</c>, so its draws are a sequence: the Endless
    /// Conveyor rolls its dish in CalculateVars and then picks the card Observe the Chef
    /// upgrades, and that pick reads the SECOND value. Building a fresh stream per call
    /// handed it the first, which upgraded the wrong card against a live capture. Keyed
    /// by entry so changing events starts a new stream rather than inheriting one.
    /// </remarks>
    private static GameRng EventRng(RunState state, string eventEntry)
    {
        if (state.EventRngStream is not null && state.EventRngName == eventEntry)
        {
            return state.EventRngStream;
        }

        uint eventSeed = unchecked(
            state.Rng.Seed + (uint)DeterministicHash.GetDeterministicHashCode(eventEntry)
        );
        state.EventRngName = eventEntry;
        state.EventRngStream = new GameRng(eventSeed);
        return state.EventRngStream;
    }

    /// <summary>
    /// An ancient's three blessings: one drawn from each of its pools, off the ancient's
    /// own Rng, in pool order.
    /// </summary>
    /// <remarks>
    /// All three of Hive's ancients share this shape and differ only in their pools and
    /// in which entries are CONDITIONAL on the run. Neow does not come through here — it
    /// has its own generator, and a different shape (a curse plus a shuffled positive
    /// list) rather than three pools.
    /// </remarks>
    public static int[] GenerateAncientOptions(RunState state, string ancient)
    {
        var rng = EventStream(state, ancient);
        return ancient switch
        {
            RunConstants.AncientDarv => DarvOptions(state, rng),
            RunConstants.AncientOrobas => OrobasOptions(state, rng),
            RunConstants.AncientPael => PaelOptions(state, rng),
            RunConstants.AncientTezcatara => TezcataraOptions(state, rng),
            RunConstants.AncientNonupeipe => ShuffledPoolOptions(
                RunConstants.NonupeipePool,
                state.Deck.Count(card => Enchantments.CanEnchant(card, Enchantment.Swift)) >= 4
                    ? RunConstants.RelicBeautifulBracelet
                    : null,
                rng
            ),
            RunConstants.AncientTanx => ShuffledPoolOptions(
                RunConstants.TanxPool,
                state.Deck.Count(card => Enchantments.CanEnchant(card, Enchantment.Instinct)) >= 3
                    ? RunConstants.RelicTriBoomerang
                    : null,
                rng
            ),
            RunConstants.AncientVakuu => VakuuOptions(rng),
            _ => [],
        };
    }

    /// <summary>
    /// Nonupeipe and Tanx: the pool in declaration order, the conditional blessing
    /// appended when its deck condition holds, one <c>UnstableShuffle</c>, the first three.
    /// </summary>
    private static int[] ShuffledPoolOptions(ReadOnlySpan<int> pool, int? conditional, GameRng rng)
    {
        var offered = pool.ToArray().ToList();
        if (conditional is int extra)
        {
            offered.Add(extra);
        }

        rng.Shuffle(offered);
        return [.. offered.Take(3)];
    }

    /// <summary>Vakuu: each of its three pools shuffled in turn, and the first of each.</summary>
    private static int[] VakuuOptions(GameRng rng)
    {
        var pools = new[]
        {
            RunConstants.VakuuPool1.ToArray().ToList(),
            RunConstants.VakuuPool2.ToArray().ToList(),
            RunConstants.VakuuPool3.ToArray().ToList(),
        };
        foreach (var pool in pools)
        {
            rng.Shuffle(pool);
        }

        return [pools[0][0], pools[1][0], pools[2][0]];
    }

    /// <summary>
    /// Orobas spends TWO draws before its pools: one picking a character other than the
    /// player's, to brand a Sea Glass with, and one deciding whether pool 1 gets the Sea
    /// Glass or a Prismatic Gem instead — <c>NextFloat() &lt; 1/3</c> for the gem.
    /// </summary>
    /// <remarks>
    /// The character draw's RESULT does not matter here (it only brands the relic, which
    /// the emulator does not model), but the draw does: skipping it would shift every pool
    /// pick after it. The list it chooses from is the unlocked characters minus the
    /// player's own, which is four on a mature profile.
    /// </remarks>
    private static int[] OrobasOptions(RunState state, GameRng rng)
    {
        // The character is kept, not just spent: if the Sea Glass is offered AND taken,
        // its cards come from that character's pool.
        state.SeaGlassCharacter = rng.NextInt(RunConstants.OtherCharacterCount);
        bool prismaticGem = rng.NextDouble() < 1.0 / 3.0;

        var pool1 = RunConstants.OrobasPool1.ToArray().ToList();
        pool1.Add(prismaticGem ? RunConstants.RelicPrismaticGemOption : RunConstants.RelicSeaGlass);

        // Pool 3 holds Touch of Orobas and Archaic Tooth, each present only if it can be
        // set up for the player -- one needs a starter relic, the other a transcendable
        // starter card, and the Ironclad has both.
        return
        [
            rng.NextItem(pool1),
            rng.NextItem(RunConstants.OrobasPool2.ToArray()),
            rng.NextItem(RunConstants.OrobasPool3.ToArray()),
        ];
    }

    /// <summary>
    /// Pael's second pool is a weighting trick: the conditional entries are added, then
    /// <c>list.AddRange(list)</c> DOUBLES everything so far, and only then is Growth
    /// appended — so Growth is half as likely as anything else in the pool.
    /// </summary>
    private static int[] PaelOptions(RunState state, GameRng rng)
    {
        var pool2 = RunConstants.PaelPool2.ToArray().ToList();
        if (state.Deck.Count(CanTakeAnEnchantment) >= 3)
        {
            pool2.Add(RunConstants.RelicPaelsClaw);
        }

        if (state.Deck.Count(IsRemovableCard) >= 5)
        {
            pool2.Add(RunConstants.RelicPaelsTooth);
        }

        pool2.AddRange(pool2);
        pool2.Add(RunConstants.RelicPaelsGrowth);

        var pool3 = RunConstants.PaelPool3.ToArray().ToList();
        // HasEventPet is not modelled, and no run has one yet.
        pool3.Add(RunConstants.RelicPaelsLegion);

        return
        [
            rng.NextItem(RunConstants.PaelPool1.ToArray()),
            rng.NextItem(pool2),
            rng.NextItem(pool3),
        ];
    }

    /// <summary>
    /// Tezcatara adds Nutritious Soup to its first pool when the deck holds a BASIC
    /// Strike — which every starting deck does, and keeps doing unless every Strike is
    /// removed or transformed.
    /// </summary>
    /// <summary>
    /// Darv: one relic from each set whose filter passes, shuffled, and the first three --
    /// or the first two plus Dusty Tome, on a coin flip.
    /// </summary>
    /// <remarks>
    /// The draw order is what makes this reproducible, and it is not obvious: ONE
    /// `NextItem` per surviving set FIRST (even for the seven sets that hold a single
    /// relic, which still spend a draw), then the shuffle, then `NextBool` for the tome.
    /// Reordering any of those gives a different three from the same seed.
    ///
    /// `UnstableShuffle` is Fisher-Yates over the list as built, so the set order in
    /// `DarvSingleRelicSets` is part of the answer.
    ///
    /// Pandora's Box drops out for a run whose modifiers clear the deck; the emulator
    /// models no such modifier, so its filter is always true here and is written as a
    /// comment rather than a condition that can never fire.
    /// </remarks>
    private static int[] DarvOptions(RunState state, GameRng rng)
    {
        var offered = new List<int>();
        foreach (int relicId in RunConstants.DarvSingleRelicSets)
        {
            // `Rng.NextItem` over a one-element array is still a draw.
            offered.Add(rng.NextItem(new[] { relicId }));
        }

        // `CurrentActIndex == 1` and `== 2`. The emulator runs act index 0, so neither
        // set's filter passes and neither spends a draw -- which is the whole reason the
        // act has to be asked rather than assumed.
        if (state.CurrentActIndex == 1)
        {
            offered.Add(rng.NextItem(RunConstants.DarvActOneSet.ToArray()));
        }

        if (state.CurrentActIndex == 2)
        {
            offered.Add(rng.NextItem(RunConstants.DarvActTwoSet.ToArray()));
        }

        rng.Shuffle(offered);

        if (rng.NextBool())
        {
            // `DustyTome.SetupForPlayer` runs as the option is BUILT, so its card is drawn off
            // `PlayerRng.Rewards` on entering Darv whether or not the tome is taken. Rolled at
            // pickup instead, a tome left on the table never drew, and the Rewards stream stayed
            // one behind (run referee, seed 6HMCK2BXTJ). The card waits in EventValue0, which
            // the ancient's screen does not otherwise use.
            state.EventValue0 = DustyTomeCard(state);
            return [.. offered.Take(2), RunConstants.DarvDustyTome];
        }

        return [.. offered.Take(3)];
    }

    private static int[] TezcataraOptions(RunState state, GameRng rng)
    {
        var pool1 = RunConstants.TezcataraPool1.ToArray().ToList();
        if (state.Deck.Any(IsBasicStrike))
        {
            pool1.Add(RunConstants.RelicNutritiousSoup);
        }

        return
        [
            rng.NextItem(pool1),
            rng.NextItem(RunConstants.TezcataraPool2.ToArray()),
            rng.NextItem(RunConstants.TezcataraPool3.ToArray()),
        ];
    }

    /// <summary>
    /// <c>c.Tags.Contains(CardTag.Strike) &amp;&amp; c.Rarity == Basic</c>. Card tags are
    /// not extracted, so this reads the entry name instead — every Strike-tagged card the
    /// game has is named for it, and at Basic rarity there is only the starter Strike.
    /// </summary>
    /// <summary>
    /// Nutritious Soup: `c.Rarity == Basic && c.Tags.Contains(CardTag.Strike)`. The tag,
    /// not a "STRIKE" substring in the entry -- the two agree among Basic cards, which is
    /// why the stand-in survived, and the tags are extracted now (E358).
    /// </summary>
    private static bool IsBasicStrike(CardInstance card)
    {
        var def = GeneratedData.Cards.Get(Math.Abs(card.DefId));
        return def.Rarity == CardRarity.Basic && def.StrikeTag;
    }

    /// <summary>
    /// `Goopy.CanEnchant`, which Pael's Claw counts against a threshold of three.
    /// </summary>
    /// <remarks>
    /// This used to be `not (Curse or Status)` behind a comment saying the emulator had no
    /// Goopy enchantment at all. It has one, and the comment outlived that: Goopy takes
    /// Defend-TAGGED cards, at any rarity, which is a far narrower set than "not a curse".
    /// The threshold made the difference unreachable -- every deck a run can hold clears
    /// three on its starting cards -- but an approximation nobody can trip is still an
    /// approximation, and it disagreed with the enchantment it was standing in for.
    /// </remarks>
    private static bool CanTakeAnEnchantment(CardInstance card) =>
        Enchantments.CanEnchant(card, Enchantment.Goopy);

    /// <summary>
    /// <c>CardModel.IsRemovable</c> (`!Eternal`) for a card whose id may carry a sign.
    /// </summary>
    private static bool IsRemovableCard(CardInstance card) =>
        !GeneratedData.Cards.Get(Math.Abs(card.DefId)).Eternal;

    /// <summary>
    /// Enchant every card in the deck that will take it — the shape Pael's Claw and
    /// Nutritious Soup share, where the relic offers no choice at all.
    /// </summary>
    private static void EnchantEveryCard(
        RunState state,
        Enchantment enchantment,
        int amount,
        Func<CardInstance, bool>? extraFilter = null
    )
    {
        for (int i = 0; i < state.Deck.Count; i++)
        {
            var card = state.Deck[i];
            if (!Enchantments.CanEnchant(card, enchantment))
            {
                continue;
            }

            if (extraFilter is not null && !extraFilter(card))
            {
                continue;
            }

            state.Deck[i] = card with { Enchantment = enchantment, EnchantAmount = amount };
        }
    }

    /// <summary>
    /// `PlayerCmd.GainGold`. The single place run-side gold is gained, because
    /// `AfterGoldGained` is a hook and a `Gold +=` cannot dispatch one.
    ///
    /// The early return is the game's, and it is load-bearing: the amount is modified
    /// FIRST and a non-positive result returns before `AfterGoldGained`. Ectoplasm zeroes
    /// the amount, so an Ectoplasm owner never fires the hook at all -- Dragon Fruit next
    /// to Ectoplasm gains no max HP, rather than gaining it off gold it did not receive.
    /// `LoseGold` is a different command and fires nothing.
    /// </summary>
    public static void GainGold(RunState state, int amount)
    {
        amount = Effects.RelicEffects.ModifyGoldGained(state.Relics, amount);
        if (amount <= 0)
        {
            return;
        }

        state.Gold += amount;
        Effects.RelicEffects.ApplyAfterGoldGained(state);
    }

    /// <summary>
    /// <c>OneOffSynchronizer.TryHandleSpoilsMap</c>, called right after the chest's own
    /// gold: when the point the run stands on carries the Spoils Map quest, every Spoils
    /// Map in the deck pays its 600 (<c>SpoilsMap.OnQuestComplete</c>, a
    /// <c>PlayerCmd.GainGold</c> like any other) and is removed from the deck.
    /// </summary>
    /// <remarks>
    /// The marker is on the map point, so it is the point that matters and not the room:
    /// a "?" that turns out to be a chest is some other point and pays nothing, and so
    /// does a chest the run reaches after the card has left the deck (the card's
    /// <c>BeforeCardRemoved</c> takes the marker off). Ending the quest clears the
    /// marker the same way.
    /// </remarks>
    public static void CompleteSpoilsMapQuest(RunState state)
    {
        if (state.SpoilsQuestCoord != state.CurrentMapCoord)
        {
            return;
        }

        for (int i = state.Deck.Count - 1; i >= 0; i--)
        {
            if (state.Deck[i].DefId == RunConstants.SpoilsMapCard)
            {
                GainGold(state, RunConstants.SpoilsMapGold);
                state.Deck.RemoveAt(i);
            }
        }

        state.SpoilsQuestCoord = null;
    }

    public static void GainMaxHp(RunState state, int amount)
    {
        state.PlayerMaxHp += amount;
        state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + amount);
    }

    /// <summary>
    /// The game's <c>CreatureCmd.LoseMaxHp</c>: current HP is only damaged by the amount
    /// the new maximum falls BELOW it, and the maximum itself never goes under 1. A
    /// player already below the new cap keeps the HP they had.
    /// </summary>
    public static void LoseMaxHp(RunState state, int amount)
    {
        int newMaxHp = Math.Max(1, state.PlayerMaxHp - amount);
        state.PlayerHp = Math.Min(state.PlayerHp, newMaxHp);
        state.PlayerMaxHp = newMaxHp;
    }

    /// <summary>
    /// Doors of Light and Dark's Light door:
    /// <c>Deck.Cards.Where(IsUpgradable).StableShuffle(base.Rng).Take(Cards)</c>.
    /// </summary>
    /// <remarks>
    /// Two things here are the whole of it, and the emulator had both wrong. The stream is
    /// the EVENT's own <c>base.Rng</c>, not <c>Rng.Niche</c> -- the same mistake as E14,
    /// where every event's transform was rolled off Niche. And <c>StableShuffle</c> sorts
    /// by ModelId before it shuffles, which compares the slugified class name as an
    /// ordinal string; sorting by the emulator's own numeric ids puts a different card
    /// under the same draw. A live capture upgraded Whirlwind where this took Shrug It Off.
    /// </remarks>
    /// <summary>
    /// <c>Deck.Where(IsUpgradable).StableShuffle(rng).Take(count)</c>, upgraded.
    /// </summary>
    /// <remarks>
    /// The two things that matter are the STREAM, which differs per caller, and the sort:
    /// <c>StableShuffle</c> orders by ModelId — the slugified class name, compared
    /// ordinally — before it shuffles, so sorting by the emulator's own numeric ids puts a
    /// different card under the same draw. Both were wrong in the Light Door once (E40).
    /// </remarks>
    /// <summary>
    /// `WarHammer.AfterCombatVictory(room.RoomType == Elite)`: `CardsVar(4)` upgradable
    /// deck cards, `StableShuffle(Rng.Niche).Take(4)`.
    /// </summary>
    /// <remarks>
    /// Takes `wasElite` rather than reading the room, for the same reason
    /// `CountEliteVictoryForSwordOfStone` does: the gate is the interesting half and a
    /// parameter is the only way to test both sides of it. On the VICTORY, so fleeing or
    /// dying in an elite pays nothing.
    /// </remarks>
    public static void ApplyWarHammerOnVictory(RunState state, bool wasElite)
    {
        if (wasElite && Effects.RelicEffects.Has(state.Relics, Effects.RelicEffects.WarHammer))
        {
            UpgradeRandomCards(state, 4, state.Rng.Niche);
        }
    }

    public static void UpgradeRandomCards(RunState state, int count, GameRng rng)
    {
        var indexes = state
            .Deck.Select((card, index) => (card, index))
            .Where(item => RunConstants.IsRunCardUpgradable(item.card))
            .Select(item => item.index)
            .ToList();
        StableShuffleDeck(state, indexes, rng);
        foreach (int index in indexes.Take(count))
        {
            state.Deck[index] = state.Deck[index] with { Upgraded = true };
        }
    }

    /// <summary>
    /// Reflections' Touch a Mirror: downgrade up to two upgraded cards, then upgrade up
    /// to four upgradable ones.
    /// </summary>
    /// <remarks>
    /// The upgradable list is derived AFTER the downgrades, not alongside them, so the
    /// two cards the mirror just knocked down are candidates to come straight back up.
    /// Each pick is <c>Rng.NextItem</c> over the remaining list with the pick removed --
    /// a draw per card, and only while the list has something in it, which is why both
    /// loops break rather than clamp.
    /// </remarks>
    public static void ReflectionsTouchAMirror(RunState state)
    {
        var rng = EventRng(state, "REFLECTIONS");

        var upgraded = Enumerable
            .Range(0, state.Deck.Count)
            .Where(index => state.Deck[index].Upgraded)
            .ToList();
        for (int i = 0; i < 2 && upgraded.Count > 0; i++)
        {
            int index = rng.NextItem(upgraded);
            upgraded.Remove(index);
            state.Deck[index] = state.Deck[index] with { Upgraded = false };
        }

        var upgradable = Enumerable
            .Range(0, state.Deck.Count)
            .Where(index => RunConstants.IsRunCardUpgradable(state.Deck[index]))
            .ToList();
        for (int i = 0; i < 4 && upgradable.Count > 0; i++)
        {
            int index = rng.NextItem(upgradable);
            upgradable.Remove(index);
            state.Deck[index] = state.Deck[index] with { Upgraded = true };
        }
    }

    /// <summary>
    /// Reflections' Shatter: a copy of every card in the deck, then a Bad Luck.
    /// </summary>
    /// <remarks>
    /// The loop runs to the deck's size as it was BEFORE any copy was added, which is
    /// what stops it copying its own copies forever. CloneCard preserves the card as it
    /// stands -- upgrades and enchantment included -- and costs no draws.
    /// </remarks>
    public static void ReflectionsShatter(RunState state)
    {
        int originalDeckSize = state.Deck.Count;
        for (int i = 0; i < originalDeckSize; i++)
        {
            // Not AddCardToDeck: this is CloneCard, so the egg relics do not get a second
            // look at a card that is already in the deck.
            state.Deck.Add(state.Deck[i]);
        }

        AddCardToDeck(state, new CardInstance(RunConstants.CardBadLuck, Upgraded: false));
    }

    /// <summary>
    /// Tinker Time's second page: two of Attack, Skill and Power.
    /// </summary>
    /// <remarks>
    /// <c>TakeRandom(2, Rng)</c> is <c>UnstableShuffle(rng).Take(2)</c>, so all three are
    /// permuted -- two draws for a list of three -- and the first two are shown. Taking
    /// only the draws the two shown cards need would leave the rider page reading the
    /// stream one value early.
    /// </remarks>
    public static void BeginTinkerCardTypePage(RunState state)
    {
        var types = new List<int>
        {
            (int)CardType.Attack,
            (int)CardType.Skill,
            (int)CardType.Power,
        };
        EventRng(state, "TINKER_TIME").Shuffle(types);
        state.EventRandomOffer = [.. types.Take(2)];
        state.EventPage = 1;
    }

    /// <summary>
    /// Tinker Time's third page: two of the three riders that belong to the chosen type.
    /// </summary>
    public static void BeginTinkerRiderPage(RunState state)
    {
        var riders = state.TinkerCardType switch
        {
            CardType.Attack => new List<int>
            {
                (int)TinkerRider.Sapping,
                (int)TinkerRider.Violence,
                (int)TinkerRider.Choking,
            },
            CardType.Skill => new List<int>
            {
                (int)TinkerRider.Energized,
                (int)TinkerRider.Wisdom,
                (int)TinkerRider.Chaos,
            },
            _ => new List<int>
            {
                (int)TinkerRider.Expertise,
                (int)TinkerRider.Curious,
                (int)TinkerRider.Improvement,
            },
        };
        // The same stream the type page used: one event, one Rng.
        EventRng(state, "TINKER_TIME").Shuffle(riders);
        state.EventRandomOffer = [.. riders.Take(2)];
        state.EventPage = 2;
    }

    public static void UpgradeTwoRandomCardsForLightDoor(RunState state)
    {
        var indexes = state
            .Deck.Select((card, index) => (card, index))
            .Where(item => RunConstants.IsRunCardUpgradable(item.card))
            .Select(item => item.index)
            .ToList();
        StableShuffleDeck(state, indexes, EventRng(state, "DOORS_OF_LIGHT_AND_DARK"));
        foreach (int index in indexes.Take(2))
        {
            state.Deck[index] = state.Deck[index] with { Upgraded = true };
        }
    }

    /// <summary>
    /// <c>CardCmd.Transform</c> on a DECK card: the original leaves its pile and the
    /// replacement is added to the deck with <c>AddInternal</c> and no index -- the bottom of
    /// the deck, not the original's place (only combat piles keep the index) -- after
    /// <c>ModifyCardBeingAddedToDeck</c> and before <c>AfterCardChangedPiles</c>, which is
    /// what <see cref="AddCardToDeck"/> runs. Writing the new card over the old one kept the
    /// deck's order wrong for the rest of the run (run referee, seed CNV3YDEJXD: Archaic
    /// Tooth's Break sat where the Bash had been).
    /// </summary>
    public static void TransformInDeck(RunState state, int deckIndex, CardInstance replacement)
    {
        state.Deck.RemoveAt(deckIndex);
        AddCardToDeck(state, replacement);
    }

    public static void TransformCardAt(RunState state, int deckIndex, GameRng rng)
    {
        if ((uint)deckIndex >= (uint)state.Deck.Count)
        {
            return;
        }

        int originalId = Math.Abs(state.Deck[deckIndex].DefId);
        // What a card can become depends on what it is: the options come from the
        // original's own pool, narrowed to the rarities a run is handed, with the
        // original and the multiplayer-only cards dropped.
        var pool = RunRewardGenerator.TransformOptionsFor(originalId);
        if (pool.Length == 0)
        {
            return;
        }

        state.Deck.RemoveAt(deckIndex);
        AddCardToDeck(state, new CardInstance(rng.NextItem(pool), Upgraded: false));
    }

    /// <summary>
    /// Leafy Poultice: the first BASIC card carrying `CardTag.Strike`, or `CardTag.Defend`.
    /// </summary>
    private static void TransformFirstTaggedBasic(RunState state, bool strike)
    {
        int index = state.Deck.FindIndex(card =>
        {
            var def = GeneratedData.Cards.Get(card.DefId);
            return def.Rarity == CardRarity.Basic && (strike ? def.StrikeTag : def.DefendTag);
        });
        if (index >= 0)
        {
            TransformCardAt(state, index, state.PlayerRng.Transformations);
        }
    }

    public static void TransformFirstCardMatching(RunState state, int cardId)
    {
        int index = state.Deck.FindIndex(card => Math.Abs(card.DefId) == cardId);
        if (index >= 0)
        {
            TransformCardAt(state, index, state.PlayerRng.Transformations);
        }
    }

    /// <summary>
    /// `PandorasBox.AfterObtained`: every BASIC Strike or Defend the deck holds that is
    /// removable, transformed off `Rng.Niche`.
    /// </summary>
    private static void PandorasBoxTransform(RunState state)
    {
        // The set is taken FIRST, then transformed. `TransformCardAt` removes the card and
        // appends the replacement, so a forward walk over a shifting list skips half of
        // them -- and the appended replacements are themselves candidates for the walk to
        // find, which a `where` over the live deck would keep transforming.
        var targets = state
            .Deck.Where(card =>
            {
                var def = GeneratedData.Cards.Get(card.DefId);
                return def.Rarity == CardRarity.Basic
                    && (def.StrikeTag || def.DefendTag)
                    && IsRemovableCard(card);
            })
            .ToList();

        foreach (var target in targets)
        {
            int index = state.Deck.IndexOf(target);
            if (index >= 0)
            {
                TransformCardAt(state, index, state.Rng.Niche);
            }
        }
    }

    /// <summary>
    /// `DustyTome.SetupForPlayer`: an ANCIENT-rarity card from the player's own pool,
    /// minus Archaic Tooth's transcendence cards, off `PlayerRng.Rewards`.
    /// </summary>
    /// <remarks>
    /// The Ironclad's Ancient cards are Break and Corruption, and Break is one of the
    /// eleven transcendence cards -- so the filtered list is a single card and the draw
    /// always lands on Corruption. The draw is still made: skipping it would shift every
    /// later value on the Rewards stream.
    /// </remarks>
    private static int DustyTomeCard(RunState state)
    {
        var candidates = RunRewardGenerator
            .OwnCardPool(state)
            .ToArray()
            .Where(id =>
                GeneratedData.Cards.Get(id).Rarity == CardRarity.Ancient
                && !RunConstants.TranscendenceCardNames.Contains(GeneratedData.Cards.Get(id).Name)
            )
            .ToArray();
        // `SetupForPlayer` is `PlayerRng.Rewards.NextItem(items)`; an empty list (no
        // character's pool is) would still be a Rewards draw.
        return state.PlayerRng.Rewards.NextItem(
            candidates.Length > 0 ? candidates : RunRewardGenerator.OwnCardPool(state).ToArray()
        );
    }

    private static void TransformAllMatching(RunState state, int cardId)
    {
        for (int i = 0; i < state.Deck.Count; i++)
        {
            if (Math.Abs(state.Deck[i].DefId) == cardId)
            {
                TransformCardAt(state, i, state.PlayerRng.Transformations);
            }
        }
    }

    /// <summary>
    /// Neow's Talisman: the LAST BASIC card carrying `CardTag.Strike`, or `CardTag.Defend`.
    /// </summary>
    private static void UpgradeLastTaggedBasic(RunState state, bool strike)
    {
        for (int i = state.Deck.Count - 1; i >= 0; i--)
        {
            var def = GeneratedData.Cards.Get(state.Deck[i].DefId);
            if (def.Rarity != CardRarity.Basic || (strike ? !def.StrikeTag : !def.DefendTag))
            {
                continue;
            }

            state.Deck[i] = state.Deck[i] with { Upgraded = true };
            return;
        }
    }

    private static void UpgradeLastCardMatching(RunState state, int cardId)
    {
        for (int i = state.Deck.Count - 1; i >= 0; i--)
        {
            if (state.Deck[i].DefId != cardId || !RunConstants.IsRunCardUpgradable(state.Deck[i]))
            {
                continue;
            }

            state.Deck[i] = state.Deck[i] with { Upgraded = true };
            return;
        }
    }
}
