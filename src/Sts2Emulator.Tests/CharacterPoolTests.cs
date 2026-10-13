using System;
using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Everything the game reads off `Player.Character` -- `CardPool`, `PotionPool`, `RelicPool` --
/// follows the combat's and the run's character instead of being the Ironclad's. Each test
/// plays one effect as a Silent (or a Necrobinder where the Silent's pool cannot show it) and
/// checks that what it hands out comes from that character's pool. Every one of them used to
/// draw from the Ironclad's.
/// </summary>
public class CharacterPoolTests
{
    private const PlayerCharacter Silent = PlayerCharacter.Silent;

    // The oracle reads the generated tables by name, not through CharacterPools, so a
    // mistake in the accessor cannot hide itself.
    private static int[] CardsOf(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Ironclad => GeneratedData.CardPools.Ironclad.ToArray(),
            PlayerCharacter.Silent => GeneratedData.CardPools.Silent.ToArray(),
            PlayerCharacter.Regent => GeneratedData.CardPools.Regent.ToArray(),
            PlayerCharacter.Necrobinder => GeneratedData.CardPools.Necrobinder.ToArray(),
            _ => GeneratedData.CardPools.Defect.ToArray(),
        };

    private static int[] PotionsOf(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Ironclad => GeneratedData.PotionPools.Ironclad.ToArray(),
            PlayerCharacter.Silent => GeneratedData.PotionPools.Silent.ToArray(),
            _ => throw new NotSupportedException(character.ToString()),
        };

    private static int[] RelicsOf(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Ironclad => GeneratedData.RelicPools.Ironclad.ToArray(),
            PlayerCharacter.Silent => GeneratedData.RelicPools.Silent.ToArray(),
            _ => throw new NotSupportedException(character.ToString()),
        };

    private static bool IsOf(PlayerCharacter character, int cardId) =>
        CardsOf(character).Contains(cardId);

    private static CombatState CombatOf(PlayerCharacter character)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Character = character;
        state.Hand.Clear();
        return state;
    }

    private static Fight FightOf(PlayerCharacter character, params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Energy(3).Enemy(hp: 80);
        fight.State.Character = character;
        return fight;
    }

    private static RunState RunOf(PlayerCharacter character, string seed = "CHARPOOL01")
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.Character = character;
        return engine.State;
    }

    // ── the accessor ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void ACombatAndARunAreTheIroncladsUntilToldOtherwise()
    {
        Assert.Equal(PlayerCharacter.Ironclad, new CombatState().Character);
        Assert.Equal(PlayerCharacter.Ironclad, new RunState().Character);
    }

    [Fact]
    public void TheRunHandsItsCharacterToTheCombatsItStarts()
    {
        var engine = new RunEngine();
        engine.Reset("CHARPOOL02");
        engine.State.Character = Silent;
        engine.State.CurrentNodeType = RunConstants.NodeNormal;

        engine.StartCombat(
            RunConstants.StarterDeckIds,
            RunConstants.SlitheringStranglerEncounterId,
            [],
            playerHp: 64,
            playerMaxHp: 80,
            potionIds: [],
            playerGold: 99
        );

        Assert.Equal(Silent, engine.State.ActiveCombat!.Character);
    }

    [Fact]
    public void CloningKeepsTheCharacterOfBothStates()
    {
        var run = RunOf(Silent);

        Assert.Equal(Silent, StateCloning.Clone(run).Character);
        Assert.Equal(Silent, StateCloning.Clone(CombatOf(Silent)).Character);
    }

    [Fact]
    public void OthersAreEveryoneButTheOwnerInTheGamesCharacterOrder()
    {
        // ModelDb.AllCharacters: Ironclad, Silent, Regent, Necrobinder, Defect.
        Assert.Equal(
            [Silent, PlayerCharacter.Regent, PlayerCharacter.Necrobinder, PlayerCharacter.Defect],
            CharacterPools.Others(PlayerCharacter.Ironclad)
        );
        Assert.Equal(
            [
                PlayerCharacter.Ironclad,
                PlayerCharacter.Regent,
                PlayerCharacter.Necrobinder,
                PlayerCharacter.Defect,
            ],
            CharacterPools.Others(Silent)
        );
    }

    // ── combat: generators that read `Owner.Character.CardPool` ──────────────────────────

    [Fact]
    public void DiscoveryOffersTheOwnersCards()
    {
        var fight = FightOf(Silent, Card(CL.Discovery));

        fight.Play();

        var offered = fight.Pending!.GeneratedCandidates;
        Assert.Equal(3, offered.Count);
        Assert.All(offered, id => Assert.True(IsOf(Silent, id), $"{id} is not a Silent card"));
    }

    [Fact]
    public void StokeReplacesTheHandWithTheOwnersCards()
    {
        var fight = FightOf(
            Silent,
            Card(CL.Calamity),
            Card(IC.Stoke),
            Card(IC.StrikeIronclad),
            Card(IC.StrikeIronclad)
        );

        fight.Play(index: 1);

        Assert.Equal(3, fight.State.Hand.Count);
        Assert.All(fight.State.Hand, card => Assert.True(IsOf(Silent, card.DefId)));
    }

    [Fact]
    public void JackpotRollsTheOwnersZeroCostCards()
    {
        var fight = FightOf(Silent, Card(CL.Jackpot));

        fight.Play();

        Assert.NotEmpty(fight.State.Hand);
        Assert.All(
            fight.State.Hand,
            card =>
            {
                Assert.True(IsOf(Silent, card.DefId));
                Assert.Equal(0, GeneratedData.Cards.Get(card.DefId).Cost);
            }
        );
    }

    [Fact]
    public void HelloWorldAddsTheOwnersCommons()
    {
        var state = CombatOf(Silent);

        CardEffects.AddDistinctCommonCardsToHand(state, 2, new Random(3));

        Assert.Equal(2, state.Hand.Count);
        Assert.All(
            state.Hand,
            card =>
            {
                Assert.True(IsOf(Silent, card.DefId));
                Assert.Equal(CardRarity.Common, GeneratedData.Cards.Get(card.DefId).Rarity);
            }
        );
    }

    [Fact]
    public void BigHatFindsEtherealCardsOnlyInThePoolOfTheOwner()
    {
        // The Ironclad's pool has no Ethereal card, so Big Hat does nothing for one; the
        // Necrobinder's has eight. The relic used to read the Ironclad's for everybody.
        var ironclad = CombatOf(PlayerCharacter.Ironclad);
        var necrobinder = CombatOf(PlayerCharacter.Necrobinder);

        CardEffects.AddDistinctEtherealCardsToHand(ironclad, 2, new Random(1));
        CardEffects.AddDistinctEtherealCardsToHand(necrobinder, 2, new Random(1));

        Assert.Empty(ironclad.Hand);
        Assert.Equal(2, necrobinder.Hand.Count);
        Assert.All(
            necrobinder.Hand,
            card =>
            {
                Assert.True(IsOf(PlayerCharacter.Necrobinder, card.DefId));
                Assert.True(GeneratedData.Cards.Get(card.DefId).Ethereal);
            }
        );
    }

    [Fact]
    public void CrossbowAddsAFreeAttackFromTheOwnersPool()
    {
        var state = CombatOf(Silent);

        CardEffects.AddFreeAttackCardToHand(state, 1, new Random(2));

        var card = Assert.Single(state.Hand);
        Assert.True(IsOf(Silent, card.DefId));
        Assert.Equal(CardType.Attack, GeneratedData.Cards.Get(card.DefId).Type);
        Assert.True(card.FreeThisTurn);
    }

    [Fact]
    public void ChoicesParadoxOffersTheOwnersCards()
    {
        var state = CombatOf(Silent);

        CardEffects.OpenChoicesParadoxOffer(state, 5, new Random(4));

        var offered = state.PendingSelection!.GeneratedCandidates;
        Assert.Equal(5, offered.Count);
        Assert.All(offered, id => Assert.True(IsOf(Silent, id)));
    }

    [Fact]
    public void CallOfTheVoidRollsFromTheOwnersPoolNotTheNecrobinders()
    {
        // CallOfTheVoidPower reads `Owner.Player.Character.CardPool`: the power is the
        // Necrobinder's card but the pool is whoever holds it.
        for (int seed = 0; seed < 25; seed++)
        {
            var state = CombatOf(Silent);

            CardEffects.AddEtherealPoolCardToHand(state, new Random(seed));

            var card = Assert.Single(state.Hand);
            Assert.True(IsOf(Silent, card.DefId));
            Assert.True(card.EtherealForCombat);
        }
    }

    [Fact]
    public void TheOwnersAttacksFeedCalamityAndInfernalBlade()
    {
        var state = CombatOf(Silent);

        CardEffects.AddRandomAttackCardsToHand(state, 3, new Random(5));

        Assert.Equal(3, state.Hand.Count);
        Assert.All(
            state.Hand,
            card =>
            {
                Assert.True(IsOf(Silent, card.DefId));
                Assert.Equal(CardType.Attack, GeneratedData.Cards.Get(card.DefId).Type);
            }
        );
    }

    [Fact]
    public void VexingPuzzleboxAndEntropyRollFromTheOwnersPool()
    {
        var puzzlebox = CombatOf(Silent);
        CardEffects.AddRandomPoolCardToHand(puzzlebox, new Random(6));
        Assert.True(IsOf(Silent, Assert.Single(puzzlebox.Hand).DefId));

        var entropy = CombatOf(Silent);
        entropy.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        CardEffects.TransformRandomCardInHand(entropy, new Random(6));
        Assert.True(IsOf(Silent, Assert.Single(entropy.Hand).DefId));
    }

    // ── combat: potions ──────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(2, CardType.Attack)]
    [InlineData(53, CardType.Skill)]
    [InlineData(48, CardType.Power)]
    public void AttackSkillAndPowerPotionOfferTheOwnersCardsOfThatType(int potionId, CardType type)
    {
        var state = CombatOf(Silent);

        PotionEffects.Apply(potionId, state, new Random(0));

        var offered = state.PendingSelection!.GeneratedCandidates;
        Assert.Equal(3, offered.Count);
        Assert.All(
            offered,
            id =>
            {
                Assert.True(IsOf(Silent, id));
                Assert.Equal(type, GeneratedData.Cards.Get(id).Type);
            }
        );
    }

    [Fact]
    public void OrobicAcidAddsOneOfEachTypeFromTheOwnersPool()
    {
        var state = CombatOf(Silent);

        PotionEffects.Apply(40, state, new Random(0));

        Assert.Equal(3, state.Hand.Count);
        Assert.All(state.Hand, card => Assert.True(IsOf(Silent, card.DefId)));
    }

    [Fact]
    public void EntropicBrewRollsFromTheOwnersPotionPoolAndTheSharedOne()
    {
        int[] pool = [.. PotionsOf(Silent), .. GeneratedData.PotionPools.Shared.ToArray()];
        var ironcladOnly = PotionsOf(PlayerCharacter.Ironclad);
        bool sawSilentOnly = false;

        for (int seed = 0; seed < 150; seed++)
        {
            var state = CombatOf(Silent);
            state.MaxPotionSlots = 3;
            state.PotionSlots = new int[6];

            PotionEffects.Apply(19, state, new Random(seed));

            foreach (int potion in state.PotionSlots.Take(3))
            {
                Assert.Contains(potion, pool);
                Assert.DoesNotContain(potion, ironcladOnly);
                sawSilentOnly |= PotionsOf(Silent).Contains(potion);
            }
        }

        Assert.True(sawSilentOnly, "a Silent run never rolled a Silent potion");
    }

    // ── combat: Splash takes the pools of everyone else ──────────────────────────────────

    [Fact]
    public void SplashOffersTheOtherCharactersAttacksWhoeverThePlayerIs()
    {
        var ironclad = CardEffects.SplashPool(CombatOf(PlayerCharacter.Ironclad));
        var silent = CardEffects.SplashPool(CombatOf(Silent));

        Assert.All(ironclad, id => Assert.False(IsOf(PlayerCharacter.Ironclad, id)));
        Assert.All(silent, id => Assert.False(IsOf(Silent, id)));
        Assert.All(silent, id => Assert.Equal(CardType.Attack, GeneratedData.Cards.Get(id).Type));

        // AllCharacters order with the owner's own pool removed: the Silent is offered the
        // Ironclad's first and the Regent's after it; the Ironclad is offered the Silent's
        // first.
        int FirstOf(List<int> pool, PlayerCharacter c) => pool.FindIndex(id => IsOf(c, id));
        Assert.Equal(0, FirstOf(silent, PlayerCharacter.Ironclad));
        Assert.True(FirstOf(silent, PlayerCharacter.Ironclad) < FirstOf(silent, PlayerCharacter.Regent));
        Assert.Equal(0, FirstOf(ironclad, Silent));
        Assert.True(FirstOf(ironclad, Silent) < FirstOf(ironclad, PlayerCharacter.Regent));
    }

    // ── run: reward pools ────────────────────────────────────────────────────────────────

    [Fact]
    public void CardRewardsComeFromTheRunsCharacter()
    {
        var state = RunOf(Silent);

        RunRewardGenerator.PopulateCardReward(state);

        Assert.All(state.RewardCards, id => Assert.True(IsOf(Silent, id)));
    }

    [Fact]
    public void ThePoolFeedingARewardFollowsTheCharacterAndPrismaticGemKeepsGameOrder()
    {
        Assert.Equal(
            CardsOf(Silent),
            RunRewardGenerator.CardRewardPool(Silent, false, false)
        );
        // `CharacterCardPools.Union(...)`: the owner sits at its AllCharacters position.
        Assert.Equal(
            Enum.GetValues<PlayerCharacter>().SelectMany(c => CardsOf(c)),
            RunRewardGenerator.CardRewardPool(Silent, false, true)
        );
    }

    [Fact]
    public void HeftyTabletOffersTheOwnersRares()
    {
        var state = RunOf(Silent);

        var offer = RunRewardGenerator.GenerateRareOnlyCardOffer(state, 3);

        Assert.All(
            offer,
            id =>
            {
                Assert.True(IsOf(Silent, id));
                Assert.Equal(CardRarity.Rare, GeneratedData.Cards.Get(id).Rarity);
            }
        );
    }

    /// <summary>
    /// Arcane Scroll used to draw from a hand-written list of the Ironclad's solo-legal
    /// rares. The derived pool is that list, in that order, for the Ironclad.
    /// </summary>
    [Fact]
    public void ArcaneScrollsRarePoolIsTheOldIroncladListWhenDerived()
    {
        int[] handWritten =
        [
            9, 29, 58, 546, 99, 113, 114, 119, 141, 183, 188, 246, 261, 272, 295, 328, 332, 334,
            339, 364, 374, 464, 494, 505, 525,
        ];

        Assert.Equal(handWritten, RunRewardGenerator.RareCardPool(RunOf(PlayerCharacter.Ironclad)));

        var silentRares = RunRewardGenerator.RareCardPool(RunOf(Silent));
        Assert.NotEmpty(silentRares);
        Assert.All(silentRares, id => Assert.True(IsOf(Silent, id)));
    }

    [Fact]
    public void PotionRollsComeFromTheRunsCharacterPlusTheSharedPool()
    {
        var state = RunOf(Silent);
        int[] allowed = [.. PotionsOf(Silent), .. GeneratedData.PotionPools.Shared.ToArray()];
        var rng = state.PlayerRng.Rewards;

        for (int i = 0; i < 60; i++)
        {
            Assert.Contains(RunRewardGenerator.NextPotion(state, rng), allowed);
        }

        int[] expected =
        [
            .. PotionsOf(PlayerCharacter.Ironclad),
            .. GeneratedData.PotionPools.Shared.ToArray(),
        ];
        Assert.Equal(expected, RunRewardGenerator.PotionRewardPool(PlayerCharacter.Ironclad));
    }

    [Fact]
    public void TheShopStocksTheRunsCharactersCards()
    {
        var state = RunOf(Silent);
        state.Gold = 1000;

        RunRewardGenerator.EnterShop(state);

        // Five slots are the character's (two Attacks, two Skills, a Power); the last two are
        // Colorless.
        Assert.All(state.ShopCards.Take(5), id => Assert.True(IsOf(Silent, id)));
        Assert.All(
            state.ShopCards.Skip(5),
            id => Assert.Contains(id, GeneratedData.CardPools.Colorless.ToArray())
        );
    }

    [Fact]
    public void ScrollBoxesBundleTheRunsCharactersCards()
    {
        var state = RunOf(Silent);

        var bundles = RunRewardGenerator.GenerateScrollBoxBundles(state);

        Assert.All(bundles.SelectMany(b => b), id => Assert.True(IsOf(Silent, id)));
    }

    [Fact]
    public void ADefectsScrollBoxesRollTheOnePercentClawBundleAndNobodyElsesDo()
    {
        // `ScrollBoxes.GenerateRandomBundles`: a Defect rolls a 1% chance per bundle, before the
        // picks, and a hit swaps the bundle for three Claws; nobody else rolls at all.
        int claw = GeneratedData.Cards.FindId("Claw")!.Value;
        string clawSeed = Enumerable
            .Range(0, 20000)
            .Select(i => $"CLAW{i:D6}")
            .First(seed => new PlayerRngSet(new RunRngSet(seed)).Rewards.NextInt(100) < 1);
        string plainSeed = Enumerable
            .Range(0, 20000)
            .Select(i => $"PLAN{i:D6}")
            .First(seed =>
            {
                var rewards = new PlayerRngSet(new RunRngSet(seed)).Rewards;
                return rewards.NextInt(100) >= 1;
            });

        var hit = RunOf(PlayerCharacter.Defect);
        hit.PlayerRng = new PlayerRngSet(new RunRngSet(clawSeed));
        Assert.Equal(new[] { claw, claw, claw }, RunRewardGenerator.GenerateScrollBoxBundles(hit)[0]);

        var miss = RunOf(PlayerCharacter.Defect);
        miss.PlayerRng = new PlayerRngSet(new RunRngSet(plainSeed));
        RunRewardGenerator.GenerateScrollBoxBundles(miss);
        Assert.Equal(6 + 2, miss.PlayerRng.Rewards.CallCount);

        // The same seed that gives a Defect its Claws gives anyone else six picks and no roll.
        var ironclad = RunOf(PlayerCharacter.Ironclad);
        ironclad.PlayerRng = new PlayerRngSet(new RunRngSet(clawSeed));
        var bundles = RunRewardGenerator.GenerateScrollBoxBundles(ironclad);
        Assert.Equal(6, ironclad.PlayerRng.Rewards.CallCount);
        Assert.DoesNotContain(claw, bundles.SelectMany(b => b));
    }

    [Fact]
    public void TheRelicBagHoldsTheRunsCharactersRelicsAndNotTheIroncladsOwn()
    {
        var state = RunOf(Silent);

        RunMapGenerator.SelectActAndGenerateRooms(state);

        var bag = state.RelicBag!.Remaining.ToArray();
        var silentRelics = RelicsOf(Silent);
        var ironcladRelics = RelicsOf(PlayerCharacter.Ironclad);
        Assert.Contains(bag, id => silentRelics.Contains(id));
        Assert.DoesNotContain(bag, id => ironcladRelics.Contains(id));
    }

    // ── run: other characters' pools ─────────────────────────────────────────────────────

    [Fact]
    public void KaleidoscopeOffersCardsFromEveryoneButTheRunsCharacter()
    {
        foreach (var own in new[] { PlayerCharacter.Ironclad, Silent, PlayerCharacter.Defect })
        {
            var state = RunOf(own);

            RunRewardGenerator.EnterOtherCharacterCardReward(state);

            Assert.All(state.RewardCards.Where(id => id != 0), id => Assert.False(IsOf(own, id)));
            Assert.Equal(3, state.RewardCards.Count(id => id != 0));
        }
    }

    [Fact]
    public void SeaGlassAndColorfulPhilosophersSkipTheOwnersPool()
    {
        var seaGlass = Enumerable
            .Range(0, RunConstants.OtherCharacterCount)
            .Select(i => RunConstants.OtherCharacterPoolFor(Silent, i).ToArray())
            .ToArray();
        var philosophers = RunNonCombatEffects.ColorfulPhilosopherPoolOrder(Silent);

        // Sea Glass walks AllCharacters; the philosophers walk their own colour order.
        Assert.Equal(
            new[]
            {
                PlayerCharacter.Ironclad,
                PlayerCharacter.Regent,
                PlayerCharacter.Necrobinder,
                PlayerCharacter.Defect,
            }.Select(c => CardsOf(c)),
            seaGlass
        );
        Assert.Equal(
            new[]
            {
                PlayerCharacter.Necrobinder,
                PlayerCharacter.Ironclad,
                PlayerCharacter.Regent,
                PlayerCharacter.Defect,
            }.Select(c => CardsOf(c)),
            philosophers
        );
    }

    // ── run: a card transforms within ITS pool ───────────────────────────────────────────

    [Fact]
    public void ACardTransformsWithinItsOwnPoolWhoeverHoldsIt()
    {
        // `CardFactory.GetDefaultTransformationOptions` starts from `original.Pool`. A Silent
        // card in an Ironclad's deck (Kaleidoscope, Sea Glass) used to find no pool and
        // fall back to the Ironclad's.
        int silentCard = CardsOf(Silent).First(id => IsOf(Silent, id)
            && GeneratedData.Cards.Get(id).Rarity == CardRarity.Common);

        var options = RunRewardGenerator.TransformOptionsFor(silentCard);

        Assert.NotEmpty(options);
        Assert.All(options, id => Assert.True(IsOf(Silent, id)));
        Assert.DoesNotContain(silentCard, options);
    }

    [Fact]
    public void QuestCardsTransformWithinColorlessLikeTheOtherPoollessOnes()
    {
        int quest = GeneratedData.Cards.FindId("SpoilsMap")!.Value;

        var options = RunRewardGenerator.TransformOptionsFor(quest);

        Assert.NotEmpty(options);
        Assert.All(options, id => Assert.Contains(id, GeneratedData.CardPools.Colorless.ToArray()));
    }
}
