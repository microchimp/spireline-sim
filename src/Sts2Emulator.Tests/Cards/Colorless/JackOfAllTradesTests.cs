using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 0-cost Skill, CardKeyword.Exhaust. MegaCrit.Sts2.Core.Models.Cards/JackOfAllTrades.cs
// rolls CardsVar(1) distinct cards from the unlocked Colorless pool (minus itself) with
// CardFactory.GetDistinctForCombat off Rng.CombatCardGeneration and adds them to hand;
// OnUpgrade raises that by 1.
public class JackOfAllTradesTests
{
    [Fact]
    public void AddsOneCardToHand()
    {
        var fight = Fight.Hand(Card(CL.JackOfAllTrades)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Single(fight.State.Hand);
    }

    [Fact]
    public void UpgradedAddsTwo()
    {
        var fight = Fight.Hand(Card(CL.JackOfAllTrades, upgraded: true)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(2, fight.State.Hand.Count);
    }

    [Fact]
    public void RollsFromTheCardGenerationStream()
    {
        var fight = Fight.Hand(Card(CL.JackOfAllTrades)).Energy(1).Enemy(hp: 40);
        var generationRng = new CountingRandom(2);
        fight.State.CardGenerationRng = generationRng;

        fight.Play();

        // A shuffle of the whole filtered pool, not one roll per card.
        Assert.Equal(PoolWithoutItself().Count - 1, generationRng.CallCount);
    }

    [Fact]
    public void CardsAreDistinctAndFollowTheShuffleOfTheColorlessPool()
    {
        var fight = Fight.Hand(Card(CL.JackOfAllTrades, upgraded: true)).Energy(1).Enemy(hp: 40);
        fight.State.CardGenerationRng = new CountingRandom(11);

        fight.Play();

        var expected = PoolWithoutItself();
        var stream = new CountingRandom(11);
        for (int i = expected.Count - 1; i > 0; i--)
        {
            int j = stream.Next(i + 1);
            (expected[i], expected[j]) = (expected[j], expected[i]);
        }

        Assert.Equal(expected.Take(2).ToList(), Fight.Ids(fight.State.Hand));
    }

    [Fact]
    public void NeverAddsItselfAndNeverRepeatsAcrossManyStreams()
    {
        for (int seed = 0; seed < 60; seed++)
        {
            var fight = Fight.Hand(Card(CL.JackOfAllTrades, upgraded: true)).Energy(1).Enemy(hp: 40);
            fight.State.CardGenerationRng = new CountingRandom(seed);
            fight.Play();

            var ids = Fight.Ids(fight.State.Hand).ToList();
            Assert.Equal(ids.Distinct().Count(), ids.Count);
            Assert.DoesNotContain(CL.JackOfAllTrades, ids);
        }
    }

    // night03 referee: the game's shuffle of this pool consumed 48 draws, i.e. 49 cards (the
    // Colorless pool minus Jack, minus the 11 MultiplayerOnly cards CardFactory.
    // FilterForPlayerCount drops in a solo run, minus the 3 that refuse combat generation).
    // The emulator shuffled 60 and drew 59. A literal count, so it cannot agree with a pool
    // that is wrong in the same way as the code under test.
    [Fact]
    public void ShufflesTheGamesFortyNineCardPoolInASoloRun()
    {
        var fight = Fight.Hand(Card(CL.JackOfAllTrades)).Energy(1).Enemy(hp: 40);
        var generationRng = new CountingRandom(2);
        fight.State.CardGenerationRng = generationRng;

        fight.Play();

        Assert.Equal(48, generationRng.CallCount);
    }

    [Fact]
    public void NeverOffersAMultiplayerOnlyCard()
    {
        for (int seed = 0; seed < 60; seed++)
        {
            var fight = Fight.Hand(Card(CL.JackOfAllTrades, upgraded: true)).Energy(1).Enemy(hp: 40);
            fight.State.CardGenerationRng = new CountingRandom(seed);
            fight.Play();

            Assert.DoesNotContain(
                Fight.Ids(fight.State.Hand),
                id => GeneratedData.Cards.Get(id).MultiplayerOnly
            );
        }
    }

    private static List<int> PoolWithoutItself() =>
        GeneratedData.CardPools.Colorless.ToArray()
            .Where(id =>
                id != CL.JackOfAllTrades
                && GeneratedData.Cards.Get(id).CanBeGeneratedInCombat
                // CardFactory.FilterForPlayerCount: a solo run drops MultiplayerOnly cards.
                && !GeneratedData.Cards.Get(id).MultiplayerOnly
                && GeneratedData.Cards.Get(id).Rarity
                    is not (CardRarity.Basic or CardRarity.Ancient or CardRarity.Event)
            )
            .ToList();
}
