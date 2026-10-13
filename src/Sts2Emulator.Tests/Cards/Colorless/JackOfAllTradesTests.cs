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

    private static List<int> PoolWithoutItself() =>
        GeneratedData.CardPools.Colorless.ToArray()
            .Where(id =>
                id != CL.JackOfAllTrades
                && GeneratedData.Cards.Get(id).CanBeGeneratedInCombat
                && GeneratedData.Cards.Get(id).Rarity
                    is not (CardRarity.Basic or CardRarity.Ancient or CardRarity.Event)
            )
            .ToList();
}
