using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Skill, CardKeyword.Exhaust. MegaCrit.Sts2.Core.Models.Cards/Discovery.cs rolls
// three distinct cards with CardFactory.GetDistinctForCombat off Rng.CombatCardGeneration
// and lets you choose one, free for the turn; OnUpgrade removes the Exhaust.
//
// The three come from Owner.Character.CardPool (the Ironclad's, through FilterForCombat),
// not the colourless pool, shuffled whole on the generation stream; the screen has canSkip.
public class DiscoveryTests
{
    [Fact]
    public void OffersThreeDistinctCards()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(CardSelectionKind.GeneratedCardToHand, fight.Pending?.Kind);
        Assert.Equal(3, fight.Pending?.Candidates.Count);
        Assert.Equal(3, fight.Pending!.GeneratedCandidates.Distinct().Count());
    }

    [Fact]
    public void TheChosenCardJoinsTheHandFreeForTheTurn()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);
        fight.Play();
        int wanted = fight.Pending!.GeneratedCandidates[1];

        fight.Choose(1);

        Assert.Equal([wanted], Fight.Ids(fight.State.Hand));
        Assert.True(fight.State.Hand[0].FreeThisTurn);
        Assert.Null(fight.Pending);
    }

    [Fact]
    public void TheOptionsComeOffTheCardGenerationStream()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);
        var generationRng = new CountingRandom(6);
        fight.State.CardGenerationRng = generationRng;

        fight.Play();

        // A shuffle of the whole filtered pool: one roll per position after the first.
        Assert.Equal(CardEffects.OwnerCharacterGenerationPool().Count - 1, generationRng.CallCount);
    }

    [Fact]
    public void OffersThreeFromTheCharactersOwnPoolInShuffleOrder()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);
        fight.State.CardGenerationRng = new CountingRandom(6);

        fight.Play();

        var offered = fight.Pending!.GeneratedCandidates;
        Assert.Equal(ShuffleTake(CardEffects.OwnerCharacterGenerationPool(), 3, 6), offered);
        var ironclad = GeneratedData.CardPools.Ironclad.ToArray();
        Assert.All(offered, id => Assert.Contains(id, ironclad));
    }

    [Fact]
    public void TheScreenCanBeSkipped()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);
        fight.Play();

        Assert.True(fight.Pending!.Skippable);
        CombatEngine.Step(fight.State, fight.Pending.Candidates.Count, new Random(0));

        Assert.Null(fight.Pending);
        Assert.Empty(fight.State.Hand);
    }

    /// <summary>`TakeRandom`: Fisher-Yates from the end over the filtered pool, take the first N.</summary>
    private static List<int> ShuffleTake(List<int> pool, int count, int seed)
    {
        var stream = new CountingRandom(seed);
        for (int i = pool.Count - 1; i > 0; i--)
        {
            int j = stream.Next(i + 1);
            (pool[i], pool[j]) = (pool[j], pool[i]);
        }

        return pool.Take(count).ToList();
    }

    [Fact]
    public void TheChosenCardCostsNoEnergyThisTurn()
    {
        var fight = Fight.Hand(Card(CL.Discovery)).Energy(1).Enemy(hp: 40);
        fight.Play();
        fight.Choose(0);
        int energyBefore = fight.State.Energy;

        fight.Play(index: 0);

        Assert.Equal(energyBefore, fight.State.Energy);
    }
}
