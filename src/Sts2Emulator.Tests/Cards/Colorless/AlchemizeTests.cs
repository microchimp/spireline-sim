using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Skill, CardKeyword.Exhaust. MegaCrit.Sts2.Core.Models.Cards/Alchemize.cs procures
// PotionFactory.CreateRandomPotionInCombat, rolled off Rng.CombatPotionGeneration;
// OnUpgrade only makes it cheaper.
public class AlchemizeTests
{
    [Fact]
    public void FillsAnEmptyPotionSlot()
    {
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.PotionSlots[0] = 0;
        fight.State.PotionSlots[1] = 0;
        fight.State.PotionSlots[2] = 0;

        fight.Play();

        Assert.NotEqual(0, fight.State.PotionSlots[0]);
        Assert.Equal(0, fight.State.PotionSlots[1]);
    }

    [Fact]
    public void RollsThePotionFromThePotionGenerationStream()
    {
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.PotionSlots[0] = 0;
        var potionRng = new CountingRandom(9);
        fight.State.PotionGenerationRng = potionRng;

        fight.Play();

        // PotionFactory.CreateRandomPotion: NextFloat for the rarity, then NextItem.
        Assert.Equal(2, potionRng.CallCount);
    }

    [Fact]
    public void TheRollIsTheFactorysRarityThenItem()
    {
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.PotionSlots[0] = 0;
        fight.State.PotionGenerationRng = new CountingRandom(13);

        fight.Play();

        // The factory's own sequence: float rarity (Rare <= 0.1, Uncommon <= 0.35, else
        // Common), then an index into that rarity's options in pool order.
        var oracle = new CountingRandom(13);
        float roll = (float)oracle.NextDouble();
        var rarity =
            roll <= 0.1f ? PotionRarity.Rare
            : roll <= 0.35f ? PotionRarity.Uncommon
            : PotionRarity.Common;
        var options = Sts2Emulator
            .Core.Run.RunRewardGenerator.PotionRewardPool(PlayerCharacter.Ironclad)
            .Where(id =>
                GeneratedData.Potions.Get(id).Rarity == rarity
                && GeneratedData.Potions.Get(id).Name
                    is not ("FairyInABottle" or "FruitJuice" or "RegenPotion")
            )
            .ToArray();

        Assert.Equal(options[oracle.Next(options.Length)], fight.State.PotionSlots[0]);
    }

    [Fact]
    public void RollsThePotionEvenWhenTheBeltIsFull()
    {
        // The potion is created before PotionCmd.TryToProcure can refuse it, so a full
        // belt still spends both draws (night03: game 2, emulator 1 on every Alchemize).
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.PotionSlots[0] = 5;
        fight.State.PotionSlots[1] = 6;
        fight.State.PotionSlots[2] = 7;
        var potionRng = new CountingRandom(9);
        fight.State.PotionGenerationRng = potionRng;

        fight.Play();

        Assert.Equal(2, potionRng.CallCount);
    }

    [Fact]
    public void RollsThePotionEvenUnderSozu()
    {
        // Sozu.ShouldProcurePotion is false for its owner: TryToProcure refuses, but only
        // after CreateRandomPotionInCombat has spent its draws.
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.Sozu));
        fight.State.PotionSlots[0] = 0;
        var potionRng = new CountingRandom(9);
        fight.State.PotionGenerationRng = potionRng;

        fight.Play();

        Assert.Equal(0, fight.State.PotionSlots[0]);
        Assert.Equal(2, potionRng.CallCount);
    }

    [Fact]
    public void NeverRollsAPotionThatCannotBeGeneratedInCombat()
    {
        // FairyInABottle, FruitJuice and RegenPotion override CanBeGeneratedInCombat.
        var banned = new HashSet<string> { "FairyInABottle", "FruitJuice", "RegenPotion" };
        for (int seed = 0; seed < 400; seed++)
        {
            var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
            fight.State.PotionSlots[0] = 0;
            fight.State.PotionGenerationRng = new CountingRandom(seed);

            fight.Play();

            Assert.DoesNotContain(
                GeneratedData.Potions.Get(fight.State.PotionSlots[0]).Name,
                banned
            );
        }
    }

    [Fact]
    public void DoesNothingWithNoFreeSlot()
    {
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);
        fight.State.PotionSlots[0] = 5;
        fight.State.PotionSlots[1] = 6;
        fight.State.PotionSlots[2] = 7;

        fight.Play();

        // The array is `PotionSlotCapacity` wide; only the first MaxPotionSlots exist.
        Assert.Equal(new[] { 5, 6, 7 }, fight.State.PotionSlots.Take(3).ToList());
    }

    [Fact]
    public void ExhaustsItself()
    {
        var fight = Fight.Hand(Card(CL.Alchemize)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal([CL.Alchemize], Fight.Ids(fight.State.ExhaustPile));
    }
}
