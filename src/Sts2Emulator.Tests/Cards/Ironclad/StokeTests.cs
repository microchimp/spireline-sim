using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Skill. MegaCrit.Sts2.Core.Models.Cards/Stoke.cs exhausts the whole hand, then
// rolls up that many replacement cards with CardFactory.GetForCombat drawing from
// Rng.CombatCardGeneration, upgrading them when the card is upgraded. There is no
// numeric upgrade.
//
// The emulator rolls from its own Ironclad pool rather than the character's unlocked
// pool, so these tests pin the shape — how many, from which stream, upgraded or not —
// rather than which specific cards come back.
public class StokeTests
{
    [Fact]
    public void ExhaustsTheHandAndReplacesItCardForCard()
    {
        var fight = Fight
            .Hand(Card(IC.Stoke), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal(2, fight.State.Hand.Count);
        Assert.Equal(2, fight.State.ExhaustPile.Count);
        Assert.DoesNotContain(fight.State.Hand, card => card.DefId == IC.Bash);
    }

    [Fact]
    public void UpgradedRollsUpgradedCards()
    {
        var fight = Fight
            .Hand(Card(IC.Stoke, upgraded: true), Card(IC.Bash))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.All(fight.State.Hand, card => Assert.True(card.Upgraded));
    }

    [Fact]
    public void RollsFromTheCardGenerationStream()
    {
        var fight = Fight
            .Hand(Card(IC.Stoke), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);
        var generationRng = new CountingRandom(3);
        fight.State.CardGenerationRng = generationRng;

        fight.Play();

        // One draw per replacement card.
        Assert.Equal(2, generationRng.CallCount);
    }

    /// <summary>
    /// The hand is a snapshot exhausted one card at a time, each leaving the hand as it goes: Dark
    /// Embrace's draw for each exhaust is not in the snapshot and stays in the hand, and the generated
    /// cards are added after it. The hand was cleared after the loop, which deleted those draws from
    /// the fight altogether (they sat in no pile).
    /// </summary>
    [Fact]
    public void DarkEmbracesDrawsStayInTheHandBesideTheGeneratedCards()
    {
        var drawn = new[] { IC.Anger, IC.Anger, IC.Anger };
        var fight = Fight
            .Hand(Card(IC.Stoke), Card(IC.Bash), Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Energy(1)
            .Draw(Card(drawn[0]), Card(drawn[1]), Card(drawn[2]), Card(IC.Bash), Card(IC.Bash))
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.DarkEmbrace, 1);

        fight.Play();

        // Three exhausted, three drawn for them, three generated.
        Assert.Equal(3, fight.State.ExhaustPile.Count);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Equal(drawn, fight.State.Hand.Take(3).Select(card => card.DefId));
        Assert.Equal(2, fight.State.DrawPile.Count);
    }

    /// <summary>
    /// All the replacement cards are rolled at once, hand space or none, and one that finds the hand
    /// full goes to the discard pile (`AddGeneratedCardsToCombat`).
    /// </summary>
    [Fact]
    public void ReplacementsThatFindTheHandFullGoToTheDiscardPile()
    {
        var hand = new List<CardInstance> { Card(IC.Stoke) };
        for (int i = 0; i < 6; i++)
        {
            hand.Add(Card(IC.StrikeIronclad));
        }

        var fight = Fight
            .Hand([.. hand])
            .Energy(1)
            .Draw([.. Enumerable.Repeat(Card(IC.DefendIronclad), 10)])
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.DarkEmbrace, 1);
        var generationRng = new CountingRandom(3);
        fight.State.CardGenerationRng = generationRng;

        fight.Play();

        // Six exhausted, six drawn, six rolled: four fit in the hand, two do not.
        Assert.Equal(6, generationRng.CallCount);
        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(2, fight.State.DiscardPile.Count(card => card.DefId != IC.Stoke));
    }

    [Fact]
    public void DoesNothingWithAnEmptyHand()
    {
        var fight = Fight.Hand(Card(IC.Stoke)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.ExhaustPile);
    }
}
