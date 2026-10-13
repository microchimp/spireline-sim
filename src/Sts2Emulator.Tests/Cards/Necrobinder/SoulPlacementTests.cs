using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// Capture Spirit, Dirge, Glimpse Beyond, Grave Warden, Reave and Severance all end in
// Soul.Create then CardPileCmd.AddGeneratedCardsToCombat(..., PileType.Draw, owner,
// CardPilePosition.Random). CardPileCmd.Add inserts each card at
// Rng.Shuffle.NextInt(drawPile.Count + 1), one draw per card, the count read after the
// previous insert. The emulator bottom-decked them and drew nothing, so the Shuffle stream
// ran short (night03 Q91FAC7EJB: game 34 draws, emulator 31) and the Souls sat at the
// bottom instead of being scattered through the pile.
public class SoulPlacementTests
{
    private const int Soul = 446;

    private static int IdOf(string name) =>
        GeneratedData.Cards.All.ToArray().First(card => card.Name == name).Id;

    private static List<CardInstance> Impatiences(int count) =>
        Pile(Enumerable.Repeat(CL.Impatience, count).ToArray());

    [Fact]
    public void CaptureSpiritInsertsEachSoulAtARandomDrawPilePositionOffTheShuffleStream()
    {
        var fight = Fight.Hand().Energy(9).Enemy(hp: 500);
        fight.State.Hand.Add(new CardInstance(IdOf("CaptureSpirit"), false));
        fight.State.DrawPile = Impatiences(4);
        fight.State.ShuffleRng = new CountingRandom(5);

        fight.Play(0, target: 0);

        var oracle = new CountingRandom(5);
        var expected = Impatiences(4);
        for (int i = 0; i < 3; i++)
        {
            expected.Insert(oracle.Next(expected.Count + 1), new CardInstance(Soul, false));
        }

        Assert.Equal(3, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(Fight.Ids(expected), Fight.Ids(fight.State.DrawPile));
    }

    [Fact]
    public void AnUpgradedCaptureSpiritMakesFourFromTheSameStream()
    {
        var fight = Fight.Hand().Energy(9).Enemy(hp: 500);
        fight.State.Hand.Add(new CardInstance(IdOf("CaptureSpirit"), true));
        fight.State.DrawPile = Impatiences(2);
        fight.State.ShuffleRng = new CountingRandom(8);

        fight.Play(0, target: 0);

        Assert.Equal(4, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(4, fight.State.DrawPile.Count(card => card.DefId == Soul));
    }

    /// <summary>One draw per Soul that lands in the draw pile, whichever card made it.</summary>
    [Theory]
    [InlineData("GlimpseBeyond", 3)]
    [InlineData("GraveWarden", 1)]
    [InlineData("Reave", 1)]
    [InlineData("Severance", 1)]
    public void EverySoulMakerRollsOneShufflePlacementPerSoulInTheDrawPile(string name, int souls)
    {
        var fight = Fight.Hand().Energy(9).Enemy(hp: 500);
        fight.State.Hand.Add(new CardInstance(IdOf(name), false));
        fight.State.DrawPile = Impatiences(2);
        fight.State.ShuffleRng = new CountingRandom(1);

        fight.Play(0, target: 0);

        Assert.Equal(souls, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(souls, fight.State.DrawPile.Count(card => card.DefId == Soul));
    }

    [Fact]
    public void DirgeRollsOnePlacementPerEnergySpent()
    {
        var fight = Fight.Hand().Energy(4).Enemy(hp: 500);
        fight.State.Hand.Add(new CardInstance(IdOf("Dirge"), false));
        fight.State.DrawPile = Impatiences(2);
        fight.State.ShuffleRng = new CountingRandom(2);

        fight.Play(0);

        Assert.Equal(4, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(4, fight.State.DrawPile.Count(card => card.DefId == Soul));
    }
}
