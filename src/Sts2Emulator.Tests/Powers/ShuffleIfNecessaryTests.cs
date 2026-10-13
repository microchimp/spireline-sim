using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardPileCmd.ShuffleIfNecessary` -- the discard pile returns only when the DRAW pile is
/// empty and the discard has something -- is the first line of `Draw`, of
/// `AutoPlayFromDrawPile` (Mayhem, Cascade, Distilled Chaos), of Foregone Conclusion's and
/// Toasty Mittens' `BeforeHandDraw`.
/// </summary>
/// <remarks>
/// Mayhem's loop (the engine's own copy of the helper Cascade already used) took the top of
/// an empty pile and did nothing: night04's Mayhem fights (KMWVQN7SVV, Y5LB1F6ZM6,
/// XNUW37WPAB...) are one reshuffle (`Rng.Shuffle` +8) short at the first end of turn after
/// Mayhem lands with the hand draw having spent the pile. There is one implementation now.
/// </remarks>
public class ShuffleIfNecessaryTests
{
    private static CardInstance Card(int id) => new(id, false);

    [Fact]
    public void MayhemReshufflesAnEmptyDrawPileBeforeItTakesTheTop()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 100)
            .Draw()
            .Discard(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad));
        fight.State.ShuffleRng = new CountingRandom(11);

        CombatEngine.AutoPlayFromDrawPile(fight.State, 1, new Random(0));

        // Three cards shuffled (two swaps), the top one played and sent back to the discard.
        Assert.Equal(2, fight.State.ShuffleRng.CallCount);
        Assert.Equal(2, fight.State.DrawPile.Count);
        Assert.Single(fight.State.DiscardPile);
        Assert.Equal(5, fight.State.PlayerBlock);
    }

    [Fact]
    public void ToastyMittensExhaustsTheTopOfTheReshuffledDiscard()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 100)
            .Draw()
            .Discard(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash), Card(IC.Anger));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.ToastyMittens));
        fight.State.ShuffleRng = new CountingRandom(5);

        RelicEffects.ApplyBeforeHandDraw(fight.State, turnNumber: 2, new Random(0));

        Assert.Single(fight.State.ExhaustPile);
        Assert.Equal(3, fight.State.DrawPile.Count);
        Assert.Empty(fight.State.DiscardPile);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>A draw that finds nothing anywhere shuffles nothing, so The Abacus pays nothing.</summary>
    [Fact]
    public void NoShuffleHappensWhenBothPilesAreEmpty()
    {
        var fight = Fight.Hand().Enemy(hp: 100).Draw().Discard();
        fight.State.Relics.Add(new RelicInstance(RelicEffects.TheAbacus));
        fight.State.PlayerBlock = 0;

        CardEffects.DrawCards(fight.State, 1, new Random(0));

        Assert.Equal(0, fight.State.PlayerBlock);
        Assert.Empty(fight.State.Hand);
    }
}
