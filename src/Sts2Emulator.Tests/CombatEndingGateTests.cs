using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The combat-ending gate (`CombatManager.IsOverOrEnding`, <see cref="CombatEngine.IsOverOrEnding" />):
/// once the last primary enemy is dead, or the player's death is pending, the commands the game
/// gates do nothing. Each stopped command is tested beside the same play while the fight goes on.
/// </summary>
public class CombatEndingGateTests
{
    private static readonly CardInstance[] SixDefends =
    [
        Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad),
        Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad),
    ];

    /// <summary>A fight whose draw pile is empty, so the next draw reshuffles the discard.</summary>
    private static Fight EmptyDrawPile(int enemyHp, params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Enemy(hp: enemyHp).PlayerHp(60, 80).Draw().Discard(SixDefends);
        fight.State.ShuffleRng = new CountingRandom(0);
        return fight;
    }

    /// <summary>
    /// `CardPileCmd.Draw` returns at once when the fight is over or ending: a Pommel Strike
    /// that kills the last enemy draws nothing, so it reshuffles nothing on the run's
    /// Shuffle stream (corpus BYL0P7VTKB, BHFCKBYSXS: Rng.Shuffle ahead of the game's).
    /// </summary>
    [Theory]
    [InlineData(5, true)]
    [InlineData(50, false)]
    public void AKillingDrawCardDrawsNothing(int enemyHp, bool ends)
    {
        var fight = EmptyDrawPile(enemyHp, Card(IC.PommelStrike));

        var result = fight.Play(0);

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 0 : 1, fight.State.Hand.Count);
        Assert.Equal(ends ? 0 : 5, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// Pillage draws until it draws a non-Attack; after the kill its first draw is refused,
    /// which ends the loop.
    /// </summary>
    [Fact]
    public void PillageDrawsNothingAfterTheKill()
    {
        var fight = EmptyDrawPile(1, Card(IC.Pillage));

        var result = fight.Play(0);

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Empty(fight.State.Hand);
        Assert.Equal(0, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// The gate is asked again before each card: Speedster's hit on the first card drawn
    /// kills, and the other two of Battle Trance's three stay in the pile.
    /// </summary>
    [Theory]
    [InlineData(5, 1)]
    [InlineData(50, 3)]
    public void ADrawStopsAtTheCardWhoseHookEndsTheFight(int enemyHp, int drawn)
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Enemy(hp: enemyHp)
            .Draw(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad));
        fight.PlayerBuff(BuffId.Speedster, 5);

        fight.Play(0);

        Assert.Equal(drawn, fight.State.Hand.Count);
        Assert.Equal(3 - drawn, fight.State.DrawPile.Count);
    }

    /// <summary>
    /// `CardPileCmd.Shuffle` does nothing once the fight is over: no Shuffle draw, and the
    /// discard stays where it is.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(30, 5)]
    public void AShuffleOnceTheFightIsOverDoesNothing(int enemyHp, int shuffleDraws)
    {
        var fight = EmptyDrawPile(enemyHp);

        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(shuffleDraws, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(shuffleDraws == 0 ? 6 : 0, fight.State.DiscardPile.Count);
    }

    /// <summary>
    /// The player's pending death shuts the gate too (`IsEnding` with a pending loss): a dead
    /// player draws nothing. A Fairy in a Bottle or an unspent Lizard Tail still to answer the
    /// death keeps it open.
    /// </summary>
    [Fact]
    public void APendingLossShutsTheGateUnlessSomethingWillRefuseTheDeath()
    {
        var fight = Fight.Hand().Enemy(hp: 30).PlayerHp(0, 80);
        Assert.True(CombatEngine.IsOverOrEnding(fight.State));

        fight.State.Relics.Add(new RelicInstance(RelicEffects.LizardTail, 0));
        Assert.False(CombatEngine.IsOverOrEnding(fight.State));

        fight.State.Relics[0] = fight.State.Relics[0] with { Counter = 1 };
        Assert.True(CombatEngine.IsOverOrEnding(fight.State));
    }

    /// <summary>
    /// A creature whose death holds the fight open keeps the gate open until its death is
    /// answered (a Gremlin Merc's reinforcements, `ShouldStopCombatFromEnding`).
    /// </summary>
    [Fact]
    public void ADeathThatHoldsTheFightOpenKeepsTheGateOpen()
    {
        var fight = Fight.Hand().Enemy(hp: 0, buffs: new BuffState(BuffId.Surprise, 1));

        Assert.False(CombatEngine.IsOverOrEnding(fight.State));
    }

    /// <summary>
    /// Tools of the Trade's card is part of the hand draw (`ToolsOfTheTradePower.ModifyHandDraw`),
    /// so a Rolling Boulder kill in `AfterPlayerTurnStart` comes after it is dealt: six cards,
    /// the sixth reshuffling the four in the discard on the run's Shuffle stream. As a separate
    /// draw after the Boulder, the gate refused it: five cards, no reshuffle, Rng.Shuffle one
    /// shuffle behind the game's.
    /// </summary>
    [Theory]
    [InlineData(3, true)]
    [InlineData(50, false)]
    public void ToolsOfTheTradesCardIsDealtBeforeATurnStartKill(int enemyHp, bool ends)
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: enemyHp)
            .PlayerHp(60, 80)
            .Draw(SixDefends[..5])
            .Discard(SixDefends[..4]);
        fight.State.ShuffleRng = new CountingRandom(0);
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);
        fight.PlayerBuff(BuffId.ToolsOfTheTrade, 1);

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Empty(fight.State.DiscardPile);
        Assert.Equal(3, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// Being the hand draw, Tools of the Trade's card is not one Fiddle refuses
    /// (`Fiddle.ShouldDraw` spares the hand draw): five, Fiddle's two and the Tools card.
    /// </summary>
    [Fact]
    public void FiddleDoesNotRefuseToolsOfTheTradesCard()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 50)
            .PlayerHp(60, 80)
            .Draw([.. SixDefends, .. SixDefends]);
        fight.State.Relics = [new RelicInstance(RelicEffects.Fiddle)];
        fight.PlayerBuff(BuffId.ToolsOfTheTrade, 1);

        fight.EndTurn();

        Assert.Equal(8, fight.State.Hand.Count);
    }

    /// <summary>
    /// Inferno's burn and Rolling Boulder are `AfterPlayerTurnStart`, which `SetupPlayerTurn`
    /// dispatches after the hand draw: a turn start whose burn kills the last enemy has
    /// already dealt the hand and reshuffled for it (corpus L80WXWJ1F5: Rng.Shuffle 16 to 28
    /// across the end turn). They ran before the draw, which the gate would have refused.
    /// </summary>
    [Theory]
    [InlineData(BuffId.InfernoSelfDamage)]
    [InlineData(BuffId.RollingBoulderPower)]
    public void ATurnStartKillComesAfterTheHandDraw(BuffId killer)
    {
        var fight = EmptyDrawPile(3);
        if (killer == BuffId.InfernoSelfDamage)
        {
            fight.PlayerBuff(BuffId.Inferno, 6);
            fight.PlayerBuff(BuffId.InfernoSelfDamage, 1);
        }
        else
        {
            fight.PlayerBuff(BuffId.RollingBoulderPower, 5);
        }

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(5, fight.State.Hand.Count);
        Assert.Equal(5, fight.State.ShuffleRng!.CallCount);
    }
}
