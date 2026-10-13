using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// An auto-play is a play: once past the gate it has STARTED (`CardPlayStarted`) and its
/// BeforeCardPlayed hooks count it. Ringing and Normality read `CardPlaysStarted` (the play in
/// progress included), Sloth its own BeforeCardPlayed count, ChainsOfBindingPower latches a
/// Bound card in BeforeCardPlayed, and Diamond Diadem counts in AfterCardPlayed -- all of
/// them auto-plays included. These run the real plays rather than setting the counters.
/// </summary>
public class AutoPlayCountersTests
{
    private const int DistilledChaos = 15;

    private static CardInstance Card(int id) => new(id, false);

    /// <summary>
    /// Havoc is the turn's first play and has started, so under Ringing the card it plays
    /// is refused: filed away (exhausted, as Havoc forces) without hitting.
    /// </summary>
    [Fact]
    public void UnderRingingTheCardHavocPlaysFirstIsRefused()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 60)
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Ringing, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(60, fight.State.Enemies[0].Hp);
        Assert.Contains(fight.State.ExhaustPile, card => card.DefId == IC.StrikeIronclad);
    }

    /// <summary>Normality: two plays done, Havoc is the third to start, its card is refused.</summary>
    [Fact]
    public void WithNormalityInHandTheCardHavocPlaysThirdIsRefused()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc), Card(ST.Normality))
            .Enemy(hp: 60)
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Energy(3);
        fight.State.CardPlaysStartedThisTurn = 2;
        fight.State.CardPlaysThisTurn = 2;

        fight.Play(0);

        Assert.Equal(60, fight.State.Enemies[0].Hp);
    }

    /// <summary>
    /// Sloth counts auto-plays: under Sloth 2, Distilled Chaos plays two Defends and the third
    /// is refused, and nothing is left to play from hand.
    /// </summary>
    [Fact]
    public void SlothCountsAutoPlays()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 60)
            .Draw(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Sloth, 2)
            .Energy(3);
        fight.State.PotionSlots[0] = DistilledChaos;

        fight.Potion(0);

        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.DoesNotContain(0, CombatEngine.ValidActions(fight.State));
    }

    /// <summary>A Bound card Havoc plays spends the turn's one Bound play.</summary>
    [Fact]
    public void ABoundCardHavocPlaysSpendsTheTurnsBoundPlay()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc), Card(IC.StrikeIronclad) with { Bound = true })
            .Enemy(hp: 60)
            .Draw(Card(IC.DefendIronclad) with { Bound = true }, Card(IC.DefendIronclad))
            .Energy(3);

        fight.Play(0);

        Assert.True(fight.State.BoundCardPlayedThisTurn);
        Assert.DoesNotContain(0, CombatEngine.ValidActions(fight.State));
    }

    /// <summary>
    /// Diamond Diadem counts every card played, auto-plays too: three Distilled Chaos plays
    /// and nothing from hand is not a quiet turn.
    /// </summary>
    [Fact]
    public void DiamondDiademCountsAutoPlays()
    {
        var fight = Fight
            .WithRelics(RelicEffects.DiamondDiadem)
            .Draw(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad));
        fight.State.PotionSlots[0] = DistilledChaos;
        fight.Potion(0);

        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(fight.State, new Random(0));

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.DiamondDiadem));
    }
}
