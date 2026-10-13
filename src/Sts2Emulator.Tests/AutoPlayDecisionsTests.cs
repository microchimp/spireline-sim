using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `DecisionsDecisions.OnPlay` calls `CardCmd.AutoPlay(card, null)` three times on the SAME
/// card: three gates, three target rolls for an AnyEnemy skill, three counted plays, and one
/// card at the end. The emulator queued it once with ReplayCount + 2: one roll, one counted
/// play, and two replays left on the card for good.
/// </summary>
public class AutoPlayDecisionsTests
{
    private const int DecisionsDecisions = 129;

    private static Fight TauntThrice(out CountingRandom targets)
    {
        var fight = Fight
            // A second Skill in hand, so the choice is asked rather than answered alone.
            .Hand(
                new CardInstance(IC.Taunt, false),
                new CardInstance(DecisionsDecisions, false),
                new CardInstance(IC.DefendIronclad, false)
            )
            .Enemy(hp: 60)
            .Enemy(hp: 60)
            .Draw(Enumerable.Repeat(new CardInstance(IC.StrikeIronclad, false), 6).ToArray());
        fight.State.Stars = 9;
        targets = new CountingRandom(7);
        fight.State.TargetRng = targets;
        return fight;
    }

    private static IEnumerable<CardInstance> AllCards(CombatState state) =>
        state.Hand.Concat(state.DrawPile).Concat(state.DiscardPile).Concat(state.ExhaustPile);

    [Fact]
    public void EachOfTheThreePlaysRollsItsOwnTargetAndCounts()
    {
        var fight = TauntThrice(out var targets);
        fight.Play(1);
        int playsBefore = fight.State.CardPlaysThisTurn;

        fight.Choose(0);

        Assert.Equal(3, targets.CallCount);
        Assert.Equal(playsBefore + 3, fight.State.CardPlaysThisTurn);
        var taunt = Assert.Single(AllCards(fight.State), card => card.DefId == IC.Taunt);
        Assert.Equal(0, taunt.ReplayCount);
        Assert.Contains(fight.State.DiscardPile, card => card.DefId == IC.Taunt);
    }

    /// <summary>
    /// Each call passes the gate on its own: under Velvet Choker (six plays a turn) with four
    /// already played, the third auto-play is refused and only two Taunts land.
    /// </summary>
    [Fact]
    public void EachOfTheThreePlaysPassesTheGateOnItsOwn()
    {
        var fight = TauntThrice(out var targets);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.VelvetChoker));
        fight.Play(1);
        fight.State.CardPlaysThisTurn = 4;

        fight.Choose(0);

        Assert.Equal(2, targets.CallCount);
        Assert.Equal(2 * 7, fight.State.PlayerBlock);
        Assert.Single(AllCards(fight.State), card => card.DefId == IC.Taunt);
    }
}
