using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardCmd.AutoPlay` (CardCmd.cs:51-): an AnyEnemy card that is given no target rolls
/// `Rng.CombatTargets.NextItem(HittableEnemies)`, even over one enemy and whatever the
/// card's type; a card that is not AnyEnemy never touches that stream there.
/// `StampedePower.AfterAutoPostPlayPhaseEntered` hands it `null`, so each Stampede attack
/// rolls once. Referee seeds X10CFTPU9L, DLADDVSPCS, K3PVPMKMU6 and C1Q25M722L diverged on
/// the stream count right after END TURN.
/// </summary>
public class AutoPlayTargetRollTests
{
    private static Fight Setup(int handCard, CountingRandom targets)
    {
        var fight = Fight
            .Hand(new CardInstance(handCard, false))
            .Enemy(hp: 60)
            .Draw(new CardInstance(IC.DefendIronclad, false), new CardInstance(IC.DefendIronclad, false));
        fight.State.TargetRng = targets;
        return fight;
    }

    [Fact]
    public void AStampedeAttackRollsTheTargetStreamOnceEvenOverOneEnemy()
    {
        var targets = new CountingRandom(7);
        var fight = Setup(IC.StrikeIronclad, targets);
        fight.PlayerBuff(BuffId.Stampede, 1);

        fight.EndTurn();

        Assert.Equal(1, targets.CallCount);
        Assert.Equal(60 - 6, fight.State.Enemies[0].Hp);
    }

    [Fact]
    public void AStampedeAttackThatHitsEveryEnemyDoesNotRollTheTargetStream()
    {
        var targets = new CountingRandom(7);
        var fight = Setup(IC.Thunderclap, targets);
        fight.PlayerBuff(BuffId.Stampede, 1);

        fight.EndTurn();

        Assert.Equal(0, targets.CallCount);
    }

    [Fact]
    public void AHavocPlayedAllEnemyAttackDoesNotRollTheTargetStream()
    {
        var targets = new CountingRandom(7);
        var fight = Fight
            .Hand(new CardInstance(IC.Havoc, false))
            .Enemy(hp: 60)
            .Draw(new CardInstance(IC.Thunderclap, false), new CardInstance(IC.DefendIronclad, false));
        fight.State.TargetRng = targets;
        fight.Energy(3);

        fight.Play(0);

        Assert.Equal(0, targets.CallCount);
    }

    [Fact]
    public void AHavocPlayedAnyEnemyAttackRollsTheTargetStreamOnce()
    {
        var targets = new CountingRandom(7);
        var fight = Fight
            .Hand(new CardInstance(IC.Havoc, false))
            .Enemy(hp: 60)
            .Draw(new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.DefendIronclad, false));
        fight.State.TargetRng = targets;
        fight.Energy(3);

        fight.Play(0);

        Assert.Equal(1, targets.CallCount);
    }

    [Theory]
    [InlineData(IC.Thunderclap, 0)]
    [InlineData(IC.StrikeIronclad, 1)]
    public void AMayhemPlayedAttackRollsTheTargetStreamOnlyWhenItAimsAtOneEnemy(int top, int rolls)
    {
        var targets = new CountingRandom(7);
        var fight = Fight
            .Hand(new CardInstance(IC.DefendIronclad, false))
            .Enemy(hp: 60)
            .Draw(
                Enumerable.Range(0, 8).Select(_ => new CardInstance(top, false)).ToArray()
            );
        fight.State.TargetRng = targets;
        fight.PlayerBuff(BuffId.MayhemPower, 1);

        fight.EndTurn();

        Assert.Equal(rolls, targets.CallCount);
    }
}
