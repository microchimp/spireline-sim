using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 2-cost Skill. MegaCrit.Sts2.Core.Models.Cards/TheBomb.cs applies TheBombPower for
// DynamicVar("Turns", 3m) turns and then SetDamage(DynamicVar("BombDamage", 40m)) on the
// power it gets back; OnUpgrade raises the damage by 10 and leaves the fuse at 3.
//
// TheBombPower is PowerInstanceType.Instanced (PowerCmd.FindExistingInstanceForStacking
// returns null for it): every play is a power of its own with its own Amount (the turns left)
// and its own damage, and TheBombPower.BeforeSideTurnEnd runs once per instance. The emulator
// keeps each as its own BuffState: Magnitude the turns, Data the damage.
public class TheBombTests
{
    private static List<BuffState> Bombs(Fight fight) =>
        fight.State.PlayerBuffs.Where(b => b.Id == BuffId.TheBombPower).ToList();

    [Fact]
    public void SetsAThreeTurnFuseForFortyDamage()
    {
        var fight = Fight.Hand(Card(CL.TheBomb)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal([new BuffState(BuffId.TheBombPower, 3, 40)], Bombs(fight));
    }

    [Fact]
    public void UpgradedCarriesFifty()
    {
        var fight = Fight.Hand(Card(CL.TheBomb, upgraded: true)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal([new BuffState(BuffId.TheBombPower, 3, 50)], Bombs(fight));
    }

    [Fact]
    public void DoesNothingToEnemiesOnThePlayItself()
    {
        var fight = Fight.Hand(Card(CL.TheBomb)).Energy(2).Enemy(hp: 60).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(60, fight.Enemy0.Hp);
        Assert.Equal(60, fight.Enemy1.Hp);
    }

    /// <summary>
    /// The mechanism's own case: a second play is a second bomb with its own fuse and damage,
    /// not a fuse of six and a damage of eighty on the first.
    /// </summary>
    [Fact]
    public void ASecondBombIsAnInstanceOfItsOwn()
    {
        var fight = Fight.Hand(Card(CL.TheBomb), Card(CL.TheBomb, upgraded: true))
            .Energy(4)
            .Enemy(hp: 200);

        fight.Play();
        fight.Play();

        Assert.Equal(
            [new BuffState(BuffId.TheBombPower, 3, 40), new BuffState(BuffId.TheBombPower, 3, 50)],
            Bombs(fight)
        );
    }

    /// <summary>
    /// Each instance counts down and goes off by itself, oldest first: a bomb played a turn
    /// later keeps its own fuse and its own damage, and the first one's blow is 40 and no more.
    /// </summary>
    [Fact]
    public void InstancesTickAndExplodeOnTheirOwnFuses()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 300);
        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.TheBombPower, 2, data: 40).PlayerBuff(BuffId.TheBombPower, 3, data: 50);

        fight.EndTurn();

        Assert.Equal(300, fight.Enemy0.Hp);
        Assert.Equal(
            [new BuffState(BuffId.TheBombPower, 1, 40), new BuffState(BuffId.TheBombPower, 2, 50)],
            Bombs(fight)
        );

        fight.EndTurn();

        Assert.Equal(260, fight.Enemy0.Hp);
        Assert.Equal([new BuffState(BuffId.TheBombPower, 1, 50)], Bombs(fight));

        fight.EndTurn();

        Assert.Equal(210, fight.Enemy0.Hp);
        Assert.Empty(Bombs(fight));
    }

    /// <summary>
    /// Two bombs on the same fuse both go off the same turn end, one blow each and to EVERY
    /// enemy, and neither takes the other's damage with it.
    /// </summary>
    [Fact]
    public void TwoBombsOnTheSameFuseDealBothBlowsToEveryEnemy()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 200).Enemy(hp: 200);
        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.TheBombPower, 1, data: 40).PlayerBuff(BuffId.TheBombPower, 1, data: 50);

        fight.EndTurn();

        Assert.Equal(110, fight.Enemy0.Hp);
        Assert.Equal(110, fight.Enemy1.Hp);
        Assert.Empty(Bombs(fight));
    }

    /// <summary>
    /// A Bomb played on a later turn is behind the first by exactly the turns between the two
    /// plays -- the case a stacked fuse of 6 gets wrong in both directions.
    /// </summary>
    [Fact]
    public void ABombPlayedATurnLaterExplodesATurnLater()
    {
        var fight = Fight.Hand(Card(CL.TheBomb)).Energy(3).Enemy(hp: 400);
        fight.State.PlayerHp = 999;
        fight.Play();
        fight.EndTurn();
        fight.State.Hand = [Card(CL.TheBomb, upgraded: true)];
        fight.Energy(3);
        fight.Play();

        Assert.Equal(
            [new BuffState(BuffId.TheBombPower, 2, 40), new BuffState(BuffId.TheBombPower, 3, 50)],
            Bombs(fight)
        );

        fight.EndTurn();
        fight.EndTurn();

        Assert.Equal(360, fight.Enemy0.Hp);
        Assert.Equal([new BuffState(BuffId.TheBombPower, 1, 50)], Bombs(fight));
    }
}
