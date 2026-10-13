using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 3-cost Power. MegaCrit.Sts2.Core.Models.Cards/RollingBoulder.cs applies PowerVar<RollingBoulderPower>(5m), damage to all enemies at turn start that grows by DynamicVar("IncrementAmount", 5m); OnUpgrade raises the starting damage by 5.
public class RollingBoulderTests
{
    [Fact]
    public void AppliesFive()
    {
        var fight = Fight.Hand(Card(CL.RollingBoulder)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.RollingBoulderPower));
    }

    [Fact]
    public void UpgradedAppliesTen()
    {
        var fight = Fight.Hand(Card(CL.RollingBoulder, upgraded: true)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(10, fight.PlayerBuffAmount(BuffId.RollingBoulderPower));
    }

    /// <summary>
    /// <c>RollingBoulderPower</c> is PowerInstanceType.Instanced: a second Boulder is a power
    /// of its own, with its own Amount and its own climb, not 10 on the first one.
    /// </summary>
    [Fact]
    public void ASecondBoulderIsAnInstanceOfItsOwn()
    {
        var fight = Fight.Hand(Card(CL.RollingBoulder), Card(CL.RollingBoulder, upgraded: true))
            .Energy(6)
            .Enemy(hp: 400);

        fight.Play();
        fight.Play();

        Assert.Equal(
            [
                new BuffState(BuffId.RollingBoulderPower, 5),
                new BuffState(BuffId.RollingBoulderPower, 10),
            ],
            fight.State.PlayerBuffs
        );
    }

    /// <summary>
    /// <c>AfterPlayerTurnStart</c> runs once per instance: each deals its own Amount to every
    /// enemy and then grows by five on its own. Stacked, the pair would hit once for 15 and
    /// climb to 20.
    /// </summary>
    [Fact]
    public void EachInstanceHitsForItsOwnAmountAndClimbsOnItsOwn()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 400).Enemy(hp: 400);
        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5).PlayerBuff(BuffId.RollingBoulderPower, 10);

        fight.EndTurn();

        Assert.Equal(385, fight.Enemy0.Hp);
        Assert.Equal(385, fight.Enemy1.Hp);
        Assert.Equal(
            [
                new BuffState(BuffId.RollingBoulderPower, 10),
                new BuffState(BuffId.RollingBoulderPower, 15),
            ],
            fight.State.PlayerBuffs.Where(b => b.Id == BuffId.RollingBoulderPower)
        );

        fight.EndTurn();

        Assert.Equal(360, fight.Enemy0.Hp);
    }

    [Fact]
    public void LeavesPlayLikeAnyPower()
    {
        var fight = Fight.Hand(Card(CL.RollingBoulder)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.DiscardPile);
    }
}
