using Sts2Emulator.Core;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// An attack card that damages its target and then applies a power to it reads
// `cardPlay.Target` once (Uppercut.cs, Strangle.cs, Hang.cs, Debilitate.cs, Squash.cs,
// Whistle.cs, SicEm.cs OnPlay). If the hit kills that target, `PowerCmd.Apply` refuses it --
// `Creature.CanReceivePowers` is false once the corpse has left the combat -- and the power
// is lost. The emulator used to look the enemy up again after the hit, which handed the
// debuff to whichever enemy stood first by then.
public class AimedDebuffTests
{
    [Theory]
    [InlineData("Uppercut")]
    [InlineData("Strangle")]
    [InlineData("Debilitate")]
    [InlineData("Hang")]
    [InlineData("Squash")]
    [InlineData("Whistle")]
    public void ADebuffAfterAKillingHitIsLostAndNotPassedToTheNextEnemy(string card)
    {
        int id = GeneratedData.Cards.FindId(card)!.Value;
        var fight = Fight.Hand(Card(id)).Energy(9).Enemy(hp: 1).Enemy(hp: 80);

        fight.Play(index: 0, target: 0);

        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(80, survivor.Hp);
        Assert.Empty(survivor.Buffs);
    }

    [Fact]
    public void SicEmsPowerIsLostWithItsKilledTarget()
    {
        int id = GeneratedData.Cards.FindId("SicEm")!.Value;
        var fight = Fight.Hand(Card(id)).Energy(9).Enemy(hp: 1).Enemy(hp: 80);
        fight.State.OstyHp = 20;

        fight.Play(index: 0, target: 0);

        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(80, survivor.Hp);
        Assert.Empty(survivor.Buffs);
    }

    /// <summary>The ordinary case still lands: the debuff goes on the enemy the card was aimed at.</summary>
    [Theory]
    [InlineData("Uppercut", BuffId.Weak)]
    [InlineData("Strangle", BuffId.Strangle)]
    [InlineData("Debilitate", BuffId.Debilitate)]
    [InlineData("Hang", BuffId.Hang)]
    [InlineData("Squash", BuffId.Vulnerable)]
    public void ASurvivingTargetStillGetsTheDebuff(string card, BuffId expected)
    {
        int id = GeneratedData.Cards.FindId(card)!.Value;
        var fight = Fight.Hand(Card(id)).Energy(9).Enemy(hp: 200).Enemy(hp: 200);

        fight.Play(index: 0, target: 1);

        Assert.Empty(fight.Enemy0.Buffs);
        Assert.True(fight.EnemyBuffAmount(expected, 1) > 0);
    }
}
