using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// MegaCrit.Sts2.Core.Commands/CreatureCmd.cs `Damage`: Hook.ModifyDamage fixes the hit's
// number, THEN Hook.BeforeDamageReceived runs, which is where ThornsPower retaliates
// (ThornsPower.BeforeDamageReceived). RupturePower.AfterDamageReceived turns the thorns
// damage (no card source) into Strength at once -- after the hit's damage is already set.
// The emulator ran Thorns first, so the Strength it gave raised the very hit that
// provoked it (seen live: a Toadpole at 15 took 7 from a Strike, the game took 6).
public class ThornsOrderTests
{
    [Fact]
    public void RuptureStrengthFromThornsDoesNotRaiseTheProvokingHit()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(1)
            .PlayerHp(48)
            .PlayerBuff(BuffId.RupturePower, 1)
            .Enemy(hp: 15, buffs: new BuffState(BuffId.Thorns, 2));

        fight.Play(0, target: 0);

        Assert.Equal(9, fight.Enemy0.Hp);
        Assert.Equal(46, fight.State.PlayerHp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>
    /// Per hit, not per card: every hit of Twin Strike fixes its number before its own
    /// Thorns retaliation, and the Strength the first retaliation gave is already in the
    /// second hit's number.
    /// </summary>
    [Fact]
    public void EveryHitOfAMultiHitCardFixesItsDamageBeforeItsOwnThorns()
    {
        var fight = Fight
            .Hand(Card(IC.TwinStrike))
            .Energy(1)
            .PlayerHp(48)
            .PlayerBuff(BuffId.RupturePower, 1)
            .Enemy(hp: 40, buffs: new BuffState(BuffId.Thorns, 2));

        fight.Play(0, target: 0);

        // 5 into the first hit, then Strength 1 from its thorns makes the second 6.
        Assert.Equal(29, fight.Enemy0.Hp);
        Assert.Equal(44, fight.State.PlayerHp);
        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>The ordering changes nothing without a Strength source.</summary>
    [Fact]
    public void ThornsStillRetaliatesOncePerPoweredHit()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(1)
            .PlayerHp(48)
            .Enemy(hp: 15, buffs: new BuffState(BuffId.Thorns, 2));

        fight.Play(0, target: 0);

        Assert.Equal(9, fight.Enemy0.Hp);
        Assert.Equal(46, fight.State.PlayerHp);
    }
}
