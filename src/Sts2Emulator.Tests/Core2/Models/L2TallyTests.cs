// game: the batch L2 AfterDamageReceived tallies Core2's damage command kept before (Beating Remnant, Hardened Shell, Lava
// Lamp) and Curl Up's arming (docs/rebuild/MIGRATION_F08.md, batch L2). With the handler deleted the tally stays empty.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;
using static Sts2Emulator.Tests.Core2.Models.L2PlayerDamageTests;

namespace Sts2Emulator.Tests.Core2.Models;

public class L2TallyTests
{
    [Fact]
    public void BeatingRemnantTalliesItsOwnersHpLossThisTurn()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.BeatingRemnant));
        var k = K(s, BeatingRemnant.Register);
        EnemyHitsPlayer(k, e, 6m);
        s.PlayerBlock = 3;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(8, s.UnblockedDamageThisTurn);
    }

    [Fact]
    public void WithoutTheirModelsTheDamageCommandKeepsNoListenerTally()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.HardenedShell, 10);
        var s = Combat(e);
        var k = K(s);
        EnemyHitsPlayer(k, e, 6m);
        Hit(k, e, 4m);
        Assert.Equal((0, false, 0), (s.UnblockedDamageThisTurn, s.TookUnblockedDamage, e.HardenedShellDamageThisTurn));
    }

    [Fact]
    public void HardenedShellTalliesTheHpItsOwnerLosesUnlessFullyBlocked()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.HardenedShell, 10);
        var s = Combat(e);
        var k = K(s, HardenedShellPower.Register);
        Hit(k, e, 4m);
        Assert.Equal(4, e.HardenedShellDamageThisTurn);
        e.Block = 10;
        Hit(k, e, 5m);
        Assert.Equal(4, e.HardenedShellDamageThisTurn);
    }

    [Fact]
    public void LavaLampLatchesOnBlockableUnblockedDamageOnly()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.LavaLamp));
        var k = K(s, LavaLamp.Register);
        SelfDamage(k, 3m);
        Assert.False(s.TookUnblockedDamage);
        EnemyHitsPlayer(k, e, 3m);
        Assert.True(s.TookUnblockedDamage);
    }

    [Fact]
    public void CurlUpArmsOnAPoweredHitFromACardAndGainsNoBlockYet()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.CurlUp, 5);
        var s = Combat(e);
        var k = K(s, CurlUpPower.Register);
        Hit(k, e, 3m);
        Assert.False(e.CurlUpArmed);
        CreatureCmd.Damage(k, [OldCreature.Of(e)], 3m, new ValueProp(ValueProps.Unpowered), OldCreature.Player,
            new CardHandle(7));
        Assert.False(e.CurlUpArmed);
        CreatureCmd.Damage(k, [OldCreature.Of(e)], 3m, ValueProps.PoweredAttack, OldCreature.Player, new CardHandle(7));
        Assert.True(e.CurlUpArmed);
        Assert.Equal(0, e.Block);
    }

    [Fact]
    public void BeatingRemnantCapsTheTurnsHpLossAt20()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.BeatingRemnant));
        var k = K(s, BeatingRemnant.Register);
        EnemyHitsPlayer(k, e, 15m);
        EnemyHitsPlayer(k, e, 10m);
        Assert.Equal((30, 20), (s.PlayerHp, s.UnblockedDamageThisTurn));
    }

    [Fact]
    public void HardenedShellCapsTheTurnsHpLossAtItsAmount()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.HardenedShell, 10);
        var s = Combat(e);
        var k = K(s, HardenedShellPower.Register);
        Hit(k, e, 15m);
        Hit(k, e, 5m);
        Assert.Equal((20, 10), (e.Hp, e.HardenedShellDamageThisTurn));
    }
}
