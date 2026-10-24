// game: the other registered pairs of the batch L2 AfterRemoved powers (docs/rebuild/MIGRATION_F08.md, batch L2): Slumber's
// AfterDamageReceived, Hex's and Dampen's AfterDeath, Burrowed's AfterBlockBroken. With the handler deleted the effect is
// missing, so each fails.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;
using static Sts2Emulator.Tests.Core2.Models.L2PlayerDamageTests;

namespace Sts2Emulator.Tests.Core2.Models;

public class L2DeathExtraTests
{
    [Fact]
    public void SlumberCountsDownPerUnblockedHitAndStunsTheBeetleAwake()
    {
        var b = Enemy("SlumberingBeetle", 30);
        BuffSystem.Apply(b.Buffs, BuffId.Slumber, 2);
        var s = Combat(b);
        var k = K(s, SlumberPower.Register);
        Hit(k, b, 3m);
        Assert.Equal((1, 0), (BuffSystem.Get(b.Buffs, BuffId.Slumber), BuffSystem.Get(b.Buffs, BuffId.Stunned)));
        Hit(k, b, 3m);
        Assert.Equal((0, 1), (BuffSystem.Get(b.Buffs, BuffId.Slumber), BuffSystem.Get(b.Buffs, BuffId.Stunned)));
    }

    [Fact]
    public void HexGoesWithTheSpectralKnight()
    {
        var knight = Enemy("SpectralKnight", 5);
        var other = Foe(5);
        var s = Combat(knight, other, Foe());
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Hex, 1);
        var k = K(s, HexPower.Register);
        Hit(k, other, 9m);
        Assert.Equal(1, BuffSystem.Get(s.PlayerBuffs, BuffId.Hex));
        Hit(k, knight, 9m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Hex));
    }

    [Fact]
    public void DampenGoesWithTheLastMagiKnight()
    {
        var a = Enemy("MagiKnight", 5);
        var b = Enemy("MagiKnight", 5);
        var s = Combat(a, b, Foe());
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Dampen, 1);
        var k = K(s, DampenPower.Register);
        Hit(k, a, 9m);
        Assert.Equal(1, BuffSystem.Get(s.PlayerBuffs, BuffId.Dampen));
        Hit(k, b, 9m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Dampen));
    }

    [Fact]
    public void BreakingTheTunnelersBlockEndsItsBurrow()
    {
        var t = Enemy("Tunneler", 30);
        BuffSystem.Apply(t.Buffs, BuffId.Burrowed, 1);
        t.Block = 5;
        var s = Combat(t);
        var k = K(s, BurrowedPower.Register);
        Hit(k, t, 3m);
        Assert.Equal(1, BuffSystem.Get(t.Buffs, BuffId.Burrowed));
        Hit(k, t, 4m);
        Assert.Equal((0, 1), (BuffSystem.Get(t.Buffs, BuffId.Burrowed), BuffSystem.Get(t.Buffs, BuffId.Stunned)));
    }

    [Fact]
    public void ShrinkTakes30PercentOffItsOwnersPoweredAttacks()
    {
        var s = Combat(Foe());
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Shrink, -1);
        var k = K(s, ShrinkPower.Register);
        ModifyDamageArgs Attack(ValueProp props) =>
            new(new CreatureRef(1), new CreatureRef(0), 10m, props, null, ModifyDamageHookType.All, PreviewModes.None);
        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(ValueProps.PoweredAttack), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(new ValueProp(ValueProps.Unpowered)), out _));
    }
}
