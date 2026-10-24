// game: the batch L2 death pairs and open pairs (docs/rebuild/MIGRATION_F08.md, batch L2): Shrink, Illusion, Imbalanced,
// Meat on the Bone, Asleep, the AfterRemoved pseudo-hook of nine powers, and the registered pairs of Reattach, Adaptable,
// Steam Eruption and Painful Stabs. A handler that changes nothing is pinned by the harness log (MonsterHarness); the
// others by their effect. With the handler deleted the log line or the effect is missing.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;
using static Sts2Emulator.Tests.Core2.Models.L2PlayerDamageTests;

namespace Sts2Emulator.Tests.Core2.Models;

public class L2DeathTests
{
    private static readonly Dictionary<string, Action<HandlerTable>> Removed = new()
    {
        ["SlumberPower"] = SlumberPower.Register, ["VitalSparkPower"] = VitalSparkPower.Register,
        ["SandpitPower"] = SandpitPower.Register, ["BurrowedPower"] = BurrowedPower.Register,
        ["HexPower"] = HexPower.Register, ["TangledPower"] = TangledPower.Register, ["RingingPower"] = RingingPower.Register,
        ["DampenPower"] = DampenPower.Register, ["ShrinkPower"] = ShrinkPower.Register,
    };

    public static IEnumerable<object[]> RemovedOnDeath() =>
    [
        ["SlumberPower", BuffId.Slumber, false], ["VitalSparkPower", BuffId.VitalSpark, false],
        ["SandpitPower", BuffId.Sandpit, false], ["BurrowedPower", BuffId.Burrowed, false],
        ["HexPower", BuffId.Hex, true], ["TangledPower", BuffId.Tangled, true], ["RingingPower", BuffId.Ringing, true],
        ["DampenPower", BuffId.Dampen, true], ["ShrinkPower", BuffId.Shrink, true],
    ];

    [Theory]
    [MemberData(nameof(RemovedOnDeath))]
    public void AfterRemovedReachesThePowerAtItsOwnersDeath(string power, BuffId buff, bool onPlayer)
    {
        var e = Foe();
        var s = Combat(e, Foe());
        BuffSystem.Apply(onPlayer ? s.PlayerBuffs : e.Buffs, buff, 1);
        var log = new List<string>();
        var k = Bind(s, log, Removed[power]);
        if (onPlayer)
            EnemyHitsPlayer(k, e, 99m);
        else
            Hit(k, e, 99m);
        Assert.Contains(log, l => l.StartsWith("AfterRemoved", StringComparison.Ordinal));
    }

    [Fact]
    public void ShrinkGoesWithTheShrinkerBeetle()
    {
        var beetle = Enemy("ShrinkerBeetle", 5);
        var other = Foe(5);
        var s = Combat(beetle, other, Foe());
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Shrink, -1);
        var k = K(s, ShrinkPower.Register);
        Hit(k, other, 9m);
        Assert.Equal(-1, BuffSystem.Get(s.PlayerBuffs, BuffId.Shrink));
        Hit(k, beetle, 9m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Shrink));
    }

    [Fact]
    public void MeatOnTheBoneHearsItsOwnersHpChange()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(Ids.Resolve(ListenerKind.Relic, "MeatOnTheBone")));
        var log = new List<string>();
        var k = Bind(s, log, MeatOnTheBone.Register);
        EnemyHitsPlayer(k, e, 5m);
        Assert.Contains(log, l => l.StartsWith("AfterCurrentHpChanged", StringComparison.Ordinal));
    }

    [Fact]
    public void AnIllusionStaysRevivingWithItsBuffsAndWithoutItsDebuffs()
    {
        var e = Foe(10);
        BuffSystem.Apply(e.Buffs, BuffId.Illusion, 1);
        BuffSystem.Apply(e.Buffs, BuffId.Strength, 3);
        BuffSystem.Apply(e.Buffs, BuffId.Vulnerable, 2);
        var s = Combat(e, Foe());
        var k = K(s, IllusionPower.Register);
        Hit(k, e, 20m);
        Assert.Contains(e, s.Enemies);
        Assert.Equal((1, 1, 3, 0), (BuffSystem.Get(e.Buffs, BuffId.Reviving), BuffSystem.Get(e.Buffs, BuffId.Illusion),
            BuffSystem.Get(e.Buffs, BuffId.Strength), BuffSystem.Get(e.Buffs, BuffId.Vulnerable)));
        Assert.False(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(1)), out _));
        Assert.True(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(2)), out _));
    }

    [Fact]
    public void AReattachSegmentRevivesWhileAnotherSegmentLives()
    {
        var a = Enemy("DecimillipedeSegment", 10);
        var b = Enemy("DecimillipedeSegment", 10);
        BuffSystem.Apply(a.Buffs, BuffId.Reattach, 1);
        BuffSystem.Apply(b.Buffs, BuffId.Reattach, 1);
        var s = Combat(a, b);
        var k = K(s, ReattachPower.Register);
        Hit(k, a, 20m);
        Assert.Contains(a, s.Enemies);
        Assert.Equal((2, 1), (BuffSystem.Get(a.Buffs, BuffId.Reviving), BuffSystem.Get(a.Buffs, BuffId.Reattach)));
        Assert.False(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(1)), out _));
        Assert.True(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(2)), out _));
    }

    [Fact]
    public void TheTestSubjectHoldsTheFightOpenAndRevives()
    {
        var t = Enemy("TestSubject", 10);
        BuffSystem.Apply(t.Buffs, BuffId.Adaptable, 1);
        var s = Combat(t);
        var k = K(s, AdaptablePower.Register);
        Assert.True(Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _));
        Hit(k, t, 20m);
        Assert.Contains(t, s.Enemies);
        Assert.Equal((1, 1), (BuffSystem.Get(t.Buffs, BuffId.Reviving), BuffSystem.Get(t.Buffs, BuffId.Adaptable)));
        Assert.False(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(1)), out _));
        Assert.True(CombatEnd.IsInProgress(k));
    }

    [Fact]
    public void TheWaterfallGiantStandsUpToBlowAndHoldsTheFightOpen()
    {
        var g = Enemy("WaterfallGiant", 10);
        BuffSystem.Apply(g.Buffs, BuffId.SteamEruption, 1);
        var s = Combat(g);
        var k = K(s, SteamEruptionPower.Register);
        Assert.True(Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _));
        Hit(k, g, 20m);
        Assert.Contains(g, s.Enemies);
        Assert.Equal(999_999_999, g.Hp);
        Assert.True(BuffSystem.Has(g.Buffs, BuffId.SteamEruption));
    }

    [Fact]
    public void APainfulStabsCorpseStaysInTheCombat()
    {
        var e = Foe(10);
        BuffSystem.Apply(e.Buffs, BuffId.PainfulStabs, 1);
        var s = Combat(e, Foe());
        var k = K(s, PainfulStabsPower.Register);
        Hit(k, e, 20m);
        Assert.Contains(e, s.Enemies);
    }

    [Fact]
    public void TheMatriarchWakesWhenStruck()
    {
        var m = Enemy("LagavulinMatriarch", 30);
        BuffSystem.Apply(m.Buffs, BuffId.Asleep, 1);
        BuffSystem.Apply(m.Buffs, BuffId.Plating, 8);
        var s = Combat(m);
        var k = K(s, AsleepPower.Register);
        Hit(k, m, 5m);
        Assert.Equal((0, 0, 1), (BuffSystem.Get(m.Buffs, BuffId.Asleep), BuffSystem.Get(m.Buffs, BuffId.Plating),
            BuffSystem.Get(m.Buffs, BuffId.Stunned)));
    }

    [Fact]
    public void TheBowlbugRockGoesOffBalanceWhenItsAttackIsFullyBlocked()
    {
        var rock = Enemy("BowlbugRock", 30);
        BuffSystem.Apply(rock.Buffs, BuffId.Imbalanced, 1);
        var s = Combat(rock);
        var k = K(s, ImbalancedPower.Register);
        EnemyHitsPlayer(k, rock, 5m);
        Assert.False(rock.OffBalance);
        s.PlayerBlock = 10;
        EnemyHitsPlayer(k, rock, 5m);
        Assert.True(rock.OffBalance);
    }
}
