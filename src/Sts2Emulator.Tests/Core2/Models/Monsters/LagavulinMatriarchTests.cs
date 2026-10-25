// game: LagavulinMatriarch: her F08/F10 handlers (Core2/Models/Monsters/LagavulinMatriarch.cs) fire in the game's pass and
// order with the game's effect, through the real damage and kill. Deleting a handler leaves its pair unregistered and its
// call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class LagavulinMatriarchTests
{
    private static EnemyState Asleep()
    {
        var m = Enemy("LagavulinMatriarch", 60);
        BuffSystem.Apply(m.Buffs, BuffId.Plating, 12);
        BuffSystem.Apply(m.Buffs, BuffId.Asleep, 3);
        return m;
    }

    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF08AndF10SwitchesOnly()
    {
        Assert.Equal([("LagavulinMatriarch", HookId.AfterDamageReceived), ("LagavulinMatriarch", HookId.AfterDeath)],
            Pairs(LagavulinMatriarch.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F08 && sw != Switch.F10);
        LagavulinMatriarch.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void AHitBelowHalfIsHeardButTheWakeIsTheAsleepPowersNotHers()
    {
        // game: LagavulinMatriarch.AfterDamageReceived stops a visual effect and opens her shell's eyes (animation only);
        // the wake (Asleep and Plating off, the stun) is AsleepPower.AfterDamageReceived, not registered here.
        var m = Asleep();
        var other = Enemy("Crusher", 40);
        var s = Combat(m, other);
        var log = new List<string>();
        var k = Bind(s, log, LagavulinMatriarch.Register);

        Hit(k, m, 35m);
        Hit(k, other, 1m);

        Assert.Equal(["AfterDamageReceived 1<-1", "AfterDamageReceived 1<-2"], log);
        Assert.Equal(25, m.Hp);
        Assert.Equal(3, BuffSystem.Get(m.Buffs, BuffId.Asleep));
        Assert.Equal(12, BuffSystem.Get(m.Buffs, BuffId.Plating));
        Assert.Equal(0, BuffSystem.Get(m.Buffs, BuffId.Stunned));
    }

    [Fact]
    public void HerKillIsAfterDeathOnlyAndChangesNothing()
    {
        // A killed receiver gets no AfterDamageReceived; AfterDeath runs on every listener, hers included.
        var m = Asleep();
        var other = Enemy("Crusher", 40);
        var s = Combat(m, other);
        var log = new List<string>();
        Hit(Bind(s, log, LagavulinMatriarch.Register), m, 99m);

        Assert.Equal(["AfterDeath 1<-1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
