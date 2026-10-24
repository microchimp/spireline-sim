// game: Crusher: its F08/F10 handlers (Core2/Models/Monsters/Crusher.cs) fire in the game's pass and order with the game's
// effect, through the real damage and kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class CrusherTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF08AndF10SwitchesOnly()
    {
        Assert.Equal([("Crusher", HookId.AfterCurrentHpChanged), ("Crusher", HookId.BeforeDeath)],
            Pairs(Crusher.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F08 && sw != Switch.F10);
        Crusher.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void ItHearsEveryHpChangeAndItsOwnDeathAndChangesNothing()
    {
        // Hook.AfterCurrentHpChanged and Hook.BeforeDeath run on every listener: the arm hears the other arm's wound and
        // its own; the bodies are animations, sound and music only.
        var crusher = Enemy("Crusher", 30);
        var rocket = Enemy("Rocket", 40);
        var s = Combat(crusher, rocket);
        var log = new List<string>();
        var k = Bind(s, log, Crusher.Register);

        Hit(k, rocket, 10m);
        Assert.Equal(["AfterCurrentHpChanged 1<-2 -10"], log);
        Assert.Equal(30, rocket.Hp);

        log.Clear();
        Hit(k, crusher, 99m);
        Assert.Equal(["AfterCurrentHpChanged 1<-1 -30", "BeforeDeath 1<-1"], log);
        Assert.Equal([rocket], s.Enemies);
        Assert.Equal(30, rocket.Hp);
        Assert.Empty(rocket.Buffs);
        Assert.Equal(50, s.PlayerHp);
    }
}
