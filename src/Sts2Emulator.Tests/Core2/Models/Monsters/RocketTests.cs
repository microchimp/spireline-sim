// game: Rocket: its F08/F10 handlers (Core2/Models/Monsters/Rocket.cs) fire in the game's pass and order with the game's
// effect, through the real damage and kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class RocketTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF08AndF10SwitchesOnly()
    {
        Assert.Equal(
            [("Rocket", HookId.AfterCurrentHpChanged), ("Rocket", HookId.BeforeDeath), ("Rocket", HookId.BeforeRemovedFromRoom)],
            Pairs(Rocket.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F08 && sw != Switch.F10);
        Rocket.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void ItsKillRunsTheHpChangeTheDeathAndTheRoomRemovalAndChangesNothing()
    {
        // The damage's HP change (every listener), BeforeDeath (every listener), then BeforeRemovedFromRoom to the leaving
        // arm only; the bodies are animations, sound and music only.
        var crusher = Enemy("Crusher", 30);
        var rocket = Enemy("Rocket", 40);
        var s = Combat(crusher, rocket);
        var log = new List<string>();
        var k = Bind(s, log, Rocket.Register);

        Hit(k, crusher, 5m);
        Assert.Equal(["AfterCurrentHpChanged 2<-1 -5"], log);

        log.Clear();
        Hit(k, rocket, 99m);
        Assert.Equal(["AfterCurrentHpChanged 2<-2 -40", "BeforeDeath 2<-2", "BeforeRemovedFromRoom 2"], log);
        Assert.Equal([crusher], s.Enemies);
        Assert.Equal(25, crusher.Hp);
        Assert.Empty(crusher.Buffs);
        Assert.Equal(50, s.PlayerHp);
    }
}
