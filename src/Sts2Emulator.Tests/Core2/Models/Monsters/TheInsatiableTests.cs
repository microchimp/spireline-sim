// game: TheInsatiable: its F08/F10 handlers (Core2/Models/Monsters/TheInsatiable.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class TheInsatiableTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("TheInsatiable", HookId.AfterDeath)], Pairs(TheInsatiable.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        TheInsatiable.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void EveryCopyIsToldOfTheDeathAndNothingChanges()
    {
        // Hook.AfterDeath runs on every listener of the combat, the dead copy included; the body is music only.
        var dies = Enemy("TheInsatiable");
        var other = Enemy("TheInsatiable", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        Hit(Bind(s, log, TheInsatiable.Register), dies, 99m);

        Assert.Equal(["AfterDeath 1<-1", "AfterDeath 2<-1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
