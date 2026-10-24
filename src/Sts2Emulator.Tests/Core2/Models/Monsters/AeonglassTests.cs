// game: Aeonglass: its F08/F10 handlers (Core2/Models/Monsters/Aeonglass.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class AeonglassTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("Aeonglass", HookId.AfterDeath)], Pairs(Aeonglass.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        Aeonglass.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void EveryCopyIsToldOfTheDeathAndNothingChanges()
    {
        // Hook.AfterDeath runs on every listener of the combat: both Aeonglass copies are told of the one death, the dead
        // one included (it is still in the combat at AfterDeath); the bodies are music only.
        var dies = Enemy("Aeonglass");
        var other = Enemy("Aeonglass", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        var k = Bind(s, log, Aeonglass.Register);

        Hit(k, dies, 99m);

        Assert.Equal(["AfterDeath 1<-1", "AfterDeath 2<-1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
