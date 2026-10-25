// game: SoulNexus: its F08/F10 handlers (Core2/Models/Monsters/SoulNexus.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class SoulNexusTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("SoulNexus", HookId.BeforeRemovedFromRoom), ("SoulNexus", HookId.Died)], Pairs(SoulNexus.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        SoulNexus.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void TheKillRunsItsDiedThenItsRoomRemovalAndNothingElse()
    {
        // Died (the event, before the removal query) and then BeforeRemovedFromRoom, each to the dying monster only;
        // both bodies are visual, so the state is the kill's alone.
        var dies = Enemy("SoulNexus");
        var other = Enemy("SoulNexus", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        var k = Bind(s, log, SoulNexus.Register);

        Hit(k, dies, 99m);

        Assert.Equal(["Died 1", "BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
