// game: SnappingJaxfruit: its F08/F10 handlers (Core2/Models/Monsters/SnappingJaxfruit.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class SnappingJaxfruitTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("SnappingJaxfruit", HookId.BeforeRemovedFromRoom)], Pairs(SnappingJaxfruit.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        SnappingJaxfruit.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void TheKillRunsItsRoomRemovalToTheLeavingMonsterOnlyAndChangesNothing()
    {
        // BeforeRemovedFromRoom goes to the leaving monster only; the body is sound or music only.
        var dies = Enemy("SnappingJaxfruit");
        var other = Enemy("SnappingJaxfruit", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        Hit(Bind(s, log, SnappingJaxfruit.Register), dies, 99m);

        Assert.Equal(["BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
