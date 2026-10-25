// game: ThievingHopper: its F08/F10 handlers (Core2/Models/Monsters/ThievingHopper.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class ThievingHopperTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("ThievingHopper", HookId.BeforeRemovedFromRoom)], Pairs(ThievingHopper.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        ThievingHopper.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void TheKillRunsItsRoomRemovalOnceAndNothingElse()
    {
        // The kill removes it from the room: BeforeRemovedFromRoom once, to the killed monster only (its twin is not asked);
        // the body is sound only, so the state is the kill's alone.
        var dies = Enemy("ThievingHopper");
        var other = Enemy("ThievingHopper", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        var k = Bind(s, log, ThievingHopper.Register);

        Hit(k, dies, 99m);

        Assert.Equal(["BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }

    [Fact]
    public void ItsEscapeRemovesItFromTheRoomTooWithNoDeathHook()
    {
        // game: CreatureCmd.Escape calls CombatManager.RemoveCreature, so the hopper's escape runs BeforeRemovedFromRoom.
        var hopper = Enemy("ThievingHopper");
        var other = Enemy("ThievingHopper", 40);
        var s = Combat(hopper, other);
        var log = new List<string>();
        var k = Bind(s, log, ThievingHopper.Register);

        CreatureCmd.Escape(k, OldCreature.Of(hopper));

        Assert.Equal(["BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
    }
}
