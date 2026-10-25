// game: WaterfallGiant: its F08/F10 handlers (Core2/Models/Monsters/WaterfallGiant.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class WaterfallGiantTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("WaterfallGiant", HookId.AfterDeath), ("WaterfallGiant", HookId.BeforeRemovedFromRoom)], Pairs(WaterfallGiant.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        WaterfallGiant.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void ItsKillRunsAfterDeathThenTheRoomRemovalAndStandingUpIsNotItsOwn()
    {
        // game: WaterfallGiant.AfterDeath is music and a sound loop; standing up to blow (999999999 HP, ABOUT_TO_BLOW) is
        // SteamEruptionPower.AfterDeath, not registered here, so the giant simply dies and leaves.
        var giant = Enemy("WaterfallGiant", 30);
        BuffSystem.Apply(giant.Buffs, BuffId.SteamEruption, 1);
        int move = giant.MoveIndex;
        var other = Enemy("WaterfallGiant", 40);
        var s = Combat(giant, other);
        var log = new List<string>();
        Hit(Bind(s, log, WaterfallGiant.Register), giant, 99m);

        Assert.Equal(["AfterDeath 1<-1", "AfterDeath 2<-1", "BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(0, giant.Hp);
        Assert.Equal(30, giant.MaxHp);
        Assert.Equal(move, giant.MoveIndex);
        Assert.Equal(40, other.Hp);
    }
}
