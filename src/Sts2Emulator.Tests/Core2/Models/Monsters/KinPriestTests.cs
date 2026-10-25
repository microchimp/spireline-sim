// game: KinPriest: its F08/F10 handlers (Core2/Models/Monsters/KinPriest.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class KinPriestTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("KinPriest", HookId.AfterDeath)], Pairs(KinPriest.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        KinPriest.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void ItHearsEachFollowersDeathAndItsOwnAndChangesNothing()
    {
        // game: KinPriest.AfterDeath: a follower's death while the priest lives (music; the last one, a talk line) and the
        // priest's own death (music) are presentation only.
        var priest = Enemy("KinPriest", 60);
        var first = Enemy("KinFollower", 20);
        var last = Enemy("KinFollower", 20);
        var s = Combat(priest, first, last);
        var log = new List<string>();
        var k = Bind(s, log, KinPriest.Register);

        Hit(k, first, 99m);
        Hit(k, last, 99m);
        // Creature refs are roster slots: the first follower has left, so the last one dies in slot 2.
        Assert.Equal(["AfterDeath 1<-2", "AfterDeath 1<-2"], log);
        Assert.Equal([priest], s.Enemies);
        Assert.Equal(60, priest.Hp);
        Assert.Empty(priest.Buffs);

        var (s2, log2) = (Combat(Enemy("KinPriest", 60), Enemy("KinFollower", 20)), new List<string>());
        Hit(Bind(s2, log2, KinPriest.Register), s2.Enemies[0], 999m);
        Assert.Equal(["AfterDeath 1<-1"], log2);
        Assert.Equal(20, s2.Enemies.Single().Hp);
    }
}
