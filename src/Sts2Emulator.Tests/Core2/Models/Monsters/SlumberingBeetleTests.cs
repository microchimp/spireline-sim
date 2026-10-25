// game: SlumberingBeetle: its F08/F10 handlers (Core2/Models/Monsters/SlumberingBeetle.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class SlumberingBeetleTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("SlumberingBeetle", HookId.Died)], Pairs(SlumberingBeetle.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        SlumberingBeetle.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void TheKillRaisesItsDiedOnceAndNothingElse()
    {
        // The Died event goes to the dying beetle only (its twin is not asked); its body stops an effect only.
        var dies = Enemy("SlumberingBeetle");
        var other = Enemy("SlumberingBeetle", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        var k = Bind(s, log, SlumberingBeetle.Register);

        Hit(k, dies, 99m);

        Assert.Equal(["Died 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }
}
