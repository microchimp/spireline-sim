// game: DecimillipedeSegment: its F08/F10 handlers (Core2/Models/Monsters/DecimillipedeSegment.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class DecimillipedeSegmentTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("DecimillipedeSegment", HookId.AfterDeath), ("DecimillipedeSegmentBack", HookId.AfterDeath), ("DecimillipedeSegmentFront", HookId.AfterDeath), ("DecimillipedeSegmentMiddle", HookId.AfterDeath)], Pairs(DecimillipedeSegment.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        DecimillipedeSegment.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void EverySegmentIsToldOfTheDeathAndNothingChanges()
    {
        // Hook.AfterDeath runs on every listener of the combat: both segment copies are told of the one death, the dead
        // one included (it is still in the combat at AfterDeath); the bodies are music only.
        var dies = Enemy("DecimillipedeSegment");
        var other = Enemy("DecimillipedeSegment", 40);
        var s = Combat(dies, other);
        var log = new List<string>();
        var k = Bind(s, log, DecimillipedeSegment.Register);

        Hit(k, dies, 99m);

        Assert.Equal(["AfterDeath 1<-1", "AfterDeath 2<-1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(40, other.Hp);
        Assert.Equal(50, s.PlayerHp);
    }

    [Fact]
    public void TheThreeGameSegmentsAreOneOldIdAndEachCopyIsAskedOnce()
    {
        // The old state lists Back, Front and Middle under one enemy id; the handler runs once per segment in the combat,
        // in roster order, each told of the middle one's death.
        var segments = new[] { Enemy("DecimillipedeSegment"), Enemy("DecimillipedeSegment"), Enemy("DecimillipedeSegment") };
        var s = Combat(segments);
        var log = new List<string>();
        var k = Bind(s, log, DecimillipedeSegment.Register);

        Hit(k, segments[1], 99m);

        Assert.Equal(["AfterDeath 1<-2", "AfterDeath 2<-2", "AfterDeath 3<-2"], log);
        Assert.Equal([segments[0], segments[2]], s.Enemies);
    }
}
