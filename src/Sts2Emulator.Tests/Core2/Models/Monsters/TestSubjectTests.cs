// game: TestSubject: its F08/F10 handlers (Core2/Models/Monsters/TestSubject.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class TestSubjectTests
{
    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("TestSubject", HookId.AfterDeath), ("TestSubject", HookId.BeforeRemovedFromRoom)], Pairs(TestSubject.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        TestSubject.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void ItsKillRunsAfterDeathThenTheRoomRemovalAndTheRespawnIsNotItsOwn()
    {
        // game: TestSubject.AfterDeath is a colour and music; the respawn (Reviving, the dead-state move) is
        // AdaptablePower.AfterDeath, not registered here, so the Test Subject simply dies and leaves.
        var subject = Enemy("TestSubject", 30);
        BuffSystem.Apply(subject.Buffs, BuffId.Adaptable, 1);
        var other = Enemy("Crusher", 40);
        var s = Combat(subject, other);
        var log = new List<string>();
        Hit(Bind(s, log, TestSubject.Register), subject, 99m);

        Assert.Equal(["AfterDeath 1<-1", "BeforeRemovedFromRoom 1"], log);
        Assert.Equal([other], s.Enemies);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(30, subject.MaxHp);
        Assert.Equal(0, BuffSystem.Get(subject.Buffs, BuffId.Reviving));
        Assert.Equal(40, other.Hp);
    }
}
