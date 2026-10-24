// game: Queen: her F08/F10 handlers (Core2/Models/Monsters/Queen.cs) fire in the game's pass and order with the game's
// effect, through the real kill. Deleting a handler leaves its pair unregistered and its call unlogged: these fail.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

public class QueenTests
{
    // The Queen's move numbers as EnemyAI.SelectIntent assigns them (CombatEngine's QueenBurnBrightMove, QueenEnrageMove).
    private const int BurnBright = 2;
    private const int Enrage = 5;

    private static (CombatState S, EnemyState Queen, EnemyState Amalgam) Fight(int queenMove = BurnBright)
    {
        var queen = Enemy("Queen", 300);
        queen.LastMove = queenMove;
        queen.CurrentIntent = new Intent(IntentType.Unknown, 0);
        var amalgam = Enemy("TorchHeadAmalgam", 50);
        return (Combat(queen, amalgam), queen, amalgam);
    }

    [Fact]
    public void ItRegistersTheGamesPairsUnderTheF10SwitchOnly()
    {
        Assert.Equal([("Queen", HookId.AfterDeath), ("Queen", HookId.BeforeRemovedFromRoom)],
            Pairs(Queen.Register).Select(h => (h.GameId, h.Hook)));
        var off = new HandlerTable(Ids, sw => sw != Switch.F10);
        Queen.Register(off);
        Assert.Equal(0, off.Count);
    }

    [Fact]
    public void TheAmalgamsDeathEnragesAQueenWhoAnnouncedBurnBrightThereAndThen()
    {
        // game: Queen.AfterDeath, SetMoveImmediate(EnragedState) when her next move is BURN_BRIGHT_FOR_ME.
        var (s, queen, amalgam) = Fight();
        var log = new List<string>();
        var k = Bind(s, log, Queen.Register);

        Hit(k, amalgam, 99m);

        Assert.Equal(["AfterDeath 1<-2"], log);
        Assert.Equal(Enrage, queen.LastMove);
        Assert.Equal(new Intent(IntentType.Buff, 2), queen.CurrentIntent);
        Assert.Equal([queen], s.Enemies);
    }

    [Fact]
    public void WithoutHerHandlerTheSwitchedOnKillLeavesTheAnnouncedMove()
    {
        // The handler is what carries the enrage under the switch (R4 dropped the old death answers there).
        var (s, queen, amalgam) = Fight();
        var k = Bind(s, []);
        Hit(k, amalgam, 99m);
        Assert.Equal(BurnBright, queen.LastMove);
    }

    [Fact]
    public void AnotherMoveAnotherDeadCreatureOrADeadQueenChangesNothing()
    {
        var (s, queen, amalgam) = Fight(queenMove: 1);
        Hit(Bind(s, [], Queen.Register), amalgam, 99m);
        Assert.Equal(1, queen.LastMove);

        // Not the amalgam: a second Queen dies, the first stays on her move.
        var other = Enemy("Queen", 10);
        var (s2, queen2, _) = Fight();
        s2.Enemies.Add(other);
        var log = new List<string>();
        Hit(Bind(s2, log, Queen.Register), other, 99m);
        Assert.Equal(BurnBright, queen2.LastMove);
        Assert.Equal(["AfterDeath 1<-3", "AfterDeath 3<-3", "BeforeRemovedFromRoom 3"], log);

        // A Queen who is not alive is not enraged: the handler asked directly, her HP at 0.
        var (s3, queen3, amalgam3) = Fight();
        var k3 = Bind(s3, [], Queen.Register);
        queen3.Hp = 0;
        Queen.AfterDeath(k3.Ctx, new ListenerRef((int)ListenerKind.Monster, Id("Queen"), 1),
            new AfterDeathArgs(new CreatureRef(2), false, 0f), null);
        Assert.Equal(BurnBright, queen3.LastMove);
        queen3.Hp = 300;
        Queen.AfterDeath(k3.Ctx, new ListenerRef((int)ListenerKind.Monster, Id("Queen"), 1),
            new AfterDeathArgs(new CreatureRef(2), false, 0f), null);
        Assert.Equal(Enrage, queen3.LastMove);
        Assert.Equal(50, amalgam3.Hp);
    }

    [Fact]
    public void HerOwnDeathRunsHerAfterDeathThenHerRoomRemovalAndChangesNothing()
    {
        var (s, queen, amalgam) = Fight();
        var log = new List<string>();
        Hit(Bind(s, log, Queen.Register), queen, 999m);
        Assert.Equal(["AfterDeath 1<-1", "BeforeRemovedFromRoom 1"], log);
        Assert.Equal([amalgam], s.Enemies);
        Assert.Equal(BurnBright, queen.LastMove);
    }

    [Fact]
    public void TheContextTellsAMonstersModelAndWhetherACreatureIsAlive()
    {
        var (s, queen, _) = Fight();
        var k = Bind(s, []);
        var c = k.Ctx;
        Assert.True(c.IsMonster(new CreatureRef(2), "TorchHeadAmalgam"));
        Assert.False(c.IsMonster(new CreatureRef(1), "TorchHeadAmalgam"));
        Assert.True(c.IsMonster(new CreatureRef(1), "Queen"));
        Assert.False(c.IsMonster(new CreatureRef(0), "Queen")); // the player
        Assert.False(c.IsMonster(new CreatureRef(7), "Queen")); // nobody there
        Assert.True(c.IsAlive(new CreatureRef(0)));
        Assert.True(c.IsAlive(new CreatureRef(1)));
        Assert.False(c.IsAlive(new CreatureRef(7)));
        // At 0 HP while her kill runs (still in the combat, marked dying) she is not alive.
        queen.Hp = 0;
        using (k.World.MarkDying(queen))
        {
            Assert.True(c.IsMonster(new CreatureRef(1), "Queen"));
            Assert.False(c.IsAlive(new CreatureRef(1)));
        }
    }
}
