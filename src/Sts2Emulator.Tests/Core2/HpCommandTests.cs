// game: CreatureCmd.Heal, SetCurrentHp, SetMaxHp, SetMaxAndCurrentHp, GainMaxHp, LoseMaxHp and Creature.HealInternal
// (docs/rebuild/MIGRATION_F08.md section 3 item 9, batch L1): each command's HP arithmetic and the AfterCurrentHpChanged it
// dispatches, heard by a listener on the combat (the Crusher's handler, logged by the monster harness), as the game's methods
// read: the hook carries the asked-for amount, SetMaxHp has none, a heal brings the dead player's hooks back.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Monsters;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2;

public class HpCommandTests
{
    private static (CombatState s, KernelState k, List<string> log, EnemyState ear) Setup(int enemyHp = 30)
    {
        var ear = Enemy("Crusher", enemyHp);
        var s = Combat(ear);
        var log = new List<string>();
        return (s, Bind(s, log, Crusher.Register), log, ear);
    }

    [Fact]
    public void HealAnnouncesTheAskedForAmountEvenAtFullHp()
    {
        var (s, k, log, _) = Setup();
        Assert.Equal(0, CreatureCmd.Heal(k, OldCreature.Player, 7));
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal(["AfterCurrentHpChanged 1<-0 7"], log);

        log.Clear();
        s.PlayerHp = 45;
        Assert.Equal(5, CreatureCmd.Heal(k, OldCreature.Player, 10));
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal(["AfterCurrentHpChanged 1<-0 10"], log);

        log.Clear();
        Assert.Equal(0, CreatureCmd.Heal(k, OldCreature.Player, 0));
        Assert.Empty(log);
    }

    [Fact]
    public void HealOfTheDeadPlayerActivatesItsHooksAgain()
    {
        var (s, k, _, _) = Setup();
        s.PlayerHp = 0;
        k.World!.DeactivatePlayerHooks();
        Assert.False(OldState.PlayerActive(k.World));
        CreatureCmd.Heal(k, OldCreature.Player, 12);
        Assert.Equal(12, s.PlayerHp);
        Assert.True(OldState.PlayerActive(k.World));
        Assert.Equal(PlayerHooks.Active, k.World.PlayerHooks);
    }

    [Fact]
    public void SetCurrentHpOnTheDeadPlayerLeavesItsHooksOff()
    {
        var (s, k, _, _) = Setup();
        s.PlayerHp = 0;
        k.World!.DeactivatePlayerHooks();
        CreatureCmd.SetCurrentHp(k, OldCreature.Player, 12);
        Assert.Equal(12, s.PlayerHp);
        Assert.Equal(PlayerHooks.Deactivated, k.World.PlayerHooks);
    }

    [Fact]
    public void SetCurrentHpAnnouncesTheAskedForChangeAndKillsAtZero()
    {
        var (s, k, log, ear) = Setup();
        CreatureCmd.SetCurrentHp(k, OldCreature.Of(ear), 40);
        Assert.Equal(30, ear.Hp);
        Assert.Equal(["AfterCurrentHpChanged 1<-1 10"], log);

        log.Clear();
        CreatureCmd.SetCurrentHp(k, OldCreature.Of(ear), 30);
        Assert.Empty(log);

        CreatureCmd.SetCurrentHp(k, OldCreature.Of(ear), 0);
        Assert.Equal(["AfterCurrentHpChanged 1<-1 -30", "BeforeDeath 1<-1"], log);
        Assert.Empty(s.Enemies);
    }

    [Fact]
    public void SetMaxHpLowersTheHpWithNoHookAndKillsAtZero()
    {
        var (s, k, log, ear) = Setup();
        Assert.Equal(-10, CreatureCmd.SetMaxHp(k, OldCreature.Of(ear), 20));
        Assert.Equal((20, 20), (ear.MaxHp, ear.Hp));
        Assert.Empty(log);
        // A max of 0 takes the HP to 0 inside SetMaxHpInternal with no hook, so the kill finds no HP to lose: no
        // AfterCurrentHpChanged at all, only the death.
        Assert.Equal(-20, CreatureCmd.SetMaxHp(k, OldCreature.Of(ear), -5));
        Assert.Equal(["BeforeDeath 1<-1"], log);
        Assert.Empty(s.Enemies);
    }

    [Fact]
    public void SetMaxAndCurrentHpIsSetMaxHpThenSetCurrentHp()
    {
        var (_, k, log, ear) = Setup();
        ear.Hp = 10;
        CreatureCmd.SetMaxAndCurrentHp(k, OldCreature.Of(ear), 25);
        Assert.Equal((25, 25), (ear.MaxHp, ear.Hp));
        Assert.Equal(["AfterCurrentHpChanged 1<-1 15"], log);
    }

    [Fact]
    public void GainMaxHpHealsByTheMaxGained()
    {
        var (s, k, log, _) = Setup();
        s.PlayerHp = 40;
        CreatureCmd.GainMaxHp(k, OldCreature.Player, 5);
        Assert.Equal((55, 45), (s.PlayerMaxHp, s.PlayerHp));
        Assert.Equal(["AfterCurrentHpChanged 1<-0 5"], log);
        Assert.Throws<ArgumentException>(() => CreatureCmd.GainMaxHp(k, OldCreature.Player, -1));
    }

    [Fact]
    public void LoseMaxHpDealsTheExcessAsDamageThenLowersTheMax()
    {
        var (s, k, log, _) = Setup();
        CreatureCmd.LoseMaxHp(k, OldCreature.Player, 10, isFromCard: false);
        Assert.Equal((40, 40), (s.PlayerMaxHp, s.PlayerHp));
        Assert.Equal(["AfterCurrentHpChanged 1<-0 -10"], log);

        log.Clear();
        s.PlayerHp = 20;
        CreatureCmd.LoseMaxHp(k, OldCreature.Player, 15, isFromCard: true);
        Assert.Equal((25, 20), (s.PlayerMaxHp, s.PlayerHp));
        Assert.Empty(log);
        Assert.Throws<ArgumentException>(() => CreatureCmd.LoseMaxHp(k, OldCreature.Player, -1, false));
    }

    [Fact]
    public void LoseMaxHpNeverTakesTheMaxUnderOne()
    {
        var (s, k, _, ear) = Setup();
        ear.Hp = 5;
        CreatureCmd.LoseMaxHp(k, OldCreature.Of(ear), 30, isFromCard: false);
        Assert.Equal(1, ear.MaxHp);
        Assert.True(ear.Hp <= 0 || !s.Enemies.Contains(ear));
    }

    [Fact]
    public void HealOfAMonsterDoesNothingWhileTheCombatIsEnding()
    {
        // game: CreatureCmd.Heal returns at once for a non-player while CombatManager.IsEnding (no living primary enemy).
        var (s, k, log, ear) = Setup();
        ear.Hp = 0;
        Assert.True(CombatEnd.IsEnding(k));
        Assert.Equal(0, CreatureCmd.Heal(k, OldCreature.Of(ear), 9));
        Assert.Equal(0, ear.Hp);
        Assert.Empty(log);
        s.PlayerHp = 30;
        Assert.Equal(9, CreatureCmd.Heal(k, OldCreature.Player, 9));
        Assert.Equal(39, s.PlayerHp);
    }
}
