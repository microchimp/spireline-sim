// Batch R7a (rebuild/f08-r7a), the R7 blockers of docs/rebuild/MIGRATION_F08.md: Die For You's pet redirect (the game's
// DieForYouPower.ModifyUnblockedDamageTarget, reached by the enemy-attack site through AttackCmd's CreatureCmd.Damage). The dealers are
// R7aDealerTests', the fork's run view ResampleLeakTests'. Game code is cited by class and
// method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class R7aTests
{
    [Fact]
    public void TheRegistrysR7aBlockHoldsDieForYouAndAllHoldsTheBlock()
    {
        var t = Table(Registry.R7a);
        AssertRegistered(t, ListenerKind.Power, "DieForYouPower", HookId.ModifyUnblockedDamageTarget);
        Assert.Single(Pairs(t));
        foreach (var r in Registry.R7a)
            Assert.Contains(r, Registry.All);
        Assert.Empty(Pairs(Table(Registry.R7a, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- Die For You

    [Fact]
    public void APoweredHitOnThePlayerGoesToALivingOstyAndItsOverkillBackToThePlayer()
    {
        // game: DieForYouPower.ModifyUnblockedDamageTarget, then CreatureCmd.Damage's overkill onto the original target.
        var s = WithOsty(Combat(40), 10);
        var k = Kernel(s, Registry.R7a);
        Hit(k, OldCreature.Player, 6m, dealer: OldCreature.Of(s.Enemies[0]));
        Assert.Equal((50, 4), (s.PlayerHp, s.OstyHp));
        Hit(k, OldCreature.Player, 7m, dealer: OldCreature.Of(s.Enemies[0]));
        Assert.Equal((47, 0), (s.PlayerHp, s.OstyHp));
    }

    [Fact]
    public void AnUnpoweredHitADeadOstyAndNoDieForYouLeaveTheHitOnThePlayer()
    {
        var s = WithOsty(Combat(40), 10);
        var k = Kernel(s, Registry.R7a);
        Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(s.Enemies[0]), props: Unpowered);
        Assert.Equal((45, 10), (s.PlayerHp, s.OstyHp));

        var dead = WithOsty(Combat(40), 10);
        dead.OstyHp = 0;
        Hit(Kernel(dead, Registry.R7a), OldCreature.Player, 5m, dealer: OldCreature.Of(dead.Enemies[0]));
        Assert.Equal((45, 0), (dead.PlayerHp, dead.OstyHp));

        var none = WithOsty(Combat(40), 10);
        Hit(Kernel(none, Without(Registry.R7a, Sts2Emulator.Core2.Models.Powers.DieForYouPower.Register)),
            OldCreature.Player, 5m, dealer: OldCreature.Of(none.Enemies[0]));
        Assert.Equal((45, 10), (none.PlayerHp, none.OstyHp));
    }

    [Fact]
    public void AnEnemysAttackOnThePlayerIsRedirectedToOstyAtTheEnemyAttackSite()
    {
        // The old suite's OstyDamageSinkTests fight, stepped under F08+F10 with every migrated model: the monster's attack
        // (AttackCmd, CreatureCmd.Damage) hits Osty for the unblocked damage and the player keeps its HP.
        var fight = Fight.Hand().Enemy(hp: 60);
        fight.State.PlayerHp = 60;
        fight.State.PlayerMaxHp = 60;
        CardEffects.SummonOsty(fight.State, 20);
        fight.State.Enemies[0].CurrentIntent = new Intent(IntentType.Attack, 8);
        _ = Kernel(fight.State, Registry.All);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, new Random(1));
        Assert.Equal(60, fight.State.PlayerHp);
        Assert.Equal(12, fight.State.OstyHp);
    }
}
