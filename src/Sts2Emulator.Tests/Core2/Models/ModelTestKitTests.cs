// One small test per member the C1 round added to ModelTestKit for the death-hook batches (C3-C5): the argument builders
// for AfterDeath, BeforeDeath and ShouldDie, a creature dying but still listed, holders on the player and on an enemy
// (the N-holder setup of docs/rebuild/CLOUD_BRIEFS.md section 3, rule 5) and Osty in the combat. The older members are
// exercised by PilotTests.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class ModelTestKitTests
{
    private static HookEvent<T> Probe<T>(List<T> seen) where T : struct =>
        (HookCtx _, ListenerRef self, in T a, ChoiceCtx? ctx) =>
        {
            seen.Add(a);
            return HookStep.Done;
        };

    /// <summary>A kernel with the player holding The Boot (a relic that has no death handler of its own) and the probe
    /// registrations <paramref name="register" /> standing in for the death hooks under test.</summary>
    private static KernelState Probed(CombatState s, Action<ModelRegistration> register)
    {
        Hold(s, RelicEffects.TheBoot);
        return Kernel(s, t => register(t.Model(ListenerKind.Relic, "TheBoot")));
    }

    [Fact]
    public void WithOstyMakesOstyCreatureOneAndMovesTheEnemiesUp()
    {
        var s = WithOsty(Combat(40, 30), hp: 7);
        var k = Kernel(s, Registry.Pilot);
        Assert.Equal((7, 7), (s.OstyHp, s.OstyMaxHp));
        Assert.Equal(new CreatureRef(1), Ref(k, OldCreature.Osty));
        Assert.Equal(new CreatureRef(2), Ref(k, OldCreature.Of(s.Enemies[0])));
        Assert.Equal(new CreatureRef(3), Ref(k, OldCreature.Of(s.Enemies[1])));
        Assert.Equal(3, IndexOf(s, s.Enemies[1]));
        var plain = Combat(40);
        Assert.Equal(1, IndexOf(plain, plain.Enemies[0]));
    }

    [Fact]
    public void GivePowerPutsTheBuffOnThePlayerOrTheEnemyAndNamesItsListener()
    {
        var s = Combat(40, 40);
        var mine = GivePower(s, "DoubleDamagePower", BuffId.DoubleDamage, 2);
        var theirs = GivePower(s, "DoubleDamagePower", BuffId.DoubleDamage, 2, s.Enemies[1]);
        Assert.Equal(Power("DoubleDamagePower", 0), mine);
        Assert.Equal(Power("DoubleDamagePower", 2), theirs);
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.DoubleDamage));
        Assert.Equal(2, BuffSystem.Get(s.Enemies[1].Buffs, BuffId.DoubleDamage));
        Assert.Equal(0, BuffSystem.Get(s.Enemies[0].Buffs, BuffId.DoubleDamage));

        // The listeners are the dispatcher's: the enemy-owned one is the one an enemy-dealt card attack changes.
        var strike = Add(s, s.Hand, 1);
        var k = Kernel(s, Registry.Pilot);
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2, card: new CardHandle(strike.Uid)),
            out var changed));
        Assert.Equal([theirs], changed);
    }

    [Fact]
    public void HoldersGiveTheModelToThePlayerAndEveryEnemyAndEachActsForItsOwnerOnly()
    {
        var s = Combat(40, 40);
        var holders = Holders(s, "DoubleDamagePower", BuffId.DoubleDamage, 2);
        Assert.Equal(
            [Power("DoubleDamagePower", 0), Power("DoubleDamagePower", 1), Power("DoubleDamagePower", 2)], holders);
        var strike = Add(s, s.Hand, 1);
        var card = new CardHandle(strike.Uid);
        var k = Kernel(s, Registry.Pilot);

        // The player's card attack doubles once (not three times) and changes the player's listener alone; the same hit
        // dealt by the second enemy changes only that enemy's listener.
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out var changed));
        Assert.Equal([holders[0]], changed);
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2, card: card), out changed));
        Assert.Equal([holders[2]], changed);
    }

    [Fact]
    public void AfterDeathOfBuildsTheArgumentsTheDispatcherPassesOn()
    {
        var seen = new List<AfterDeathArgs>();
        var k = Probed(Combat(40, 40), t => t.On(HookId.AfterDeath, Probe(seen)));
        Hooks.AfterDeath(k, AfterDeathOf(2));
        Hooks.AfterDeath(k, AfterDeathOf(1, removalPrevented: true, animLength: 1.5f));
        Assert.Equal([new AfterDeathArgs(new CreatureRef(2), false, 0f), new AfterDeathArgs(new CreatureRef(1), true, 1.5f)],
            seen);
    }

    [Fact]
    public void BeforeDeathOfBuildsTheArgumentsTheDispatcherPassesOn()
    {
        var seen = new List<BeforeDeathArgs>();
        var k = Probed(Combat(40, 40), t => t.On(HookId.BeforeDeath, Probe(seen)));
        Hooks.BeforeDeath(k, BeforeDeathOf(2));
        Assert.Equal([new BeforeDeathArgs(new CreatureRef(2))], seen);
    }

    [Fact]
    public void ShouldDieOfBuildsTheArgumentsTheDispatcherPassesOn()
    {
        var seen = new List<ShouldDieArgs>();
        var k = Probed(Combat(40, 40), t => t.On(HookId.ShouldDie,
            (HookPredicate<ShouldDieArgs>)((HookCtx _, ListenerRef self, in ShouldDieArgs a) =>
            {
                seen.Add(a);
                return false;
            })));
        // The probe answers "does not die": the dispatcher reports it and names the listener that kept the creature alive.
        Assert.False(Hooks.ShouldDie(k, ShouldDieOf(1), out var preventer));
        Assert.Equal([new ShouldDieArgs(new CreatureRef(1))], seen);
        Assert.NotNull(preventer);
    }

    [Fact]
    public void DyingKeepsADeadCreatureListedUntilTheScopeEnds()
    {
        var s = Combat(40, 40);
        var dead = s.Enemies[0];
        var k = Kernel(s, Registry.Pilot);

        using (Dying(k, dead))
        {
            // The state a death override sees: 0 HP, marked dying, and still a creature of the combat at its place.
            Assert.Equal(0, dead.Hp);
            Assert.True(k.World!.IsDying(dead));
            Assert.Equal(new CreatureRef(1), OldCreatures.TryRef(k.World, OldCreature.Of(dead)));
            Assert.Equal(2, OldCreatures.Enemies(k.World).Count);
        }

        Assert.Equal(40, dead.Hp);
        Assert.False(k.World!.IsDying(dead));
        Assert.Equal(2, OldCreatures.Enemies(k.World).Count);
        // Without the mark a creature at 0 HP is gone from the combat.
        dead.Hp = 0;
        Assert.Null(OldCreatures.TryRef(k.World, OldCreature.Of(dead)));
        Assert.Equal(1, OldCreatures.Enemies(k.World).Count);
    }
}
