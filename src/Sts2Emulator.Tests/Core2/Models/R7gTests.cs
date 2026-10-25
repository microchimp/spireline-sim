// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): the F08/F10 pairs `port_list --unreached` listed, which the gate now
// counts. Shared test rules 1-5 (docs/rebuild/CLOUD_BRIEFS.md section 3). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class R7gTests
{
    private static int Buff(List<BuffState> buffs, BuffId id) => BuffSystem.Get(buffs, id);

    private static readonly (ListenerKind, string, HookId)[] Expected =
    [
        (ListenerKind.Power, "EnvenomPower", HookId.AfterDamageGiven),
        (ListenerKind.Power, "MonarchsGazePower", HookId.AfterDamageGiven),
        (ListenerKind.Power, "NecroMasteryPower", HookId.AfterCurrentHpChanged),
        (ListenerKind.Power, "ReaperFormPower", HookId.AfterDamageGiven),
        (ListenerKind.Power, "ReflectPower", HookId.AfterDamageReceived),
        (ListenerKind.Power, "SwordSagePower", HookId.AfterRemoved),
        (ListenerKind.Power, "DieForYouPower", HookId.ShouldAllowHitting),
        (ListenerKind.Power, "DieForYouPower", HookId.ShouldCreatureBeRemovedFromCombatAfterDeath),
        (ListenerKind.Power, "DieForYouPower", HookId.ShouldPowerBeRemovedAfterOwnerDeath),
        (ListenerKind.Relic, "BookRepairKnife", HookId.AfterDiedToDoom),
        (ListenerKind.Relic, "EmotionChip", HookId.AfterDamageReceived),
    ];

    // ------------------------------------------------------------------ rules 1 and 4, the table test

    [Fact]
    public void TheBlockHoldsTheElevenPairsAndAllHoldsTheBlock()
    {
        var t = Table(Registry.R7g);
        Assert.Equal(Expected.OrderBy(p => p.ToString()), Pairs(t).Select(p => (p.Kind, p.Model, p.Hook)).OrderBy(p => p.ToString()));
        foreach (var (kind, model, hook) in Expected)
            AssertRegistered(t, kind, model, hook);
        var all = Pairs(Table(Registry.All)).Select(p => (p.Kind, p.Model, p.Hook)).ToHashSet();
        Assert.All(Expected, p => Assert.Contains(p, all));
    }

    [Fact]
    public void WithF08AndF10OffThePairsAreAbsent()
    {
        var t = Table(Registry.R7g, AllButDamage);
        foreach (var (kind, model, hook) in Expected)
            AssertAbsent(t, kind, model, hook);
    }

    // ------------------------------------------------------------------ rules 2, 3 and 5: behaviour, mutant, N holders

    [Fact]
    public void EnvenomPoisonsTheTargetOfTheOwnersPoweredHitThatGetsThrough()
    {
        // game: EnvenomPower.AfterDamageGiven: dealer == Owner, powered, unblocked > 0. Held by the player and every enemy
        // (rule 5): the player's hit poisons once, for the player's Envenom alone.
        var s = Combat(40, 40);
        Holders(s, "EnvenomPower", BuffId.Envenom, 2);
        Hit(Kernel(s, Registry.R7g), OldCreature.Of(s.Enemies[0]), 5m);
        Assert.Equal((35, 2, 0), (s.Enemies[0].Hp, Buff(s.Enemies[0].Buffs, BuffId.Poison), Buff(s.Enemies[1].Buffs, BuffId.Poison)));

        var blocked = Combat(40);
        blocked.Enemies[0].Block = 10;
        GivePower(blocked, "EnvenomPower", BuffId.Envenom, 2);
        Hit(Kernel(blocked, Registry.R7g), OldCreature.Of(blocked.Enemies[0]), 5m);
        Assert.Equal(0, Buff(blocked.Enemies[0].Buffs, BuffId.Poison));

        var mutant = Combat(40);
        GivePower(mutant, "EnvenomPower", BuffId.Envenom, 2);
        Hit(Kernel(mutant, Without(Registry.R7g, EnvenomPower.Register)), OldCreature.Of(mutant.Enemies[0]), 5m);
        Assert.Equal(0, Buff(mutant.Enemies[0].Buffs, BuffId.Poison));
    }

    [Fact]
    public void MonarchsGazeTakesStrengthOffWhatTheOwnersPoweredHitStruck()
    {
        // game: MonarchsGazePower.AfterDamageGiven: dealer == Owner, powered, whatever the damage (a blocked hit too).
        var s = Combat(40);
        s.Enemies[0].Block = 10;
        GivePower(s, "MonarchsGazePower", BuffId.MonarchsGaze, 3);
        Hit(Kernel(s, Registry.R7g), OldCreature.Of(s.Enemies[0]), 5m);
        var direct = Combat(40);
        CardEffects.ApplyTemporaryStrengthDownTo(direct.Enemies[0], 3);
        Assert.Equal(Buff(direct.Enemies[0].Buffs, BuffId.Strength), Buff(s.Enemies[0].Buffs, BuffId.Strength));
        Assert.Equal(-3, Buff(s.Enemies[0].Buffs, BuffId.Strength));

        var unpowered = Combat(40);
        GivePower(unpowered, "MonarchsGazePower", BuffId.MonarchsGaze, 3);
        Hit(Kernel(unpowered, Registry.R7g), OldCreature.Of(unpowered.Enemies[0]), 5m, props: Unpowered);
        Assert.Equal(0, Buff(unpowered.Enemies[0].Buffs, BuffId.Strength));

        var mutant = Combat(40);
        GivePower(mutant, "MonarchsGazePower", BuffId.MonarchsGaze, 3);
        Hit(Kernel(mutant, Without(Registry.R7g, MonarchsGazePower.Register)), OldCreature.Of(mutant.Enemies[0]), 5m);
        Assert.Equal(0, Buff(mutant.Enemies[0].Buffs, BuffId.Strength));
    }

    [Fact]
    public void ReaperFormDoomsTheTotalDamageTimesItsAmount()
    {
        // game: ReaperFormPower.AfterDamageGiven: result.TotalDamage * Amount, blocked damage included.
        var s = Combat(40);
        s.Enemies[0].Block = 2;
        GivePower(s, "ReaperFormPower", BuffId.ReaperForm, 2);
        Hit(Kernel(s, Registry.R7g), OldCreature.Of(s.Enemies[0]), 5m);
        Assert.Equal((37, 10), (s.Enemies[0].Hp, Buff(s.Enemies[0].Buffs, BuffId.Doom)));

        var mutant = Combat(40);
        GivePower(mutant, "ReaperFormPower", BuffId.ReaperForm, 2);
        Hit(Kernel(mutant, Without(Registry.R7g, ReaperFormPower.Register)), OldCreature.Of(mutant.Enemies[0]), 5m);
        Assert.Equal(0, Buff(mutant.Enemies[0].Buffs, BuffId.Doom));
    }

    [Fact]
    public void ReflectSendsTheBlockedPartOfAPoweredHitBackToTheDealer()
    {
        // game: ReflectPower.AfterDamageReceived: target == Owner, blocked > 0, powered, dealer known.
        var s = Combat(40);
        s.PlayerBlock = 4;
        GivePower(s, "ReflectPower", BuffId.Reflect, 1);
        Hit(Kernel(s, Registry.R7g), OldCreature.Player, 6m, dealer: OldCreature.Of(s.Enemies[0]));
        Assert.Equal((48, 36), (s.PlayerHp, s.Enemies[0].Hp));

        var mutant = Combat(40);
        mutant.PlayerBlock = 4;
        GivePower(mutant, "ReflectPower", BuffId.Reflect, 1);
        Hit(Kernel(mutant, Without(Registry.R7g, ReflectPower.Register)), OldCreature.Player, 6m,
            dealer: OldCreature.Of(mutant.Enemies[0]));
        Assert.Equal(40, mutant.Enemies[0].Hp);
    }

    [Fact]
    public void NecroMasteryHitsEveryEnemyForWhatOstyLost()
    {
        // game: NecroMasteryPower.AfterCurrentHpChanged: delta < 0 on the owner's Osty, -delta * Amount at every hittable
        // enemy, unblockable. Osty had 3: a hit of 5 takes 3, so 6 each.
        var s = WithOsty(Combat(20, 20), hp: 3);
        s.Enemies[1].Block = 5;
        GivePower(s, "NecroMasteryPower", BuffId.NecroMastery, 2);
        OldEntry.HitOsty(Kernel(s, Registry.R7g), 5);
        Assert.Equal((0, 14, 14), (s.OstyHp, s.Enemies[0].Hp, s.Enemies[1].Hp));

        var mutant = WithOsty(Combat(20), hp: 3);
        GivePower(mutant, "NecroMasteryPower", BuffId.NecroMastery, 2);
        OldEntry.HitOsty(Kernel(mutant, Without(Registry.R7g, NecroMasteryPower.Register)), 5);
        Assert.Equal(20, mutant.Enemies[0].Hp);
    }

    [Fact]
    public void SwordSageTakesItsReplaysBackFromTheBladesWhenRemoved()
    {
        // game: SwordSagePower.AfterRemoved: every Sovereign Blade loses Amount replays. The sweep's call, made directly.
        var s = Combat(40);
        var blade = Add(s, s.Hand, 448);
        s.Hand[0] = blade with { ReplayCount = 3 };
        var k = Kernel(s, Registry.R7g);
        SwordSagePower.AfterRemoved(k.Ctx, Power("SwordSagePower", 0), new AfterRemovedArgs(OldCreature.Player, BuffId.SwordSage,
            [new BuffState(BuffId.SwordSage, 2)]), null);
        Assert.Equal(1, s.Hand[0].ReplayCount);
        SwordSagePower.AfterRemoved(k.Ctx, Power("SwordSagePower", 1), new AfterRemovedArgs(OldCreature.Of(s.Enemies[0]),
            BuffId.SwordSage, [new BuffState(BuffId.SwordSage, 2)]), null);
        Assert.Equal(1, s.Hand[0].ReplayCount);
    }

    [Fact]
    public void BookRepairKnifeHealsPerFatalDoomDeathAsTheOldSiteDoes()
    {
        // game: BookRepairKnife.AfterDiedToDoom: the heal times the creatures not the owner whose death is fatal. No Core2
        // command dispatches the hook yet (the old KillDoomedEnemies heals, once); the dispatch equals the old call.
        var s = Combat(40, 40);
        s.PlayerHp = 30;
        Hold(s, RelicEffects.BookRepairKnife);
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Minion, 1));
        var k = Kernel(s, Registry.R7g);
        Hooks.AfterDiedToDoom(k, new AfterDiedToDoomArgs([Ref(k, OldCreature.Of(s.Enemies[0])), Ref(k, OldCreature.Of(s.Enemies[1]))]));
        var direct = Combat(40);
        direct.PlayerHp = 30;
        Hold(direct, RelicEffects.BookRepairKnife);
        RelicEffects.ApplyAfterDiedToDoom(direct, 1);
        Assert.Equal(direct.PlayerHp, s.PlayerHp);
        Assert.Equal(33, s.PlayerHp);

        var mutant = Combat(40);
        mutant.PlayerHp = 30;
        Hold(mutant, RelicEffects.BookRepairKnife);
        var km = Kernel(mutant, Without(Registry.R7g, BookRepairKnife.Register));
        Hooks.AfterDiedToDoom(km, new AfterDiedToDoomArgs([Ref(km, OldCreature.Of(mutant.Enemies[0]))]));
        Assert.Equal(30, mutant.PlayerHp);
    }

    [Fact]
    public void DieForYouAllowsHittingTheLivingOnlyAndKeepsItsDeadOsty()
    {
        // game: DieForYouPower.ShouldAllowHitting (creature.IsAlive), ShouldCreatureBeRemovedFromCombatAfterDeath (false
        // for the owner) and ShouldPowerBeRemovedAfterOwnerDeath (false).
        var s = WithOsty(Combat(40, 40));
        s.Enemies[1].Hp = 0;
        var k = Kernel(s, Registry.R7g);
        var osty = Ref(k, OldCreature.Osty);
        var dead = new CreatureRef(3); // Osty is 1, the enemies 2 and 3
        Assert.True(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(new CreatureRef(2)), out _));
        Assert.False(Hooks.ShouldAllowHitting(k, new ShouldAllowHittingArgs(dead), out _));
        Assert.False(Hooks.ShouldCreatureBeRemovedFromCombatAfterDeath(k, new ShouldCreatureBeRemovedFromCombatAfterDeathArgs(osty), out _));
        Assert.True(Hooks.ShouldCreatureBeRemovedFromCombatAfterDeath(k, new ShouldCreatureBeRemovedFromCombatAfterDeathArgs(dead), out _));
        Assert.False(DieForYouPower.ShouldPowerBeRemovedAfterOwnerDeath(k.Ctx, Power("DieForYouPower", 1),
            new PowerDeathArgs(OldCreature.Osty, BuffId.Minion)));

        var km = Kernel(s, Without(Registry.R7g, DieForYouPower.RegisterDeath));
        Assert.True(Hooks.ShouldAllowHitting(km, new ShouldAllowHittingArgs(new CreatureRef(3)), out _));
    }

    [Fact]
    public void EmotionChipsDamageHookOnlyPresents()
    {
        // game: EmotionChip.AfterDamageReceived sets the relic's status (presentation): no state to differ, no mutant.
        var s = Combat(40);
        Hold(s, RelicEffects.EmotionChip);
        Hit(Kernel(s, Registry.R7g), OldCreature.Player, 6m, dealer: OldCreature.Of(s.Enemies[0]));
        Assert.Equal(44, s.PlayerHp);
    }
}
