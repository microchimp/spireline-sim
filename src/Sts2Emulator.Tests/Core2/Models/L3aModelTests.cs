// Batch L3a (docs/rebuild/CLOUD_BRIEFS.md section 3, the shared rules; the LOCAL models whose old bodies were private or
// acted for every holder): Thorns, Flame Barrier, Crab Rage, Surrounded and Ravenous under Core2/Models/Powers, each
// tested as the rules ask: (1) its pairs are in the handler table and none is a stub, (2) the dispatch gives what the old
// function gives for the same state, (3) the delete-mutant gives the no-handler value, (4) with F08 off its pairs are
// absent, (5) with the model on several creatures a dispatch about one acts for that holder alone, once. The death hooks
// are dispatched inside Dying (the fallen at 0 HP, still listed). Thorns and Flame Barrier are re-entries: their old
// bodies deal damage, which under the switch is one more CreatureCmd.Damage. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class L3aModelTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> L3a = Registry.L3a;

    private static int Buff(List<BuffState> buffs, BuffId id) => BuffSystem.Get(buffs, id);

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list's rows for the five models.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] L3aPairs =
    [
        (ListenerKind.Power, "ThornsPower", HookId.BeforeDamageReceived),
        (ListenerKind.Power, "FlameBarrierPower", HookId.AfterDamageReceived),
        (ListenerKind.Power, "CrabRagePower", HookId.AfterDeath),
        (ListenerKind.Power, "SurroundedPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "SurroundedPower", HookId.AfterDeath),
        (ListenerKind.Power, "RavenousPower", HookId.AfterDeath),
    ];

    [Fact]
    public void TheRegistrysL3aBlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(L3a);
        Assert.Equal(L3aPairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal)
            .ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal), Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
        AssertRegistered(t, ListenerKind.Power, "ThornsPower", HookId.BeforeDamageReceived);
        AssertRegistered(t, ListenerKind.Power, "FlameBarrierPower", HookId.AfterDamageReceived);
        AssertRegistered(t, ListenerKind.Power, "CrabRagePower", HookId.AfterDeath);
        AssertRegistered(t, ListenerKind.Power, "SurroundedPower", HookId.ModifyDamageMultiplicative, HookId.AfterDeath);
        AssertRegistered(t, ListenerKind.Power, "RavenousPower", HookId.AfterDeath);
    }

    [Fact]
    public void TheRegistrysAllHoldsTheL3aBlock()
    {
        foreach (var model in L3a)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in L3aPairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffL3aRegistersNothing()
    {
        var t = Table(L3a, AllButDamage);
        Assert.Empty(Pairs(t));
        AssertAbsent(t, ListenerKind.Power, "ThornsPower", HookId.BeforeDamageReceived);
        AssertAbsent(t, ListenerKind.Power, "SurroundedPower", HookId.ModifyDamageMultiplicative, HookId.AfterDeath);
    }

    // ---------------------------------------------------------------------------------------- Thorns

    [Fact]
    public void AnEnemysThornsHitsThePlayerBackThroughBlockAsTheOldBodyDoes()
    {
        // game: ThornsPower.BeforeDamageReceived; old: CardEffects.TriggerEnemyThorns on the player's powered hit. The
        // slug's Thorns 3 against the player's 1 block: 2 HP lost, as the old body (called directly, no switch) gives.
        var s = Combat(40);
        var slug = s.Enemies[0];
        GivePower(s, "ThornsPower", BuffId.Thorns, 3, slug);
        s.PlayerBlock = 1;
        var k = Kernel(s, L3a);

        Hit(k, OldCreature.Of(slug), 6m);

        var oracle = Combat(40);
        oracle.Enemies[0].Buffs.Add(new BuffState(BuffId.Thorns, 3));
        oracle.PlayerBlock = 1;
        CardEffects.TriggerEnemyThorns(oracle, oracle.Enemies[0]);
        Assert.Equal((oracle.PlayerHp, oracle.PlayerBlock), (s.PlayerHp, s.PlayerBlock));
        Assert.Equal((48, 0), (s.PlayerHp, s.PlayerBlock));
        Assert.Equal(34, slug.Hp);

        // The delete-mutant: no handler, no retaliation.
        var s2 = Combat(40);
        GivePower(s2, "ThornsPower", BuffId.Thorns, 3, s2.Enemies[0]);
        s2.PlayerBlock = 1;
        Hit(Kernel(s2, Without(L3a, ThornsPower.Register)), OldCreature.Of(s2.Enemies[0]), 6m);
        Assert.Equal((50, 1), (s2.PlayerHp, s2.PlayerBlock));
    }

    [Fact]
    public void ThornsAnswersAPoweredAttackOrOmnisliceAndNothingElse()
    {
        // The game's test (and the old sites'): a powered attack, or Omnislice's unpowered splash. Another unpowered hit
        // (a relic's, a power's) runs no Thorns.
        var s = Combat(40);
        var slug = s.Enemies[0];
        GivePower(s, "ThornsPower", BuffId.Thorns, 3, slug);
        var k = Kernel(s, L3a);
        Hit(k, OldCreature.Of(slug), 4m, props: Unpowered);
        Assert.Equal(50, s.PlayerHp);

        var omni = Add(s, s.Hand, CL.Omnislice);
        Hit(k, OldCreature.Of(slug), 4m, props: Unpowered, card: new CardHandle(omni.Uid));
        Assert.Equal(47, s.PlayerHp);
    }

    [Fact]
    public void ThePlayersThornsHitsTheAttackingEnemyBackAsTheOldBodyDoes()
    {
        // old: EnemyAI.ApplyPlayerThorns(enemy) on the enemy's attack. The player's Thorns 4 against the attacking slug.
        var s = Combat(40);
        var slug = s.Enemies[0];
        GivePower(s, "ThornsPower", BuffId.Thorns, 4);
        var k = Kernel(s, L3a);

        Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(slug));

        var oracle = Combat(40);
        oracle.PlayerBuffs.Add(new BuffState(BuffId.Thorns, 4));
        EnemyAI.ApplyPlayerThorns(oracle.Enemies[0], oracle);
        Assert.Equal(oracle.Enemies[0].Hp, slug.Hp);
        Assert.Equal(36, slug.Hp);
        Assert.Equal(45, s.PlayerHp);

        var s2 = Combat(40);
        GivePower(s2, "ThornsPower", BuffId.Thorns, 4);
        Hit(Kernel(s2, Without(L3a, ThornsPower.Register)), OldCreature.Player, 5m, dealer: OldCreature.Of(s2.Enemies[0]));
        Assert.Equal(40, s2.Enemies[0].Hp);
    }

    [Fact]
    public void ThornsActsForTheHitCreaturesCopyOnlyAmongThreeHolders()
    {
        // The N-holder test: Thorns 2 on the player and on both slugs. The player's hit on slug A runs A's Thorns alone
        // (2 back, not 6); slug B's attack on the player runs the player's alone, at B (A untouched).
        var s = Combat(40, 40);
        var (a, b) = (s.Enemies[0], s.Enemies[1]);
        Holders(s, "ThornsPower", BuffId.Thorns, 2);
        var k = Kernel(s, L3a);

        Hit(k, OldCreature.Of(a), 5m);
        Assert.Equal((48, 35, 40), (s.PlayerHp, a.Hp, b.Hp));

        Hit(k, OldCreature.Player, 3m, dealer: OldCreature.Of(b));
        Assert.Equal((45, 35, 38), (s.PlayerHp, a.Hp, b.Hp));
    }

    [Fact]
    public void TheOldPlayerDamageEntryCarriesTheEnemyDealerUnderTheSwitch()
    {
        // The re-entry: an enemy's Thorns reaches CreatureCmd.Damage with its owner as dealer, as ThornsPower's call
        // passes. Seen through Damage's dead-dealer rule: a dead dealer's damage does nothing, no dealer's lands.
        var s = Combat(40, 40);
        s.Enemies[0].Hp = 0;
        _ = Kernel(s, L3a);
        CardEffects.DealDamageToPlayer(s, 3, dealer: s.Enemies[0]);
        Assert.Equal(50, s.PlayerHp);
        CardEffects.DealDamageToPlayer(s, 3);
        Assert.Equal(47, s.PlayerHp);
    }

    // ---------------------------------------------------------------------------------------- Flame Barrier

    [Fact]
    public void FlameBarrierHitsTheAttackerBackEvenThroughFullBlockAsTheOldBodyDoes()
    {
        // game: FlameBarrierPower.AfterDamageReceived; old: EnemyAI.ApplyPlayerFlameBarrier(enemy) per hit of the enemy's
        // attack, absorbed or not. Flame Barrier 4, the 5 swallowed by 10 block: the slug takes 4.
        var s = Combat(40);
        var slug = s.Enemies[0];
        GivePower(s, "FlameBarrierPower", BuffId.FlameBarrier, 4);
        s.PlayerBlock = 10;
        var k = Kernel(s, L3a);

        Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(slug));

        var oracle = Combat(40);
        oracle.PlayerBuffs.Add(new BuffState(BuffId.FlameBarrier, 4));
        EnemyAI.ApplyPlayerFlameBarrier(oracle.Enemies[0], oracle);
        Assert.Equal(oracle.Enemies[0].Hp, slug.Hp);
        Assert.Equal((36, 50), (slug.Hp, s.PlayerHp));

        // An unpowered hit from the slug: nothing.
        Hit(k, OldCreature.Player, 2m, dealer: OldCreature.Of(slug), props: Unpowered);
        Assert.Equal(36, slug.Hp);

        var s2 = Combat(40);
        GivePower(s2, "FlameBarrierPower", BuffId.FlameBarrier, 4);
        s2.PlayerBlock = 10;
        Hit(Kernel(s2, Without(L3a, FlameBarrierPower.Register)), OldCreature.Player, 5m,
            dealer: OldCreature.Of(s2.Enemies[0]));
        Assert.Equal(40, s2.Enemies[0].Hp);
    }

    [Fact]
    public void FlameBarrierActsForThePlayersCopyOnlyAmongTwoHolders()
    {
        // The N-holder test: Flame Barrier 4 on the player and on the slug. The slug's attack on the player costs the slug 4
        // once; the player's hit on the slug runs no retaliation (the old engine's power is the player's alone).
        var s = Combat(40);
        var slug = s.Enemies[0];
        Holders(s, "FlameBarrierPower", BuffId.FlameBarrier, 4);
        s.PlayerBlock = 10;
        var k = Kernel(s, L3a);

        Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(slug));
        Assert.Equal((36, 50), (slug.Hp, s.PlayerHp));

        Hit(k, OldCreature.Of(slug), 6m);
        Assert.Equal((30, 50, 5), (slug.Hp, s.PlayerHp, s.PlayerBlock));
    }

    // ---------------------------------------------------------------------------------------- Crab Rage

    private static CombatState CrabCombat(int crabs)
    {
        var s = Combat(Enumerable.Repeat(30, crabs).ToArray());
        foreach (var e in s.Enemies)
            e.Buffs.Add(new BuffState(BuffId.CrabRage, 1));
        return s;
    }

    [Fact]
    public void CrabRageEnragesTheSurvivorWhenItsPartnerDiesAsTheOldBodyDoes()
    {
        // game: CrabRagePower.AfterDeath; old: CombatEngine.TryEnrageCrabPartner(fallen). The partner gains Strength and
        // Block and the rage is spent. Dispatched inside Dying.
        var s = CrabCombat(2);
        var survivor = s.Enemies[1];
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var oracle = CrabCombat(2);
        oracle.Enemies[0].Hp = 0;
        Assert.True(CombatEngine.TryEnrageCrabPartner(oracle, oracle.Enemies[0]));
        Assert.Equal(oracle.Enemies[1].Buffs, survivor.Buffs);
        Assert.Equal(oracle.Enemies[1].Block, survivor.Block);
        Assert.Equal(RunConstants.CrabRageStrength, Buff(survivor.Buffs, BuffId.Strength));
        Assert.Equal(RunConstants.CrabRageBlock, survivor.Block);
        Assert.Equal(0, Buff(survivor.Buffs, BuffId.CrabRage));

        var s2 = CrabCombat(2);
        var k2 = Kernel(s2, Without(L3a, CrabRagePower.Register));
        using (Dying(k2, s2.Enemies[0]))
            Hooks.AfterDeath(k2, AfterDeathOf(1));
        Assert.Equal((0, 0, 1), (Buff(s2.Enemies[1].Buffs, BuffId.Strength), s2.Enemies[1].Block,
            Buff(s2.Enemies[1].Buffs, BuffId.CrabRage)));
    }

    [Fact]
    public void CrabRageKeepsTheOldTestOfTheFallen()
    {
        // The old body answers the death of a Crab Rage holder only (the game's test is any ally's death).
        var s = CrabCombat(2);
        s.Enemies[0].Buffs.Clear();
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(1, Buff(s.Enemies[1].Buffs, BuffId.CrabRage));
        Assert.Equal(0, s.Enemies[1].Block);
    }

    [Fact]
    public void CrabRageEnragesEachSurvivorOnceAmongFourHolders()
    {
        // The N-holder test: Crab Rage on the player and three slugs; slug A dies. Each surviving slug is enraged once (the
        // rage is spent, so the second survivor's handler finds nothing left), the player's copy does nothing (the old
        // body enrages enemies), and the state equals one direct old call.
        var s = Combat(30, 30, 30);
        Holders(s, "CrabRagePower", BuffId.CrabRage, 1);
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var oracle = CrabCombat(3);
        oracle.PlayerBuffs.Add(new BuffState(BuffId.CrabRage, 1));
        oracle.Enemies[0].Hp = 0;
        CombatEngine.TryEnrageCrabPartner(oracle, oracle.Enemies[0]);
        for (int i = 1; i < 3; i++)
        {
            Assert.Equal(oracle.Enemies[i].Buffs, s.Enemies[i].Buffs);
            Assert.Equal(RunConstants.CrabRageStrength, Buff(s.Enemies[i].Buffs, BuffId.Strength));
            Assert.Equal(RunConstants.CrabRageBlock, s.Enemies[i].Block);
        }

        Assert.Equal(oracle.PlayerBuffs, s.PlayerBuffs);
        Assert.Equal(0, s.PlayerBlock);
    }

    // ---------------------------------------------------------------------------------------- Surrounded

    /// <summary>The Kaiser Crab's layout on two slugs: the player faces right, slug A is at the player's back (Back Attack
    /// Left), slug B in front (Back Attack Right).</summary>
    private static CombatState KaiserCombat()
    {
        var s = Combat(30, 30);
        s.PlayerBuffs.Add(new BuffState(BuffId.Surrounded, RunConstants.FacingRight));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.BackAttackLeft, 1));
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.BackAttackRight, 1));
        return s;
    }

    [Fact]
    public void SurroundedMultipliesAPoweredHitFromBehindAsTheOldBodyDoes()
    {
        // game: SurroundedPower.ModifyDamageMultiplicative; old: BuffSystem.Surrounded_ModifyDamageMultiplicative from the
        // old powered fold. Slug A, behind: 1.5x (15 of 10); slug B, in front: 1x; an unpowered hit from behind: 1x.
        var s = KaiserCombat();
        var k = Kernel(s, L3a);
        decimal factor = (decimal)BuffSystem.Surrounded_ModifyDamageMultiplicative(s.Enemies[0].Buffs, s.PlayerBuffs);

        Assert.Equal(10m * factor, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, dealer: 1), out var changed));
        Assert.Equal(15m, 10m * factor);
        Assert.Equal([Power("SurroundedPower", 0)], changed);
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, dealer: 2), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, Unpowered, dealer: 1), out _));

        var k2 = Kernel(KaiserCombat(), Without(L3a, SurroundedPower.Register));
        Assert.Equal(10m, Hooks.ModifyDamage(k2, Attack(target: 0, damage: 10m, dealer: 1), out var none));
        Assert.Empty(none);
    }

    [Fact]
    public void SurroundedMultipliesForTheTargetsCopyOnlyAmongThreeHolders()
    {
        // The N-holder test: Surrounded (facing right) on the player and both slugs. Slug A's hit on the player changes the
        // player's listener alone and lands 1.5x once (15, not 22.5 or more).
        var s = KaiserCombat();
        s.PlayerBuffs.Clear();
        var holders = Holders(s, "SurroundedPower", BuffId.Surrounded, RunConstants.FacingRight);
        var k = Kernel(s, L3a);

        Assert.Equal(15m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, dealer: 1), out var changed));
        Assert.Equal([holders[0]], changed);
    }

    [Fact]
    public void SurroundedTurnsThePlayerToFaceTheSurvivorAsTheOldBodyDoes()
    {
        // game: SurroundedPower.AfterDeath; old: CombatEngine.TurnPlayerToFaceSurvivor. Slug B (in front) dies, slug A
        // (behind, Back Attack Left) is every enemy left: the player turns to face left. Dispatched inside Dying.
        var s = KaiserCombat();
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));

        var oracle = KaiserCombat();
        oracle.Enemies[1].Hp = 0;
        CombatEngine.TurnPlayerToFaceSurvivor(oracle);
        Assert.Equal(oracle.PlayerBuffs, s.PlayerBuffs);
        Assert.Equal(RunConstants.FacingLeft, Buff(s.PlayerBuffs, BuffId.Surrounded));

        // A prevented removal turns nothing; the delete-mutant turns nothing.
        var s2 = KaiserCombat();
        var k2 = Kernel(s2, L3a);
        using (Dying(k2, s2.Enemies[1]))
            Hooks.AfterDeath(k2, AfterDeathOf(2, removalPrevented: true));
        Assert.Equal(RunConstants.FacingRight, Buff(s2.PlayerBuffs, BuffId.Surrounded));
        var s3 = KaiserCombat();
        var k3 = Kernel(s3, Without(L3a, SurroundedPower.Register));
        using (Dying(k3, s3.Enemies[1]))
            Hooks.AfterDeath(k3, AfterDeathOf(2));
        Assert.Equal(RunConstants.FacingRight, Buff(s3.PlayerBuffs, BuffId.Surrounded));
    }

    [Fact]
    public void SurroundedTurnsThePlayerOnceAmongThreeHolders()
    {
        // The N-holder test: Surrounded on the player and both slugs. Slug B's death turns the player once (left; a second
        // turn would face right again) and leaves the slugs' copies as they were.
        var s = KaiserCombat();
        s.PlayerBuffs.Clear();
        Holders(s, "SurroundedPower", BuffId.Surrounded, RunConstants.FacingRight);
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));

        Assert.Equal(RunConstants.FacingLeft, Buff(s.PlayerBuffs, BuffId.Surrounded));
        Assert.Equal(RunConstants.FacingRight, Buff(s.Enemies[0].Buffs, BuffId.Surrounded));
        Assert.Equal(RunConstants.FacingRight, Buff(s.Enemies[1].Buffs, BuffId.Surrounded));
    }

    // ---------------------------------------------------------------------------------------- Ravenous

    private static CombatState RavenousCombat(int slugs)
    {
        var s = Combat(Enumerable.Repeat(20, slugs).ToArray());
        foreach (var e in s.Enemies)
            e.Buffs.Add(new BuffState(BuffId.Ravenous, 3));
        return s;
    }

    [Fact]
    public void RavenousStunsAndStrengthensTheSurvivorAsTheOldBodyDoes()
    {
        // game: RavenousPower.AfterDeath; old: CombatEngine.Ravenous_AfterDeath (the loop over the living holders). Slug A
        // dies: slug B gains Strength 3 and a Stun and announces Unknown. Dispatched inside Dying.
        var s = RavenousCombat(2);
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var oracle = RavenousCombat(2);
        oracle.Enemies[0].Hp = 0;
        CombatEngine.Ravenous_AfterDeath(oracle);
        Assert.Equal(oracle.Enemies[1].Buffs, s.Enemies[1].Buffs);
        Assert.Equal(oracle.Enemies[1].CurrentIntent, s.Enemies[1].CurrentIntent);
        Assert.Equal((3, 1), (Buff(s.Enemies[1].Buffs, BuffId.Strength), Buff(s.Enemies[1].Buffs, BuffId.Stunned)));
        Assert.Equal(IntentType.Unknown, s.Enemies[1].CurrentIntent.Type);
        Assert.Equal(0, Buff(s.Enemies[0].Buffs, BuffId.Strength));

        // A prevented removal and the delete-mutant: nothing.
        var s2 = RavenousCombat(2);
        var k2 = Kernel(s2, L3a);
        using (Dying(k2, s2.Enemies[0]))
            Hooks.AfterDeath(k2, AfterDeathOf(1, removalPrevented: true));
        Assert.Equal(0, Buff(s2.Enemies[1].Buffs, BuffId.Strength));
        var s3 = RavenousCombat(2);
        var k3 = Kernel(s3, Without(L3a, RavenousPower.Register));
        using (Dying(k3, s3.Enemies[0]))
            Hooks.AfterDeath(k3, AfterDeathOf(1));
        Assert.Equal(0, Buff(s3.Enemies[1].Buffs, BuffId.Strength));
    }

    [Fact]
    public void RavenousActsForEachSurvivorOnceAmongFourHolders()
    {
        // The N-holder test (the review's 3.3): Ravenous 3 on the player and three slugs; slug A dies. Slugs B and C each
        // gain Strength 3 and one Stun (N owners' effects, not N squared: the old loop called once per listener would give
        // 6 and 2), the player's copy does nothing, and the state equals one direct call of the per-owner part per owner.
        var s = Combat(20, 20, 20);
        Holders(s, "RavenousPower", BuffId.Ravenous, 3);
        var k = Kernel(s, L3a);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var oracle = RavenousCombat(3);
        oracle.Enemies[0].Hp = 0;
        CombatEngine.Ravenous_AfterDeathOf(oracle.Enemies[1]);
        CombatEngine.Ravenous_AfterDeathOf(oracle.Enemies[2]);
        for (int i = 1; i < 3; i++)
        {
            Assert.Equal(oracle.Enemies[i].Buffs, s.Enemies[i].Buffs);
            Assert.Equal((3, 1), (Buff(s.Enemies[i].Buffs, BuffId.Strength), Buff(s.Enemies[i].Buffs, BuffId.Stunned)));
        }

        Assert.Equal(0, Buff(s.PlayerBuffs, BuffId.Strength));
        Assert.Equal(0, Buff(s.Enemies[0].Buffs, BuffId.Strength));
    }

    [Fact]
    public void TheOldRavenousLoopIsThePerOwnerPartForEachLivingHolder()
    {
        // The split in Core/: one call of the old loop equals one call of the per-owner part for each living holder.
        var a = RavenousCombat(3);
        a.Enemies[0].Hp = 0;
        CombatEngine.Ravenous_AfterDeath(a);
        var b = RavenousCombat(3);
        b.Enemies[0].Hp = 0;
        foreach (var e in b.Enemies.Where(e => e.Hp > 0))
            CombatEngine.Ravenous_AfterDeathOf(e);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(b.Enemies[i].Buffs, a.Enemies[i].Buffs);
            Assert.Equal(b.Enemies[i].CurrentIntent, a.Enemies[i].CurrentIntent);
        }
    }
}
