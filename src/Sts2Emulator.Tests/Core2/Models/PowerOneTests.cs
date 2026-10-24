// The C2 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C2"): the port list's power-1 rows as migrated models
// under Core2/Models/Powers, each tested the way the shared rules ask: (1) its pair is in the handler table and is no
// stub, (2) the dispatch gives what the old function gives for the same state, with the changed set, (3) the
// delete-mutant (the model's registration left out) gives the no-handler value, (4) with F08 off its pair is absent,
// (5) with the model on several creatures a dispatch about one changes that holder's listener and no other. The table
// test pins the batch's (model, hook) set, standing in for EmuDump kernel-handlers and the port list's gate, which need
// the manifest and a built EmuDump. The damage folds also get the integer-versus-decimal check: a handler returns its
// addend or factor as a decimal and the one cast is CreatureCmd.Damage's. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PowerOneTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Power1 = Registry.Power1;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list's power-1 rows that are CLOUD (CLOUD_BRIEFS
    /// section 1). Not here, by the stop rule (their old bodies are private, so no handler can call them):
    /// power:THORNS_POWER BeforeDamageReceived (CardEffects.TriggerEnemyThorns, EnemyAI.ApplyPlayerThorns) and
    /// power:FLAME_BARRIER_POWER AfterDamageReceived (EnemyAI.ApplyPlayerFlameBarrier).</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Power1Pairs =
    [
        (ListenerKind.Power, "StrengthPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "VulnerablePower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "WeakPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "VigorPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "TankPower", HookId.ModifyDamageMultiplicative),
    ];

    [Fact]
    public void TheRegistrysPower1BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Power1);
        Assert.Equal(Power1Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePower1BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; the batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Power1)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Power1Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffPower1RegistersNothing()
    {
        Assert.Empty(Pairs(Table(Power1, AllButDamage)));
    }

    [Fact]
    public void TheDamageFoldsKeepDecimalsAndTheDamageCommandCastsOnce()
    {
        // The integer-versus-decimal check over the batch's folds: Strength 3 and Weak 1 on the player, Vulnerable 1 and
        // Tank 1 on the slug. (6 + 3) x 0.75 x 2 x 1.5 = 20.25, kept as a decimal by the fold (no handler casts), 20 lost
        // by the one cast in CreatureCmd.Damage; the old float fold casts once at its end too and gives 20. Each listener
        // changed it, in the dispatcher's order (the player's powers, then the slug's, each in application order).
        var s = Combat(40);
        var slug = s.Enemies[0];
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, 1));
        slug.Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        slug.Buffs.Add(new BuffState(BuffId.Tank, 1));
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(20.25m, v);
        Assert.Equal([Power("StrengthPower", 0), Power("WeakPower", 0), Power("VulnerablePower", 1), Power("TankPower", 1)], changed);
        Assert.Equal(20, BuffSystem.IncomingDamage(6, s.PlayerBuffs, slug.Buffs));
        Assert.Equal(20, Assert.Single(Hit(k, OldCreature.Of(slug), 6m)).UnblockedDamage);
        Assert.Equal(20, slug.Hp);
        // A 4: (4 + 3) x 0.75 x 2 x 1.5 = 15.75, 15 lost, the old fold's 15.
        Assert.Equal(15.75m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m), out _));
        Assert.Equal(15, BuffSystem.IncomingDamage(4, s.PlayerBuffs, slug.Buffs));
        Assert.Equal(15, Assert.Single(Hit(k, OldCreature.Of(slug), 4m)).UnblockedDamage);
        Assert.Equal(5, slug.Hp);
    }

    [Fact]
    public void TheDealerSideFoldsDoNotTouchTheOwnersPetsHit()
    {
        // Strength 3, Vigor 2 and Weak 1 on the player; Osty (creature 1) hits the slug (creature 2): the dealer is not the
        // owner, so none applies and the hit stays 6 (game: the pet is the attacker of its own hits, which the player's
        // Strength does not raise, as the old engine's note at CardEffects.OstyAttackDamage records). The old engine's Osty
        // path (CardEffects.DealOstyDamage into DealDamageToEnemy) folds the hit with the player's list instead,
        // (6 + 3 + 2) x 0.75 = 8.25 -> 8: the R7 check the PR lists. The player's own hit takes all three.
        var s = Combat(40);
        WithOsty(s);
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.PlayerBuffs.Add(new BuffState(BuffId.Vigor, 2));
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, 1));
        var k = Kernel(s, Power1);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(8, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));

        Assert.Equal(8.25m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m), out changed));
        Assert.Equal([Power("StrengthPower", 0), Power("VigorPower", 0), Power("WeakPower", 0)], changed);
    }

    // ---------------------------------------------------------------------------------------- Strength (power-1)

    /// <summary>The player with Strength 3, one slug.</summary>
    private static CombatState StrengthCombat(int amount = 3)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, amount));
        return s;
    }

    [Fact]
    public void StrengthIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power1), ListenerKind.Power, "StrengthPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void StrengthAddsTheOldFunctionsNumberToTheOwnersPoweredAttack()
    {
        // game: StrengthPower.ModifyDamageAdditive; old: BuffSystem.IncomingDamage adds
        // Strength_ModifyDamageAdditive(attackerBuffs) with no condition at the site.
        var s = StrengthCombat();
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m + BuffSystem.Strength_ModifyDamageAdditive(s.PlayerBuffs), v);
        Assert.Equal(9m, v);
        Assert.Equal([Power("StrengthPower", 0)], changed);
        Assert.Equal(9, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));

        // A negative Strength subtracts: the old site adds the number as it is.
        s.PlayerBuffs[0] = new BuffState(BuffId.Strength, -2);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Equal([Power("StrengthPower", 0)], changed);
        Assert.Equal(4, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        s.PlayerBuffs[0] = new BuffState(BuffId.Strength, 3);

        // Where the condition fails: the slug's attack on the player (the owner is the target, not the dealer), an
        // unpowered hit, no dealer; at 0 the addend is 0 and nothing changed.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Strength, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void StrengthOnAnEnemyRaisesItsOwnAttack()
    {
        // The slug's Strength 2 on its attack on the player: 6 + 2 by the slug's listener; the old enemy-attack path's fold.
        var s = Combat(40);
        var slug = s.Enemies[0];
        var theirs = GivePower(s, "StrengthPower", BuffId.Strength, 2, slug);
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(8m, v);
        Assert.Equal([theirs], changed);
        Assert.Equal(8, BuffSystem.IncomingDamage(6, slug.Buffs, s.PlayerBuffs));
    }

    [Fact]
    public void StrengthActsForItsOwnHolderAmongThree()
    {
        // The N-holder test: the player and both slugs hold Strength 3. The player's hit on the first slug takes the player's
        // listener alone, once; the second slug's hit on the player takes that slug's alone.
        var s = Combat(40, 40);
        var holders = Holders(s, "StrengthPower", BuffId.Strength, 3);
        var k = Kernel(s, Power1);

        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[0]], changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[2]], changed);
    }

    [Fact]
    public void StrengthDeleteMutantGivesTheNoHandlerValue()
    {
        var s = StrengthCombat();
        var k = Kernel(s, Without(Power1, StrengthPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void StrengthIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power1, AllButDamage), ListenerKind.Power, "StrengthPower", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Vulnerable (power-1)

    /// <summary>The player, one slug with Vulnerable 2.</summary>
    private static CombatState VulnerableCombat(int amount = 2)
    {
        var s = Combat(40);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, amount));
        return s;
    }

    [Fact]
    public void VulnerableIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power1), ListenerKind.Power, "VulnerablePower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void VulnerableMultipliesAHitOnItsOwnerByTheOldFunctionsFactor()
    {
        // game: VulnerablePower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage multiplies by
        // Vulnerable_ModifyDamageMultiplicative(attackerBuffs, defenderBuffs, vulnerableDelta) when the defender's
        // Vulnerable > 0; the card path (CardEffects.DealDamageToEnemy) passes the player's list and the Paper Phrog delta.
        var s = VulnerableCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m * (decimal)BuffSystem.Vulnerable_ModifyDamageMultiplicative(s.PlayerBuffs, slug.Buffs, 0f), v);
        Assert.Equal(9m, v);
        Assert.Equal([Power("VulnerablePower", 1)], changed);
        Assert.Equal(9, BuffSystem.IncomingDamage(6, s.PlayerBuffs, slug.Buffs));

        // The inputs the old function reads, each against the old fold with the card path's arguments: the Phrog's shift
        // (1.75: 10.5 kept, 10 by the cast), Cruelty on the attacker (2), Debilitate on the owner (2).
        Hold(s, RelicEffects.PaperPhrog);
        Assert.Equal(10.5m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Equal([Power("VulnerablePower", 1)], changed);
        Assert.Equal(10, BuffSystem.IncomingDamage(6, s.PlayerBuffs, slug.Buffs,
            vulnerableDelta: RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(s)));
        s.Relics.Clear();
        s.PlayerBuffs.Add(new BuffState(BuffId.CrueltyPower, 50));
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Equal(12, BuffSystem.IncomingDamage(6, s.PlayerBuffs, slug.Buffs));
        s.PlayerBuffs.Clear();
        slug.Buffs.Add(new BuffState(BuffId.Debilitate, 1));
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Equal(12, BuffSystem.IncomingDamage(6, s.PlayerBuffs, slug.Buffs));
        slug.Buffs.RemoveAt(1);

        // Where the condition fails: the slug's own attack on the player (the owner is the dealer, not the target), an
        // unpowered hit, the amount at 0. No dealer: the owner is the target, so the factor applies with no attacker's list.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null), out changed));
        Assert.Equal([Power("VulnerablePower", 1)], changed);
        slug.Buffs[0] = new BuffState(BuffId.Vulnerable, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VulnerableOnThePlayerTakesTheEnemyAttackPathsInputs()
    {
        // The old enemy-attack path (EnemyAI.DealAttackDamage) passes the attacking enemy's list and no Phrog delta: a
        // Vulnerable player holding the Phrog takes 1.5 from the slug, not 1.75; the slug's Cruelty, which the old
        // function reads off any attacker's list, raises it.
        var s = Combat(40);
        var slug = s.Enemies[0];
        var mine = GivePower(s, "VulnerablePower", BuffId.Vulnerable, 1);
        Hold(s, RelicEffects.PaperPhrog);
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(9m, v);
        Assert.Equal([mine], changed);
        Assert.Equal(9, BuffSystem.IncomingDamage(6, slug.Buffs, s.PlayerBuffs));
        slug.Buffs.Add(new BuffState(BuffId.CrueltyPower, 50));
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Equal([mine], changed);
        Assert.Equal(12, BuffSystem.IncomingDamage(6, slug.Buffs, s.PlayerBuffs));
    }

    [Fact]
    public void VulnerableFoldsAnOstyHitWithThePlayersListAsTheOldCardPathDoes()
    {
        // The old card path (CardEffects.DealOstyDamage into DealDamageToEnemy) folds Osty's hit with the player's list,
        // so the player's Cruelty raises it; the handler passes that list for a player-side dealer. The game reads the
        // dealer's own powers, and Osty has none in the old state: the R7 check the PR lists. Osty is creature 1, the slug 2.
        var s = VulnerableCombat();
        WithOsty(s);
        s.PlayerBuffs.Add(new BuffState(BuffId.CrueltyPower, 50));
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(12m, v);
        Assert.Equal([Power("VulnerablePower", 2)], changed);
        Assert.Equal(12, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
    }

    [Fact]
    public void VulnerableActsForItsOwnHolderAmongThree()
    {
        // The N-holder test: the player and both slugs hold Vulnerable 2. The player's hit on the first slug takes that
        // slug's listener alone; the second slug's hit on the player takes the player's alone.
        var s = Combat(40, 40);
        var holders = Holders(s, "VulnerablePower", BuffId.Vulnerable, 2);
        var k = Kernel(s, Power1);

        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);
    }

    [Fact]
    public void VulnerableDeleteMutantGivesTheNoHandlerValue()
    {
        var s = VulnerableCombat();
        var k = Kernel(s, Without(Power1, VulnerablePower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VulnerableIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power1, AllButDamage), ListenerKind.Power, "VulnerablePower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Weak (power-1)

    /// <summary>The player with Weak 1, one slug.</summary>
    private static CombatState WeakCombat(int amount = 1)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, amount));
        return s;
    }

    [Fact]
    public void WeakIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power1), ListenerKind.Power, "WeakPower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void WeakMultipliesTheOwnersPoweredAttackByTheOldFunctionsFactor()
    {
        // game: WeakPower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage multiplies by
        // Weak_ModifyDamageMultiplicative(attackerBuffs, weakDelta) when the attacker's Weak > 0; the card path passes no
        // delta. 4.5 kept as a decimal, 4 by the old fold's cast.
        var s = WeakCombat();
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m * (decimal)BuffSystem.Weak_ModifyDamageMultiplicative(s.PlayerBuffs, 0f), v);
        Assert.Equal(4.5m, v);
        Assert.Equal([Power("WeakPower", 0)], changed);
        Assert.Equal(4, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        Assert.Equal(BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs), (int)v);

        // Debilitate on the owner: the old function's half instead of its three quarters.
        s.PlayerBuffs.Add(new BuffState(BuffId.Debilitate, 1));
        Assert.Equal(3m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Equal([Power("WeakPower", 0)], changed);
        Assert.Equal(3, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        s.PlayerBuffs.RemoveAt(1);

        // Where the condition fails: the slug's attack on the Weak player (the owner is the target, not the dealer), an
        // unpowered hit, the amount at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Weak, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void WeakOnAnEnemyTakesThePaperKraneShiftOnItsAttackOnThePlayer()
    {
        // The Krane's shift the old enemy-attack path passes (EnemyAI.DealAttackDamage: WeakMultiplierDeltaAgainstPlayer):
        // a Weak slug hits the Krane's holder for 0.6 of 8, 4.8 kept as a decimal, 4 by the old fold's cast; 6 without the
        // Krane. The player's own Weak attacking the slug takes no shift (the card path passes 0), Krane or not.
        var s = Combat(40);
        var slug = s.Enemies[0];
        var theirs = GivePower(s, "WeakPower", BuffId.Weak, 1, slug);
        Hold(s, RelicEffects.PaperKrane);
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 8m, dealer: 1), out var changed);

        float delta = RelicEffects.WeakMultiplierDeltaAgainstPlayer(s);
        Assert.Equal(8m * (decimal)BuffSystem.Weak_ModifyDamageMultiplicative(slug.Buffs, delta), v);
        Assert.Equal(4.8m, v);
        Assert.Equal([theirs], changed);
        Assert.Equal(4, BuffSystem.IncomingDamage(8, slug.Buffs, s.PlayerBuffs, weakDelta: delta));
        Assert.Equal(BuffSystem.IncomingDamage(8, slug.Buffs, s.PlayerBuffs, weakDelta: delta), (int)v);

        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 8m, dealer: 1), out changed));
        Assert.Equal([theirs], changed);

        var mine = GivePower(s, "WeakPower", BuffId.Weak, 1);
        Hold(s, RelicEffects.PaperKrane);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m), out changed));
        Assert.Equal([mine], changed);
        Assert.Equal(6, BuffSystem.IncomingDamage(8, s.PlayerBuffs, slug.Buffs));
    }

    [Fact]
    public void WeakActsForItsOwnHolderAmongThree()
    {
        // The N-holder test: the player and both slugs hold Weak 1. The player's hit on the first slug takes the player's
        // listener alone (the target's Weak is not the dealer's); the second slug's hit on the player takes that slug's alone.
        var s = Combat(40, 40);
        var holders = Holders(s, "WeakPower", BuffId.Weak, 1);
        var k = Kernel(s, Power1);

        Assert.Equal(4.5m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[0]], changed);
        Assert.Equal(4.5m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[2]], changed);
    }

    [Fact]
    public void WeakDeleteMutantGivesTheNoHandlerValue()
    {
        var s = WeakCombat();
        var k = Kernel(s, Without(Power1, WeakPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void WeakIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power1, AllButDamage), ListenerKind.Power, "WeakPower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Vigor (power-1)

    /// <summary>The player with Vigor 2, one slug.</summary>
    private static CombatState VigorCombat(int amount = 2)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Vigor, amount));
        return s;
    }

    [Fact]
    public void VigorIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power1), ListenerKind.Power, "VigorPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void VigorAddsTheOldFunctionsNumberToTheOwnersPoweredAttack()
    {
        // game: VigorPower.ModifyDamageAdditive; old: BuffSystem.IncomingDamage adds Vigor_ModifyDamageAdditive(attackerBuffs)
        // with no condition at the site. The spending of the power after the attack is another hook's.
        var s = VigorCombat();
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m + BuffSystem.Vigor_ModifyDamageAdditive(s.PlayerBuffs), v);
        Assert.Equal(8m, v);
        Assert.Equal([Power("VigorPower", 0)], changed);
        Assert.Equal(8, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));

        // Where the condition fails: the slug's attack on the player (the owner is the target, not the dealer), an
        // unpowered hit, no dealer; at 0 the addend is 0 and nothing changed.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Vigor, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VigorOnAnEnemyRaisesItsOwnAttack()
    {
        // The slug's Vigor 6 on its attack on the player: 6 + 6 by the slug's listener; the old enemy-attack path's fold.
        var s = Combat(40);
        var slug = s.Enemies[0];
        var theirs = GivePower(s, "VigorPower", BuffId.Vigor, 6, slug);
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(12m, v);
        Assert.Equal([theirs], changed);
        Assert.Equal(12, BuffSystem.IncomingDamage(6, slug.Buffs, s.PlayerBuffs));
    }

    [Fact]
    public void VigorActsForItsOwnHolderAmongThree()
    {
        // The N-holder test: the player and both slugs hold Vigor 2. The player's hit on the first slug takes the player's
        // listener alone, once; the second slug's hit on the player takes that slug's alone.
        var s = Combat(40, 40);
        var holders = Holders(s, "VigorPower", BuffId.Vigor, 2);
        var k = Kernel(s, Power1);

        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[0]], changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[2]], changed);
    }

    [Fact]
    public void VigorDeleteMutantGivesTheNoHandlerValue()
    {
        var s = VigorCombat();
        var k = Kernel(s, Without(Power1, VigorPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VigorIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power1, AllButDamage), ListenerKind.Power, "VigorPower", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Tank (power-1)

    [Fact]
    public void TankIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power1), ListenerKind.Power, "TankPower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void TankDoublesAPoweredAttackOnItsOwnerWhateverItsStackCount()
    {
        // game: TankPower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage multiplies by
        // Tank_ModifyDamageMultiplicative() (a flat 2) when the defender's Tank > 0.
        var s = Combat(40);
        var slug = s.Enemies[0];
        var mine = GivePower(s, "TankPower", BuffId.Tank, 1);
        var k = Kernel(s, Power1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(6m * (decimal)BuffSystem.Tank_ModifyDamageMultiplicative(), v);
        Assert.Equal(12m, v);
        Assert.Equal([mine], changed);
        Assert.Equal(12, BuffSystem.IncomingDamage(6, slug.Buffs, s.PlayerBuffs));
        s.PlayerBuffs[0] = new BuffState(BuffId.Tank, 2);
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Equal([mine], changed);

        // Where the condition fails: the owner's own attack on the slug (the owner is the dealer, not the target), an
        // unpowered hit, the amount at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, props: Unpowered), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Tank, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TankActsForItsOwnHolderAmongThree()
    {
        // The N-holder test: the player and both slugs hold Tank. The player's hit on the first slug takes that slug's
        // listener alone; the second slug's hit on the player takes the player's alone.
        var s = Combat(40, 40);
        var holders = Holders(s, "TankPower", BuffId.Tank, 1);
        var k = Kernel(s, Power1);

        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);
    }

    [Fact]
    public void TankDeleteMutantGivesTheNoHandlerValue()
    {
        var s = Combat(40);
        GivePower(s, "TankPower", BuffId.Tank, 1);
        var k = Kernel(s, Without(Power1, TankPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TankIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power1, AllButDamage), ListenerKind.Power, "TankPower", HookId.ModifyDamageMultiplicative);
    }
}
