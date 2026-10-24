// The C5 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C5"): the port list's power-4 rows that are CLOUD
// (Conqueror is ported locally) as migrated models under Core2/Models/Powers, each tested the way the shared rules ask:
// (1) its pairs are in the handler table and none is a stub, (2) the dispatch gives what the old function gives for the
// same state, (3) the delete-mutant (the model's registration left out) gives the no-handler value, (4) with F08 off its
// pairs are absent, (5) with two holders a dispatch about one creature changes that holder's listener alone. The death
// hooks are dispatched inside the kit's Dying scope (the fallen at 0 HP, still listed, as the game's overrides see it) and
// through CreatureCmd.Kill; Infested's hold is also read through CombatEnd (R3b's boundary check). The table test pins
// the batch's (model, hook) set, standing in for EmuDump kernel-handlers and the port list's gate, which need the
// manifest and a built EmuDump. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PowerFourTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Power4 = Registry.Power4;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list rows power-4 that are CLOUD (Conqueror's
    /// ModifyDamageMultiplicative is ported locally, round R4c).</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Power4Pairs =
    [
        (ListenerKind.Power, "CalcifyPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "AccuracyPower", HookId.ModifyDamageAdditive),
    ];

    private static IOrderedEnumerable<(ListenerKind Kind, string Model, HookId Hook)> Sorted(
        IEnumerable<(ListenerKind Kind, string Model, HookId Hook)> pairs) =>
        pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal);

    [Fact]
    public void TheRegistrysPower4BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Power4);
        Assert.Equal(Sorted(Power4Pairs), Sorted(Pairs(t)));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePower4BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; the batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Power4)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Power4Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Power4, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- Calcify (power-4)

    /// <summary>The player with Calcify 3 and Osty (10 of 10, creature 1), one slug (creature 2).</summary>
    private static CombatState CalcifyCombat(int amount = 3)
    {
        var s = WithOsty(Combat(40));
        GivePower(s, "CalcifyPower", BuffId.Calcify, amount);
        return s;
    }

    [Fact]
    public void CalcifyIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power4), ListenerKind.Power, "CalcifyPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void CalcifyAddsTheOldFunctionsNumberToOstysPoweredAttack()
    {
        // game: CalcifyPower.ModifyDamageAdditive (a powered attack whose dealer is Osty, the owner's pet); old:
        // CardEffects.OstyAttackDamage adds Calcify_ModifyDamageAdditive(state) to every Osty attack it costs.
        var s = CalcifyCombat();
        var k = Kernel(s, Power4);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(6m + CardEffects.Calcify_ModifyDamageAdditive(s), v);
        Assert.Equal(9m, v);
        Assert.Equal([Power("CalcifyPower", 0)], changed);
        // The same hit through the damage command: the slug loses 9.
        Assert.Equal(9, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 6m, dealer: OldCreature.Osty)).UnblockedDamage);
        Assert.Equal(31, s.Enemies[0].Hp);

        // Where the condition fails: the player's own attack, the slug's attack on the player, an unpowered hit by Osty,
        // no dealer, the amount at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, props: Unpowered, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: null), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Calcify, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void CalcifyAddsNothingToCreatureOneWhenItIsNoPet()
    {
        // Without Osty creature 1 is the slug: its attack on the player is no pet's.
        var s = Combat(40);
        GivePower(s, "CalcifyPower", BuffId.Calcify, 3);
        var k = Kernel(s, Power4);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void CalcifyActsForThePetsOwnerOnlyAmongTwoHolders()
    {
        // The N-holder test: Osty is the player's pet, so the slug's Calcify adds nothing to Osty's hit (and the addend
        // lands once: 9, not 12).
        var s = CalcifyCombat();
        var mine = Power("CalcifyPower", 0);
        var theirs = GivePower(s, "CalcifyPower", BuffId.Calcify, 3, s.Enemies[0]);
        var k = Kernel(s, Power4);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(9m, v);
        Assert.Equal([mine], changed);
        Assert.DoesNotContain(theirs, changed);
    }

    [Fact]
    public void CalcifyDeleteMutantGivesTheNoHandlerValue()
    {
        var s = CalcifyCombat();
        var k = Kernel(s, Without(Power4, CalcifyPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(6, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 6m, dealer: OldCreature.Osty)).UnblockedDamage);
        Assert.Equal(34, s.Enemies[0].Hp);
    }

    [Fact]
    public void CalcifyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power4, AllButDamage), ListenerKind.Power, "CalcifyPower", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Accuracy (power-4)

    private const int ShivDef = 430;
    private const int HiddenDaggersDef = 249;

    /// <summary>The player with Accuracy 4 (the old ShivDamage buff), a Shiv, a Hidden Daggers and a Strike in hand, one
    /// slug.</summary>
    private static (CombatState S, CardHandle Shiv, CardHandle Daggers, CardHandle Strike) AccuracyCombat(int amount = 4)
    {
        var s = Combat(40);
        GivePower(s, "AccuracyPower", BuffId.ShivDamage, amount);
        var shiv = Add(s, s.Hand, ShivDef);
        var daggers = Add(s, s.Hand, HiddenDaggersDef);
        var strike = Add(s, s.Hand, 1);
        return (s, new CardHandle(shiv.Uid), new CardHandle(daggers.Uid), new CardHandle(strike.Uid));
    }

    [Fact]
    public void AccuracyIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power4), ListenerKind.Power, "AccuracyPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void AccuracyAddsTheOldFunctionsNumberToTheOwnersShivs()
    {
        // game: AccuracyPower.ModifyDamageAdditive; old: CardEffects.ShivDamage adds Accuracy_ModifyDamageAdditive(state)
        // to the card it costs, which the two Shiv sites reach for a card named Shiv or HiddenDaggers, every Shiv of the
        // turn (Phantom Blades' first-Shiv test is not Accuracy's).
        var (s, shiv, daggers, strike) = AccuracyCombat();
        var k = Kernel(s, Power4);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: shiv), out var changed);

        Assert.Equal(4m + CardEffects.Accuracy_ModifyDamageAdditive(s), v);
        Assert.Equal(8m, v);
        Assert.Equal([Power("AccuracyPower", 0)], changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: daggers), out changed));
        Assert.Equal([Power("AccuracyPower", 0)], changed);
        s.ShivsPlayedThisTurn = 2;
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: shiv), out changed));
        Assert.Equal([Power("AccuracyPower", 0)], changed);

        // Where the condition fails: another card, no card, an unpowered hit, an enemy dealer, the amount at 0.
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: strike), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, props: Unpowered, card: shiv), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 4m, dealer: 1, card: shiv), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.ShivDamage, 0);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: shiv), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void AccuracyActsForTheDealerOnlyAmongTwoHolders()
    {
        // The N-holder test: the player's Shiv is the player's hit, so the slug's Accuracy adds nothing (8, not 12).
        var (s, shiv, _, _) = AccuracyCombat();
        var mine = Power("AccuracyPower", 0);
        var theirs = GivePower(s, "AccuracyPower", BuffId.ShivDamage, 4, s.Enemies[0]);
        var k = Kernel(s, Power4);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: shiv), out var changed);

        Assert.Equal(8m, v);
        Assert.Equal([mine], changed);
        Assert.DoesNotContain(theirs, changed);
    }

    [Fact]
    public void AccuracyDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, shiv, _, _) = AccuracyCombat();
        var k = Kernel(s, Without(Power4, AccuracyPower.Register));
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: shiv), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void AccuracyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power4, AllButDamage), ListenerKind.Power, "AccuracyPower", HookId.ModifyDamageAdditive);
    }

}
