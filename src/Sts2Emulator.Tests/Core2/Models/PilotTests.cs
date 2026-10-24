// The C1 pilot handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C1 pilot"): the port list's power-5, relic-2 and
// card-1 rows as migrated models under Core2/Models, each tested the way the shared rules ask: (1) its pairs are in the
// handler table and none is a stub, (2) the dispatch gives what the old function gives for the same state, (3) the
// delete-mutant (the model's registration left out) gives the no-handler value, (4) with F08 off its pairs are absent.
// The table test at the end pins the batch's (model, hook) set, standing in for EmuDump kernel-handlers and the port
// list's gate, which need the manifest and a built EmuDump. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Power;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PilotTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Pilot = Registry.Pilot;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The pilot's (model, hook) pairs, written out: the port list rows power-5, relic-2 and card-1.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] PilotPairs =
    [
        (ListenerKind.Power, "PhantomBladesPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "LethalityPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "TrackingPower", HookId.ModifyDamageMultiplicative),
    ];

    [Fact]
    public void TheRegistrysPilotBlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Pilot);
        Assert.Equal(PilotPairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePilotBlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; the pilot's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Pilot)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in PilotPairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffThePilotRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Pilot, AllButDamage)));
    }

    [Fact]
    public void TheKitsDeleteMutantLeavesOneRegistrationOut()
    {
        Action<HandlerTable> a = static t => { };
        Action<HandlerTable> b = static t => { };
        Assert.Equal([b], Without([a, b], a));
        Assert.Equal(Pilot.Count, Without(Pilot, a).Count);
    }

    // ---------------------------------------------------------------------------------------- Phantom Blades (power-5)

    private const int ShivDef = 430;

    /// <summary>The player with Phantom Blades 3, a Shiv in hand, no Shiv finished this turn, one slug.</summary>
    private static (CombatState S, CardInstance Shiv) PhantomBladesCombat(int amount = 3)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.PhantomBlades, amount));
        var shiv = Add(s, s.Hand, ShivDef);
        return (s, shiv);
    }

    [Fact]
    public void PhantomBladesIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Power, "PhantomBladesPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void PhantomBladesAddsTheOldFunctionsNumberToTheFirstShivOfTheTurn()
    {
        // game: PhantomBladesPower.ModifyDamageAdditive; old: CardEffects.ShivDamage adds
        // PhantomBlades_ModifyDamageAdditive(state) to a Shiv while ShivsPlayedThisTurn is 0.
        var (s, shiv) = PhantomBladesCombat();
        var k = Kernel(s, Pilot);
        var source = new CardHandle(shiv.Uid);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: source), out var changed);

        Assert.Equal(4m + CardEffects.PhantomBlades_ModifyDamageAdditive(s), v);
        Assert.Equal(7m, v);
        Assert.Equal([Power("PhantomBladesPower", 0)], changed);

        // Where the old site's condition fails: a Shiv already finished this turn, another card, no card, an unpowered
        // hit, an enemy dealer.
        s.ShivsPlayedThisTurn = 1;
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: source), out changed));
        Assert.Empty(changed);
        s.ShivsPlayedThisTurn = 0;
        var strike = Add(s, s.Hand, 1);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: new CardHandle(strike.Uid)), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, props: Unpowered, card: source), out changed));
        Assert.Empty(changed);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 4m, dealer: 1, card: source), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PhantomBladesDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, shiv) = PhantomBladesCombat();
        var k = Kernel(s, Without(Pilot, PhantomBladesPower.Register));
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: new CardHandle(shiv.Uid)), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PhantomBladesIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Power, "PhantomBladesPower", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Lethality (power-5)

    /// <summary>The player with Lethality 50 and the turn's lethal attack play flagged, one slug.</summary>
    private static CombatState LethalityCombat(int amount = 50, bool lethalPlay = true)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Lethality, amount));
        s.LethalAttackPlay = lethalPlay;
        return s;
    }

    [Fact]
    public void LethalityIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Power, "LethalityPower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void LethalityMultipliesByTheOldFunctionsFactorOnTheFlaggedPlay()
    {
        // game: LethalityPower.ModifyDamageMultiplicative; old: CardEffects.DealDamageToEnemy passes
        // Lethality_ModifyDamageMultiplicative(lethality) as the card multiplier when lethality > 0 && LethalAttackPlay.
        var s = LethalityCombat();
        var k = Kernel(s, Pilot);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m * (decimal)CardEffects.Lethality_ModifyDamageMultiplicative(50), v);
        Assert.Equal(9m, v);
        Assert.Equal([Power("LethalityPower", 0)], changed);
        // The same state through the old fold: 6 x 1.5 = 9.
        Assert.Equal(9, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs,
            CardEffects.Lethality_ModifyDamageMultiplicative(50)));

        // Where the old site's condition fails: the flag down, an unpowered hit, an enemy dealer, the amount at 0.
        s.LethalAttackPlay = false;
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.LethalAttackPlay = true;
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Lethality, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void LethalityDeleteMutantGivesTheNoHandlerValue()
    {
        var s = LethalityCombat();
        var k = Kernel(s, Without(Pilot, LethalityPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void LethalityIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Power, "LethalityPower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Tracking (power-5)

    /// <summary>The player with Tracking 2 and a Strike in hand, one slug with Weak 1.</summary>
    private static (CombatState S, CardHandle Card) TrackingCombat(int amount = 2)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Tracking, amount));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Weak, 1));
        var strike = Add(s, s.Hand, 1);
        return (s, new CardHandle(strike.Uid));
    }

    [Fact]
    public void TrackingIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Power, "TrackingPower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void TrackingMultipliesByItsAmountAgainstAWeakTarget()
    {
        // game: TrackingPower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage multiplies by
        // Tracking_ModifyDamageMultiplicative(tracking) when the attacker's Tracking > 0 and the defender has Weak.
        var (s, card) = TrackingCombat();
        var k = Kernel(s, Pilot);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out var changed);

        Assert.Equal(6m * BuffSystem.Tracking_ModifyDamageMultiplicative(2), v);
        Assert.Equal(12m, v);
        Assert.Equal([Power("TrackingPower", 0)], changed);
        // The same state through the old fold (the player is not Weak, the slug is): 6 x 2 = 12.
        Assert.Equal(12, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));

        // Where the condition fails: a target without Weak, an unpowered hit, an enemy dealer, no card source.
        s.Enemies[0].Buffs.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out changed));
        Assert.Empty(changed);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Weak, 1));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: card), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: card), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TrackingDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, card) = TrackingCombat();
        var k = Kernel(s, Without(Pilot, TrackingPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TrackingIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Power, "TrackingPower", HookId.ModifyDamageMultiplicative);
    }
}
