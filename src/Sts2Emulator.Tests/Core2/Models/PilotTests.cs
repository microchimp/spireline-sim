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
using Sts2Emulator.Core2.Models.Cards;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
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
        (ListenerKind.Card, "Melancholy", HookId.AfterDeath),
        (ListenerKind.Power, "PhantomBladesPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "LethalityPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "TrackingPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "DoubleDamagePower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Relic, "VitruvianMinion", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Relic, "HandDrill", HookId.AfterDamageGiven),
        (ListenerKind.Relic, "TheBoot", HookId.AfterModifyingHpLostAfterOsty),
        (ListenerKind.Relic, "TheBoot", HookId.ModifyHpLostAfterOstyLate),
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

    // ---------------------------------------------------------------------------------------- Double Damage (power-5)

    /// <summary>The player with Double Damage (two stacks: still double) and a Strike in hand, one slug.</summary>
    private static (CombatState S, CardHandle Card) DoubleDamageCombat(int amount = 2)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.DoubleDamage, amount));
        var strike = Add(s, s.Hand, 1);
        return (s, new CardHandle(strike.Uid));
    }

    [Fact]
    public void DoubleDamageIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Power, "DoubleDamagePower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void DoubleDamageDoublesTheOwnersCardAttackWhateverItsStackCount()
    {
        // game: DoubleDamagePower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage multiplies by
        // DoubleDamage_ModifyDamageMultiplicative() (a flat 2) when the attacker's Double Damage > 0.
        var (s, card) = DoubleDamageCombat();
        var k = Kernel(s, Pilot);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out var changed);

        Assert.Equal(6m * BuffSystem.DoubleDamage_ModifyDamageMultiplicative(), v);
        Assert.Equal(12m, v);
        Assert.Equal([Power("DoubleDamagePower", 0)], changed);
        Assert.Equal(12, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));

        // Where the condition fails: an unpowered hit, an enemy dealer, no card source, the amount at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: card), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: card), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.DoubleDamage, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void DoubleDamageDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, card) = DoubleDamageCombat();
        var k = Kernel(s, Without(Pilot, DoubleDamagePower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: card), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void DoubleDamageIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Power, "DoubleDamagePower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Vitruvian Minion (relic-2)

    private const int MinionStrikeDef = 310;

    /// <summary>The player holding Vitruvian Minion with a Minion Strike and a Strike in hand, one slug.</summary>
    private static (CombatState S, ListenerRef Relic, CardHandle Minion, CardHandle Strike) VitruvianMinionCombat()
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.VitruvianMinion);
        var minion = Add(s, s.Hand, MinionStrikeDef);
        var strike = Add(s, s.Hand, 1);
        return (s, relic, new CardHandle(minion.Uid), new CardHandle(strike.Uid));
    }

    [Fact]
    public void VitruvianMinionIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Relic, "VitruvianMinion", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void VitruvianMinionDoublesAMinionTaggedCardsDamage()
    {
        // game: VitruvianMinion.ModifyDamageMultiplicative; old: RelicEffects.CardDamageMultiplier returns
        // VitruvianMinion_ModifyDamageMultiplicative() for a def with MinionTag while the relic is held.
        var (s, relic, minion, strike) = VitruvianMinionCombat();
        var k = Kernel(s, Pilot);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: minion), out var changed);

        Assert.Equal(6m * RelicEffects.CardDamageMultiplier(s, GeneratedData.Cards.Get(MinionStrikeDef)), v);
        Assert.Equal(12m, v);
        Assert.Equal([relic], changed);

        // Where the condition fails: a card without the tag, no card source, the relic not held.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
        Assert.Equal(1, RelicEffects.CardDamageMultiplier(s, GeneratedData.Cards.Get(1)));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: minion), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VitruvianMinionDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _, minion, _) = VitruvianMinionCombat();
        var k = Kernel(s, Without(Pilot, VitruvianMinion.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: minion), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void VitruvianMinionIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Relic, "VitruvianMinion", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- The Boot (relic-2)

    /// <summary>The player holding The Boot, one slug.</summary>
    private static (CombatState S, ListenerRef Relic) BootCombat(int enemyHp = 40)
    {
        var s = Combat(enemyHp);
        return (s, Hold(s, RelicEffects.TheBoot));
    }

    [Fact]
    public void TheBootIsRegisteredForItsTwoPairs()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Relic, "TheBoot", HookId.ModifyHpLostAfterOstyLate,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    [Fact]
    public void TheBootFloorsOneToFourLostToFiveInTheLateAfterOstyPass()
    {
        // game: TheBoot.ModifyHpLostAfterOstyLate; old: CardEffects.LandDamageOnEnemy sets hpLoss =
        // RelicEffects.BootDamageFloor(state, hpLoss) for a powered hit. The fold's changed set is by the truncated value.
        var (s, relic) = BootCombat();
        var k = Kernel(s, Pilot);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m), out var changed);

        Assert.Equal(RelicEffects.BootDamageFloor(s, 3), v);
        Assert.Equal(5m, v);
        Assert.Equal([relic], changed);

        // Five or more, and nothing lost: the old function leaves them; so does the handler, decimals kept.
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out changed));
        Assert.Empty(changed);
        Assert.Equal(5.5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 5.5m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0m), out changed));
        Assert.Empty(changed);
        // A decimal between the bounds is decided on its truncation: 4.5 lost is 4 lost, floored to 5; 0.5 is 0, kept.
        Assert.Equal(5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 4.5m), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0.5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0.5m), out changed));
        Assert.Empty(changed);

        // Where the old site's condition fails: the before-Osty phase (another pass), an unpowered hit, an enemy's hit
        // on the player, a hit on a creature that is no enemy, the relic not held.
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 0), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TheBootsFloorLandsThroughTheDamageCommandAfterBlock()
    {
        // game: CreatureCmd.Damage runs Hook.ModifyHpLost after the block absorption and the pet redirect, then
        // LoseHpInternal. A 3 hit on 40: 5 lost. Block 2 against 5: 3 unblocked, floored to 5. Block 5 against 3: fully
        // blocked, 0 lost, no floor. An unpowered 3 on no block: 3 lost.
        var (s, _) = BootCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Pilot);

        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(slug), 3m)).UnblockedDamage);
        Assert.Equal(35, slug.Hp);
        slug.Block = 2;
        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        Assert.Equal(30, slug.Hp);
        slug.Block = 5;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Of(slug), 3m)).UnblockedDamage);
        Assert.Equal(30, slug.Hp);
        slug.Block = 0;
        Assert.Equal(3, Assert.Single(Hit(k, OldCreature.Of(slug), 3m, props: Unpowered)).UnblockedDamage);
        Assert.Equal(27, slug.Hp);
    }

    [Fact]
    public void TheBootsAfterModifyingHpLostAfterOstyDoesNothingToTheState()
    {
        // game: TheBoot.AfterModifyingHpLostAfterOsty, reached by the filter-set pass when the relic modified the loss.
        // The old engine has no body at this point; the handler returns Done and the state is as it was.
        // NO DELETE-MUTANT for this pair (CLOUD_BRIEFS section 3, rule 3, does not apply): the game's override only flashes
        // the relic (presentation, no combat state), so with the handler deleted the state is the same and no assertion on
        // it could fail. The pair is pinned instead by TheBootIsRegisteredForItsTwoPairs and the table test. A mutant is
        // not faked here.
        var (s, relic) = BootCombat();
        var k = Kernel(s, Pilot);
        int hp = s.Enemies[0].Hp;
        var status = Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([relic]));
        Assert.False(status.IsPaused);
        Assert.Equal(hp, s.Enemies[0].Hp);
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal([new RelicInstance(RelicEffects.TheBoot)], s.Relics);
    }

    [Fact]
    public void TheBootDeleteMutantGivesTheNoHandlerValue()
    {
        // The late pair (ModifyHpLostAfterOstyLate) only: the after-Osty pair has no mutant, see above.
        var (s, _) = BootCombat();
        var k = Kernel(s, Without(Pilot, TheBoot.Register));
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m), out var changed));
        Assert.Empty(changed);
        Assert.Equal(3, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 3m)).UnblockedDamage);
        Assert.Equal(37, s.Enemies[0].Hp);
    }

    [Fact]
    public void TheBootIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Relic, "TheBoot", HookId.ModifyHpLostAfterOstyLate,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    // ---------------------------------------------------------------------------------------- Hand Drill (relic-2)

    /// <summary>The player holding Hand Drill, one slug with 3 block.</summary>
    private static (CombatState S, ListenerRef Relic) HandDrillCombat(int block = 3)
    {
        var s = Combat(40);
        s.Enemies[0].Block = block;
        return (s, Hold(s, RelicEffects.HandDrill));
    }

    private static int Vulnerable(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.Vulnerable);

    [Fact]
    public void HandDrillIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Relic, "HandDrill", HookId.AfterDamageGiven);
    }

    [Fact]
    public void HandDrillAppliesTheOldBodysVulnerableWhenAHitBreaksBlock()
    {
        // game: HandDrill.AfterDamageGiven on result.WasBlockBroken; old: CardEffects.LandDamageOnEnemy calls
        // RelicEffects.ApplyAfterBlockBroken(state, target) when block was there before the hit and is 0 after it.
        var (s, _) = HandDrillCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Pilot);

        var r = Assert.Single(Hit(k, OldCreature.Of(slug), 5m));

        Assert.True(r.WasBlockBroken);
        Assert.Equal(38, slug.Hp);
        var oracle = HandDrillCombat().S;
        RelicEffects.ApplyAfterBlockBroken(oracle, oracle.Enemies[0]);
        Assert.Equal(Vulnerable(oracle.Enemies[0]), Vulnerable(slug));
        Assert.Equal(2, Vulnerable(slug));

        // A hit that leaves block standing, one on no block, and one exactly the size of the block (broken, fully
        // blocked): the old site's rule.
        var (s2, _) = HandDrillCombat(block: 9);
        var k2 = Kernel(s2, Pilot);
        Assert.False(Assert.Single(Hit(k2, OldCreature.Of(s2.Enemies[0]), 5m)).WasBlockBroken);
        Assert.Equal(0, Vulnerable(s2.Enemies[0]));
        var (s3, _) = HandDrillCombat(block: 0);
        var k3 = Kernel(s3, Pilot);
        Assert.False(Assert.Single(Hit(k3, OldCreature.Of(s3.Enemies[0]), 5m)).WasBlockBroken);
        Assert.Equal(0, Vulnerable(s3.Enemies[0]));
        var (s4, _) = HandDrillCombat(block: 5);
        var k4 = Kernel(s4, Pilot);
        Assert.True(Assert.Single(Hit(k4, OldCreature.Of(s4.Enemies[0]), 5m)).WasBlockBroken);
        Assert.Equal(2, Vulnerable(s4.Enemies[0]));
    }

    [Fact]
    public void HandDrillDoesNothingForAnEnemysHitOrAKilledTarget()
    {
        // An enemy breaking the player's block is not the relic's owner's hit; a target the hit kills is not alive for
        // the old body (its Hp > 0 check), so no Vulnerable lands on the corpse.
        var (s, _) = HandDrillCombat();
        s.PlayerBlock = 3;
        var slug = s.Enemies[0];
        var k = Kernel(s, Pilot);
        Assert.True(Assert.Single(Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(slug))).WasBlockBroken);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Vulnerable));

        var (s2, _) = HandDrillCombat();
        var dying = s2.Enemies[0];
        dying.Hp = 2;
        var k2 = Kernel(s2, Pilot);
        k2.Commands.KillStandIn = (_, _) => { }; // the kill would strip the corpse's powers: keep them to look
        var r = Assert.Single(Hit(k2, OldCreature.Of(dying), 9m));
        Assert.True(r.WasBlockBroken);
        Assert.True(r.WasTargetKilled);
        Assert.Equal(0, Vulnerable(dying));
    }

    [Fact]
    public void HandDrillDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = HandDrillCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Without(Pilot, HandDrill.Register));
        Assert.True(Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).WasBlockBroken);
        Assert.Equal(0, Vulnerable(slug));
    }

    [Fact]
    public void HandDrillIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Relic, "HandDrill", HookId.AfterDamageGiven);
    }

    // ---------------------------------------------------------------------------------------- Melancholy (card-1)

    private const int MelancholyDef = 301;

    /// <summary>Three Melancholy copies (hand, draw, exhaust; the draw one upgraded, the hand one already bumped) among
    /// Strikes in every pile, two slugs.</summary>
    private static CombatState MelancholyCombat()
    {
        var s = Combat(40, 40);
        Add(s, s.Hand, 1);
        var hand = Add(s, s.Hand, MelancholyDef);
        s.Hand[^1] = hand with { CostBump = 2 };
        Add(s, s.DrawPile, MelancholyDef, upgraded: true);
        Add(s, s.DrawPile, 1);
        Add(s, s.DiscardPile, 1);
        Add(s, s.ExhaustPile, MelancholyDef);
        return s;
    }

    /// <summary>Every pile's cards by (uid, def), pile by pile: a move or a swap of a copy changes it, a cost bump does not.</summary>
    private static string Layout(CombatState s) =>
        string.Join("|", new[] { s.Hand, s.DrawPile, s.DiscardPile, s.ExhaustPile }
            .Select(p => string.Join(",", p.Select(c => $"{c.Uid}:{c.DefId}"))));

    private static IEnumerable<int> Bumps(CombatState s) =>
        new[] { s.Hand, s.DrawPile, s.DiscardPile, s.ExhaustPile }.SelectMany(p => p).Select(c => c.CostBump);

    [Fact]
    public void MelancholyIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Pilot), ListenerKind.Card, "Melancholy", HookId.AfterDeath);
    }

    [Fact]
    public void MelancholyCheapensEachCopyOnceOnADeathAsTheOldFunctionDoesInOneCall()
    {
        // game: Melancholy.AfterDeath (EnergyCost.AddThisCombat on the card, once per copy, for a death not prevented);
        // old: CardEffects.CheapenMelancholyForDeath(state) lowers every copy's CostBump by one in one call. The
        // dispatcher visits each copy's listener once, so the piles equal the old function's result.
        var s = MelancholyCombat();
        var k = Kernel(s, Pilot);
        var oracle = MelancholyCombat();

        string layout = Layout(s);
        Hooks.AfterDeath(k, AfterDeathOf(1));
        CardEffects.CheapenMelancholyForDeath(oracle);

        Assert.Equal(Bumps(oracle), Bumps(s));
        Assert.Equal([0, 1, -1, 0, 0, -1], Bumps(s));
        Assert.Equal(layout, Layout(s)); // the copies stay where they were, in every pile
        Assert.Equal([MelancholyDef], s.ExhaustPile.Select(c => c.DefId));

        // A second death: one more off each copy; a prevented death: nothing.
        Hooks.AfterDeath(k, AfterDeathOf(2));
        Assert.Equal([0, 0, -2, 0, 0, -2], Bumps(s));
        Hooks.AfterDeath(k, AfterDeathOf(1, removalPrevented: true));
        Assert.Equal([0, 0, -2, 0, 0, -2], Bumps(s));
    }

    [Fact]
    public void MelancholyDeleteMutantGivesTheNoHandlerValue()
    {
        var s = MelancholyCombat();
        var k = Kernel(s, Without(Pilot, Melancholy.Register));
        Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal([0, 2, 0, 0, 0, 0], Bumps(s));
    }

    [Fact]
    public void MelancholyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Pilot, AllButDamage), ListenerKind.Card, "Melancholy", HookId.AfterDeath);
    }
}
