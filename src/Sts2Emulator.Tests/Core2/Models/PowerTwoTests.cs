// The C3 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C3"): the port list's power-2 rows as migrated models
// under Core2/Models/Powers, each tested the way the shared rules ask: (1) its pairs are in the handler table and none is
// a stub, (2) the dispatch gives what the old function gives for the same state, (3) the delete-mutant (the model's
// registration left out) gives the no-handler value, (4) with F08 off its pairs are absent, (5) a holder acts for its
// owner alone. The death hooks are dispatched with the kit's builders inside Dying (the dead creature at 0 HP, marked
// dying, still listed), and once through CreatureCmd.Kill. The table test at the end pins the batch's (model, hook)
// set, standing in for EmuDump kernel-handlers and the port list's gate, which need the manifest and a built EmuDump.
// Surprise is not in the batch: its old AfterDeath body takes the step's Random, which no handler has (the stop rule;
// the PR lists it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PowerTwoTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Power2 = Registry.Power2;

    /// <summary>A basic Attack card (StrikeIronclad) and a basic Skill (DefendIronclad), by def id.</summary>
    private const int StrikeDef = 472;
    private const int DefendDef = 131;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list rows power-2 of the models the batch ports.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Power2Pairs =
    [
        (ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive),
        (ListenerKind.Power, "GigantificationPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "IntangiblePower", HookId.ModifyDamageCap),
        (ListenerKind.Power, "IntangiblePower", HookId.ModifyHpLostAfterOsty),
        (ListenerKind.Power, "IntangiblePower", HookId.AfterModifyingHpLostAfterOsty),
        (ListenerKind.Power, "SlipperyPower", HookId.ModifyHpLostAfterOsty),
        (ListenerKind.Power, "SlipperyPower", HookId.AfterDamageReceived),
        (ListenerKind.Power, "BufferPower", HookId.ModifyHpLostAfterOstyLate),
        (ListenerKind.Power, "BufferPower", HookId.AfterModifyingHpLostAfterOsty),
        (ListenerKind.Power, "PossessSpeedPower", HookId.AfterDeath),
    ];

    [Fact]
    public void TheRegistrysPower2BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Power2);
        Assert.Equal(Power2Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePower2BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; this batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Power2)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Power2Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Power2, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- Tainted (power-2)

    /// <summary>The player with Tainted 2, one slug.</summary>
    private static CombatState TaintedCombat(int amount = 2)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Tainted, amount));
        return s;
    }

    [Fact]
    public void TaintedIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void TaintedAddsTheOldFunctionsNumberToAPoweredAttackOnItsOwner()
    {
        // game: TaintedPower.ModifyDamageAdditive; old: BuffSystem.IncomingDamage adds
        // Tainted_ModifyDamageAdditive(defenderBuffs) to every powered hit, whoever deals it.
        var s = TaintedCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(6m + BuffSystem.Tainted_ModifyDamageAdditive(s.PlayerBuffs), v);
        Assert.Equal(8m, v);
        Assert.Equal([Power("TaintedPower", 0)], changed);
        // The same state through the old fold (the slug attacks the player): 6 + 2 = 8.
        Assert.Equal(8, BuffSystem.IncomingDamage(6, s.Enemies[0].Buffs, s.PlayerBuffs));
        // A hit with no dealer counts too: the old function reads the defender alone.
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: null), out changed));
        Assert.Equal([Power("TaintedPower", 0)], changed);

        // Where the condition fails: a hit on another creature (the owner's own attack: the old fold reads the slug's
        // Tainted there, 0), an unpowered hit on the owner, the stack at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        Assert.Equal(6, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, props: Unpowered, dealer: 1), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Tainted, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TaintedActsForItsOwnerAloneAmongThreeHolders()
    {
        // The N-holder test: the player and both slugs hold Tainted 2. A hit on the first slug takes that slug's addend
        // once (8, not 12) and changes its listener alone; the second slug's hit on the player changes the player's.
        var s = Combat(40, 40);
        var holders = Holders(s, "TaintedPower", BuffId.Tainted, 2);
        var k = Kernel(s, Power2);

        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);
    }

    [Fact]
    public void TaintedDeleteMutantGivesTheNoHandlerValue()
    {
        var s = TaintedCombat();
        var k = Kernel(s, Without(Power2, TaintedPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TaintedIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Gigantification (power-2)

    /// <summary>The player with Gigantification 1, an Attack card and a Skill card in hand, one slug.</summary>
    private static (CombatState S, CardHandle Attack, CardHandle Skill) GigantificationCombat(int amount = 1)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Gigantification, amount));
        var strike = Add(s, s.Hand, StrikeDef);
        var defend = Add(s, s.Hand, DefendDef);
        return (s, new CardHandle(strike.Uid), new CardHandle(defend.Uid));
    }

    [Fact]
    public void GigantificationIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "GigantificationPower", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void GigantificationTriplesTheOwnersAttackCardsDamage()
    {
        // game: GigantificationPower.ModifyDamageMultiplicative; old: CardEffects.DmgFrom multiplies the card's damage by
        // Gigantification_ModifyDamageMultiplicative() (a flat 3) for an Attack def while the player's Gigantification > 0.
        var (s, attack, skill) = GigantificationCombat();
        var k = Kernel(s, Power2);
        Assert.Equal(CardType.Attack, GeneratedData.Cards.Get(StrikeDef).Type);
        Assert.Equal(CardType.Skill, GeneratedData.Cards.Get(DefendDef).Type);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: attack), out var changed);

        Assert.Equal(6m * CardEffects.Gigantification_ModifyDamageMultiplicative(), v);
        Assert.Equal(18m, v);
        Assert.Equal([Power("GigantificationPower", 0)], changed);
        // Two stacks still triple once: the number is flat.
        s.PlayerBuffs[0] = new BuffState(BuffId.Gigantification, 2);
        Assert.Equal(18m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: attack), out changed));
        Assert.Equal([Power("GigantificationPower", 0)], changed);

        // Where the condition fails: a Skill as the card source, no card source, an unpowered hit, the amount at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: skill), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: attack), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Gigantification, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: attack), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void GigantificationFollowsTheOwnersCardNotTheDealerAndNeverAnEnemysHolder()
    {
        // The factor follows the card (the old site reads no dealer; the game triples the latched command's card source), so
        // the owner's card dealt by its pet counts (Osty is creature 1, the slug 2). The N-holder case: a slug holding
        // Gigantification does not triple the player's card (the card's owner is the player), and the factor lands once;
        // the slug's own hit has no card source and is not tripled.
        var (s, attack, _) = GigantificationCombat();
        WithOsty(s);
        var theirs = GivePower(s, "GigantificationPower", BuffId.Gigantification, 1, s.Enemies[0]);
        var k = Kernel(s, Power2);

        Assert.Equal(18m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1, card: attack), out var changed));
        Assert.Equal([Power("GigantificationPower", 0)], changed);
        Assert.Equal(18m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, card: attack), out changed));
        Assert.Equal([Power("GigantificationPower", 0)], changed);
        Assert.DoesNotContain(theirs, changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void GigantificationDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, attack, _) = GigantificationCombat();
        var k = Kernel(s, Without(Power2, GigantificationPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: attack), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void GigantificationIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "GigantificationPower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Intangible (power-2)

    /// <summary>One slug with Intangible 1 (40 HP).</summary>
    private static (CombatState S, ListenerRef Power) IntangibleCombat(int amount = 1)
    {
        var s = Combat(40);
        return (s, GivePower(s, "IntangiblePower", BuffId.Intangible, amount, s.Enemies[0]));
    }

    [Fact]
    public void IntangibleIsRegisteredForItsThreePairs()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "IntangiblePower", HookId.ModifyDamageCap, HookId.ModifyHpLostAfterOsty,
            HookId.AfterModifyingHpLostAfterOsty);
        // The row the port list marks not modelled has no handler.
        AssertAbsent(Table(Power2), ListenerKind.Power, "IntangiblePower", HookId.AfterModifyingDamageAmount);
    }

    [Fact]
    public void IntangibleCapsTheDamageAimedAtItsOwnerAtTheOldFunctionsOne()
    {
        // game: IntangiblePower.ModifyDamageCap; old: BuffSystem.IncomingDamage (and the unpowered and player-side sites)
        // end with CapIncomingDamage(damage, defenderBuffs): Math.Min(damage, 1) while the defender's Intangible > 0. The
        // cap pass lists the power when its cap lowered the damage.
        var (s, power) = IntangibleCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m), out var changed);

        Assert.Equal(BuffSystem.CapIncomingDamage(9, s.Enemies[0].Buffs), v);
        Assert.Equal(1m, v);
        Assert.Equal([power], changed);
        Assert.Equal(1, BuffSystem.IncomingDamage(9, s.PlayerBuffs, s.Enemies[0].Buffs));
        // The cap reads no props and no dealer: an unpowered hit, a hit with no dealer.
        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m, props: Unpowered), out changed));
        Assert.Equal([power], changed);
        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m, dealer: null), out changed));
        Assert.Equal([power], changed);
        // A damage the cap does not lower is not changed: exactly 1, 0, a decimal under 1 (kept, decimals and all);
        // 1.5 is above the cap and lowered to 1 (the old site caps its int 1 to 1: both land 1).
        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 1m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 0m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0.5m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 0.5m), out changed));
        Assert.Empty(changed);
        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 1.5m), out changed));
        Assert.Equal([power], changed);
        // Where the condition fails: a hit on another creature, the stack at 0.
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 9m, dealer: 1), out changed));
        Assert.Empty(changed);
        s.Enemies[0].Buffs[0] = new BuffState(BuffId.Intangible, 0);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m), out changed));
        Assert.Empty(changed);
        Assert.Equal(9, BuffSystem.CapIncomingDamage(9, s.Enemies[0].Buffs));
    }

    [Fact]
    public void IntangibleCapsTheHpLostByItsOwnerAtTheOldFunctionsOne()
    {
        // game: IntangiblePower.ModifyHpLostAfterOsty; old: CardEffects.LosePlayerHp runs CapHpLoss(hpLoss, PlayerBuffs):
        // Math.Min(hpLoss, 1) while the owner's Intangible > 0. The fold's changed set is by the truncated value.
        var (s, power) = IntangibleCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m), out var changed);

        Assert.Equal(BuffSystem.CapHpLoss(9, s.Enemies[0].Buffs), v);
        Assert.Equal(1m, v);
        Assert.Equal([power], changed);
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m, props: Unpowered, dealer: null), out changed));
        Assert.Equal([power], changed);
        // A loss the cap does not lower is left as it was, decimals kept: 1, 0, 0.5, and 1.5 (its truncation is 1 already).
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 1m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0.5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0.5m), out changed));
        Assert.Empty(changed);
        Assert.Equal(1.5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 1.5m), out changed));
        Assert.Empty(changed);
        // Where the condition fails: the before-Osty phase (another pass), a loss of another creature, the stack at 0.
        Assert.Equal(9m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(9m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 9m, dealer: 1), out changed));
        Assert.Empty(changed);
        s.Enemies[0].Buffs[0] = new BuffState(BuffId.Intangible, 0);
        Assert.Equal(9m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void IntangiblesTwoCapsLandThroughTheDamageCommandOnAnEnemyAndOnThePlayer()
    {
        // game: CreatureCmd.Damage runs Hook.ModifyDamage (the cap) before the block and Hook.ModifyHpLost after it. A 9 hit
        // on an intangible slug: 1 lost; an unpowered 9: 1 lost; block 3 against 9: the capped 1 is absorbed, 0 lost,
        // block 2. The intangible player against the slug's 9: 1 lost, as the old CardEffects.DealDamageToPlayer leaves it.
        var (s, _) = IntangibleCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power2);

        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Of(slug), 9m)).UnblockedDamage);
        Assert.Equal(39, slug.Hp);
        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Of(slug), 9m, props: Unpowered)).UnblockedDamage);
        Assert.Equal(38, slug.Hp);
        slug.Block = 3;
        var r = Assert.Single(Hit(k, OldCreature.Of(slug), 9m));
        Assert.Equal((0, 1, 2, 38), (r.UnblockedDamage, r.BlockedDamage, slug.Block, slug.Hp));

        s.PlayerBuffs.Add(new BuffState(BuffId.Intangible, 1));
        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Player, 9m, dealer: OldCreature.Of(slug))).UnblockedDamage);
        Assert.Equal(49, s.PlayerHp);
        var oracle = Combat(40);
        oracle.PlayerBuffs.Add(new BuffState(BuffId.Intangible, 1));
        CardEffects.DealDamageToPlayer(oracle, 9);
        Assert.Equal(oracle.PlayerHp, s.PlayerHp);
    }

    [Fact]
    public void IntangiblesAfterModifyingHpLostAfterOstyDoesNothingToTheState()
    {
        // game: IntangiblePower.AfterModifyingHpLostAfterOsty, reached by the filter-set pass when the power lowered the
        // loss. The old engine has no body at this point (the row's locations are the two cap numbers); the handler
        // returns Done and the state is as it was. NO DELETE-MUTANT for this pair (CLOUD_BRIEFS section 3, rule 3, does
        // not apply): the old engine changes no state at this point, so with the handler deleted the state is the same
        // and no assertion on it could fail. The pair is pinned by IntangibleIsRegisteredForItsThreePairs and the table
        // test; what the game's override does is for the decompile review. A mutant is not faked here.
        var (s, power) = IntangibleCombat();
        var k = Kernel(s, Power2);
        var status = Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([power]));
        Assert.False(status.IsPaused);
        Assert.Equal(40, s.Enemies[0].Hp);
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal([new BuffState(BuffId.Intangible, 1)], s.Enemies[0].Buffs);
        Assert.Empty(s.PlayerBuffs);
    }

    [Fact]
    public void IntangibleActsForItsOwnerAloneAmongThreeHolders()
    {
        // The N-holder test: the player and both slugs are intangible. A hit on the first slug is capped by that slug's
        // power alone; the second slug's hit on the player by the player's; the HP-loss fold the same.
        var s = Combat(40, 40);
        var holders = Holders(s, "IntangiblePower", BuffId.Intangible, 1);
        var k = Kernel(s, Power2);

        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(1m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 9m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 2, 9m), out changed));
        Assert.Equal([holders[2]], changed);
    }

    [Fact]
    public void IntangibleDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = IntangibleCombat();
        var k = Kernel(s, Without(Power2, IntangiblePower.Register));
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 9m), out var changed));
        Assert.Empty(changed);
        Assert.Equal(9m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m), out changed));
        Assert.Empty(changed);
        Assert.Equal(9, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 9m)).UnblockedDamage);
        Assert.Equal(31, s.Enemies[0].Hp);
    }

    [Fact]
    public void IntangibleIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "IntangiblePower", HookId.ModifyDamageCap,
            HookId.ModifyHpLostAfterOsty, HookId.AfterModifyingHpLostAfterOsty);
    }

    // ---------------------------------------------------------------------------------------- Slippery (power-2)

    /// <summary>One slug with Slippery 2 (40 HP).</summary>
    private static (CombatState S, ListenerRef Power) SlipperyCombat(int amount = 2)
    {
        var s = Combat(40);
        return (s, GivePower(s, "SlipperyPower", BuffId.Slippery, amount, s.Enemies[0]));
    }

    private static int SlipperyOf(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.Slippery);

    [Fact]
    public void SlipperyIsRegisteredForItsTwoPairs()
    {
        // The row registers the stack's spending (Slippery_AfterModifyingHpLostAfterOsty) under AfterDamageReceived.
        AssertRegistered(Table(Power2), ListenerKind.Power, "SlipperyPower", HookId.ModifyHpLostAfterOsty, HookId.AfterDamageReceived);
        AssertAbsent(Table(Power2), ListenerKind.Power, "SlipperyPower", HookId.AfterModifyingHpLostAfterOsty);
    }

    [Fact]
    public void SlipperySetsALossOfOneOrMoreToTheOldFunctionsOne()
    {
        // game: SlipperyPower.ModifyHpLostAfterOsty; old: CardEffects.LandDamageOnEnemy sets hpLoss =
        // Slippery_ModifyHpLostAfterOsty() when the target's Slippery > 0 and hpLoss >= 1, powered or not.
        var (s, power) = SlipperyCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out var changed);

        Assert.Equal(CardEffects.Slippery_ModifyHpLostAfterOsty(), v);
        Assert.Equal(1m, v);
        Assert.Equal([power], changed);
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m, props: Unpowered, dealer: null), out changed));
        Assert.Equal([power], changed);
        // Exactly 1 is left as it is; a loss under 1 too (it truncates to 0, where the old site's int is 0); 1.5 becomes 1
        // (its truncation was 1 already, so the fold lists no modifier).
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 1m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0.5m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0.5m), out changed));
        Assert.Empty(changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 0m), out changed));
        Assert.Empty(changed);
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 1.5m), out changed));
        Assert.Empty(changed);
        // Where the condition fails: the before-Osty phase, a loss of another creature, the stack at 0.
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1), out changed));
        Assert.Empty(changed);
        s.Enemies[0].Buffs[0] = new BuffState(BuffId.Slippery, 0);
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void SlipperySpendsAStackThroughTheDamageCommandWhenAHitTakesHp()
    {
        // game: SlipperyPower.AfterDamageReceived; old: CardEffects.LandDamageOnEnemy calls
        // Slippery_AfterModifyingHpLostAfterOsty(target) beside the number above, under the same condition. A 7 hit on 40
        // with Slippery 2: 1 lost, Slippery 1; an unpowered 7: 1 lost, Slippery gone; a third 7: 7 lost.
        var (s, _) = SlipperyCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power2);

        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Of(slug), 7m)).UnblockedDamage);
        Assert.Equal((39, 1), (slug.Hp, SlipperyOf(slug)));
        var oracle = SlipperyCombat().S.Enemies[0];
        oracle.Hp -= CardEffects.Slippery_ModifyHpLostAfterOsty();
        CardEffects.Slippery_AfterModifyingHpLostAfterOsty(oracle);
        Assert.Equal((oracle.Hp, SlipperyOf(oracle)), (slug.Hp, SlipperyOf(slug)));

        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Of(slug), 7m, props: Unpowered)).UnblockedDamage);
        Assert.Equal((38, 0), (slug.Hp, SlipperyOf(slug)));
        Assert.Equal(7, Assert.Single(Hit(k, OldCreature.Of(slug), 7m)).UnblockedDamage);
        Assert.Equal(31, slug.Hp);

        // Block: a fully blocked 7 takes no HP and spends no stack; block 3 against 7 leaves 4, set to 1, a stack spent.
        // The slug's hit on the player is not a hit on the owner.
        var (s2, _) = SlipperyCombat();
        var slug2 = s2.Enemies[0];
        slug2.Block = 9;
        var k2 = Kernel(s2, Power2);
        Assert.Equal(0, Assert.Single(Hit(k2, OldCreature.Of(slug2), 7m)).UnblockedDamage);
        Assert.Equal((40, 2), (slug2.Hp, SlipperyOf(slug2)));
        slug2.Block = 3;
        Assert.Equal(1, Assert.Single(Hit(k2, OldCreature.Of(slug2), 7m)).UnblockedDamage);
        Assert.Equal((39, 1), (slug2.Hp, SlipperyOf(slug2)));
        Assert.Equal(7, Assert.Single(Hit(k2, OldCreature.Player, 7m, dealer: OldCreature.Of(slug2))).UnblockedDamage);
        Assert.Equal(1, SlipperyOf(slug2));
    }

    [Fact]
    public void SlipperyActsForItsOwnerAloneAmongTwoHolders()
    {
        // The N-holder test: two slippery slugs. A hit on the first sets its loss and spends its stack; the second keeps
        // both, and its own hit changes its own listener.
        var s = Combat(40, 40);
        var first = GivePower(s, "SlipperyPower", BuffId.Slippery, 2, s.Enemies[0]);
        var second = GivePower(s, "SlipperyPower", BuffId.Slippery, 2, s.Enemies[1]);
        var k = Kernel(s, Power2);

        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out var changed));
        Assert.Equal([first], changed);
        Assert.Equal(1, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 7m)).UnblockedDamage);
        Assert.Equal((39, 1), (s.Enemies[0].Hp, SlipperyOf(s.Enemies[0])));
        Assert.Equal((40, 2), (s.Enemies[1].Hp, SlipperyOf(s.Enemies[1])));
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 2, 7m), out changed));
        Assert.Equal([second], changed);
    }

    [Fact]
    public void SlipperyDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = SlipperyCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Without(Power2, SlipperyPower.Register));
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out var changed));
        Assert.Empty(changed);
        Assert.Equal(7, Assert.Single(Hit(k, OldCreature.Of(slug), 7m)).UnblockedDamage);
        Assert.Equal((33, 2), (slug.Hp, SlipperyOf(slug)));
    }

    [Fact]
    public void SlipperyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "SlipperyPower", HookId.ModifyHpLostAfterOsty,
            HookId.AfterDamageReceived);
    }

    // ---------------------------------------------------------------------------------------- Buffer (power-2)

    /// <summary>The player with Buffer 2, one slug.</summary>
    private static (CombatState S, ListenerRef Power) BufferCombat(int amount = 2)
    {
        var s = Combat(40);
        return (s, GivePower(s, "BufferPower", BuffId.Buffer, amount));
    }

    private static int BufferOf(CombatState s) => BuffSystem.Get(s.PlayerBuffs, BuffId.Buffer);

    [Fact]
    public void BufferIsRegisteredForItsTwoPairs()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "BufferPower", HookId.ModifyHpLostAfterOstyLate,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    [Fact]
    public void BufferTakesTheOwnersLossToTheOldFunctionsZeroInTheLateAfterOstyPass()
    {
        // game: BufferPower.ModifyHpLostAfterOstyLate; old: CardEffects.LosePlayerHp returns
        // Buffer_ModifyHpLostAfterOstyLate() for a loss above 0 while the player's Buffer > 0, whatever the props. The
        // fold's changed set is by the truncated value, so a loss that truncates to 0 does not list the power (the old
        // site returns before Buffer for it).
        var (s, power) = BufferCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1), out var changed);

        Assert.Equal(CardEffects.Buffer_ModifyHpLostAfterOstyLate(), v);
        Assert.Equal(0m, v);
        Assert.Equal([power], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, props: Unpowered, dealer: null), out changed));
        Assert.Equal([power], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 1m, dealer: 1), out changed));
        Assert.Equal([power], changed);
        // A loss under 1 goes to the same 0, with the power listed as no modifier (its truncation was 0 already).
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0.5m, dealer: 1), out changed));
        Assert.Empty(changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0m, dealer: 1), out changed));
        Assert.Empty(changed);
        // Where the condition fails: the before-Osty phase (another pass), a loss of another creature, the stack at 0.
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Buffer, 0);
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void BuffersAfterModifyingHpLostAfterOstySpendsTheOldFunctionsStack()
    {
        // game: BufferPower.AfterModifyingHpLostAfterOsty, reached by the filter-set pass when the power is among the
        // modifiers: one stack spent (2 to 1), and at 1 the power removed, as Buffer_AfterModifyingHpLostAfterOsty does.
        // Dispatched with the power absent from the modifiers, the pass does not reach it.
        var (s, power) = BufferCombat();
        var k = Kernel(s, Power2);
        var oracle = BufferCombat().S;

        Assert.False(Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([power])).IsPaused);
        CardEffects.Buffer_AfterModifyingHpLostAfterOsty(oracle, 2);

        Assert.Equal(BufferOf(oracle), BufferOf(s));
        Assert.Equal(1, BufferOf(s));
        Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([]));
        Assert.Equal(1, BufferOf(s));
        Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([power]));
        CardEffects.Buffer_AfterModifyingHpLostAfterOsty(oracle, 1);
        Assert.Equal(0, BufferOf(s));
        Assert.Equal(oracle.PlayerBuffs, s.PlayerBuffs);
        Assert.DoesNotContain(s.PlayerBuffs, b => b.Id == BuffId.Buffer);
    }

    [Fact]
    public void BufferSpendsAStackThroughTheDamageCommandOnlyWhenHpWouldHaveComeOff()
    {
        // game: Hook.ModifyHpLost lists Buffer as a modifier when its 0 moved the amount, and AfterModifyingHpLostAfterOsty
        // then reaches BufferPower.AfterModifyingHpLostAfterOsty; old: CardEffects.LosePlayerHp spends a stack (removing at
        // 1) and returns 0. The slug's 7 on the player with Buffer 2: 0 lost, Buffer 1; an unpowered 7: 0 lost, Buffer
        // gone; a third 7: 7 lost. A fully blocked 7 spends nothing.
        var (s, _) = BufferCombat();
        var slug = OldCreature.Of(s.Enemies[0]);
        var k = Kernel(s, Power2);

        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 7m, dealer: slug)).UnblockedDamage);
        Assert.Equal((50, 1), (s.PlayerHp, BufferOf(s)));
        var oracle = BufferCombat().S;
        Assert.Equal(0, CardEffects.LosePlayerHp(oracle, 7));
        Assert.Equal((oracle.PlayerHp, BufferOf(oracle)), (s.PlayerHp, BufferOf(s)));

        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 7m, dealer: slug, props: Unpowered)).UnblockedDamage);
        Assert.Equal((50, 0), (s.PlayerHp, BufferOf(s)));
        Assert.DoesNotContain(s.PlayerBuffs, b => b.Id == BuffId.Buffer);
        Assert.Equal(7, Assert.Single(Hit(k, OldCreature.Player, 7m, dealer: slug)).UnblockedDamage);
        Assert.Equal(43, s.PlayerHp);

        var (s2, _) = BufferCombat();
        s2.PlayerBlock = 9;
        var k2 = Kernel(s2, Power2);
        Assert.Equal(0, Assert.Single(Hit(k2, OldCreature.Player, 7m, dealer: OldCreature.Of(s2.Enemies[0]))).UnblockedDamage);
        Assert.Equal((50, 2, 2), (s2.PlayerHp, BufferOf(s2), s2.PlayerBlock));
    }

    [Fact]
    public void BufferLeavesAnEnemysLossAlone()
    {
        // The per-owner rule where the model has one holder in the old engine: the old function acts on the player's list
        // and the old enemy site (CardEffects.LandDamageOnEnemy) has no Buffer, so a Buffer put on a slug takes nothing off
        // a hit on it and spends nothing, on either side; the player's own loss is still taken by the player's.
        var (s, mine) = BufferCombat();
        var theirs = GivePower(s, "BufferPower", BuffId.Buffer, 2, s.Enemies[0]);
        var k = Kernel(s, Power2);

        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out var changed));
        Assert.Empty(changed);
        Assert.Equal(7, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 7m)).UnblockedDamage);
        Assert.Equal((33, 2, 2), (s.Enemies[0].Hp, BuffSystem.Get(s.Enemies[0].Buffs, BuffId.Buffer), BufferOf(s)));
        Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([theirs]));
        Assert.Equal((2, 2), (BuffSystem.Get(s.Enemies[0].Buffs, BuffId.Buffer), BufferOf(s)));
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1), out changed));
        Assert.Equal([mine], changed);
    }

    [Fact]
    public void BufferDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = BufferCombat();
        var k = Kernel(s, Without(Power2, BufferPower.Register));
        Assert.Equal(7m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 7m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(7, Assert.Single(Hit(k, OldCreature.Player, 7m, dealer: OldCreature.Of(s.Enemies[0]))).UnblockedDamage);
        Assert.Equal((43, 2), (s.PlayerHp, BufferOf(s)));
    }

    [Fact]
    public void BufferIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "BufferPower", HookId.ModifyHpLostAfterOstyLate,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    // ---------------------------------------------------------------------------------------- Possess Speed (power-2)

    private static int Dexterity(CombatState s) => BuffSystem.Get(s.PlayerBuffs, BuffId.Dexterity);
    private static int PossessedSpeed(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.PossessSpeed);

    /// <summary>Two slugs, the first with Possess Speed 3 (the Dexterity it took), the player at Dexterity -3.</summary>
    private static (CombatState S, ListenerRef Power) PossessSpeedCombat(int amount = 3)
    {
        var s = Combat(40, 40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Dexterity, -amount));
        return (s, GivePower(s, "PossessSpeedPower", BuffId.PossessSpeed, amount, s.Enemies[0]));
    }

    [Fact]
    public void PossessSpeedIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "PossessSpeedPower", HookId.AfterDeath);
    }

    [Fact]
    public void PossessSpeedGivesTheDexterityBackOnItsOwnersDeath()
    {
        // game: PossessSpeedPower.AfterDeath; old: CombatEngine.AnswerDeaths calls PossessSpeed_AfterDeath(state, fallen) for
        // a fallen creature whose Possess Speed > 0: the player gains the amount as Dexterity and the power leaves the
        // corpse. Dispatched inside Dying (the owner at 0 HP, marked dying, still listed), as the kill runs the hook.
        var (s, _) = PossessSpeedCombat();
        var dead = s.Enemies[0];
        var k = Kernel(s, Power2);
        var oracle = PossessSpeedCombat().S;

        using (Dying(k, dead))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        CombatEngine.PossessSpeed_AfterDeath(oracle, oracle.Enemies[0]);

        Assert.Equal(Dexterity(oracle), Dexterity(s));
        Assert.Equal(0, Dexterity(s));
        Assert.Equal(0, PossessedSpeed(dead));
        Assert.Equal(oracle.Enemies[0].Buffs, dead.Buffs);

        // Through the kill: CreatureCmd.Kill runs AfterDeath(false) for the dead owner before it strips the corpse.
        var (s2, _) = PossessSpeedCombat();
        var k2 = Kernel(s2, Power2);
        CreatureCmd.Kill(k2, [OldCreature.Of(s2.Enemies[0])]);
        Assert.Equal(0, Dexterity(s2));
        Assert.Equal(0, PossessedSpeed(s2.Enemies[0]));
    }

    [Fact]
    public void PossessSpeedDoesNothingForAnotherCreaturesDeathOrAPreventedOne()
    {
        // The handler's condition: the dead creature is its owner, and the death was not prevented (the old site answers
        // real deaths). The second slug's death, and the owner's prevented death, leave the Dexterity and the power.
        var (s, _) = PossessSpeedCombat();
        var k = Kernel(s, Power2);

        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));
        Assert.Equal((-3, 3), (Dexterity(s), PossessedSpeed(s.Enemies[0])));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1, removalPrevented: true));
        Assert.Equal((-3, 3), (Dexterity(s), PossessedSpeed(s.Enemies[0])));
    }

    [Fact]
    public void PossessSpeedGivesBackEachOwnersOwnShareAmongTwoHolders()
    {
        // The N-holder test: both slugs took 3 (the player at -6). The first's death gives 3 back (not 6) and strips the
        // first alone; the second's death gives the other 3: one direct call of the old body per owner.
        var (s, _) = PossessSpeedCombat();
        GivePower(s, "PossessSpeedPower", BuffId.PossessSpeed, 3, s.Enemies[1]);
        s.PlayerBuffs[0] = new BuffState(BuffId.Dexterity, -6);
        var k = Kernel(s, Power2);

        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal((-3, 0, 3), (Dexterity(s), PossessedSpeed(s.Enemies[0]), PossessedSpeed(s.Enemies[1])));
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));
        Assert.Equal((0, 0, 0), (Dexterity(s), PossessedSpeed(s.Enemies[0]), PossessedSpeed(s.Enemies[1])));
    }

    [Fact]
    public void PossessSpeedDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = PossessSpeedCombat();
        var k = Kernel(s, Without(Power2, PossessSpeedPower.Register));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal((-3, 3), (Dexterity(s), PossessedSpeed(s.Enemies[0])));
    }

    [Fact]
    public void PossessSpeedIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "PossessSpeedPower", HookId.AfterDeath);
    }
}
