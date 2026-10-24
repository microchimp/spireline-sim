// The C6 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C6"): the port list's relic-1 rows as migrated models
// under Core2/Models/Relics, each tested the way the shared rules ask: (1) its pairs are in the handler table and none is
// a stub, (2) the dispatch gives what the old function gives for the same state, (3) the delete-mutant (the model's
// registration left out) gives the no-handler value, (4) with F08 off its pairs are absent, (5) the holder rule: a relic
// has one holder, the player, and no second copy in a combat, so the two-holder setup of rule 5 has no relic case; what it
// asks, that a handler acts for its owner alone, is pinned per model by the dispatches about another creature (Tungsten
// Rod on an enemy's or Osty's loss, Gremlin Horn on a death on the owner's side) and by the two relics of one old function
// each landing once (Strike Dummy with Fake Strike Dummy). The table test at the top pins the batch's (model, hook) set,
// standing in for EmuDump kernel-handlers and the port list's gate, which need the manifest and a built EmuDump. Game
// code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Relics;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class RelicOneTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Relic1 = Registry.Relic1;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list's relic-1 rows the briefs mark CLOUD.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Relic1Pairs =
    [
        (ListenerKind.Relic, "TungstenRod", HookId.AfterModifyingHpLostAfterOsty),
        (ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty),
        (ListenerKind.Relic, "StrikeDummy", HookId.ModifyDamageAdditive),
        (ListenerKind.Relic, "PenNib", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Relic, "GremlinHorn", HookId.AfterDeath),
        (ListenerKind.Relic, "MiniatureCannon", HookId.ModifyDamageAdditive),
        (ListenerKind.Relic, "MysticLighter", HookId.ModifyDamageAdditive),
        (ListenerKind.Relic, "FakeStrikeDummy", HookId.ModifyDamageAdditive),
    ];

    [Fact]
    public void TheRegistrysRelic1BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Relic1);
        Assert.Equal(Relic1Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsTheRelic1BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; this batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Relic1)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Relic1Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Relic1, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- the cards the tests play

    /// <summary>An Attack with the Strike tag (StrikeIronclad).</summary>
    private const int StrikeDef = 472;

    /// <summary>An Attack without it (Bash).</summary>
    private const int BashDef = 30;

    /// <summary>A Skill (DefendIronclad).</summary>
    private const int DefendDef = 131;

    private static CardDef Def(int id) => GeneratedData.Cards.Get(id);

    // ---------------------------------------------------------------------------------------- Tungsten Rod (relic-1)

    /// <summary>The player holding Tungsten Rod, one slug.</summary>
    private static (CombatState S, ListenerRef Relic) TungstenRodCombat(int enemyHp = 40)
    {
        var s = Combat(enemyHp);
        return (s, Hold(s, RelicEffects.TungstenRod));
    }

    [Fact]
    public void TungstenRodIsRegisteredForItsTwoPairs()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    [Fact]
    public void TungstenRodTakesOneOffTheOwnersHpLossInTheAfterOstyPass()
    {
        // game: TungstenRod.ModifyHpLostAfterOsty; old: CardEffects.LosePlayerHp runs RelicEffects.ModifyHpLost, which
        // applies TungstenRod_ModifyHpLostAfterOsty to the player's loss while the relic is held, whatever brought the loss.
        var (s, relic) = TungstenRodCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out var changed);

        Assert.Equal(RelicEffects.TungstenRod_ModifyHpLostAfterOsty(3), v);
        Assert.Equal(2m, v);
        Assert.Equal([relic], changed);
        // The same state through the old fold: 3 becomes 2.
        Assert.Equal(2, RelicEffects.ModifyHpLost(s, 3));

        // Every route: an unpowered loss (a Burn, Poison, Thorns), a loss with no dealer, a card's own cost.
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, props: Unpowered, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: null), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, props: Unpowered, dealer: 0), out changed));
        Assert.Equal([relic], changed);

        // The floor at zero: one lost becomes nothing; nothing lost stays nothing and counts as no change.
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 1m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0m, dealer: 1), out changed));
        Assert.Empty(changed);

        // Decimals: the running value moves by one and keeps its fraction (3.5 to 2.5, truncated to 2 at the loss; 1.5
        // to 0.5, truncated to 0, which the trunc-ne criterion counts as a change from 1); a loss under 1 is floored to 0
        // (the old engine's cast had made it 0 already) and, truncating to 0 either way, counts as no change.
        Assert.Equal(2.5m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3.5m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0.5m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 1.5m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0.5m, dealer: 1), out changed));
        Assert.Empty(changed);

        // Where the condition fails: the before-Osty phase (another pass), a loss of an enemy (the player's hit on the
        // slug), the relic not held.
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, dealer: 0), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TungstenRodActsForItsOwnerAndNotForOstyOrAnEnemy()
    {
        // The per-owner condition (CLOUD_BRIEFS section 3, rule 5, for a relic: one holder): a loss of Osty (creature 1
        // once it is in the combat) or of the slug (creature 2) is another creature's, and the rod leaves it.
        var (s, relic) = TungstenRodCombat();
        WithOsty(s);
        var k = Kernel(s, Relic1);

        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, dealer: 2), out var changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 2, 3m, dealer: 0), out changed));
        Assert.Empty(changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 2), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void TungstenRodsReductionLandsThroughTheDamageCommandAfterBlock()
    {
        // game: CreatureCmd.Damage runs Hook.ModifyHpLost after the block absorption and the pet redirect, then
        // LoseHpInternal with the one cast. The slug hits the player: 3 on no block, 2 lost. Block 2 against 5: 3 unblocked,
        // 2 lost. Block 5 against 3: fully blocked, nothing lost. 1 on no block: nothing lost. 3.5: 2.5 after the rod,
        // 2 lost at the cast. 0.5: nothing. An unpowered 3 (a Burn's): 2 lost.
        var (s, _) = TungstenRodCombat();
        var slug = OldCreature.Of(s.Enemies[0]);
        var k = Kernel(s, Relic1);

        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug)).UnblockedDamage);
        Assert.Equal(48, s.PlayerHp);
        s.PlayerBlock = 2;
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        s.PlayerBlock = 5;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        s.PlayerBlock = 0;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 1m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3.5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(44, s.PlayerHp);
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 0.5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(44, s.PlayerHp);
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug, props: Unpowered)).UnblockedDamage);
        Assert.Equal(42, s.PlayerHp);

        // The old route for the first hit: CardEffects.LosePlayerHp takes 2 off 50 and reports 2.
        var (oracle, _) = TungstenRodCombat();
        Assert.Equal(2, CardEffects.LosePlayerHp(oracle, 3));
        Assert.Equal(48, oracle.PlayerHp);
    }

    [Fact]
    public void TungstenRodsAfterModifyingHpLostAfterOstyDoesNothingToTheState()
    {
        // game: TungstenRod.AfterModifyingHpLostAfterOsty, reached by the filter-set pass when the relic modified the loss.
        // The old engine has no body at this point; the handler returns Done and the state is as it was.
        // NO DELETE-MUTANT for this pair (CLOUD_BRIEFS section 3, rule 3, does not apply): the game's override only flashes
        // the relic (presentation, no combat state), so with the handler deleted the state is the same and no assertion on
        // it could fail. The pair is pinned instead by TungstenRodIsRegisteredForItsTwoPairs and the table test. A mutant
        // is not faked here.
        var (s, relic) = TungstenRodCombat();
        var k = Kernel(s, Relic1);
        var status = Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([relic]));
        Assert.False(status.IsPaused);
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal(40, s.Enemies[0].Hp);
        Assert.Equal([new RelicInstance(RelicEffects.TungstenRod)], s.Relics);
    }

    [Fact]
    public void TungstenRodDeleteMutantGivesTheNoHandlerValue()
    {
        // The fold pair (ModifyHpLostAfterOsty) only: the after-modifying pair has no mutant, see above.
        var (s, _) = TungstenRodCombat();
        var k = Kernel(s, Without(Relic1, TungstenRod.Register));
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(3, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: OldCreature.Of(s.Enemies[0]))).UnblockedDamage);
        Assert.Equal(47, s.PlayerHp);
    }

    [Fact]
    public void TungstenRodIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    // ---------------------------------------------------------------------------------------- Strike Dummy (relic-1)

    /// <summary>The player holding Strike Dummy with a Strike, a Bash and a Defend in hand, one slug.</summary>
    private static (CombatState S, ListenerRef Relic, CardHandle Strike, CardHandle Bash, CardHandle Defend) StrikeDummyCombat()
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.StrikeDummy);
        var strike = Add(s, s.Hand, StrikeDef);
        var bash = Add(s, s.Hand, BashDef);
        var defend = Add(s, s.Hand, DefendDef);
        return (s, relic, new CardHandle(strike.Uid), new CardHandle(bash.Uid), new CardHandle(defend.Uid));
    }

    [Fact]
    public void StrikeDummyIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "StrikeDummy", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void StrikeDummyAddsTheOldFunctionsNumberToAStrikeTaggedAttack()
    {
        // game: StrikeDummy.ModifyDamageAdditive; old: RelicEffects.CardDamageBonus adds StrikeDummy_ModifyDamageAdditive()
        // for an Attack def with StrikeTag while the relic is held.
        var (s, relic, strike, bash, defend) = StrikeDummyCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed);

        Assert.Equal(6m + RelicEffects.CardDamageBonus(s, Def(StrikeDef), upgraded: false), v);
        Assert.Equal(6m + RelicEffects.StrikeDummy_ModifyDamageAdditive(), v);
        Assert.Equal(9m, v);
        Assert.Equal([relic], changed);

        // Where the condition fails: an Attack without the tag, a Skill, no card source, the relic not held.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: bash), out changed));
        Assert.Empty(changed);
        Assert.Equal(0, RelicEffects.CardDamageBonus(s, Def(BashDef), upgraded: false));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: defend), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void StrikeDummyAddsNothingToAnUnpoweredHitWithACardSource()
    {
        // game: StrikeDummy.ModifyDamageAdditive answers 0 off a powered attack (ValueProp.IsPoweredAttack). The old site read
        // no props, so an unpowered hit with a card source (a Strike's self-damage, thorns off a card) got the bonus; the
        // handler follows the game, and the no-handler value is the damage as it was.
        var (s, _, strike, _, _) = StrikeDummyCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: strike), out var changed));
        Assert.Empty(changed);
        // The same card as a powered attack: the bonus lands.
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
    }

    [Fact]
    public void StrikeDummyRefusesOnlyWhenBothTheDealerAndTheCardsOwnerAreNotItsOwner()
    {
        // game: StrikeDummy.ModifyDamageAdditive answers 0 when the dealer is not the owner's creature AND the card is not
        // the owner's. The old state holds one card owner, the player (HookCtx.CardOwner), so a hit by an enemy or by no
        // dealer with the player's Strike still gets the bonus (the card test is met), as the game's test gives; the
        // refusal of both failing is not reachable until the kernel has several card owners.
        var (s, relic, strike, _, _) = StrikeDummyCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: strike), out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null, card: strike), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void StrikeDummyDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _, strike, _, _) = StrikeDummyCombat();
        var k = Kernel(s, Without(Relic1, StrikeDummy.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void StrikeDummyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "StrikeDummy", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Pen Nib (relic-1)

    /// <summary>The player holding Pen Nib at <paramref name="counter" />, the combat's arming flag as said, a Strike and a
    /// Defend in hand, one slug. The default is the armed state: the counter wrapped to 0 on the tenth Attack.</summary>
    private static (CombatState S, ListenerRef Relic, CardHandle Strike, CardHandle Defend) PenNibCombat(int counter = 0, bool armed = true)
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.PenNib, counter);
        s.PenNibArmed = armed;
        var strike = Add(s, s.Hand, StrikeDef);
        var defend = Add(s, s.Hand, DefendDef);
        return (s, relic, new CardHandle(strike.Uid), new CardHandle(defend.Uid));
    }

    [Fact]
    public void PenNibIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "PenNib", HookId.ModifyDamageMultiplicative);
    }

    [Fact]
    public void PenNibDoublesAnAttackWhileArmed()
    {
        // game: PenNib.ModifyDamageMultiplicative; old: RelicEffects.CardDamageMultiplier returns
        // PenNib_ModifyDamageMultiplicative() for an Attack def while the relic's counter is 0 and PenNibArmed is up.
        var (s, relic, strike, defend) = PenNibCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed);

        Assert.Equal(6m * RelicEffects.CardDamageMultiplier(s, Def(StrikeDef)), v);
        Assert.Equal(6m * RelicEffects.PenNib_ModifyDamageMultiplicative(), v);
        Assert.Equal(12m, v);
        Assert.Equal([relic], changed);

        // Where the condition fails: the flag down, the counter off 0 with the flag up, a Skill, no card source, the relic
        // not held.
        s.PenNibArmed = false;
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
        s.PenNibArmed = true;
        s.Relics[0] = s.Relics[0] with { Counter = 1 };
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
        s.Relics[0] = s.Relics[0] with { Counter = 0 };
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: defend), out changed));
        Assert.Empty(changed);
        Assert.Equal(1, RelicEffects.CardDamageMultiplier(s, Def(DefendDef)));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PenNibFollowsTheOldCountersArmingCycle()
    {
        // old: RelicEffects.BeforeCardPlayedRelics counts an Attack on the relic's counter, wraps it at ten and arms the
        // flag on the wrap; a card that is no Attack lowers the flag. The handler reads the counter and the flag, so the
        // tenth Attack is doubled, the eleventh is not, and a Skill played in between disarms.
        var (s, relic, strike, _) = PenNibCombat(counter: 9, armed: false);
        var k = Kernel(s, Relic1);

        RelicEffects.BeforeCardPlayedRelics(s, Def(StrikeDef));
        Assert.Equal(0, s.Relics[0].Counter);
        Assert.True(s.PenNibArmed);
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed));
        Assert.Equal([relic], changed);

        RelicEffects.BeforeCardPlayedRelics(s, Def(StrikeDef));
        Assert.Equal(1, s.Relics[0].Counter);
        Assert.False(s.PenNibArmed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);

        s.Relics[0] = s.Relics[0] with { Counter = 0 };
        s.PenNibArmed = true;
        RelicEffects.BeforeCardPlayedRelics(s, Def(DefendDef));
        Assert.Equal(0, s.Relics[0].Counter);
        Assert.False(s.PenNibArmed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PenNibDoublesNothingOnAnUnpoweredHitAndNothingForAnotherDealer()
    {
        // game: PenNib.ModifyDamageMultiplicative answers 1 off a powered attack (ValueProp.IsPoweredAttack) and for a
        // dealer that is neither the owner's creature nor its pet (HookCtx.IsOwnerOrPet). The old site read neither, so
        // an unpowered hit with a card source, or an enemy's hit carrying the player's card, was doubled; the handler
        // follows the game, and the no-handler value is the damage as it was.
        var (s, relic, strike, _) = PenNibCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: strike), out var changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: strike), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null, card: strike), out changed));
        Assert.Empty(changed);
        // Controls: the owner's powered attack is doubled.
        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void PenNibDoublesTheOwnersPetsHitAndNotAnEnemys()
    {
        // game: PenNib.ModifyDamageMultiplicative accepts a dealer that is the owner's creature or its Osty. Osty is
        // creature 1 and the slug creature 2 here.
        var s = WithOsty(Combat(40));
        var relic = Hold(s, RelicEffects.PenNib);
        s.PenNibArmed = true;
        var strike = new CardHandle(Add(s, s.Hand, StrikeDef).Uid);
        var k = Kernel(s, Relic1);

        Assert.Equal(12m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: 1, card: strike), out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2, card: strike), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PenNibDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _, strike, _) = PenNibCombat();
        var k = Kernel(s, Without(Relic1, PenNib.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void PenNibIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "PenNib", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- Gremlin Horn (relic-1)

    /// <summary>The player holding Gremlin Horn with 3 energy, a Strike, a Bash and a Defend in the draw pile (in that
    /// order) and nothing in hand, one slug per HP given.</summary>
    private static (CombatState S, ListenerRef Relic) GremlinHornCombat(params int[] enemyHp)
    {
        var s = Combat(enemyHp);
        var relic = Hold(s, RelicEffects.GremlinHorn);
        s.Energy = 3;
        Add(s, s.DrawPile, StrikeDef);
        Add(s, s.DrawPile, BashDef);
        Add(s, s.DrawPile, DefendDef);
        return (s, relic);
    }

    /// <summary>A pile's cards by (uid, def) in order.</summary>
    private static string Layout(List<CardInstance> pile) => string.Join(",", pile.Select(c => $"{c.Uid}:{c.DefId}"));

    [Fact]
    public void GremlinHornIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "GremlinHorn", HookId.AfterDeath);
    }

    [Fact]
    public void GremlinHornPaysTheOldBodysEnergyAndCardForAnEnemysDeath()
    {
        // game: GremlinHorn.AfterDeath for a death on the other side; old: CombatEngine.AnswerDeaths calls
        // GremlinHorn_AfterDeath(state, rng) once per fallen enemy while the relic is held. Dispatched with the slug dead and
        // still listed (the state the game's override sees), the state equals the old body's on the same state.
        var (s, _) = GremlinHornCombat(40, 40);
        var k = Kernel(s, Relic1);
        var (oracle, _) = GremlinHornCombat(40, 40);
        var slug = s.Enemies[0];

        using (Dying(k, slug))
            Hooks.AfterDeath(k, AfterDeathOf(IndexOf(s, slug)));
        oracle.Enemies[0].Hp = 0;
        CombatEngine.GremlinHorn_AfterDeath(oracle, new Random(0));

        Assert.Equal(oracle.Energy, s.Energy);
        Assert.Equal(4, s.Energy);
        Assert.Equal(Layout(oracle.Hand), Layout(s.Hand));
        Assert.Equal([StrikeDef], s.Hand.Select(c => c.DefId));
        Assert.Equal(Layout(oracle.DrawPile), Layout(s.DrawPile));
        Assert.Equal([BashDef, DefendDef], s.DrawPile.Select(c => c.DefId));

        // A death whose removal was prevented pays too: the old site, as the game, reads the side alone.
        using (Dying(k, slug))
            Hooks.AfterDeath(k, AfterDeathOf(IndexOf(s, slug), removalPrevented: true));
        Assert.Equal(5, s.Energy);
        Assert.Equal([StrikeDef, BashDef], s.Hand.Select(c => c.DefId));
    }

    [Fact]
    public void GremlinHornPaysNothingForADeathOnItsOwnersSide()
    {
        // The per-owner condition: the dead creature must be on the other side from the holder (the player). The player's
        // own death (creature 0) and Osty's (creature 1 once it is in the combat) pay nothing; the slug's (creature 2) does.
        var (s, _) = GremlinHornCombat(40, 40);
        WithOsty(s);
        var k = Kernel(s, Relic1);

        Hooks.AfterDeath(k, AfterDeathOf(0));
        Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(3, s.Energy);
        Assert.Empty(s.Hand);

        var slug = s.Enemies[0];
        using (Dying(k, slug))
            Hooks.AfterDeath(k, AfterDeathOf(IndexOf(s, slug)));
        Assert.Equal(4, s.Energy);
        Assert.Equal([StrikeDef], s.Hand.Select(c => c.DefId));
    }

    [Fact]
    public void GremlinHornOnTheLastEnemysDeathPaysTheEnergyAndTheOldBodysDrawRefuses()
    {
        // old: CardEffects.DrawCards draws nothing once the fight is over or ending (CombatEngine.IsOverOrEnding), and the
        // last slug dead makes it so; the energy still comes. The handler equals the old body on the same state.
        var (s, _) = GremlinHornCombat(40);
        var k = Kernel(s, Relic1);
        var (oracle, _) = GremlinHornCombat(40);

        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        oracle.Enemies[0].Hp = 0;
        CombatEngine.GremlinHorn_AfterDeath(oracle, new Random(0));

        Assert.Equal(oracle.Energy, s.Energy);
        Assert.Equal(4, s.Energy);
        Assert.Equal(Layout(oracle.Hand), Layout(s.Hand));
        Assert.Empty(s.Hand);
        Assert.Equal(3, s.DrawPile.Count);
    }

    [Fact]
    public void GremlinHornsDrawReshufflesOnTheCombatsShuffleStream()
    {
        // The old body takes a Random for the reshuffle and CardEffects.ShuffleIfNecessary prefers the combat's Shuffle
        // stream to it; the handler passes that stream. With the draw pile empty, the discard is shuffled into it on the
        // stream, and the result equals the old body's on a combat with the same stream (the same seed, the same count of
        // draws after).
        var (s, _) = GremlinHornCombat(40, 40);
        var (oracle, _) = GremlinHornCombat(40, 40);
        foreach (var c in new[] { s, oracle })
        {
            c.DiscardPile.AddRange(c.DrawPile);
            c.DrawPile.Clear();
            c.ShuffleRng = new CountingRandom(7);
        }
        var k = Kernel(s, Relic1);

        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        oracle.Enemies[0].Hp = 0;
        CombatEngine.GremlinHorn_AfterDeath(oracle, new Random(0));

        Assert.Equal(oracle.Energy, s.Energy);
        Assert.Equal(Layout(oracle.Hand), Layout(s.Hand));
        Assert.Single(s.Hand);
        Assert.Equal(Layout(oracle.DrawPile), Layout(s.DrawPile));
        Assert.Equal(2, s.DrawPile.Count);
        Assert.Empty(s.DiscardPile);
        Assert.Equal(oracle.ShuffleRng!.CallCount, s.ShuffleRng!.CallCount);
        Assert.True(s.ShuffleRng.CallCount > 0);
    }

    [Fact]
    public void GremlinHornDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = GremlinHornCombat(40, 40);
        var k = Kernel(s, Without(Relic1, GremlinHorn.Register));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(3, s.Energy);
        Assert.Empty(s.Hand);
        Assert.Equal(3, s.DrawPile.Count);
    }

    [Fact]
    public void GremlinHornIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "GremlinHorn", HookId.AfterDeath);
    }

    // ---------------------------------------------------------------------------------------- Miniature Cannon (relic-1)

    /// <summary>The player holding Miniature Cannon with an upgraded Strike, a plain Strike, an upgraded Bash and an
    /// upgraded Defend in hand, one slug.</summary>
    private static (CombatState S, ListenerRef Relic, CardHandle StrikePlus, CardHandle Strike, CardHandle BashPlus, CardHandle DefendPlus)
        MiniatureCannonCombat()
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.MiniatureCannon);
        var strikePlus = Add(s, s.Hand, StrikeDef, upgraded: true);
        var strike = Add(s, s.Hand, StrikeDef);
        var bashPlus = Add(s, s.Hand, BashDef, upgraded: true);
        var defendPlus = Add(s, s.Hand, DefendDef, upgraded: true);
        return (s, relic, new CardHandle(strikePlus.Uid), new CardHandle(strike.Uid), new CardHandle(bashPlus.Uid),
            new CardHandle(defendPlus.Uid));
    }

    [Fact]
    public void MiniatureCannonIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "MiniatureCannon", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void MiniatureCannonAddsTheOldFunctionsNumberToAnUpgradedAttack()
    {
        // game: MiniatureCannon.ModifyDamageAdditive; old: RelicEffects.CardDamageBonus adds
        // MiniatureCannon_ModifyDamageAdditive() for an Attack def when the copy is upgraded and the relic is held.
        var (s, relic, strikePlus, strike, bashPlus, defendPlus) = MiniatureCannonCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strikePlus), out var changed);

        Assert.Equal(6m + RelicEffects.CardDamageBonus(s, Def(StrikeDef), upgraded: true), v);
        Assert.Equal(6m + RelicEffects.MiniatureCannon_ModifyDamageAdditive(), v);
        Assert.Equal(9m, v);
        Assert.Equal([relic], changed);
        // Any upgraded Attack: the tag is not read.
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: bashPlus), out changed));
        Assert.Equal([relic], changed);
        // The copy in play (in flight, off the hand: the card's view is Play, read from the world): the same answer.
        var flying = s.Hand.Single(x => x.Uid == strikePlus.Uid);
        s.Hand.Remove(flying);
        using (k.World!.Flying(flying))
        {
            Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strikePlus), out changed));
            Assert.Equal([relic], changed);
        }
        s.Hand.Insert(0, flying);

        // Where the condition fails: a plain copy, an upgraded Skill, no card source, the relic not held.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
        Assert.Equal(0, RelicEffects.CardDamageBonus(s, Def(StrikeDef), upgraded: false));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: defendPlus), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strikePlus), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void MiniatureCannonAddsNothingToAnUnpoweredHitWithACardSource()
    {
        // game: MiniatureCannon.ModifyDamageAdditive answers 0 off a powered attack (ValueProp.IsPoweredAttack); the old site
        // read no props, so an upgraded Attack's unpowered hit got the bonus. The no-handler value is the damage as it was.
        var (s, _, strikePlus, _, _, _) = MiniatureCannonCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: strikePlus), out var changed));
        Assert.Empty(changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strikePlus), out changed));
    }

    [Fact]
    public void MiniatureCannonRefusesOnlyWhenBothTheDealerAndTheCardsOwnerAreNotItsOwner()
    {
        // game: MiniatureCannon.ModifyDamageAdditive answers 0 when the dealer is not the owner's creature AND the card is
        // not the owner's. The old state holds one card owner, the player (HookCtx.CardOwner), so a hit by an enemy or by no
        // dealer with the player's upgraded card still gets the bonus; the refusal of both failing is not reachable until
        // the kernel has several card owners.
        var (s, relic, strikePlus, _, _, _) = MiniatureCannonCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: strikePlus), out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(9m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null, card: strikePlus), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void MiniatureCannonDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _, strikePlus, _, _, _) = MiniatureCannonCombat();
        var k = Kernel(s, Without(Relic1, MiniatureCannon.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strikePlus), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void MiniatureCannonIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "MiniatureCannon", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Mystic Lighter (relic-1)

    /// <summary>The copy of <paramref name="card" /> in hand enchanted, the same uid (the old state keeps the enchantment
    /// on the CardInstance).</summary>
    private static CardInstance Enchant(CombatState s, CardInstance card, Enchantment enchantment, int amount)
    {
        int i = s.Hand.FindIndex(c => c.Uid == card.Uid);
        s.Hand[i] = s.Hand[i] with { Enchantment = enchantment, EnchantAmount = amount };
        return s.Hand[i];
    }

    /// <summary>The player holding Mystic Lighter with a Sharp 2 Bash, a plain Bash and a Nimble 2 Defend in hand, one
    /// slug.</summary>
    private static (CombatState S, ListenerRef Relic, CardInstance Sharp, CardHandle Plain, CardHandle NimbleDefend) MysticLighterCombat()
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.MysticLighter);
        var sharp = Enchant(s, Add(s, s.Hand, BashDef), Enchantment.Sharp, 2);
        var plain = Add(s, s.Hand, BashDef);
        var nimble = Enchant(s, Add(s, s.Hand, DefendDef), Enchantment.Nimble, 2);
        return (s, relic, sharp, new CardHandle(plain.Uid), new CardHandle(nimble.Uid));
    }

    [Fact]
    public void MysticLighterIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "MysticLighter", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void MysticLighterAddsTheOldFunctionsNumberOnceBesideTheEnchantmentsOwnBonus()
    {
        // game: MysticLighter.ModifyDamageAdditive; old: CardEffects.DmgFrom adds the card's own enchantment bonus
        // (EnchantDamageAdditive) and, apart from it, RelicEffects.EnchantedCardDamageBonus for an enchanted Attack. In the
        // kernel the enchantment's own bonus is the dispatcher's enchant-add step (Steps.EnchantAdd), run before the
        // additive pass, and the relic's is this handler: a Sharp 2 Bash at 8 is 8 + 2 + 9, the 9 once.
        var (s, relic, sharp, plain, nimbleDefend) = MysticLighterCombat();
        var k = Kernel(s, Relic1);
        var args = Attack(target: 1, damage: 8m, card: new CardHandle(sharp.Uid));

        decimal v = Hooks.ModifyDamage(k, args, out var changed);

        Assert.Equal(8m + CardEffects.EnchantDamageAdditive(sharp) + RelicEffects.EnchantedCardDamageBonus(s), v);
        Assert.Equal(19m, v);
        Assert.Equal([relic], changed);
        // The two parts, apart: the step adds the enchantment's 2 and nothing of the relic's; the old function is the 9.
        Assert.Equal(10m, Steps.EnchantAdd(k, HookId.Dispatch_ModifyDamage, in args, 8m));
        Assert.Equal(2, CardEffects.EnchantDamageAdditive(sharp));
        Assert.Equal(9, RelicEffects.EnchantedCardDamageBonus(s));
        // The copy in play (in flight, off the hand: the card's view is Play, read from the world): the same answer.
        s.Hand.Remove(sharp);
        using (k.World!.Flying(sharp))
        {
            Assert.Equal(19m, Hooks.ModifyDamage(k, args, out changed));
            Assert.Equal([relic], changed);
        }
        s.Hand.Insert(0, sharp);

        // Where the condition fails: a card without an enchantment, an enchanted Skill, no card source, the relic not held
        // (the enchantment's own 2 stays, the relic's 9 goes).
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m, card: plain), out changed));
        Assert.Empty(changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m, card: nimbleDefend), out changed));
        Assert.Empty(changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(10m, Hooks.ModifyDamage(k, args, out changed));
        Assert.Empty(changed);
        Assert.Equal(0, RelicEffects.EnchantedCardDamageBonus(s));
    }

    [Fact]
    public void MysticLighterPaysForAnyEnchantmentOnAnAttack()
    {
        // Any enchantment, not only a damage one: a Swift Strike (an enchantment with no damage bonus of its own) gets
        // the 9 and nothing from the step.
        var (s, relic, _, _, _) = MysticLighterCombat();
        var swift = Enchant(s, Add(s, s.Hand, StrikeDef), Enchantment.Swift, 1);
        var k = Kernel(s, Relic1);
        var args = Attack(target: 1, damage: 6m, card: new CardHandle(swift.Uid));

        Assert.Equal(15m, Hooks.ModifyDamage(k, args, out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(6m, Steps.EnchantAdd(k, HookId.Dispatch_ModifyDamage, in args, 6m));
    }

    [Fact]
    public void MysticLighterAddsNothingToAnUnpoweredHitWithACardSource()
    {
        // game: MysticLighter.ModifyDamageAdditive answers 0 off a powered attack (ValueProp.IsPoweredAttack); the old site
        // read no props, so an enchanted Attack's unpowered hit got the 9. The enchantment's own step answers 0 off a
        // powered attack too, so the no-handler value is the damage as it was.
        var (s, _, sharp, _, _) = MysticLighterCombat();
        var k = Kernel(s, Relic1);
        var args = Attack(target: 1, damage: 8m, props: Unpowered, card: new CardHandle(sharp.Uid));

        Assert.Equal(8m, Hooks.ModifyDamage(k, args, out var changed));
        Assert.Empty(changed);
        Assert.Equal(8m, Steps.EnchantAdd(k, HookId.Dispatch_ModifyDamage, in args, 8m));
    }

    [Fact]
    public void MysticLighterReadsTheCardsOwnerAndNotTheDealer()
    {
        // game: MysticLighter.ModifyDamageAdditive answers 0 for a card of another owner and reads no dealer. The old state
        // holds one card owner, the player (HookCtx.CardOwner), so a hit by an enemy or by no dealer with the player's
        // enchanted card still gets the 9; the refusal for another card owner is not reachable until the kernel has several.
        var (s, relic, sharp, _, _) = MysticLighterCombat();
        var k = Kernel(s, Relic1);
        var card = new CardHandle(sharp.Uid);

        Assert.Equal(19m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 8m, dealer: 1, card: card), out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(19m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m, dealer: null, card: card), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void MysticLighterDeleteMutantGivesTheNoHandlerValue()
    {
        // The enchantment's own bonus is the dispatcher's step, not the model's: it stays; the relic's 9 goes.
        var (s, _, sharp, _, _) = MysticLighterCombat();
        var k = Kernel(s, Without(Relic1, MysticLighter.Register));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 8m, card: new CardHandle(sharp.Uid)), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void MysticLighterIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "MysticLighter", HookId.ModifyDamageAdditive);
    }

    // ---------------------------------------------------------------------------------------- Fake Strike Dummy (relic-1)

    /// <summary>The player holding Fake Strike Dummy with a Strike, a Bash and a Defend in hand, one slug.</summary>
    private static (CombatState S, ListenerRef Relic, CardHandle Strike, CardHandle Bash, CardHandle Defend) FakeStrikeDummyCombat()
    {
        var s = Combat(40);
        var relic = Hold(s, RelicEffects.FakeStrikeDummy);
        var strike = Add(s, s.Hand, StrikeDef);
        var bash = Add(s, s.Hand, BashDef);
        var defend = Add(s, s.Hand, DefendDef);
        return (s, relic, new CardHandle(strike.Uid), new CardHandle(bash.Uid), new CardHandle(defend.Uid));
    }

    [Fact]
    public void FakeStrikeDummyIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "FakeStrikeDummy", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void FakeStrikeDummyAddsTheOldFunctionsNumberToAStrikeTaggedAttack()
    {
        // game: FakeStrikeDummy.ModifyDamageAdditive; old: RelicEffects.CardDamageBonus adds
        // FakeStrikeDummy_ModifyDamageAdditive() for an Attack def with StrikeTag while the relic is held.
        var (s, relic, strike, bash, defend) = FakeStrikeDummyCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed);

        Assert.Equal(6m + RelicEffects.CardDamageBonus(s, Def(StrikeDef), upgraded: false), v);
        Assert.Equal(6m + RelicEffects.FakeStrikeDummy_ModifyDamageAdditive(), v);
        Assert.Equal(7m, v);
        Assert.Equal([relic], changed);

        // Where the condition fails: an Attack without the tag, a Skill, no card source, the relic not held.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: bash), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: defend), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void BothDummiesHeldEachAddTheirOwnNumberOnce()
    {
        // The two relics of one old function (RelicEffects.CardDamageBonus adds both when both are held): two listeners,
        // each landing its own number once, 6 + 3 + 1, as the old function gives for the same state.
        var (s, dummy, strike, _, _) = StrikeDummyCombat();
        var fake = Hold(s, RelicEffects.FakeStrikeDummy);
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed);

        Assert.Equal(6m + RelicEffects.CardDamageBonus(s, Def(StrikeDef), upgraded: false), v);
        Assert.Equal(10m, v);
        Assert.Equal([dummy, fake], changed.OrderBy(l => l.Instance).ToList());
    }

    [Fact]
    public void FakeStrikeDummyAddsNothingToAnUnpoweredHitWithACardSource()
    {
        // game: FakeStrikeDummy.ModifyDamageAdditive answers 0 off a powered attack (ValueProp.IsPoweredAttack); the old site
        // read no props. The no-handler value is the damage as it was.
        var (s, _, strike, _, _) = FakeStrikeDummyCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered, card: strike), out var changed));
        Assert.Empty(changed);
        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out changed));
    }

    [Fact]
    public void FakeStrikeDummyRefusesOnlyWhenBothTheDealerAndTheCardsOwnerAreNotItsOwner()
    {
        // game: FakeStrikeDummy.ModifyDamageAdditive answers 0 when the dealer is not the owner's creature AND the card is
        // not the owner's; with one card owner in the old state (HookCtx.CardOwner) a hit by an enemy or by no dealer with
        // the player's Strike still gets the bonus, as for Strike Dummy.
        var (s, relic, strike, _, _) = FakeStrikeDummyCombat();
        var k = Kernel(s, Relic1);

        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1, card: strike), out var changed));
        Assert.Equal([relic], changed);
        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null, card: strike), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void FakeStrikeDummyDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _, strike, _, _) = FakeStrikeDummyCombat();
        var k = Kernel(s, Without(Relic1, FakeStrikeDummy.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, card: strike), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void FakeStrikeDummyIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "FakeStrikeDummy", HookId.ModifyDamageAdditive);
    }
}
