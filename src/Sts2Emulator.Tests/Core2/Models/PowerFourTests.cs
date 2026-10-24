// The C5 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C5"): the port list's power-4 rows that are CLOUD
// (Conqueror is ported locally) as migrated models under Core2/Models/Powers, each tested the way the shared rules ask:
// (1) its pairs are in the handler table and none is a stub, (2) the dispatch gives what the old function gives for the
// same state, (3) the delete-mutant (the model's registration left out) gives the no-handler value, (4) with F08 off its
// pairs are absent, (5) with two holders a dispatch about one creature changes that holder's listener alone. The death
// hooks are dispatched inside the kit's Dying scope (the fallen at 0 HP, still listed, as the game's overrides see it) and
// through CreatureCmd.Kill. The table test pins
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
        (ListenerKind.Power, "SicEmPower", HookId.AfterDamageGiven),
        (ListenerKind.Power, "ConstrictPower", HookId.AfterDeath),
        (ListenerKind.Power, "HeistPower", HookId.BeforeDeath),
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

    // ---------------------------------------------------------------------------------------- Sic 'Em (power-4)

    /// <summary>The player with Osty (10 of 10, creature 1) and a card in hand (the Osty attack's source), one slug
    /// (creature 2) under Sic 'Em 3.</summary>
    private static (CombatState S, EnemyState Slug, CardHandle Card) SicEmCombat(int amount = 3, int slugHp = 40)
    {
        var s = WithOsty(Combat(slugHp));
        GivePower(s, "SicEmPower", BuffId.SicEm, amount, s.Enemies[0]);
        var card = Add(s, s.Hand, 1);
        return (s, s.Enemies[0], new CardHandle(card.Uid));
    }

    private static (int Hp, int MaxHp) Osty(CombatState s) => (s.OstyHp, s.OstyMaxHp);

    /// <summary>Osty's hit on <paramref name="target" /> from the player's card <paramref name="card" />.</summary>
    private static IReadOnlyList<DamageResult> OstyHit(KernelState k, EnemyState target, CardHandle card, decimal amount = 5m) =>
        Hit(k, OldCreature.Of(target), amount, dealer: OldCreature.Osty, card: card);

    [Fact]
    public void SicEmIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power4), ListenerKind.Power, "SicEmPower", HookId.AfterDamageGiven);
    }

    [Fact]
    public void SicEmSummonsOstyForItsAmountWhenOstyHitsTheHolder()
    {
        // game: SicEmPower.AfterDamageGiven (the dealer is Osty, the target the holder); old: CardEffects.DealOstyDamage
        // calls SicEm_AfterDamageGiven(state, sicEm) after Osty's hit when the target's SicEm is above 0; a living Osty
        // grows by it (CardEffects.SummonOsty).
        var (s, slug, card) = SicEmCombat();
        var k = Kernel(s, Power4);

        var r = Assert.Single(OstyHit(k, slug, card));

        Assert.Equal(5, r.UnblockedDamage);
        Assert.Equal(35, slug.Hp);
        var (oracle, _, _) = SicEmCombat();
        CardEffects.SicEm_AfterDamageGiven(oracle, 3);
        Assert.Equal(Osty(oracle), Osty(s));
        Assert.Equal((13, 13), Osty(s));
        // A fully blocked hit summons too: the old site asks nothing of the result, and the game's hook runs for every
        // result.
        slug.Block = 9;
        Assert.True(Assert.Single(OstyHit(k, slug, card)).WasFullyBlocked);
        Assert.Equal((16, 16), Osty(s));
        slug.Block = 0;

        // Where the condition fails: the player's hit on the holder from the same card, the holder's hit on the player, a
        // hit with no card source (the old site is reached from card plays alone), the amount at 0, a dead Osty (the
        // command deals nothing for a dead dealer).
        Hit(k, OldCreature.Of(slug), 5m, card: card);
        Assert.Equal((16, 16), Osty(s));
        Hit(k, OldCreature.Player, 5m, dealer: OldCreature.Of(slug));
        Assert.Equal((16, 16), Osty(s));
        Hit(k, OldCreature.Of(slug), 5m, dealer: OldCreature.Osty);
        Assert.Equal((16, 16), Osty(s));
        slug.Buffs[0] = new BuffState(BuffId.SicEm, 0);
        OstyHit(k, slug, card);
        Assert.Equal((16, 16), Osty(s));
        slug.Buffs[0] = new BuffState(BuffId.SicEm, 3);
        s.OstyHp = 0;
        Assert.Equal(0, Assert.Single(OstyHit(k, slug, card)).UnblockedDamage);
        Assert.Equal((0, 16), Osty(s));
    }

    [Fact]
    public void SicEmStillSummonsOnTheHitThatKillsTheHolder()
    {
        // The dying holder is listed until the kill (CreatureCmd.Damage's dying mark), so AfterDamageGiven visits its
        // listener, as the old site reads the corpse's SicEm before the death answer.
        var (s, slug, card) = SicEmCombat(slugHp: 3);
        var k = Kernel(s, Power4);
        var r = Assert.Single(OstyHit(k, slug, card));
        Assert.True(r.WasTargetKilled);
        Assert.Equal((13, 13), Osty(s));
        Assert.DoesNotContain(slug, s.Enemies);
    }

    [Fact]
    public void SicEmActsForTheHolderHitOnlyAmongTwoHolders()
    {
        // The N-holder test: two slugs under Sic 'Em 3 and 5; Osty's hit on the first grows Osty by 3 alone (one call of
        // the old body for that holder, none for the other), its hit on the second by 5.
        var s = WithOsty(Combat(40, 40));
        GivePower(s, "SicEmPower", BuffId.SicEm, 3, s.Enemies[0]);
        GivePower(s, "SicEmPower", BuffId.SicEm, 5, s.Enemies[1]);
        var card = new CardHandle(Add(s, s.Hand, 1).Uid);
        var k = Kernel(s, Power4);
        OstyHit(k, s.Enemies[0], card);
        Assert.Equal((13, 13), Osty(s));
        OstyHit(k, s.Enemies[1], card);
        Assert.Equal((18, 18), Osty(s));
    }

    [Fact]
    public void SicEmDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, slug, card) = SicEmCombat();
        var k = Kernel(s, Without(Power4, SicEmPower.Register));
        Assert.Equal(5, Assert.Single(OstyHit(k, slug, card)).UnblockedDamage);
        Assert.Equal((10, 10), Osty(s));
    }

    [Fact]
    public void SicEmIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power4, AllButDamage), ListenerKind.Power, "SicEmPower", HookId.AfterDamageGiven);
    }

    // ---------------------------------------------------------------------------------------- Constrict (power-4)

    /// <summary>The player under Constrict 3, a Slithering Strangler (creature 1) and a slug (creature 2).</summary>
    private static (CombatState S, EnemyState Strangler) ConstrictCombat(int amount = 3)
    {
        var s = Combat(30, 40);
        s.Enemies[0].DefId = KE.SlitheringStrangler;
        GivePower(s, "ConstrictPower", BuffId.Constrict, amount);
        return (s, s.Enemies[0]);
    }

    private static int Constrict(CombatState s) => BuffSystem.Get(s.PlayerBuffs, BuffId.Constrict);

    [Fact]
    public void ConstrictIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power4), ListenerKind.Power, "ConstrictPower", HookId.AfterDeath);
    }

    [Fact]
    public void ConstrictLeavesThePlayerWhenAStranglerDiesAsTheOldFunctionDoes()
    {
        // game: ConstrictPower.AfterDeath; old: CombatEngine.AnswerDeaths calls Constrict_AfterDeath(state) for a fallen
        // whose def is the Slithering Strangler. Dispatched inside Dying: the fallen at 0 HP, still listed.
        var (s, strangler) = ConstrictCombat();
        var k = Kernel(s, Power4);
        using (Dying(k, strangler))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var (oracle, _) = ConstrictCombat();
        CombatEngine.Constrict_AfterDeath(oracle);
        Assert.Equal(Constrict(oracle), Constrict(s));
        Assert.Equal(0, Constrict(s));
        Assert.Empty(s.PlayerBuffs);

        // Where the condition fails: the slug's death, a prevented death of the Strangler.
        var (s2, strangler2) = ConstrictCombat();
        var k2 = Kernel(s2, Power4);
        using (Dying(k2, s2.Enemies[1]))
            Hooks.AfterDeath(k2, AfterDeathOf(2));
        Assert.Equal(3, Constrict(s2));
        using (Dying(k2, strangler2))
            Hooks.AfterDeath(k2, AfterDeathOf(1, removalPrevented: true));
        Assert.Equal(3, Constrict(s2));
    }

    [Fact]
    public void ConstrictLeavesThePlayerThroughTheKillCommand()
    {
        // game: CreatureCmd.Kill runs AfterDeath(false) for a death not prevented, with the Strangler at 0 HP and listed;
        // the corpse then leaves the roster.
        var (s, strangler) = ConstrictCombat();
        var k = Kernel(s, Power4);
        CreatureCmd.Kill(k, [OldCreature.Of(strangler)]);
        Assert.Equal(0, Constrict(s));
        Assert.Equal([40], s.Enemies.Select(e => e.Hp));
    }

    [Fact]
    public void ConstrictActsForThePlayersCopyOnlyAmongThreeHolders()
    {
        // The N-holder test: Constrict on the player, the Strangler and the slug (Holders); the Strangler's death removes
        // the player's (one call of the old body for that owner) and leaves the enemies' copies, which the old function
        // never touches.
        var s = Combat(30, 40);
        s.Enemies[0].DefId = KE.SlitheringStrangler;
        Holders(s, "ConstrictPower", BuffId.Constrict, 3);
        var k = Kernel(s, Power4);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(0, Constrict(s));
        Assert.Equal(3, BuffSystem.Get(s.Enemies[0].Buffs, BuffId.Constrict));
        Assert.Equal(3, BuffSystem.Get(s.Enemies[1].Buffs, BuffId.Constrict));
    }

    [Fact]
    public void ConstrictDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, strangler) = ConstrictCombat();
        var k = Kernel(s, Without(Power4, ConstrictPower.Register));
        using (Dying(k, strangler))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(3, Constrict(s));
    }

    [Fact]
    public void ConstrictIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power4, AllButDamage), ListenerKind.Power, "ConstrictPower", HookId.AfterDeath);
    }

    // ---------------------------------------------------------------------------------------- Heist (power-4)

    /// <summary>A Fat Gremlin (creature 1) holding Heist for 25 stolen gold, one slug (creature 2).</summary>
    private static (CombatState S, EnemyState Gremlin) HeistCombat(int gold = 25)
    {
        var s = Combat(30, 40);
        var gremlin = s.Enemies[0];
        gremlin.DefId = KE.FatGremlin;
        gremlin.HeistGold = gold;
        GivePower(s, "HeistPower", BuffId.Heist, gold, gremlin);
        return (s, gremlin);
    }

    /// <summary>The gold banked for the reward, the gold the thief still holds, and its Heist amount.</summary>
    private static (int Banked, int Held, int Power) Gold(CombatState s, EnemyState thief) =>
        (s.StolenBackGold, thief.HeistGold, BuffSystem.Get(thief.Buffs, BuffId.Heist));

    [Fact]
    public void HeistIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power4), ListenerKind.Power, "HeistPower", HookId.BeforeDeath);
    }

    [Fact]
    public void HeistBanksTheStolenGoldBeforeTheThiefsDeathAsTheOldFunctionDoes()
    {
        // game: HeistPower.BeforeDeath; old: CombatEngine.AnswerDeaths calls Heist_BeforeDeath(state, fallen) for a Fat
        // Gremlin that has not escaped and holds stolen gold. Dispatched inside Dying: the thief at 0 HP, still listed.
        var (s, gremlin) = HeistCombat();
        var k = Kernel(s, Power4);
        using (Dying(k, gremlin))
            Assert.False(Hooks.BeforeDeath(k, BeforeDeathOf(1)).IsPaused);

        var (oracle, thief) = HeistCombat();
        CombatEngine.Heist_BeforeDeath(oracle, thief);
        Assert.Equal(Gold(oracle, thief), Gold(s, gremlin));
        Assert.Equal((25, 0, 0), Gold(s, gremlin));

        // Where the condition fails: the slug's death; an escaped Gremlin; one holding no gold; one of another def.
        var (s2, g2) = HeistCombat();
        var k2 = Kernel(s2, Power4);
        using (Dying(k2, s2.Enemies[1]))
            Hooks.BeforeDeath(k2, BeforeDeathOf(2));
        Assert.Equal((0, 25, 25), Gold(s2, g2));
        g2.Escaped = true;
        using (Dying(k2, g2))
            Hooks.BeforeDeath(k2, BeforeDeathOf(1));
        Assert.Equal((0, 25, 25), Gold(s2, g2));
        g2.Escaped = false;
        g2.HeistGold = 0;
        using (Dying(k2, g2))
            Hooks.BeforeDeath(k2, BeforeDeathOf(1));
        Assert.Equal((0, 0, 25), Gold(s2, g2));
        g2.HeistGold = 25;
        g2.DefId = Slug;
        using (Dying(k2, g2))
            Hooks.BeforeDeath(k2, BeforeDeathOf(1));
        Assert.Equal((0, 25, 25), Gold(s2, g2));
    }

    [Fact]
    public void HeistBanksTheGoldThroughTheKillCommandBeforeTheDeath()
    {
        // game: CreatureCmd.Kill runs BeforeDeath first (step 4 of the kill sequence), before ShouldDie and the powers'
        // removal; the thief then leaves the roster.
        var (s, gremlin) = HeistCombat();
        var k = Kernel(s, Power4);
        CreatureCmd.Kill(k, [OldCreature.Of(gremlin)]);
        Assert.Equal((25, 0, 0), Gold(s, gremlin));
        Assert.DoesNotContain(gremlin, s.Enemies);
    }

    [Fact]
    public void HeistActsForTheDyingThiefOnlyAmongTwoHolders()
    {
        // The N-holder test: two Fat Gremlins holding 25 and 40; the first's death banks 25 alone (one call of the old
        // body for that owner) and leaves the second's gold and power.
        var s = Combat(30, 30);
        foreach (var (thief, gold) in new[] { (s.Enemies[0], 25), (s.Enemies[1], 40) })
        {
            thief.DefId = KE.FatGremlin;
            thief.HeistGold = gold;
            GivePower(s, "HeistPower", BuffId.Heist, gold, thief);
        }

        var k = Kernel(s, Power4);
        using (Dying(k, s.Enemies[0]))
            Hooks.BeforeDeath(k, BeforeDeathOf(1));
        Assert.Equal((25, 0, 0), Gold(s, s.Enemies[0]));
        Assert.Equal((25, 40, 40), Gold(s, s.Enemies[1]));
    }

    [Fact]
    public void HeistDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, gremlin) = HeistCombat();
        var k = Kernel(s, Without(Power4, HeistPower.Register));
        using (Dying(k, gremlin))
            Hooks.BeforeDeath(k, BeforeDeathOf(1));
        Assert.Equal((0, 25, 25), Gold(s, gremlin));
    }

    [Fact]
    public void HeistIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power4, AllButDamage), ListenerKind.Power, "HeistPower", HookId.BeforeDeath);
    }
}
