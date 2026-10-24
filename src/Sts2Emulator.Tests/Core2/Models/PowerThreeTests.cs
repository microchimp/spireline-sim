// The C4 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C4"): the port list's power-3 rows as migrated models
// under Core2/Models/Powers, each tested the way the shared rules ask: (1) its pairs are in the handler table and none is
// a stub, (2) the dispatch gives what the old function gives for the same state, (3) the delete-mutant (the model's
// registration left out) gives the no-handler value, (4) with F08 off its pairs are absent, (5) among several holders a
// dispatch about one creature changes that holder's listener alone, once. The table test at the start pins the batch's
// (model, hook) set, standing in for EmuDump kernel-handlers and the port list's gate, which need the manifest and a
// built EmuDump. Not in the batch: Crab Rage and Surrounded (their AfterDeath bodies, CombatEngine.TryEnrageCrabPartner,
// FacePast and TurnPlayerToFaceSurvivor, are private: the stop rule of section 3), Stock and Asleep (the local batch
// L2's). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PowerThreeTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Power3 = Registry.Power3;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list's power-3 rows the batch ports.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Power3Pairs =
    [
        (ListenerKind.Power, "FlutterPower", HookId.AfterDamageReceived),
        (ListenerKind.Power, "FlutterPower", HookId.ModifyDamageMultiplicative),
        (ListenerKind.Power, "PersonalHivePower", HookId.AfterDamageReceived),
        (ListenerKind.Power, "PlowPower", HookId.AfterDamageReceived),
    ];

    [Fact]
    public void TheRegistrysPower3BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Power3);
        Assert.Equal(Power3Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal)
            .ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal), Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePower3BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; this batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Power3)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Power3Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Power3, AllButDamage)));
    }

    private static int Stunned(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.Stunned);

    // ---------------------------------------------------------------------------------------- Flutter (power-3)

    /// <summary>One slug with Flutter 2 (a hopper's two hits before the stun); returns its listener.</summary>
    private static (CombatState S, ListenerRef Flutter) FlutterCombat(int amount = 2)
    {
        var s = Combat(40);
        return (s, GivePower(s, "FlutterPower", BuffId.Flutter, amount, s.Enemies[0]));
    }

    private static int Flutter(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.Flutter);

    [Fact]
    public void FlutterIsRegisteredForItsTwoPairs()
    {
        AssertRegistered(Table(Power3), ListenerKind.Power, "FlutterPower", HookId.ModifyDamageMultiplicative,
            HookId.AfterDamageReceived);
    }

    [Fact]
    public void FlutterHalvesAPoweredAttackOnItsOwner()
    {
        // game: FlutterPower.ModifyDamageMultiplicative; old: BuffSystem.IncomingDamage (the powered-attack path)
        // multiplies by Flutter_ModifyDamageMultiplicative() when the defender's Flutter > 0.
        var (s, me) = FlutterCombat();
        var k = Kernel(s, Power3);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(6m * (decimal)BuffSystem.Flutter_ModifyDamageMultiplicative(), v);
        Assert.Equal(3m, v);
        Assert.Equal([me], changed);
        // The same state through the old fold: 6 x 0.5 = 3.
        Assert.Equal(3, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        // The dealer is not read: a hit with none on the owner halves too.
        Assert.Equal(3m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, dealer: null), out changed));
        Assert.Equal([me], changed);

        // Where the old site's condition fails: an unpowered hit (thorns, poison), a hit on another creature, the amount
        // at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, props: Unpowered), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
        s.Enemies[0].Buffs[0] = new BuffState(BuffId.Flutter, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void FlutterLosesOneStackPerUnblockedPoweredHitAndStunsItsOwnerAtZero()
    {
        // game: FlutterPower.AfterDamageReceived; old: CardEffects.LandDamageOnEnemy calls
        // Flutter_AfterDamageReceived(target) for a powered hit that lost the target HP while its Flutter > 0. Through
        // CreatureCmd.Damage the fold halves the 5 to 2.5, lost as 2 (the command's one cast), then the event takes a
        // stack. The oracle is the old body called on an equal slug.
        var (s, _) = FlutterCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power3);
        var oracle = FlutterCombat().S.Enemies[0];

        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        CardEffects.Flutter_AfterDamageReceived(oracle);

        Assert.Equal(38, slug.Hp);
        Assert.Equal(oracle.Buffs, slug.Buffs);
        Assert.Equal((1, 0), (Flutter(slug), Stunned(slug)));

        // The second hit takes the last stack and stuns; a third finds no Flutter (the site's condition, and no halving
        // either) and changes nothing.
        Hit(k, OldCreature.Of(slug), 5m);
        CardEffects.Flutter_AfterDamageReceived(oracle);
        Assert.Equal(oracle.Buffs, slug.Buffs);
        Assert.Equal((0, 1), (Flutter(slug), Stunned(slug)));
        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        Assert.Equal([new BuffState(BuffId.Stunned, 1)], slug.Buffs);
    }

    [Fact]
    public void FlutterKeepsItsStacksOnABlockedOrUnpoweredHit()
    {
        // The old site asks for hpLoss > 0 and a powered hit: block that absorbs the hit, or an unpowered hit (poison,
        // thorns) that lands, takes no stack.
        var (s, _) = FlutterCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power3);

        slug.Block = 9;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        Assert.Equal((2, 0), (Flutter(slug), Stunned(slug)));
        slug.Block = 0;
        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(slug), 5m, props: Unpowered)).UnblockedDamage);
        Assert.Equal((2, 0), (Flutter(slug), Stunned(slug)));
    }

    [Fact]
    public void FlutterActsForItsOwnerAloneAmongThreeHolders()
    {
        // Rule 5: the player and both slugs hold Flutter 2. A powered hit on the first slug is halved once, by that slug's
        // listener, and takes one stack off that slug alone; the second slug's hit on the player is halved by the
        // player's listener.
        var s = Combat(40, 40);
        var holders = Holders(s, "FlutterPower", BuffId.Flutter, 2);
        var k = Kernel(s, Power3);

        Assert.Equal(3m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(3m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);

        Hit(k, OldCreature.Of(s.Enemies[0]), 6m);
        Assert.Equal(37, s.Enemies[0].Hp);
        Assert.Equal([1, 2], s.Enemies.Select(Flutter));
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.Flutter));

        // The player's Flutter halves the slug's hit and keeps its stacks: the old event body takes an enemy (the old site
        // is the enemy side's landing of a hit, and the old player side has no Flutter), so the handler acts for an enemy
        // owner only, as the old engine does.
        Hit(k, OldCreature.Player, 6m, dealer: OldCreature.Of(s.Enemies[1]));
        Assert.Equal(47, s.PlayerHp);
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.Flutter));
        Assert.Equal([1, 2], s.Enemies.Select(Flutter));
    }

    [Fact]
    public void FlutterDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = FlutterCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Without(Power3, FlutterPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        Assert.Equal(35, slug.Hp);
        Assert.Equal((2, 0), (Flutter(slug), Stunned(slug)));
    }

    [Fact]
    public void FlutterIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power3, AllButDamage), ListenerKind.Power, "FlutterPower", HookId.ModifyDamageMultiplicative,
            HookId.AfterDamageReceived);
    }

    // ---------------------------------------------------------------------------------------- Personal Hive (power-3)

    private const int DazedDef = ST.Dazed;

    /// <summary>One slug with Personal Hive 2, three Strikes in the player's draw pile; returns the hive's listener.</summary>
    private static (CombatState S, ListenerRef Hive) PersonalHiveCombat(int amount = 2)
    {
        var s = Combat(40);
        var hive = GivePower(s, "PersonalHivePower", BuffId.PersonalHive, amount, s.Enemies[0]);
        for (int i = 0; i < 3; i++)
            Add(s, s.DrawPile, 1);
        return (s, hive);
    }

    /// <summary>The draw pile's cards by (uid, def), top first: where each Dazed landed.</summary>
    private static string Draw(CombatState s) => string.Join(",", s.DrawPile.Select(c => $"{c.Uid}:{c.DefId}"));

    private static int Dazed(CombatState s) => s.DrawPile.Count(c => c.DefId == DazedDef);

    [Fact]
    public void PersonalHiveIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power3), ListenerKind.Power, "PersonalHivePower", HookId.AfterDamageReceived);
    }

    [Fact]
    public void PersonalHiveShufflesItsAmountOfDazedIntoTheDrawPileOnAPoweredHit()
    {
        // game: PersonalHivePower.AfterDamageReceived; old: CardEffects.LandDamageOnEnemy calls
        // PersonalHive_AfterDamageReceived(state, personalHive) for a powered hit on a target with Personal Hive > 0 that
        // is alive after it; no damage result is read. The old body draws each card's place from state.ShuffleRng or, for
        // a state made without one, a fixed stream, so the oracle's pile equals the dispatch's card for card, uids included.
        var (s, _) = PersonalHiveCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power3);
        var oracle = PersonalHiveCombat().S;

        Hit(k, OldCreature.Of(slug), 5m);
        CardEffects.PersonalHive_AfterDamageReceived(oracle, 2);

        Assert.Equal(35, slug.Hp);
        Assert.Equal(Draw(oracle), Draw(s));
        Assert.Equal(5, s.DrawPile.Count);
        Assert.Equal(2, Dazed(s));

        // A fully blocked hit counts too (no result is read); an unpowered one does not; nor does a hit that kills the
        // owner (the old site's Hp > 0; CreatureCmd.Damage runs AfterDamageReceived for a target it did not kill).
        slug.Block = 9;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        CardEffects.PersonalHive_AfterDamageReceived(oracle, 2);
        Assert.Equal(Draw(oracle), Draw(s));
        Assert.Equal(4, Dazed(s));
        Hit(k, OldCreature.Of(slug), 5m, props: Unpowered);
        Assert.Equal(4, Dazed(s));
        k.Commands.KillStandIn = (_, _) => { }; // the kill would sweep the corpse: the pile is what is looked at
        Assert.True(Assert.Single(Hit(k, OldCreature.Of(slug), 99m)).WasTargetKilled);
        Assert.Equal(4, Dazed(s));
    }

    [Fact]
    public void PersonalHiveActsForItsOwnerAloneAmongTwoHolders()
    {
        // Rule 5: both slugs hold Personal Hive, at 2 and at 3. A hit on the first adds its two Dazed once, not the
        // second's three as well; a hit on the second adds three.
        var s = Combat(40, 40);
        Add(s, s.DrawPile, 1);
        GivePower(s, "PersonalHivePower", BuffId.PersonalHive, 2, s.Enemies[0]);
        GivePower(s, "PersonalHivePower", BuffId.PersonalHive, 3, s.Enemies[1]);
        var k = Kernel(s, Power3);

        Hit(k, OldCreature.Of(s.Enemies[0]), 5m);
        Assert.Equal(2, Dazed(s));
        Hit(k, OldCreature.Of(s.Enemies[1]), 5m);
        Assert.Equal(5, Dazed(s));
        Assert.Equal(6, s.DrawPile.Count);
    }

    [Fact]
    public void PersonalHiveDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = PersonalHiveCombat();
        var k = Kernel(s, Without(Power3, PersonalHivePower.Register));
        string before = Draw(s);
        Hit(k, OldCreature.Of(s.Enemies[0]), 5m);
        Assert.Equal(35, s.Enemies[0].Hp);
        Assert.Equal(before, Draw(s));
        Assert.Equal(0, Dazed(s));
    }

    [Fact]
    public void PersonalHiveIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power3, AllButDamage), ListenerKind.Power, "PersonalHivePower", HookId.AfterDamageReceived);
    }

    // ---------------------------------------------------------------------------------------- Plow (power-3)

    /// <summary>One slug at <paramref name="hp" /> with Plow at <paramref name="threshold" />, Strength 4 and Temporary
    /// Strength 2, an attack announced; returns Plow's listener.</summary>
    private static (CombatState S, ListenerRef Plow) PlowCombat(int hp = 20, int threshold = 10)
    {
        var s = Combat(hp);
        var slug = s.Enemies[0];
        var plow = GivePower(s, "PlowPower", BuffId.Plow, threshold, slug);
        slug.Buffs.Add(new BuffState(BuffId.Strength, 4));
        slug.Buffs.Add(new BuffState(BuffId.TemporaryStrength, 2));
        slug.CurrentIntent = new Intent(IntentType.Attack, 9);
        return (s, plow);
    }

    private static (int Plow, int Strength, int Temporary, int Stunned) Plowed(EnemyState e) =>
        (BuffSystem.Get(e.Buffs, BuffId.Plow), BuffSystem.Get(e.Buffs, BuffId.Strength),
            BuffSystem.Get(e.Buffs, BuffId.TemporaryStrength), Stunned(e));

    [Fact]
    public void PlowIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power3), ListenerKind.Power, "PlowPower", HookId.AfterDamageReceived);
    }

    [Fact]
    public void PlowStripsTheOwnersStrengthAndStunsItOnceAHitLeavesItAtTheThreshold()
    {
        // game: PlowPower.AfterDamageReceived; old: CardEffects.LandDamageOnEnemy calls EnemyAI.TriggerPlowIfWounded(target)
        // for a hit with hpLoss > 0, and the body fires once the HP is at or under the threshold: Plow, Strength and
        // Temporary Strength removed, Stunned 1, the announced intent replaced. The oracle is the body called on an equal
        // slug at the HP the hit leaves.
        var (s, _) = PlowCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Power3);

        // 20 to 14: above the threshold, the body returns.
        Assert.Equal(6, Assert.Single(Hit(k, OldCreature.Of(slug), 6m)).UnblockedDamage);
        Assert.Equal((10, 4, 2, 0), Plowed(slug));
        Assert.Equal(new Intent(IntentType.Attack, 9), slug.CurrentIntent);

        // 14 to 10: at the threshold.
        Hit(k, OldCreature.Of(slug), 4m);
        var oracle = PlowCombat(hp: 10).S.Enemies[0];
        EnemyAI.TriggerPlowIfWounded(oracle);

        Assert.Equal(10, slug.Hp);
        Assert.Equal(oracle.Buffs, slug.Buffs);
        Assert.Equal(oracle.CurrentIntent, slug.CurrentIntent);
        Assert.Equal(oracle.MoveIndex, slug.MoveIndex);
        Assert.Equal((0, 0, 0, 1), Plowed(slug));
        Assert.Equal(IntentType.Unknown, slug.CurrentIntent.Type);
    }

    [Fact]
    public void PlowFiresOnAnUnpoweredHitTooButNotOnABlockedOne()
    {
        // The old site reads no props for Plow (a poison tick counts), only hpLoss > 0: a fully blocked hit leaves it.
        var (s, _) = PlowCombat(hp: 12);
        var slug = s.Enemies[0];
        var k = Kernel(s, Power3);

        slug.Block = 9;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Of(slug), 5m)).UnblockedDamage);
        Assert.Equal((10, 4, 2, 0), Plowed(slug));
        slug.Block = 0;
        Assert.Equal(3, Assert.Single(Hit(k, OldCreature.Of(slug), 3m, props: Unpowered)).UnblockedDamage);
        Assert.Equal(9, slug.Hp);
        Assert.Equal((0, 0, 0, 1), Plowed(slug));
    }

    [Fact]
    public void PlowActsForItsOwnerAloneAmongThreeHolders()
    {
        // Rule 5: the player and both slugs (at 20) hold Plow 10, the slugs Strength 4 too. A hit that brings the first
        // slug to the threshold strips and stuns that slug alone; a hit that brings the player under it changes nothing
        // of the player's (the old body takes an enemy: the old site is the enemy side's landing of a hit).
        var s = Combat(20, 20);
        Holders(s, "PlowPower", BuffId.Plow, 10);
        foreach (var e in s.Enemies)
            e.Buffs.Add(new BuffState(BuffId.Strength, 4));
        var k = Kernel(s, Power3);

        Hit(k, OldCreature.Of(s.Enemies[0]), 10m);
        Assert.Equal((0, 0, 0, 1), Plowed(s.Enemies[0]));
        Assert.Equal((10, 4, 0, 0), Plowed(s.Enemies[1]));
        Hit(k, OldCreature.Player, 45m, dealer: OldCreature.Of(s.Enemies[1]));
        Assert.Equal(5, s.PlayerHp);
        Assert.Equal(10, BuffSystem.Get(s.PlayerBuffs, BuffId.Plow));
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Stunned));
    }

    [Fact]
    public void PlowDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, _) = PlowCombat();
        var slug = s.Enemies[0];
        var k = Kernel(s, Without(Power3, PlowPower.Register));
        Hit(k, OldCreature.Of(slug), 10m);
        Assert.Equal(10, slug.Hp);
        Assert.Equal((10, 4, 2, 0), Plowed(slug));
        Assert.Equal(new Intent(IntentType.Attack, 9), slug.CurrentIntent);
    }

    [Fact]
    public void PlowIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power3, AllButDamage), ListenerKind.Power, "PlowPower", HookId.AfterDamageReceived);
    }
}
