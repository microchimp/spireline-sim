// Batch L3b (docs/rebuild/MIGRATION_F08.md): the LOCAL models that needed a kernel addition. Surprise, Infested and Stock
// draw the game's Niche stream (HookCtx.Niche, the combat's NicheHpRng), Stock lists a new Axebot with Core2's
// CreatureCmd.Add, and Hang's multiplier is its fold (the card path's multiply is gated off under the switch). Each
// model is tested as CLOUD_BRIEFS section 3 asks: (1) its pairs are registered and none is a stub, (2) the dispatch gives
// what the old function gives for the same state, (3) the delete-mutant gives the no-handler value, (4) with F08 off its
// pairs are absent, (5) with two holders a dispatch about one creature changes that holder's listener alone. The death hooks
// run inside the kit's Dying scope and through CreatureCmd.Kill. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class L3bTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> L3b = Registry.L3b;
    private const int Seed = 7;
    private const int HangCard = 236;
    private const int OtherCard = 1;
    private const int Strike = 473;

    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] L3bPairs =
    [
        (ListenerKind.Power, "SurprisePower", HookId.AfterDeath),
        (ListenerKind.Power, "SurprisePower", HookId.ShouldStopCombatFromEnding),
        (ListenerKind.Power, "InfestedPower", HookId.AfterDeath),
        (ListenerKind.Power, "InfestedPower", HookId.ShouldStopCombatFromEnding),
        (ListenerKind.Power, "StockPower", HookId.AfterDeath),
        (ListenerKind.Power, "StockPower", HookId.ShouldStopCombatFromEnding),
        (ListenerKind.Power, "HangPower", HookId.ModifyDamageMultiplicative),
    ];

    private static IOrderedEnumerable<(ListenerKind Kind, string Model, HookId Hook)> Sorted(
        IEnumerable<(ListenerKind Kind, string Model, HookId Hook)> pairs) =>
        pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal);

    /// <summary>A kernel bound to the combat, whose Niche stream (NicheHpRng) is a fresh CountingRandom, and to a run whose
    /// open combat stream is another one that no handler draws (<see cref="RunStream" />).</summary>
    private static (KernelState K, CountingRandom Rng) InRun(CombatState s, IEnumerable<Action<HandlerTable>> models)
    {
        var rng = new CountingRandom(Seed);
        s.NicheHpRng = rng;
        var run = new RunState { ActiveCombatRng = new CountingRandom(Seed) };
        return (new Sts2Emulator.Core2.Kernel(new KernelOptions(DamageOnly), () => Ids, models).Bind(s, run), rng);
    }

    private static CountingRandom RunStream(KernelState k) => k.World!.Run!.ActiveCombatRng!;

    private static List<(int DefId, int Hp, int MaxHp)> Roster(CombatState s) =>
        s.Enemies.Select(e => (e.DefId, e.Hp, e.MaxHp)).ToList();

    // ---------------------------------------------------------------------------------------- the batch's table

    [Fact]
    public void TheRegistrysL3bBlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(L3b);
        Assert.Equal(Sorted(L3bPairs), Sorted(Pairs(t)));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsTheL3bBlock()
    {
        foreach (var model in L3b)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in L3bPairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        var t = Table(L3b, AllButDamage);
        Assert.Empty(Pairs(t));
        AssertAbsent(t, ListenerKind.Power, "SurprisePower", HookId.AfterDeath, HookId.ShouldStopCombatFromEnding);
        AssertAbsent(t, ListenerKind.Power, "InfestedPower", HookId.AfterDeath, HookId.ShouldStopCombatFromEnding);
        AssertAbsent(t, ListenerKind.Power, "StockPower", HookId.AfterDeath, HookId.ShouldStopCombatFromEnding);
        AssertAbsent(t, ListenerKind.Power, "HangPower", HookId.ModifyDamageMultiplicative);
    }

    // ---------------------------------------------------------------------------------------- the kernel additions

    [Fact]
    public void NicheIsTheCombatsNicheStreamElseItsShuffleStreamAndThrowsWithNeither()
    {
        var (k, rng) = InRun(Combat(10), L3b);
        Assert.Same(rng, k.Ctx.Niche);
        Assert.NotSame(RunStream(k), k.Ctx.Niche);
        var handMade = Combat(10);
        handMade.ShuffleRng = new CountingRandom(Seed);
        Assert.Same(handMade.ShuffleRng, Kernel(handMade, L3b).Ctx.Niche);
        var bare = Kernel(Combat(10), L3b);
        Assert.Throws<InvalidOperationException>(() => bare.Ctx.Niche);
    }

    [Fact]
    public void AddListsTheCreatureAtTheRostersEndAndRefusesOneAlreadyListed()
    {
        var s = Combat(10, 20);
        var k = Kernel(s, L3b);
        var bot = CombatEngine.NewRespawnedAxebot(2, 30, s.AscensionLevel);
        Assert.Equal(OldCreature.Of(bot), CreatureCmd.Add(k, bot));
        Assert.Same(bot, s.Enemies[^1]);
        Assert.Equal(3, s.Enemies.Count);
        Assert.Throws<InvalidOperationException>(() => CreatureCmd.Add(k, bot));
        Assert.Throws<InvalidOperationException>(() => CreatureCmd.Add(k, s.Enemies[0]));
    }

    // ---------------------------------------------------------------------------------------- Surprise and Infested

    public static IEnumerable<object[]> Spawners() =>
    [
        ["SurprisePower", BuffId.Surprise],
        ["InfestedPower", BuffId.Infested],
    ];

    private static void OldBody(string model, CombatState s, Random rng, EnemyState fallen)
    {
        if (model == "SurprisePower")
            CombatEngine.Surprise_AfterDeath(s, rng, fallen);
        else
            CombatEngine.Infested_AfterDeath(s, rng, fallen);
    }

    private static Action<HandlerTable> Registration(string model) =>
        model == "SurprisePower" ? SurprisePower.Register : InfestedPower.Register;

    /// <summary>The old body on an oracle combat built like the test's: the fallen at 0 HP while it runs.</summary>
    private static (CombatState S, CountingRandom Rng) Oracle(BuffId buff, int fallen, params int[] holders)
    {
        var oracle = Combat(5, 40);
        foreach (int h in holders)
            oracle.Enemies[h].Buffs.Add(new BuffState(buff, 1));
        var stream = new CountingRandom(Seed);
        oracle.NicheHpRng = stream;
        var dead = oracle.Enemies[fallen];
        int hp = dead.Hp;
        dead.Hp = 0;
        OldBody(buff == BuffId.Surprise ? "SurprisePower" : "InfestedPower", oracle, stream, dead);
        dead.Hp = hp;
        return (oracle, stream);
    }

    [Theory]
    [MemberData(nameof(Spawners))]
    public void TheSpawnersDeathRunsTheOldBodyOnTheOldSitesStream(string model, BuffId buff)
    {
        var s = Combat(5, 40);
        GivePower(s, model, buff, 1, s.Enemies[0]);
        var (k, rng) = InRun(s, L3b);
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));

        var (oracle, stream) = Oracle(buff, 0, 0);
        Assert.Equal(Roster(oracle), Roster(s));
        Assert.True(s.Enemies.Count > 2);
        // The HP rolls drew the combat's Niche stream, as the old body's do; the run's stream is untouched.
        Assert.Equal(stream.CallCount, rng.CallCount);
        Assert.True(rng.CallCount > 0);
        Assert.Equal(0, RunStream(k).CallCount);

        // Removal prevented, or another creature's death: nothing.
        var s2 = Combat(5, 40);
        GivePower(s2, model, buff, 1, s2.Enemies[0]);
        var (k2, rng2) = InRun(s2, L3b);
        using (Dying(k2, s2.Enemies[0]))
            Hooks.AfterDeath(k2, AfterDeathOf(1, removalPrevented: true));
        using (Dying(k2, s2.Enemies[1]))
            Hooks.AfterDeath(k2, AfterDeathOf(2));
        Assert.Equal(2, s2.Enemies.Count);
        Assert.Equal(0, rng2.CallCount);
    }

    [Theory]
    [MemberData(nameof(Spawners))]
    public void TheSpawnersDeathThroughKillSpawnsAndTheCorpseLeaves(string model, BuffId buff)
    {
        var s = Combat(5, 40);
        var spawner = s.Enemies[0];
        GivePower(s, model, buff, 1, spawner);
        var (k, rng) = InRun(s, L3b);
        Hit(k, OldCreature.Of(spawner), 99m);
        Assert.DoesNotContain(spawner, s.Enemies);
        Assert.True(s.Enemies.Count > 1);
        Assert.True(rng.CallCount > 0);
        Assert.True(CombatEnd.IsInProgress(k));
    }

    [Theory]
    [MemberData(nameof(Spawners))]
    public void TheSpawnersDeleteMutantSpawnsNothing(string model, BuffId buff)
    {
        var s = Combat(5, 40);
        GivePower(s, model, buff, 1, s.Enemies[0]);
        var (k, rng) = InRun(s, Without(L3b, Registration(model)));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(2, s.Enemies.Count);
        Assert.Equal(0, rng.CallCount);
        Assert.False(Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _));
    }

    [Theory]
    [MemberData(nameof(Spawners))]
    public void TheSpawnersActForTheDeadHolderOnlyAmongTwoHolders(string model, BuffId buff)
    {
        var s = Combat(5, 40);
        GivePower(s, model, buff, 1, s.Enemies[0]);
        GivePower(s, model, buff, 1, s.Enemies[1]);
        var (k, rng) = InRun(s, L3b);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));

        var (oracle, stream) = Oracle(buff, 1, 0, 1);
        Assert.Equal(Roster(oracle), Roster(s));
        Assert.Equal(stream.CallCount, rng.CallCount);
        Assert.Equal(1, BuffSystem.Get(s.Enemies[0].Buffs, buff));
    }

    [Theory]
    [MemberData(nameof(Spawners))]
    public void TheSpawnersHoldTheFightOpenAsTheOldFunctionSays(string model, BuffId buff)
    {
        var s = Combat(5, 40);
        GivePower(s, model, buff, 1, s.Enemies[0]);
        var k = Kernel(s, L3b);
        bool old = model == "SurprisePower"
            ? CombatEngine.Surprise_ShouldStopCombatFromEnding(s.Enemies[0])
            : CombatEngine.Infested_ShouldStopCombatFromEnding(s.Enemies[0]);
        Assert.True(old);
        Assert.Equal(old, Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _));
        Assert.False(Hooks.ShouldStopCombatFromEnding(Kernel(Combat(5, 40), L3b), new ShouldStopCombatFromEndingArgs(), out _));
    }

    // ---------------------------------------------------------------------------------------- Stock

    private static CombatState StockCombat(int stock, params int[] hp)
    {
        var s = Combat(hp);
        s.Enemies[0].DefId = KE.Axebot;
        GivePower(s, "StockPower", BuffId.Stock, stock, s.Enemies[0]);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Strength, 3));
        return s;
    }

    [Fact]
    public void AnAxebotsDeathListsANewAxebotAtTheRostersEndAsTheOldRespawnBuildsIt()
    {
        var s = StockCombat(2, 5, 40);
        var dead = s.Enemies[0];
        var other = s.Enemies[1];
        var (k, rng) = InRun(s, L3b);
        Hit(k, OldCreature.Of(dead), 99m);

        // The old in-place respawn on the same stream: the old death answer for a death that holds the fight open.
        var oracle = StockCombat(2, 5, 40);
        oracle.Enemies[0].Hp = 0;
        var stream = new CountingRandom(Seed);
        CombatEngine.AnswerDeathsHoldingTheFightOpen(oracle, stream);
        var revived = oracle.Enemies[0];
        Assert.Equal(KE.Axebot, revived.DefId);

        Assert.DoesNotContain(dead, s.Enemies);
        Assert.Equal(2, s.Enemies.Count);
        Assert.Same(other, s.Enemies[0]);
        var bot = s.Enemies[1];
        Assert.Equal((revived.DefId, revived.Block, revived.MoveIndex, revived.LastMove),
            (bot.DefId, bot.Block, bot.MoveIndex, bot.LastMove));
        // The HP: CreateCreature's unique roll on the combat's Niche stream (the old respawn drew its site's stream flat).
        var expect = StockCombat(2, 5, 40);
        expect.Enemies[0].Hp = 0;
        int hp = CombatEngine.RollAddedAxebotHp(expect, new CountingRandom(Seed), expect.Enemies[0], expect.AscensionLevel);
        Assert.Equal((hp, hp), (bot.Hp, bot.MaxHp));
        Assert.Equal(0, RunStream(k).CallCount);
        Assert.Equal(revived.CurrentIntent, bot.CurrentIntent);
        Assert.Equal(revived.Buffs, bot.Buffs);
        Assert.Equal([new BuffState(BuffId.Stock, 1)], bot.Buffs);
        Assert.Equal(stream.CallCount, rng.CallCount);
        Assert.Equal(1, rng.CallCount);
        Assert.True(CombatEnd.IsInProgress(k));
    }

    [Fact]
    public void TheLastAxebotsLastStockBringsBackABotWithNoneAndTheFightGoesOn()
    {
        var s = StockCombat(1, 5);
        var (k, _) = InRun(s, L3b);
        var dead = s.Enemies[0];
        Hit(k, OldCreature.Of(dead), 99m);
        Assert.Single(s.Enemies);
        Assert.NotSame(dead, s.Enemies[0]);
        Assert.Equal(KE.Axebot, s.Enemies[0].DefId);
        Assert.True(s.Enemies[0].Hp > 0);
        Assert.Empty(s.Enemies[0].Buffs);
        Assert.True(CombatEnd.IsInProgress(k));
    }

    [Fact]
    public void StockDoesNothingForAnotherDeathOrAPreventedRemoval()
    {
        var s = StockCombat(2, 5, 40);
        var (k, rng) = InRun(s, L3b);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1, removalPrevented: true));
        Assert.Equal(2, s.Enemies.Count);
        Assert.Equal(0, rng.CallCount);
    }

    [Fact]
    public void StocksDeleteMutantLosesTheRespawn()
    {
        var s = StockCombat(2, 5, 40);
        var (k, rng) = InRun(s, Without(L3b, StockPower.Register));
        Hit(k, OldCreature.Of(s.Enemies[0]), 99m);
        Assert.Single(s.Enemies);
        Assert.NotEqual(KE.Axebot, s.Enemies[0].DefId);
        Assert.Equal(0, rng.CallCount);
    }

    [Fact]
    public void StockActsForTheDeadBotOnlyAmongTwoHolders()
    {
        var s = StockCombat(2, 5, 40);
        s.Enemies[1].DefId = KE.Axebot;
        GivePower(s, "StockPower", BuffId.Stock, 2, s.Enemies[1]);
        var survivor = s.Enemies[1];
        var (k, rng) = InRun(s, L3b);
        Hit(k, OldCreature.Of(s.Enemies[0]), 99m);
        Assert.Equal(2, s.Enemies.Count);
        Assert.Same(survivor, s.Enemies[0]);
        Assert.Equal(2, BuffSystem.Get(survivor.Buffs, BuffId.Stock));
        Assert.Equal(1, BuffSystem.Get(s.Enemies[1].Buffs, BuffId.Stock));
        Assert.Equal(1, rng.CallCount);
    }

    [Fact]
    public void TheNewAxebotsHpIsUniqueAgainstTheLivingAndTheDyingOne()
    {
        var s = StockCombat(2, 5, 40);
        var band = GeneratedData.Enemies.Get(KE.Axebot).HpBand(s.AscensionLevel);
        Assert.True(band.Max - band.Min >= 2);
        s.Enemies[0].MaxHp = band.Min;
        s.Enemies[0].Hp = 0;
        s.Enemies[1].MaxHp = band.Max;
        var corpse = new EnemyState { DefId = KE.Axebot, Hp = 0, MaxHp = band.Min + 1 };
        s.Enemies.Add(corpse);
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 64; seed++)
            seen.Add(CombatEngine.RollAddedAxebotHp(s, new CountingRandom(seed), s.Enemies[0], s.AscensionLevel));
        Assert.DoesNotContain(band.Min, seen);
        Assert.DoesNotContain(band.Max, seen);
        // A corpse the game has already removed is no part of the side.
        Assert.Contains(band.Min + 1, seen);
        Assert.All(seen, v => Assert.InRange(v, band.Min, band.Max));
    }

    /// <summary>A Strike (a powered card) kills an Axebot through the old card path under the switch: the death runs from
    /// Kill, and the new bot's HP draws the combat's Niche stream once, not the Shuffle stream the old in-blow answer passes
    /// and not the run's stream.</summary>
    [Fact]
    public void APoweredCardKillOfAnAxebotRespawnsItOnTheCombatsNicheStream()
    {
        var fight = Fight.Hand(new CardInstance(Strike, false)).Energy(9).Enemy(hp: 1, defId: KE.Axebot);
        var s = fight.State;
        var dead = s.Enemies[0];
        GivePower(s, "StockPower", BuffId.Stock, 2, dead);
        var (k, niche) = InRun(s, L3b);
        var shuffle = s.ShuffleRng?.CallCount;
        fight.Play(0, target: 0);
        Assert.DoesNotContain(dead, s.Enemies);
        var bot = Assert.Single(s.Enemies);
        Assert.Equal(KE.Axebot, bot.DefId);
        Assert.True(bot.Hp > 1);
        Assert.Equal([new BuffState(BuffId.Stock, 1)], bot.Buffs);
        Assert.Equal(1, niche.CallCount);
        Assert.Equal(shuffle, s.ShuffleRng?.CallCount);
        Assert.Equal(0, RunStream(k).CallCount);
    }

    [Fact]
    public void StockHoldsTheFightOpenAsTheOldFunctionSays()
    {
        var s = StockCombat(2, 5);
        var k = Kernel(s, L3b);
        Assert.True(CombatEngine.Stock_ShouldStopCombatFromEnding(s.Enemies[0]));
        Assert.True(Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _));
        var mutant = Kernel(StockCombat(2, 5), Without(L3b, StockPower.Register));
        Assert.False(Hooks.ShouldStopCombatFromEnding(mutant, new ShouldStopCombatFromEndingArgs(), out _));
    }

    // ---------------------------------------------------------------------------------------- Hang

    private static (CombatState S, CardHandle Hang, CardHandle Other) HangCombat(int stacks = 4)
    {
        var s = Combat(200, 200);
        GivePower(s, "HangPower", BuffId.Hang, stacks, s.Enemies[0]);
        var hang = Add(s, s.Hand, HangCard);
        var other = Add(s, s.Hand, OtherCard);
        return (s, new CardHandle(hang.Uid), new CardHandle(other.Uid));
    }

    [Fact]
    public void HangMultipliesItsOwnersHangDamageByTheAmount()
    {
        var (s, hang, other) = HangCombat();
        var k = Kernel(s, L3b);
        Assert.Equal(40m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m, card: hang), out var changed));
        Assert.Equal([Power("HangPower", 1)], changed);
        // Not its owner, not Hang, no card: the identity.
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 10m, card: hang), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m, card: other), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m), out _));
    }

    [Fact]
    public void HangThroughTheDamageCommandLandsOnceMultiplied()
    {
        var (s, hang, _) = HangCombat();
        var k = Kernel(s, L3b);
        Hit(k, OldCreature.Of(s.Enemies[0]), 10m, card: hang);
        Assert.Equal(160, s.Enemies[0].Hp);
    }

    /// <summary>The Hang card played through the old card path (CardEffects' Hang case) under F08: its pre-multiply is gated
    /// off and HangPower's fold multiplies, so 10 lands as 40 against 4 stacks, as with the switch off; twice would be 160.
    /// </summary>
    [Fact]
    public void TheHangCardUnderTheSwitchMultipliesItsDamageExactlyOnce()
    {
        Fight Played(bool switched)
        {
            var fight = Fight.Hand().Energy(9).Enemy(hp: 500);
            // With a uid, as every card of a combat holds one: the damage's card source is found by it.
            fight.State.Hand.Add(fight.State.Fresh(new CardInstance(HangCard, false)));
            GivePower(fight.State, "HangPower", BuffId.Hang, 4, fight.Enemy0);
            if (switched)
                Kernel(fight.State, L3b);
            fight.Play(0, target: 0);
            return fight;
        }

        var on = Played(switched: true);
        Assert.NotNull(on.State.Kernel);
        Assert.Equal(460, on.Enemy0.Hp);
        Assert.Equal(8, on.EnemyBuffAmount(BuffId.Hang));
        var off = Played(switched: false);
        Assert.Equal(off.Enemy0.Hp, on.Enemy0.Hp);
    }

    [Fact]
    public void HangsDeleteMutantGivesTheNoHandlerValue()
    {
        var (s, hang, _) = HangCombat();
        var k = Kernel(s, Without(L3b, HangPower.Register));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m, card: hang), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void HangActsForItsOwnHolderOnlyAmongTwoHolders()
    {
        var (s, hang, _) = HangCombat();
        GivePower(s, "HangPower", BuffId.Hang, 2, s.Enemies[1]);
        var k = Kernel(s, L3b);
        Assert.Equal(40m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m, card: hang), out var one));
        Assert.Equal(20m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 10m, card: hang), out var two));
        Assert.Equal([Power("HangPower", 1)], one);
        Assert.Equal([Power("HangPower", 2)], two);
    }
}
