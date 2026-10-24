// The step-API acceptance test of the F08+F10 migration (docs/reviews/2026-10-24-f08-migration.md section 3.2 and
// recommendation 1; docs/rebuild/MIGRATION_F08.md section 1). Two fights traced on the game (Ironclad A10, the weak Corpse Slug
// pair each holding Ravenous, the player at Strength 40), played through the old step API as the lockstep drives it
// (CombatEngine.Step; with a kernel bound, RunEngine.StepCombat, the queue entry the same step goes through), with the real
// model Registry. Every number and id below is a value of the trace: nothing is inferred.
//
//   Headbutt, step 3 (Defend, Defend, Headbutt on slug 0; slugs 28 and 27 HP): the kill, the death hooks, the survivor's
//     Strength and the corpse's removal all come before the discard-pile screen opens. At the stopped step the roster holds the
//     survivor alone, with Strength +5 on top of its Ravenous; the player has spent the energy, kept the Defends' block, and
//     the played card is still in flight (the discard pile holds the two Defends, the draw pile its six).
//   Twin Strike, step 1 (slugs 29 and 27 HP): one hit only. The first kills (one CreatureAttacked, one DamageReceived), the
//     second finds no living target of its own, and the card's end shows the survivor at full HP with Strength +5.
//
// What the trace has and has not. The trace carries the survivor's powers at each mark and its Strength gain as an event before
// the screen, so the roster and Strength are pinned at the stopped step. The survivor's Stunned state and intent (move STUNNED,
// intent Stun) are in the trace only in the state at the end of the step, after the screen is answered: they are asserted
// there (in the old engine the Stun intent is the Stunned buff: OldGameView writes the buff as the game's Stun intent), not at
// the stop. Gremlin Horn, which the review's section 3.2 names, is not in these fights (no horn in the relics, the hand
// does not grow): it is not pinned here.
//
// The order of the death events is the game's hook sequence for the killing hit: BeforeDeath, ShouldDie (asked twice),
// ShouldCreatureBeRemovedFromCombatAfterDeath, AfterDeath (Ravenous runs here, with the dying slug still listed), then
// AfterAttack, whose listener list already lacks the dead slug. Under the switch the kernel dispatches those hooks, so a
// logging observer on the player's relic sees them in that order; the old engine runs no such passes and has nothing to log.
//
// What runs when. Each test has two kinds of assertion. The fixture and what holds in the old order and the new alike (block
// 10 and energy 1 before Headbutt; the screen's kind and candidate count; HP, hand and piles; the end state after the answer)
// run outside `Pinned`, so a wrong fixture or a wrong pin fails the test whatever the switch. Only what differs by order is
// inside `Pinned`: at the Headbutt stop, the roster, the survivor's Strength, the kernel's kill count and the event order;
// for Twin Strike, the kill count and the event order (its end state is the same in the old order).
//   Switch off (no kernel bound, the old order): the Headbutt stop assertions are a strict expected failure (the corpse is
//   still in the roster at the screen: the G2891c25e0d shape). Twin Strike's switch-off test has nothing inside Pinned: an
//   ordinary pin of the end state, and the order of the events cannot be asserted without a kernel.
//   Switch on (F08+F10 bound): strict expected failures until the fights' whole path exists, then they must pass. That is R4
//   (the reroute of the old damage and kill sites onto Core2's commands) AND the handlers these fights run through: at least
//   Ravenous's AfterDeath (batch power-1, LOCAL and not ported yet: Registry.All is empty on main) and Strength's damage fold;
//   R4 alone does not make them pass, because with a pair unported the dispatcher finds no handler and does nothing. A
//   switched-on test that unexpectedly passes fails the suite ("flip RerouteLanded"). The environment variable
//   STS2_STEP_TEST=strict runs the switched-on assertions as plain ones, to read their failures while R4 and the handlers are
//   built (every R4 and handler commit runs them that way).
//
// Kernel-less xunit has no expected-failure marker; the strict xfail is `Pinned` below.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using Xunit.Sdk;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Core2;

public class StepApiKillTests
{
    /// <summary>
    /// Flip to true in the commit after which the switched-on tests must pass: R4 (the reroute) together with the handlers the
    /// two fights touch (Ravenous's AfterDeath and Strength's damage fold at least, both of batch power-1). Not R4 alone.
    /// </summary>
    private const bool RerouteLanded = true;  // R7d: the strict run passes (AfterAttack logged at its bridge point)

    private const string FlipOn =
        "the switched-on values now hold: if R4 and the handlers these fights run through (at least Ravenous's AfterDeath and " +
        "Strength's damage fold) have landed, set RerouteLanded = true";

    private const string FlipOff =
        "the old order now matches the game at the stopped step: the old engine changed, or the pinned values are wrong";

    private static bool SwitchedOnMustPass =>
        RerouteLanded || Environment.GetEnvironmentVariable("STS2_STEP_TEST") == "strict";

    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly JsonObject IdMap = LoadIdMap();
    private static readonly string PlayerRelic = Name("relics", 1);

    private static JsonObject LoadIdMap()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        return JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!.AsObject();
    }

    private static string Name(string kind, int id) => IdMap[kind]!.AsObject().First(kv => (int)kv.Value! == id).Key;

    // ------------------------------------------------------------------------------------------ the observer and the run

    private static readonly string[] GameDeathOrder =
        ["BeforeDeath", "ShouldDie", "ShouldRemove", "AfterDeath", "AfterAttack"];

    private static Action<HandlerTable> Observer(List<string> log) => t =>
    {
        t.Model(ListenerKind.Relic, PlayerRelic)
            .On(HookId.BeforeDeath, Event<BeforeDeathArgs>(log, "BeforeDeath"))
            .On(HookId.ShouldDie, Ask<ShouldDieArgs>(log, "ShouldDie"))
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath, Ask<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>(log, "ShouldRemove"))
            .On(HookId.AfterDeath, Event<AfterDeathArgs>(log, "AfterDeath"))
            .On(HookId.AfterAttack, Event<AfterAttackArgs>(log, "AfterAttack"));
    };

    private static HookEvent<T> Event<T>(List<string> log, string name) where T : struct =>
        (HookCtx _, ListenerRef self, in T a, ChoiceCtx? c) =>
        {
            log.Add(name);
            return HookStep.Done;
        };

    private static HookPredicate<T> Ask<T>(List<string> log, string name) where T : struct =>
        (HookCtx _, ListenerRef self, in T a) =>
        {
            log.Add(name);
            return true;
        };

    /// <summary>The log with consecutive repeats folded (the game asks ShouldDie twice for the one kill).</summary>
    private static List<string> Folded(List<string> log)
    {
        var o = new List<string>();
        foreach (var e in log)
            if (o.Count == 0 || o[^1] != e)
                o.Add(e);
        return o;
    }

    private sealed class Play
    {
        public required CombatState S { get; init; }
        public KernelState? K { get; init; }
        public RunEngine? E { get; init; }
        public Random Rng { get; } = new CountingRandom(0);
        public List<string> Log { get; init; } = [];

        public StepResult Step(int action, int target = -1) =>
            E is null ? CombatEngine.Step(S, action, Rng, target) : E.StepCombat(S, action, Rng, target);
    }

    private static Play Build(Fight fight, bool switchOn)
    {
        var s = fight.State;
        if (!switchOn)
            return new Play { S = s };
        var log = new List<string>();
        var kernel = new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => Ids, Registry.All.Append(Observer(log)).ToArray());
        // AfterAttack is F06's: with F06 off the attack's end runs it as the bridge, which no handler sees, so its point
        // is logged through the kernel's bridge seam, in order with the F10 hooks the kernel dispatches.
        kernel.BridgePoint = h =>
        {
            if (h == HookId.AfterAttack)
                log.Add("AfterAttack");
        };
        var engine = new RunEngine { Kernel = kernel };
        engine.BindCombat(s);
        return new Play { S = s, K = s.Kernel, E = engine, Log = log };
    }

    private static Fight Slugs(int hp0, int hp1, params CardInstance[] hand)
    {
        var f = Fight.Encounter(CombatFactory.ActOneEncounter.CorpseSlugs, ascension: 10, seed: 0, 1)
            .PlayerHp(64, 80)
            .PlayerBuff(BuffId.Strength, 40)
            .Energy(3)
            .Discard();
        f.State.Hand = [.. hand];
        (int hp, EnemyState e)[] roster = [(hp0, f.State.Enemies[0]), (hp1, f.State.Enemies[1])];
        foreach (var (hp, e) in roster)
        {
            e.Hp = hp;
            e.MaxHp = hp;
            if (BuffSystem.Get(e.Buffs, BuffId.Ravenous) == 0)
                e.Buffs.Add(new BuffState(BuffId.Ravenous, 5));
        }

        Assert.Equal(2, f.State.Enemies.Count);
        return f;
    }

    private static int Strength(EnemyState e) => BuffSystem.Get(e.Buffs, BuffId.Strength);

    /// <summary>
    /// A strict expected failure, for the assertions that differ by order only (never the fixture, never what holds in both
    /// orders): with <paramref name="mustPass" /> false the body has to fail an assertion (an exception of another type is a
    /// broken harness and surfaces as itself); a body that holds is the signal to flip the expectation.
    /// </summary>
    private static void Pinned(bool mustPass, string flip, Action body)
    {
        if (mustPass)
        {
            body();
            return;
        }

        var ex = Record.Exception(body);
        Assert.True(ex is XunitException, ex is null
            ? $"the pinned game values hold but this test is marked expected-to-fail: {flip}"
            : $"expected an assertion failure, got {ex.GetType().Name}: {ex.Message}");
    }

    // ------------------------------------------------------------------------------------------ Headbutt, trace step 3

    private static void HeadbuttKill(bool switchOn, bool mustPass, string flip)
    {
        var fight = Slugs(28, 27,
            Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash),
            Card(IC.Headbutt), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .Draw(Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad).ToArray());
        var s = fight.State;
        var dead = s.Enemies[0];
        var survivor = s.Enemies[1];
        var p = Build(fight, switchOn);

        // the fixture (outside Pinned): trace step 1-2, two Defends
        Assert.Equal((64, 80, 0, 3), (s.PlayerHp, s.PlayerMaxHp, s.PlayerBlock, s.Energy));
        Assert.Equal((28, 27), (dead.Hp, survivor.Hp));
        p.Step(0);                // Defend
        p.Step(0);                // Defend
        Assert.Equal((10, 1), (s.PlayerBlock, s.Energy));
        Assert.Equal(6, s.Hand.Count);
        p.Step(3, 0);             // Headbutt on slug 0: 49 damage into 28 HP

        // the stopped step, the discard-pile screen open: what holds whichever order answers the death
        Assert.Equal(CardSelectionKind.DiscardToDrawPileTop, s.PendingSelection?.Kind);
        Assert.Equal(2, s.PendingSelection?.Candidates.Count);
        Assert.Equal((64, 80, 10, 0), (s.PlayerHp, s.PlayerMaxHp, s.PlayerBlock, s.Energy));
        Assert.Equal([IC.StrikeIronclad, IC.DefendIronclad, IC.Bash, IC.DefendIronclad, IC.DefendIronclad], Fight.Ids(s.Hand));
        Assert.Equal((6, 2), (s.DrawPile.Count, s.DiscardPile.Count));
        Assert.Equal((27, 27, 5), (survivor.Hp, survivor.MaxHp, BuffSystem.Get(survivor.Buffs, BuffId.Ravenous)));

        // what differs by order: the game has answered the death and removed the corpse before the screen
        Pinned(mustPass, flip, () =>
        {
            Assert.DoesNotContain(dead, s.Enemies);
            Assert.Equal([survivor], s.Enemies);
            Assert.Equal(5, Strength(survivor));
            if (switchOn)
            {
                Assert.Equal(1, p.K!.Commands.KillCalls);
                Assert.Equal(GameDeathOrder, Folded(p.Log));
            }
        });

        // the answer (outside Pinned: the end of the step is the game's in either order): the chosen Defend goes on top of the
        // draw pile, Headbutt to the discard pile; the slug is stunned (the game's move STUNNED, intent Stun)
        p.Step(0);
        Assert.Null(s.PendingSelection);
        Assert.Equal((7, 2), (s.DrawPile.Count, s.DiscardPile.Count));
        Assert.Equal([IC.DefendIronclad, IC.Headbutt], Fight.Ids(s.DiscardPile));
        Assert.Equal(5, s.Hand.Count);
        // read the roster afresh: answering a screen replays the step on a copy of the state, so `survivor` may be a stale object
        var left = Assert.Single(s.Enemies);
        Assert.Equal((27, 27, 5), (left.Hp, left.MaxHp, Strength(left)));
        Assert.Equal(5, BuffSystem.Get(left.Buffs, BuffId.Ravenous));
        Assert.Equal(1, BuffSystem.Get(left.Buffs, BuffId.Stunned));
        Assert.Equal((64, 10, 0), (s.PlayerHp, s.PlayerBlock, s.Energy));
    }

    [Fact]
    public void HeadbuttKillRunsTheDeathHooksAndRemovesTheCorpseBeforeItsScreen_SwitchOff_ExpectedToFail() =>
        HeadbuttKill(false, false, FlipOff);

    // Switched on, the whole body is the strict expected failure, not only `Pinned`: R4 has landed and the power-1 handlers
    // (Ravenous's AfterDeath, Strength) have not, so the kernel's kill skips the old Ravenous site and nothing answers it, and
    // the survivor's Strength and Ravenous read differently outside `Pinned` too. The fixture is still checked by the switch-off
    // tests, which run the same body with `mustPass` for it; STS2_STEP_TEST=strict (or RerouteLanded) runs this body plain.
    [Fact]
    public void HeadbuttKillRunsTheDeathHooksAndRemovesTheCorpseBeforeItsScreen_F08F10() =>
        Pinned(SwitchedOnMustPass, FlipOn, () => HeadbuttKill(true, true, FlipOn));

    // ------------------------------------------------------------------------------------------ Twin Strike, trace step 1

    private static void TwinStrikeKill(bool switchOn, bool mustPass, string flip)
    {
        var fight = Slugs(29, 27,
            Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.Inflame), Card(IC.StrikeIronclad),
            Card(IC.TwinStrike))
            .Draw(Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad,
                IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad).ToArray());
        var s = fight.State;
        var dead = s.Enemies[0];
        var survivor = s.Enemies[1];
        var p = Build(fight, switchOn);

        Assert.Equal((64, 80, 0, 3), (s.PlayerHp, s.PlayerMaxHp, s.PlayerBlock, s.Energy));
        Assert.Equal((29, 27), (dead.Hp, survivor.Hp));
        p.Step(5, 0);             // Twin Strike on slug 0: 45 damage into 29 HP

        // the card's end: the same in the old order and the new (the old engine answers the death after the card)
        Assert.Null(s.PendingSelection);
        Assert.DoesNotContain(dead, s.Enemies);
        Assert.Equal([survivor], s.Enemies);
        // one hit: the second finds no living target of its own, so the survivor keeps its HP
        Assert.Equal((27, 27), (survivor.Hp, survivor.MaxHp));
        Assert.Equal((5, 5), (BuffSystem.Get(survivor.Buffs, BuffId.Ravenous), Strength(survivor)));
        Assert.Equal(1, BuffSystem.Get(survivor.Buffs, BuffId.Stunned));
        Assert.Equal((64, 80, 0, 2), (s.PlayerHp, s.PlayerMaxHp, s.PlayerBlock, s.Energy));
        Assert.Equal([IC.StrikeIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.Inflame, IC.StrikeIronclad], Fight.Ids(s.Hand));
        Assert.Equal((9, 1), (s.DrawPile.Count, s.DiscardPile.Count));
        Assert.Equal([IC.TwinStrike], Fight.Ids(s.DiscardPile));

        // what differs by order (a kernel is needed to see it): one kill inside the hit, in the game's hook order
        if (switchOn)
            Pinned(mustPass, flip, () =>
            {
                Assert.Equal(1, p.K!.Commands.KillCalls);  // one Damage call ran: the card's second hit never reaches a dead target
                Assert.Equal(GameDeathOrder, Folded(p.Log));
            });
    }

    /// <summary>The old order ends the card in the game's state (it answers the death after the card, not inside the hit), so
    /// this pin passes today and must keep passing; only the order of the events differs, and that needs a kernel to see.</summary>
    [Fact]
    public void TwinStrikeLandsOneHitWhenTheFirstKills_SwitchOff_EndStateMatchesTheGame() =>
        TwinStrikeKill(false, true, FlipOff);

    // Switched on, the whole body is the strict expected failure, not only `Pinned`: R4 has landed and the power-1 handlers
    // (Ravenous's AfterDeath, Strength) have not, so the kernel's kill skips the old Ravenous site and nothing answers it, and
    // the survivor's Strength and Ravenous read differently outside `Pinned` too. The fixture is still checked by the switch-off
    // tests, which run the same body with `mustPass` for it; STS2_STEP_TEST=strict (or RerouteLanded) runs this body plain.
    [Fact]
    public void TwinStrikeLandsOneHitWhenTheFirstKills_F08F10() =>
        Pinned(SwitchedOnMustPass, FlipOn, () => TwinStrikeKill(true, true, FlipOn));
}
