using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

/// <summary>
/// The generated dispatchers (Core2/Hooks.g.cs, from tools/kernel/hooks.json) against a fake runtime that returns
/// scripted listeners: each pass op's loop, its changed set, the run-time pass flags, pauses by context (a stopping
/// context stops the dispatch until resumed, per-listener moves on) and wait points, the listener cursor, mutate and
/// try-or-null handlers, the zero-listener fast path, and a named step that is not implemented yet saying so.
/// </summary>
public class HookShapeTests
{
    private sealed class FakeRuntime : HookRuntime
    {
        private readonly Dictionary<HookId, List<(ListenerRef Who, Delegate Fn)>> _handlers = new();
        public readonly List<string> Log = new();
        public readonly List<PauseSite> Pauses = new();
        /// <summary>Listeners that have left the combat: skipped when visited.</summary>
        public readonly HashSet<ListenerRef> Gone = new();
        public int Snapshots;
        private long _actions;

        public FakeRuntime On(HookId hook, ListenerRef who, Delegate fn)
        {
            if (!_handlers.TryGetValue(hook, out var list))
                _handlers[hook] = list = new();
            list.Add((who, fn));
            return this;
        }

        public override bool HasAny(HookId hook) => _handlers.ContainsKey(hook);

        public override HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook)
        {
            Snapshots++;
            if (!_handlers.TryGetValue(hook, out var list))
                return HookCursor<THandler>.Empty();
            return new HookCursor<THandler>(list.Select(x => new HookListener<THandler>(x.Who, (THandler)x.Fn)).ToList(),
                member: who => !Gone.Contains(who));
        }

        public override ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who) => new(dispatcher, who);

        public override void OnPaused(in PauseSite site, HookStep step)
        {
            Pauses.Add(site);
            _actions++;
            Log.Add($"paused {Name(site.Who)}");
        }

        public override long HookActionMark() => _actions;

        public override void WaitAll(HookId dispatcher, long mark) => Log.Add($"wait {dispatcher} from {mark}");
    }

    private sealed record PickResume(int Site) : HookResume;

    private static readonly ListenerRef A = new(1, 10, 0), B = new(1, 11, 0), C = new(2, 12, 0), D = new(3, 13, 0);

    private static string Name(ListenerRef who) =>
        who == A ? "A" : who == B ? "B" : who == C ? "C" : who == D ? "D" : who.ToString();

    private static ModifyPowerAmountGivenArgs PowerArgs(decimal amount) =>
        new(new PowerRef(new CreatureRef(0), 1, 0), new CreatureRef(0), amount, null, null);

    private static HookFold<ModifyPowerAmountGivenArgs, decimal> Fold(FakeRuntime rt, ListenerRef who, decimal r) =>
        (ListenerRef self, in ModifyPowerAmountGivenArgs a, decimal current) =>
        {
            rt.Log.Add($"{Name(self)} sees {current}");
            return r;
        };

    [Fact]
    public void FoldAddThenFoldMul_ReturnsTheValueAndTheChangedSet()
    {
        var rt = new FakeRuntime();
        rt.On(HookId.ModifyPowerAmountGivenAdditive, A, Fold(rt, A, 2m))
          .On(HookId.ModifyPowerAmountGivenAdditive, B, Fold(rt, B, 0m))
          .On(HookId.ModifyPowerAmountGivenMultiplicative, C, Fold(rt, C, 1.5m))
          .On(HookId.ModifyPowerAmountGivenMultiplicative, D, Fold(rt, D, 1m));

        decimal v = Hooks.ModifyPowerAmountGiven(rt, PowerArgs(10m), out var changed);

        Assert.Equal(18m, v);
        Assert.Equal(new[] { A, C }, changed);
        Assert.Equal(new[] { "A sees 10", "B sees 12", "C sees 12", "D sees 18.0" }, rt.Log);
    }

    [Fact]
    public void ZeroListeners_TakeTheFastPath()
    {
        var rt = new FakeRuntime();
        decimal v = Hooks.ModifyPowerAmountGiven(rt, PowerArgs(7m), out var changed);
        Assert.Equal(7m, v);
        Assert.Empty(changed);
        Assert.Equal(0, rt.Snapshots);
    }

    private static ModifyDamageArgs DamageArgs(decimal damage, ModifyDamageHookType type) =>
        new(new CreatureRef(1), new CreatureRef(0), damage, new ValueProp(0), null, type, new PreviewMode(0));

    private static HookCap<ModifyDamageArgs, decimal> Cap(decimal cap) =>
        (ListenerRef self, in ModifyDamageArgs a) => cap;

    [Fact]
    public void FoldCap_KeepsTheRunningMinimumAndTheListenersThatLowered()
    {
        var rt = new FakeRuntime();
        bool additiveCalled = false;
        rt.On(HookId.ModifyDamageAdditive, D, (HookFold<ModifyDamageArgs, decimal>)(
              (ListenerRef self, in ModifyDamageArgs a, decimal current) => { additiveCalled = true; return 100m; }))
          .On(HookId.ModifyDamageCap, A, Cap(8m))
          .On(HookId.ModifyDamageCap, B, Cap(12m))
          .On(HookId.ModifyDamageCap, C, Cap(5m))
          .On(HookId.ModifyDamageCap, D, Cap(7m));

        decimal v = Hooks.ModifyDamagePasses(rt, DamageArgs(10m, ModifyDamageHookType.Cap), 10m, out var changed);
        Assert.Equal(5m, v);
        Assert.Equal(new[] { A, C }, changed);
        Assert.False(additiveCalled); // the Additive pass is not selected by the run-time flag

        decimal low = Hooks.ModifyDamagePasses(rt, DamageArgs(3m, ModifyDamageHookType.Cap), 3m, out var none);
        Assert.Equal(3m, low);
        Assert.Empty(none);
    }

    [Fact]
    public void PredicateAll_ReturnsTheFirstPreventer()
    {
        var rt = new FakeRuntime();
        HookPredicate<ShouldDieArgs> Says(ListenerRef who, bool answer) => (ListenerRef self, in ShouldDieArgs a) =>
        {
            rt.Log.Add(Name(self));
            return answer;
        };
        rt.On(HookId.ShouldDie, A, Says(A, true))
          .On(HookId.ShouldDieLate, B, Says(B, true))
          .On(HookId.ShouldDieLate, C, Says(C, false))
          .On(HookId.ShouldDieLate, D, Says(D, false));

        bool dies = Hooks.ShouldDie(rt, new ShouldDieArgs(new CreatureRef(1)), out var preventer);

        Assert.False(dies);
        Assert.Equal(C, preventer);
        Assert.Equal(new[] { "A", "B", "C" }, rt.Log);
    }

    [Fact]
    public void PredicateAny_ReturnsTheFirstTrue()
    {
        var rt = new FakeRuntime();
        HookPredicate<ShouldStopCombatFromEndingArgs> Says(bool answer) =>
            (ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
            {
                rt.Log.Add(Name(self));
                return answer;
            };
        rt.On(HookId.ShouldStopCombatFromEnding, A, Says(false))
          .On(HookId.ShouldStopCombatFromEnding, B, Says(true))
          .On(HookId.ShouldStopCombatFromEnding, C, Says(true));

        Assert.True(Hooks.ShouldStopCombatFromEnding(rt, new ShouldStopCombatFromEndingArgs(), out var first));
        Assert.Equal(B, first);
        Assert.Equal(new[] { "A", "B" }, rt.Log);

        Assert.False(Hooks.ShouldStopCombatFromEnding(new FakeRuntime(), new ShouldStopCombatFromEndingArgs(),
            out var nobody));
        Assert.Null(nobody);
    }

    [Fact]
    public void PassedContextPause_StopsTheDispatchUntilResumedFromPassAndIndex()
    {
        var rt = new FakeRuntime();
        var ctx = new ChoiceCtx();
        var pausing = new HashSet<ListenerRef> { A, C };
        HookEvent<AfterCardDrawnArgs> Run() => (ListenerRef self, in AfterCardDrawnArgs a, ChoiceCtx? c) =>
        {
            Assert.Same(ctx, c);
            Assert.Equal(self, c!.Models[^1]); // pushed for the listener's turn
            rt.Log.Add(Name(self));
            return pausing.Remove(self) ? HookStep.Paused("choose a card", new PickResume(1)) : HookStep.Done;
        };
        rt.On(HookId.AfterCardDrawnEarly, A, Run())
          .On(HookId.AfterCardDrawnEarly, B, Run())
          .On(HookId.AfterCardDrawn, D, Run())
          .On(HookId.AfterCardDrawn, C, Run());
        var args = new AfterCardDrawnArgs(ctx, new CardHandle(5), true);

        // The game awaits A's whole task: B and the later pass wait for the answer.
        HookStatus s = Hooks.AfterCardDrawn(rt, args);
        Assert.True(s.IsPaused);
        Assert.Equal(new[] { "A", "paused A" }, rt.Log);
        Assert.Equal(new PauseSite(HookId.Dispatch_AfterCardDrawn, HookId.AfterCardDrawnEarly, 0, 0, A,
            ContextMode.Passed, ctx), s.Site);
        Assert.Equal(new[] { A }, ctx.Models); // A stays pushed while its choice is open

        // Answered: the rest of pass 0 from its pass-start cursor, then pass 1 until C pauses at index 1.
        s = Hooks.AfterCardDrawnResume(rt, args, s);
        Assert.True(s.IsPaused);
        Assert.Equal((1, 1, C), (s.Site!.Value.Pass, s.Site.Value.Index, s.Site.Value.Who));
        Assert.Equal(new[] { "A", "paused A", "B", "D", "C", "paused C" }, rt.Log);
        Assert.Equal(new[] { C }, ctx.Models);
        // K4: the pause keeps pass 0's snapshot as plain data and the resume rebuilds its cursor from one fresh listing
        // (handlers and membership); the order is still the pass-start snapshot's.
        Assert.Equal(3, rt.Snapshots);

        Assert.Equal(HookStatus.Done, Hooks.AfterCardDrawnResume(rt, args, s));
        Assert.Empty(ctx.Models);
        Assert.Throws<InvalidOperationException>(() => Hooks.AfterCardDrawnResume(rt, args, HookStatus.Done));
    }

    [Fact]
    public void PassedContextResume_ChecksMembershipWhenTheLaterListenersAreVisited()
    {
        var rt = new FakeRuntime();
        var ctx = new ChoiceCtx();
        HookEvent<AfterCardDrawnArgs> Run(bool pause) => (ListenerRef self, in AfterCardDrawnArgs a, ChoiceCtx? c) =>
        {
            rt.Log.Add(Name(self));
            return pause ? HookStep.Paused("choose", new PickResume(1)) : HookStep.Done;
        };
        rt.On(HookId.AfterCardDrawnEarly, A, Run(true)).On(HookId.AfterCardDrawnEarly, B, Run(false))
          .On(HookId.AfterCardDrawnEarly, C, Run(false));
        var args = new AfterCardDrawnArgs(ctx, new CardHandle(5), true);

        var s = Hooks.AfterCardDrawn(rt, args);
        rt.Gone.Add(B); // B left the combat while A's choice was open
        Assert.Equal(HookStatus.Done, Hooks.AfterCardDrawnResume(rt, args, s));
        Assert.Equal(new[] { "A", "paused A", "C" }, rt.Log);
    }

    [Fact]
    public void PerListenerPause_RecordsAHookActionAndMovesOnToTheNextListener()
    {
        var rt = new FakeRuntime();
        HookEvent<AfterTurnEndArgs> Run(bool pause) => (ListenerRef self, in AfterTurnEndArgs a, ChoiceCtx? c) =>
        {
            Assert.Equal(self, c!.Owner);
            rt.Log.Add(Name(self));
            return pause ? HookStep.Paused("choose", new PickResume(2)) : HookStep.Done;
        };
        rt.On(HookId.AfterSideTurnEnd, A, Run(false))
          .On(HookId.AfterSideTurnEnd, B, Run(true))
          .On(HookId.AfterSideTurnEnd, D, Run(false))
          .On(HookId.AfterSideTurnEndLate, C, Run(false));

        Hooks.AfterTurnEndPasses(rt, new AfterTurnEndArgs(Side.Player, Array.Empty<CreatureRef>()));

        Assert.Equal(new[]
        {
            "A", "B", "paused B", "D", "wait Dispatch_AfterTurnEnd from 0", "C", "wait Dispatch_AfterTurnEnd from 0",
        }, rt.Log);
        var site = Assert.Single(rt.Pauses);
        Assert.Equal((ContextMode.PerListener, 0, 1), (site.Context, site.Pass, site.Index));
        Assert.Equal(HookId.Dispatch_AfterTurnEnd, site.Ctx!.Dispatcher);
    }

    [Fact]
    public void Cursor_BuildsTheLatePartWhenTheFirstEndsAndSkipsNonMembersAtVisit()
    {
        HookListener<Action> L(ListenerRef who) => new(who, () => { });
        var gone = new HashSet<ListenerRef>();
        int built = 0;
        var cursor = new HookCursor<Action>(new[] { L(A), L(B) },
            late: () => { built++; gone.Add(D); return new[] { L(C), L(D), L(A) }; },
            member: who => !gone.Contains(who));
        var seen = new List<(ListenerRef, int)>();

        Assert.True(cursor.Next(out var e));
        seen.Add((e.Who, cursor.Index));
        gone.Add(B); // removed mid-pass, before its visit
        Assert.Equal(0, built);
        while (cursor.Next(out e))
            seen.Add((e.Who, cursor.Index));

        Assert.Equal(1, built); // the deck part ended, then the combat part was built, once
        Assert.Equal(new[] { (A, 0), (C, 2), (A, 4) }, seen);
        Assert.False(cursor.Next(out _));
        Assert.False(HookCursor<Action>.Empty().Next(out _));
    }

    [Fact]
    public void Mutate_HandlerReordersTheCardsInPlace()
    {
        var rt = new FakeRuntime();
        rt.On(HookId.ModifyShuffleOrder, A, (HookMutate<ModifyShuffleOrderArgs>)(
            (ListenerRef self, in ModifyShuffleOrderArgs a) =>
            {
                var top = a.Cards[a.Cards.Count - 1];
                a.Cards.RemoveAt(a.Cards.Count - 1);
                a.Cards.Insert(0, top);
            }));
        var cards = new List<CardHandle> { new(1), new(2), new(3) };

        Hooks.ModifyShuffleOrder(rt, new ModifyShuffleOrderArgs(new PlayerRef(0), cards, false));

        Assert.Equal(new CardHandle[] { new(3), new(1), new(2) }, cards);
    }

    [Fact]
    public void TryWithOutNotNull_ATrueAnswerWithoutACardChangesNothing()
    {
        var rt = new FakeRuntime();
        HookTryOrNull<ModifyCardBeingAddedToDeckArgs, CardHandle> Gives(bool answer, CardHandle? card) =>
            (ListenerRef self, in ModifyCardBeingAddedToDeckArgs a, CardHandle current, out CardHandle? next) =>
            {
                next = card;
                return answer;
            };
        rt.On(HookId.TryModifyCardBeingAddedToDeck, A, Gives(true, null))
          .On(HookId.TryModifyCardBeingAddedToDeck, B, Gives(true, new CardHandle(9)))
          .On(HookId.TryModifyCardBeingAddedToDeckLate, C, Gives(false, new CardHandle(4)));

        var v = Hooks.ModifyCardBeingAddedToDeck(rt, new ModifyCardBeingAddedToDeckArgs(new CardHandle(1)),
            out var changed);

        Assert.Equal(new CardHandle(9), v);
        Assert.Equal(new[] { B }, changed);
    }

    [Fact]
    public void UnimplementedStep_NamesTheStepAndTheGameMethod()
    {
        // ModifyDamage's steps are implemented (F08 R2): off the kernel's own runtime none applies and the clamp runs.
        var rt = new FakeRuntime();
        Assert.Equal(10m, Hooks.ModifyDamage(rt, DamageArgs(10m, ModifyDamageHookType.None), out _));

        var e = Assert.Throws<NotImplementedException>(() => Hooks.ModifyBlock(rt,
            new ModifyBlockArgs(new CreatureRef(0), 5m, new ValueProp(0), new CardHandle(2), null), out _));
        Assert.Contains("enchant-add", e.Message); // typed by ModifyBlockArgs
        Assert.Contains("Hook.ModifyBlock", e.Message);
    }
}
