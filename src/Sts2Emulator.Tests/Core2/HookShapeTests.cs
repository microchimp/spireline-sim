using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

/// <summary>
/// The generated dispatchers (Core2/Hooks.g.cs, from tools/kernel/hooks.json) against a fake runtime that returns
/// scripted listeners: each pass op's loop, its changed set, the run-time pass flags, pauses and wait points, the
/// zero-listener fast path, and a named step that is not implemented yet saying so.
/// </summary>
public class HookShapeTests
{
    private sealed class FakeRuntime : HookRuntime
    {
        private readonly Dictionary<HookId, List<(ListenerRef Who, Delegate Fn)>> _handlers = new();
        public readonly List<string> Log = new();
        public readonly List<PauseSite> Pauses = new();
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

        public override IReadOnlyList<HookListener<THandler>> Listeners<THandler>(HookIterator iterator, HookId hook)
        {
            Snapshots++;
            if (!_handlers.TryGetValue(hook, out var list))
                return Array.Empty<HookListener<THandler>>();
            return list.Select(x => new HookListener<THandler>(x.Who, (THandler)x.Fn)).ToList();
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
    public void EventPassThatPauses_RecordsThePauseAndContinuesWithTheNextListener()
    {
        var rt = new FakeRuntime();
        var ctx = new ChoiceCtx();
        HookEvent<AfterCardDrawnArgs> Run(bool pause) => (ListenerRef self, in AfterCardDrawnArgs a, ChoiceCtx? c) =>
        {
            Assert.Same(ctx, c);
            Assert.Equal(self, c!.Models[^1]); // pushed for the listener's turn
            rt.Log.Add(Name(self));
            return pause ? HookStep.Paused("choose a card", new PickResume(1)) : HookStep.Done;
        };
        rt.On(HookId.AfterCardDrawnEarly, A, Run(true))
          .On(HookId.AfterCardDrawnEarly, B, Run(false))
          .On(HookId.AfterCardDrawn, C, Run(false));

        Hooks.AfterCardDrawn(rt, new AfterCardDrawnArgs(ctx, new CardHandle(5), true));

        Assert.Equal(new[] { "A", "paused A", "B", "C" }, rt.Log);
        var site = Assert.Single(rt.Pauses);
        Assert.Equal(new PauseSite(HookId.Dispatch_AfterCardDrawn, HookId.AfterCardDrawnEarly, 0, A,
            ContextMode.Passed, ctx), site);
        Assert.Empty(ctx.Models); // popped after each listener
    }

    [Fact]
    public void PerListenerPass_RecordsAHookActionAndDrainsAtTheWaitPoints()
    {
        var rt = new FakeRuntime();
        HookEvent<AfterTurnEndArgs> Run(bool pause) => (ListenerRef self, in AfterTurnEndArgs a, ChoiceCtx? c) =>
        {
            Assert.Equal(self, c!.Owner);
            rt.Log.Add(Name(self));
            return pause ? HookStep.Paused("choose", new PickResume(2)) : HookStep.Done;
        };
        rt.On(HookId.AfterSideTurnEnd, A, Run(true))
          .On(HookId.AfterSideTurnEnd, B, Run(false))
          .On(HookId.AfterSideTurnEndLate, C, Run(false));

        Hooks.AfterTurnEndPasses(rt, new AfterTurnEndArgs(Side.Player, Array.Empty<CreatureRef>()));

        Assert.Equal(new[]
        {
            "A", "paused A", "B", "wait Dispatch_AfterTurnEnd from 0", "C", "wait Dispatch_AfterTurnEnd from 0",
        }, rt.Log);
        var site = Assert.Single(rt.Pauses);
        Assert.Equal(ContextMode.PerListener, site.Context);
        Assert.Equal(HookId.Dispatch_AfterTurnEnd, site.Ctx!.Dispatcher);
    }

    [Fact]
    public void UnimplementedStep_NamesTheStepAndTheGameMethod()
    {
        var rt = new FakeRuntime();
        var e = Assert.Throws<NotImplementedException>(() =>
            Hooks.ModifyDamage(rt, DamageArgs(10m, ModifyDamageHookType.None), out _));
        Assert.Contains("preview-hittable", e.Message);
        Assert.Contains("Hook.ModifyDamage", e.Message);
    }
}
