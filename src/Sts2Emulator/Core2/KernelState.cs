// The kernel's per-combat state (docs/rebuild/TIER1_KERNEL.md section 4): what one combat (or one run, outside a combat)
// owns of the kernel, as the HookRuntime Hooks.g.cs calls. Held by CombatState.Kernel and RunState.Kernel and copied with
// the state by Clone and CopyTo, so a clone or a replayed step (HookChoiceReplay) carries its own marks, binding and
// recorded actions and never writes into another state's. The shared part (switches, handler table, ids) is Kernel.
// With no switch on every member a site calls is a no-op that writes nothing.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Core2;

public sealed class KernelState : HookRuntime
{
    private readonly List<RecordedHookAction> _actions = [];
    private readonly List<PauseSite> _stops = [];
    private readonly List<WaitPoint> _waits = [];
    private long _nextAction;
    private CombatState? _combat;
    private RunState? _run;
    private OldWorld? _world;
    private OldListeners? _listeners;

    internal KernelState(Kernel shared, CombatState? combat, RunState? run)
    {
        Shared = shared;
        _combat = combat;
        _run = run;
    }

    /// <summary>The kernel's shared, immutable part.</summary>
    public Kernel Shared { get; }

    public KernelOptions Options => Shared.Options;

    /// <summary>Whether any switch is on. When false this state builds no world and records nothing.</summary>
    public bool AnyOn => Shared.AnyOn;

    /// <summary>Whether the switch of <paramref name="family" />'s group is on.</summary>
    public bool On(Switch family) => Shared.On(family);

    /// <summary>The shared handler table; null with no switch on.</summary>
    public HandlerTable? Handlers => Shared.Handlers;

    /// <summary>The old state the listeners are read from, with this state's marks; null with no switch on.</summary>
    public OldWorld? World => AnyOn ? _world ??= new OldWorld(_combat, _run, Shared.Ids!) : null;

    /// <summary>The listener cursor of the passes; null with no switch on.</summary>
    public OldListeners? Passes => AnyOn ? _listeners ??= new OldListeners(World!, Handlers!) : null;

    /// <summary>Hook actions per-listener passes recorded since this state began (K4's queue runs them).</summary>
    public IReadOnlyList<RecordedHookAction> HookActions => _actions;

    /// <summary>Pauses of dispatchers whose context awaits each listener (the dispatch stopped there).</summary>
    public IReadOnlyList<PauseSite> Stops => _stops;

    public IReadOnlyList<WaitPoint> Waits => _waits;

    /// <summary>Binds the world to the combat (and run) a site runs on; returns this.</summary>
    public KernelState At(CombatState? combat, RunState? run = null)
    {
        _combat = combat;
        if (run is not null)
            _run = run;
        if (_world is { } w)
        {
            w.Combat = combat;
            if (run is not null)
                w.Run = run;
        }

        return this;
    }

    // ------------------------------------------------------------------------------------- copying

    /// <summary>
    /// This state as a copy of the state it is on carries it: the same switches, marks, recorded actions, stops, waits
    /// and action counter, bound to <paramref name="combat" /> (and <paramref name="run" />; null keeps this state's
    /// run). The copy fills <paramref name="into" /> when that is a state of the same kernel (the in-place restore of
    /// <c>CopyTo</c>, whose holder is mid-step and keeps its object), else a new one. Nothing is shared with this state:
    /// the lists and pause contexts are copied, and a scope open on this state restores this state only.
    /// </summary>
    internal KernelState CopyTo(CombatState? combat, RunState? run, KernelState? into)
    {
        if (ReferenceEquals(into, this))
            return this;
        var k = into is not null && ReferenceEquals(into.Shared, Shared) ? into : new KernelState(Shared, combat, run);
        k._combat = combat;
        k._run = run ?? _run;
        k._actions.Clear();
        k._actions.AddRange(_actions.Select(a => a with { Site = Detach(a.Site) }));
        k._stops.Clear();
        k._stops.AddRange(_stops.Select(Detach));
        k._waits.Clear();
        k._waits.AddRange(_waits);
        k._nextAction = _nextAction;
        k._listeners = null;
        if (_world is not null || k._world is not null)
            (k._world ??= new OldWorld(null, null, Shared.Ids!)).CopyFrom(_world, k._combat, k._run);
        return k;
    }

    private static PauseSite Detach(PauseSite site) => site with { Ctx = site.Ctx?.Copy() };

    // ------------------------------------------------------------------------------------- HookRuntime

    public override bool HasAny(HookId hook) => AnyOn && Handlers!.HasAny(hook);

    public override HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook) =>
        AnyOn ? Passes!.Cursor<THandler>(iterator, hook) : HookCursor<THandler>.Empty();

    public override ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who) => new(dispatcher, who);

    /// <summary>
    /// Per-listener contexts: the listener's choice becomes a recorded hook action (K4's queue runs it) and the dispatch
    /// moves on. Every other context: the dispatch stops; the stop is recorded for the action that owns the context.
    /// With no switch on nothing is recorded.
    /// </summary>
    public override void OnPaused(in PauseSite site, HookStep step)
    {
        if (!AnyOn)
            return;
        if (site.Context == ContextMode.PerListener)
            _actions.Add(new RecordedHookAction(++_nextAction, site, step));
        else
            _stops.Add(site);
    }

    public override long HookActionMark() => _nextAction;

    /// <summary>K3 records the wait point and leaves the hook actions pending; K4's executor drains them here. With no
    /// switch on nothing is recorded.</summary>
    public override void WaitAll(HookId dispatcher, long mark)
    {
        if (AnyOn)
            _waits.Add(new WaitPoint(dispatcher, mark, _actions.Count(a => a.Id > mark)));
    }
}
