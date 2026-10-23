// The kernel over the old engine (docs/rebuild/TIER1_KERNEL.md sections 1 and 4): the HookRuntime Hooks.g.cs calls,
// tying the listener cursor (Listeners.cs, K2), the handler table and the switches together, and the caller marks the
// old engine's sites set. With no switch on the kernel holds no world, no table and no recorded action: every member
// the old engine's sites call is a no-op that writes nothing, so an engine with this kernel attached is the old engine.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Core2;

/// <summary>The switches one engine instance runs with (families; a group switches whole).</summary>
public sealed class KernelOptions
{
    public static readonly KernelOptions None = new([]);

    public KernelOptions(IEnumerable<Switch> switches, IReadOnlyList<Switch[]>? groups = null)
    {
        Switches = switches.ToHashSet();
        Groups = groups ?? SwitchGroups.Groups;
        if (Switches.Contains(Switch.None))
            throw new ArgumentException("Switch.None is no family");
        foreach (var g in Groups)
        {
            var on = g.Where(Switches.Contains).ToList();
            if (on.Count > 0 && on.Count < g.Length)
                throw new ArgumentException(
                    $"families {string.Join("+", g)} share an old site and switch together (tools/kernel/sites.json); " +
                    $"on: {string.Join("+", on)}, missing: {string.Join("+", g.Except(on))}");
        }

        foreach (var s in Switches)
            if (!Groups.Any(g => g.Contains(s)))
                throw new ArgumentException($"{s} is in no switch group");
    }

    public IReadOnlySet<Switch> Switches { get; }

    public IReadOnlyList<Switch[]> Groups { get; }

    /// <summary>Options from family names ("F06", as tools/fidelity/migrated_families.json lists them).</summary>
    public static KernelOptions FromFamilies(IEnumerable<string> families) =>
        new(families.Select(f => Enum.TryParse<Switch>(f, out var s) && s != Switch.None
            ? s
            : throw new ArgumentException($"unknown family '{f}'")));
}

/// <summary>A hook action a per-listener pass recorded (the listener paused at a choice); K4's queue runs it.</summary>
public readonly record struct RecordedHookAction(long Id, PauseSite Site, HookStep Step);

/// <summary>A wait_all point a dispatcher reached: the hook actions it started since <c>Mark</c> are pending.</summary>
public readonly record struct WaitPoint(HookId Dispatcher, long Mark, int Pending);

public sealed class Kernel : HookRuntime
{
    [ThreadStatic] private static Kernel? _current;
    private readonly bool[] _on = new bool[256];
    private readonly Func<OldModelIds>? _ids;
    private readonly IReadOnlyList<Action<HandlerTable>> _models;
    private HandlerTable? _table;
    private OldWorld? _world;
    private OldListeners? _listeners;
    private readonly List<RecordedHookAction> _actions = [];
    private readonly List<PauseSite> _stops = [];
    private readonly List<WaitPoint> _waits = [];
    private long _nextAction;

    /// <param name="options">The switches on.</param>
    /// <param name="ids">The old model ids (read only when a switch is on).</param>
    /// <param name="models">The migrated models' Register methods; each registers only its switched-on hooks.</param>
    public Kernel(KernelOptions options, Func<OldModelIds>? ids = null, IEnumerable<Action<HandlerTable>>? models = null)
    {
        Options = options;
        foreach (var s in options.Switches)
            _on[(int)s] = true;
        AnyOn = options.Switches.Count > 0;
        _ids = ids;
        _models = models?.ToList() ?? [];
    }

    public KernelOptions Options { get; }

    /// <summary>Whether any switch is on. When false the kernel never builds its world or table.</summary>
    public bool AnyOn { get; }

    /// <summary>Whether the switch of <paramref name="family" />'s group is on.</summary>
    public bool On(Switch family) => _on[(int)family];

    /// <summary>The handler table, built with the first use when a switch is on; null with none on.</summary>
    public HandlerTable? Handlers => AnyOn ? _table ??= BuildTable() : null;

    /// <summary>The old state the listeners are read from; null with no switch on.</summary>
    public OldWorld? World => AnyOn ? _world ??= new OldWorld(null, null, Ids()) : null;

    /// <summary>The listener cursor of the passes; null with no switch on.</summary>
    public OldListeners? Passes => AnyOn ? _listeners ??= new OldListeners(World!, Handlers!) : null;

    public IReadOnlyList<RecordedHookAction> HookActions => _actions;

    /// <summary>Pauses of dispatchers whose context awaits each listener (the dispatch stopped there).</summary>
    public IReadOnlyList<PauseSite> Stops => _stops;

    public IReadOnlyList<WaitPoint> Waits => _waits;

    // ------------------------------------------------------------------------------------- attachment to the old engine

    /// <summary>The kernel the old engine's sites on this thread reach (one engine per thread; EmuDump: per process).</summary>
    public static Kernel? Current => _current;

    /// <summary>Makes <paramref name="k" /> the current kernel until the scope ends (the previous one is restored).</summary>
    public static IDisposable Enter(Kernel? k)
    {
        var prev = _current;
        _current = k;
        return new Scope(() => _current = prev);
    }

    /// <summary>Attaches <paramref name="k" /> to this thread with no scope (an engine's lifetime; null detaches).</summary>
    public static void Attach(Kernel? k) => _current = k;

    /// <summary>
    /// A site's gate: the current kernel bound to the combat the site runs on when <paramref name="family" />'s
    /// switch is on, else null (the site runs its old code). <c>if (Kernel.If(Switch.F06, combat) is { } k) ... else old</c>.
    /// </summary>
    public static Kernel? If(Switch family, CombatState combat, RunState? run = null) =>
        _current is { } k && k.On(family) ? k.At(combat, run) : null;

    /// <summary>Binds the world to the combat (and run) a site runs on; returns this.</summary>
    public Kernel At(CombatState? combat, RunState? run = null)
    {
        if (World is { } w)
        {
            w.Combat = combat;
            if (run is not null)
                w.Run = run;
        }

        return this;
    }

    // ------------------------------------------------------------------------------------- caller marks (K2's OldWorld)
    // Scoped setters for the old engine's call sites (TIER1_KERNEL.md section 3, "K3 obligations, the marks"):
    // `using (Kernel.Marks.Flying(card)) { ... }`. With no kernel or no switch on they return a shared no-op and write
    // nothing.

    public static class Marks
    {
        public static IDisposable Flying(CardInstance card) => W() is { } w ? w.Flying(card) : Noop.Instance;

        public static IDisposable Setup() => W() is { } w ? w.Setup() : Noop.Instance;

        public static IDisposable MarkDying(EnemyState enemy) => W() is { } w ? w.MarkDying(enemy) : Noop.Instance;

        public static IDisposable KeepPlayerHooks() => W() is { } w ? w.KeepPlayerHooks() : Noop.Instance;

        public static IDisposable DeactivatePlayerHooks() =>
            W() is { } w ? w.DeactivatePlayerHooks() : Noop.Instance;

        private static OldWorld? W() => _current is { AnyOn: true } k ? k.World : null;
    }

    // ------------------------------------------------------------------------------------- HookRuntime

    public override bool HasAny(HookId hook) => AnyOn && Handlers!.HasAny(hook);

    public override HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook) =>
        AnyOn ? Passes!.Cursor<THandler>(iterator, hook) : HookCursor<THandler>.Empty();

    public override ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who) => new(dispatcher, who);

    /// <summary>
    /// Per-listener contexts: the listener's choice becomes a recorded hook action (K4's queue runs it) and the dispatch
    /// moves on. Every other context: the dispatch stops; the stop is recorded for the action that owns the context.
    /// </summary>
    public override void OnPaused(in PauseSite site, HookStep step)
    {
        if (site.Context == ContextMode.PerListener)
            _actions.Add(new RecordedHookAction(++_nextAction, site, step));
        else
            _stops.Add(site);
    }

    public override long HookActionMark() => _nextAction;

    /// <summary>K3 records the wait point and leaves the hook actions pending; K4's executor drains them here.</summary>
    public override void WaitAll(HookId dispatcher, long mark) =>
        _waits.Add(new WaitPoint(dispatcher, mark, _actions.Count(a => a.Id > mark)));

    private HandlerTable BuildTable()
    {
        var t = new HandlerTable(Ids(), On);
        foreach (var register in _models)
            register(t);
        return t;
    }

    private OldModelIds Ids() =>
        _ids?.Invoke() ?? throw new InvalidOperationException("a kernel with a switch on needs the old model ids");

    private sealed class Scope(Action undo) : IDisposable
    {
        private Action? _undo = undo;

        public void Dispose()
        {
            _undo?.Invoke();
            _undo = null;
        }
    }

    private sealed class Noop : IDisposable
    {
        public static readonly Noop Instance = new();

        public void Dispose()
        {
        }
    }
}
