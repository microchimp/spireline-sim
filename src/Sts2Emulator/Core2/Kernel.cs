// The kernel over the old engine (docs/rebuild/TIER1_KERNEL.md sections 1 and 4): the HookRuntime Hooks.g.cs calls,
// tying the listener cursor (Listeners.cs, K2), the handler table and the switches together, and the caller marks the
// old engine's sites set. The kernel is two parts: Kernel here, shared and immutable (switches, handler table, ids), and
// KernelState (KernelState.cs), one per combat, held by the CombatState (and the RunState) and copied with it. With no
// switch on the kernel holds no world, no table and no recorded action, and a state with none is the old engine's.
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

/// <summary>
/// The part of the kernel every combat of an engine shares: the switches, the handler table and the model ids. Immutable
/// once made (the table and the ids are built once, on first use, when a switch is on) and safe to share between
/// threads; no combat's state lives here (that is <see cref="KernelState" />, held by the combat). An engine holds one
/// and binds it to the states it makes (<see cref="Bind" />); a state with none is the old engine's.
/// </summary>
public sealed class Kernel
{
    private readonly bool[] _on = new bool[256];
    private readonly Lazy<OldModelIds> _ids;
    private readonly Lazy<HandlerTable> _table;

    /// <param name="options">The switches on.</param>
    /// <param name="ids">The old model ids (read only when a switch is on).</param>
    /// <param name="models">The migrated models' Register methods; each registers only its switched-on hooks.</param>
    public Kernel(KernelOptions options, Func<OldModelIds>? ids = null, IEnumerable<Action<HandlerTable>>? models = null)
    {
        Options = options;
        foreach (var s in options.Switches)
            _on[(int)s] = true;
        AnyOn = options.Switches.Count > 0;
        var registrations = models?.ToList() ?? [];
        _ids = new Lazy<OldModelIds>(() =>
            ids?.Invoke() ?? throw new InvalidOperationException("a kernel with a switch on needs the old model ids"));
        _table = new Lazy<HandlerTable>(() =>
        {
            var t = new HandlerTable(_ids.Value, On);
            foreach (var register in registrations)
                register(t);
            return t;
        });
    }

    public KernelOptions Options { get; }

    /// <summary>
    /// Test seam: the attack's F06 points while F06 is off. There the bridge runs the old bodies where the game runs
    /// Hook.BeforeAttack and Hook.AfterAttack (AttackCmd.Execute, OldEntry.AttackContext), outside the kernel's dispatch, so
    /// no handler sees them; this is called with the hook at that point, in order with the kernel's own dispatches. Unset
    /// outside tests.
    /// </summary>
    internal Action<HookId>? BridgePoint { get; set; }

    /// <summary>Whether any switch is on. When false the kernel never builds its table and no state records anything.</summary>
    public bool AnyOn { get; }

    /// <summary>Whether the switch of <paramref name="family" />'s group is on.</summary>
    public bool On(Switch family) => _on[(int)family];

    /// <summary>The handler table, built with the first use when a switch is on; null with none on.</summary>
    public HandlerTable? Handlers => AnyOn ? _table.Value : null;

    /// <summary>The old model ids; null with no switch on.</summary>
    public OldModelIds? Ids => AnyOn ? _ids.Value : null;

    // ------------------------------------------------------------------------------------- attachment to the states

    /// <summary>
    /// A new per-combat state of this kernel, bound to <paramref name="combat" /> (and <paramref name="run" />) and held
    /// by it (<see cref="CombatState.Kernel" />); with no combat, held by the run (<see cref="RunState.Kernel" />).
    /// </summary>
    public KernelState Bind(CombatState? combat, RunState? run = null)
    {
        var k = new KernelState(this, combat, run);
        if (combat is not null)
            combat.Kernel = k;
        else if (run is not null)
            run.Kernel = k;
        return k;
    }

    /// <summary>What an engine attaches to the states it makes: <see cref="Bind" /> when a switch is on, else null (the
    /// state stays the old engine's, with no object and no field written).</summary>
    public KernelState? Attach(CombatState? combat, RunState? run = null) => AnyOn ? Bind(combat, run) : null;

    /// <summary>
    /// A site's gate: the combat's kernel state, bound to the combat (and run) the site runs on, when
    /// <paramref name="family" />'s switch is on, else null (the site runs its old code; the old engine's combats hold
    /// none). <c>if (Kernel.If(Switch.F06, combat) is { } k) Hooks.Y(k, ...) else old</c>.
    /// </summary>
    public static KernelState? If(Switch family, CombatState combat, RunState? run = null) =>
        combat.Kernel is { } k && k.On(family) ? k.At(combat, run) : null;

    /// <summary>The gate of a run-level site: the active combat's state while one is open (its marks), else the run's.</summary>
    public static KernelState? If(Switch family, RunState run) =>
        (run.ActiveCombat?.Kernel ?? run.Kernel) is { } k && k.On(family) ? k.At(run.ActiveCombat, run) : null;

    // ------------------------------------------------------------------------------------- caller marks (K2's OldWorld)
    // Scoped setters for the old engine's call sites (TIER1_KERNEL.md section 3, "K3 obligations, the marks"):
    // `using (Kernel.Marks.Flying(combat, card)) { ... }`. They set the mark on the combat's own kernel state. A combat
    // with none (the old engine's) or with no switch on gets a shared no-op that writes nothing.

    public static class Marks
    {
        public static IDisposable Flying(CombatState combat, CardInstance card) =>
            W(combat) is { } w ? w.Flying(card) : Noop.Instance;

        public static IDisposable Setup(CombatState combat) => W(combat) is { } w ? w.Setup() : Noop.Instance;

        public static IDisposable MarkDying(CombatState combat, EnemyState enemy) =>
            W(combat) is { } w ? w.MarkDying(enemy) : Noop.Instance;

        public static IDisposable KeepPlayerHooks(CombatState combat) =>
            W(combat) is { } w ? w.KeepPlayerHooks() : Noop.Instance;

        public static IDisposable DeactivatePlayerHooks(CombatState combat) =>
            W(combat) is { } w ? w.DeactivatePlayerHooks() : Noop.Instance;

        private static OldWorld? W(CombatState combat) => combat.Kernel is { AnyOn: true } k ? k.World : null;
    }

    private sealed class Noop : IDisposable
    {
        public static readonly Noop Instance = new();

        public void Dispose()
        {
        }
    }
}
