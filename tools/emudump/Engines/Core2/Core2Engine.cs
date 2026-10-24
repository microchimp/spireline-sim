// The core2 engine (docs/rebuild/TIER1_KERNEL.md sections 1 and 6): the old engine with the Tier 1 kernel attached and
// the switches of tools/fidelity/migrated_families.json. With no switch on the kernel builds nothing and binds nothing to
// the states, so core2 is the old engine byte for byte (tools/fidelity/byte_proof.py --b-args "--engine core2"). The
// kernel is reached through the states the engine makes (CombatState.Kernel), never through the thread: an old engine
// made later on the same thread runs the old code.
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Core2;

namespace EmuDump.Engines.Core2;

public sealed class Core2Engine : OldEngine
{
    // Read once per process: the switches of migrated_families.json, the model ids, and the kernel built from them
    // (its handler table too), shared by every default core2 engine.
    private static readonly Lazy<KernelOptions> Options = new(ReadOptions);
    private static readonly Lazy<OldModelIds> Ids = new(ReadIds);
    private static readonly Lazy<Kernel> Default = new(() => Make(Options.Value));

    public Core2Engine() : this(Default.Value)
    {
    }

    public Core2Engine(KernelOptions options) : this(Make(options))
    {
    }

    public Core2Engine(Kernel kernel)
    {
        Kernel = kernel;
        Run.Kernel = kernel;
    }

    public override string Name => "core2";

    /// <summary>With F10 on, while the combat is in progress: the old ending predicate against the kernel's.</summary>
    public override string? EndingDisagreement()
    {
        if (Run.State.ActiveCombat is not { Kernel: { } k } c || !k.On(Switch.F10) || !CombatEnd.IsInProgress(k))
            return null;
        bool old = Sts2Emulator.Core.CombatEngine.OldIsOverOrEnding(c);
        bool core2 = CombatEnd.IsOverOrEnding(k.At(c, Run.State));
        return old == core2 ? null : $"old {(old ? "ending" : "open")}, core2 {(core2 ? "ending" : "open")}";
    }

    /// <summary>The old engine's game view plus the kernel's own state when a switch is on (Core2GameView); with none, old's bytes.</summary>
    public override JsonObject GameView()
    {
        var view = base.GameView();
        Core2GameView.Add(view, Run);
        return view;
    }

    /// <summary>The game-model to old-id resolution (id_map.json and power_aliases.json next to EmuDump; read once).</summary>
    internal static OldModelIds ModelIds => Ids.Value;

    /// <summary>The kernel this engine's states are bound to (shared; each combat holds its own KernelState).</summary>
    public Kernel Kernel { get; }

    /// <summary>The switches tools/fidelity/migrated_families.json lists (copied next to EmuDump; read once).</summary>
    public static KernelOptions Load() => Options.Value;

    private static Kernel Make(KernelOptions options) =>
        new(options, () => Ids.Value, Sts2Emulator.Core2.Models.Registry.All);

    private static KernelOptions ReadOptions()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "migrated_families.json");
        var migrated = JsonNode.Parse(File.ReadAllText(path))!["migrated"]!.AsArray();
        return KernelOptions.FromFamilies(migrated.Select(f => (string)f!));
    }

    private static OldModelIds ReadIds() =>
        OldModelIds.Load(Path.Combine(AppContext.BaseDirectory, "id_map.json"),
            Path.Combine(AppContext.BaseDirectory, "power_aliases.json"));
}
