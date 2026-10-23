// The core2 engine (docs/rebuild/TIER1_KERNEL.md sections 1 and 6): the old engine with the Tier 1 kernel attached and
// the switches of tools/fidelity/migrated_families.json. With no switch on the kernel builds nothing and every site
// gate is false, so core2 is the old engine byte for byte (tools/fidelity/byte_proof.py --b-args "--engine core2").
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Core2;

namespace EmuDump.Engines.Core2;

public sealed class Core2Engine : OldEngine
{
    public Core2Engine() : this(Load())
    {
    }

    public Core2Engine(KernelOptions options)
    {
        Kernel = new Kernel(options, LoadIds, Sts2Emulator.Core2.Models.Registry.All);
        Kernel.Attach(Kernel);
    }

    public override string Name => "core2";

    /// <summary>The kernel attached to this engine (and to this thread while it is the latest engine made).</summary>
    public Kernel Kernel { get; }

    /// <summary>The switches tools/fidelity/migrated_families.json lists (copied next to EmuDump).</summary>
    public static KernelOptions Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "migrated_families.json");
        var migrated = JsonNode.Parse(File.ReadAllText(path))!["migrated"]!.AsArray();
        return KernelOptions.FromFamilies(migrated.Select(f => (string)f!));
    }

    static OldModelIds LoadIds() =>
        OldModelIds.Load(Path.Combine(AppContext.BaseDirectory, "id_map.json"),
            Path.Combine(AppContext.BaseDirectory, "power_aliases.json"));
}
