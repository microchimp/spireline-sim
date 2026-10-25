// The engine a native handle runs (docs/rebuild/NATIVE_KERNEL.md sections 3 and 4): the old engine, or core2, the old
// engine with the Tier 1 kernel attached (TIER1_KERNEL.md), built as EmuDump's Core2Engine builds it: the switches of
// tools/fidelity/migrated_families.json (or the families the spec names), the model ids of data/id_map.json and
// tools/fidelity/power_aliases.json, and the handler registrations of Core2.Models.Registry.All. The three files are
// embedded in the library from the same sources EmuDump copies next to itself, so the two front ends read one list.
using System.Reflection;
using System.Text.Json.Nodes;
using Sts2Emulator.Core2;

namespace Sts2Emulator.Interop;

public sealed class NativeEngine
{
    /// <summary>The default: no kernel, the old engine byte for byte.</summary>
    public static readonly NativeEngine Old = new("old", null);

    private static readonly Lazy<OldModelIds> Ids = new(ReadIds);
    private static readonly Lazy<NativeEngine> Shipped = new(() => Make(ReadShippedOptions()));
    private static readonly Dictionary<string, NativeEngine> Named = new(StringComparer.Ordinal);

    private NativeEngine(string name, Kernel? kernel)
    {
        Name = name;
        Kernel = kernel;
    }

    /// <summary>The canonical name: "old", or "core2:" and the families on, in order ("core2:" with none on).</summary>
    public string Name { get; }

    /// <summary>The kernel every run of this engine is bound to (shared; each combat holds its own KernelState); null for old.</summary>
    public Kernel? Kernel { get; }

    /// <summary>Whether a switch is on: the paths the kernel does not reach yet are refused (NATIVE_KERNEL.md section 2).</summary>
    public bool KernelOn => Kernel is { AnyOn: true };

    /// <summary>
    /// The engine a spec names: "old"; "core2", the families of the shipped migrated_families.json; "core2:F08,F10", the
    /// families named (a switch group is still all or nothing). Built once per spec. Throws ArgumentException for any other.
    /// </summary>
    public static NativeEngine Parse(string spec)
    {
        spec = spec.Trim();
        if (spec == "old")
        {
            return Old;
        }

        if (spec == "core2")
        {
            return Shipped.Value;
        }

        if (!spec.StartsWith("core2:", StringComparison.Ordinal))
        {
            throw new ArgumentException($"unknown engine '{spec}' (old, core2 or core2:F08,F10)");
        }

        var families = spec["core2:".Length..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        lock (Named)
        {
            string key = string.Join(",", families);
            if (!Named.TryGetValue(key, out var engine))
            {
                engine = Make(KernelOptions.FromFamilies(families));
                Named[key] = engine;
            }

            return engine;
        }
    }

    private static NativeEngine Make(KernelOptions options) =>
        new("core2:" + string.Join(",", options.Switches.OrderBy(s => (int)s)),
            new Kernel(options, () => Ids.Value, Core2.Models.Registry.All));

    /// <summary>The families tools/fidelity/migrated_families.json lists, as embedded at build.</summary>
    internal static KernelOptions ReadShippedOptions()
    {
        var migrated = Embedded("migrated_families.json")["migrated"]!.AsArray();
        return KernelOptions.FromFamilies(migrated.Select(f => (string)f!));
    }

    private static OldModelIds ReadIds() => new(Embedded("id_map.json"), Embedded("power_aliases.json"));

    private static JsonObject Embedded(string name)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Sts2Emulator." + name)
            ?? throw new InvalidOperationException($"{name} is not embedded in the library (Sts2Emulator.csproj)");
        return JsonNode.Parse(stream)!.AsObject();
    }
}
