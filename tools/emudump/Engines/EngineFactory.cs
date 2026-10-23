using EmuDump.Engine;
using EmuDump.Engines.Old;

namespace EmuDump.Engines;

/// <summary>Makes the engine a process runs: one per fight, by name (<c>--engine old|core2</c> later).</summary>
static class EngineFactory
{
    /// <summary>The engine <see cref="Create"/> makes without a name.</summary>
    internal static string Selected = "old";

    internal static IEngine Create(string? name = null) => (name ?? Selected) switch
    {
        "old" => new OldEngine(),
        var other => throw new ArgumentException($"unknown engine '{other}' (known: old)"),
    };
}
