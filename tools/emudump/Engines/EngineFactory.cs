using EmuDump.Engine;
using EmuDump.Engines.Core2;
using EmuDump.Engines.Old;

namespace EmuDump.Engines;

/// <summary>Makes the engine a process runs: one per fight, by name (<c>--engine old|core2</c>).</summary>
static class EngineFactory
{
    /// <summary>The engine names <see cref="Create"/> knows.</summary>
    internal static readonly string[] Known = ["old", "core2"];

    /// <summary>The modes that make their engine here, and so take <c>--engine</c>.</summary>
    internal static readonly string[] Modes = ["fight", "reanchor", "hookprobe", "setupinfo", "fuzz"];

    /// <summary>The engine <see cref="Create"/> makes without a name.</summary>
    internal static string Selected = "old";

    internal static IEngine Create(string? name = null) => (name ?? Selected) switch
    {
        "old" => new OldEngine(),
        "core2" => new Core2Engine(),
        var other => throw new ArgumentException($"unknown engine '{other}' (known: {string.Join(", ", Known)})"),
    };

    /// <summary>
    /// Takes <c>--engine NAME</c> out of the arguments (<paramref name="engine"/>: the name, null without the flag); null
    /// (with the reason in <paramref name="error"/>) when the name is unknown, missing, or given to a mode that does not
    /// make its engine here.
    /// </summary>
    internal static string[]? TakeFlag(string[] args, out string? engine, out string? error)
    {
        error = null;
        engine = null;
        int i = Array.IndexOf(args, "--engine");
        if (i < 0) return args;
        if (i + 1 >= args.Length || !Known.Contains(args[i + 1]))
        {
            error = $"--engine needs one of {string.Join(", ", Known)}";
            return null;
        }

        if (i == 0 || !Modes.Contains(args[0]))
        {
            error = $"--engine applies to the modes {string.Join(", ", Modes)} only";
            return null;
        }

        engine = args[i + 1];
        return [.. args[..i], .. args[(i + 2)..]];
    }
}
