// Engine faults (docs/ENGINE_FAULTS.md): what the copy throws where the game's own code does not finish a step, and the
// record a native entry point keeps of any exception that escapes an engine step, so a worker process gets a counted
// refusal rather than an unhandled exception that ends it.
namespace Sts2Emulator.Core;

/// <summary>
/// A state where the game's own code never finishes the step: its loop has no exit there, and the game would draw or
/// play forever. The copy stops and says so instead of running until memory is gone. Pillage drawing for a Hellraiser
/// whose every auto-play is refused is the one known (engine fault class 2).
/// </summary>
public sealed class GameHangs(string message) : Exception(message);

/// <summary>
/// A step the copy will not follow further although the game might finish it: a chain deeper than the copy's stack
/// allows (Unceasing Top's draws, each a Strike Hellraiser plays into another empty hand).
/// </summary>
public sealed class EngineLimit(string message) : Exception(message);

/// <summary>
/// An exception that escaped an engine step at a native entry point: its class, the top frame inside the engine, and its
/// message. The entry point returns the fault code with this as the last error, and the handle refuses every later step
/// with it: the combat is left wherever the exception found it, part way through the step, and is not stepped again.
/// </summary>
public sealed record EngineFault(string Class, string Frame, string Message)
{
    /// <summary>The fault code the step exports return (the refused answer is -1, a step 0 or 1).</summary>
    public const int Code = -2;

    public static EngineFault Of(Exception e) =>
        new(e.GetType().Name, TopEngineFrame(e.StackTrace), e.Message.Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' '));

    /// <summary>What the last error carries: class, frame and message, tab-separated.</summary>
    public string Text => $"{Class}\t{Frame}\t{Message}";

    /// <summary>
    /// The first frame of <paramref name="stackTrace" /> in the engine's own code (a list's or the runtime's frames above it
    /// skipped), as Type.Method: "CardEffects.DrawCards". "unknown" when there is none.
    /// </summary>
    internal static string TopEngineFrame(string? stackTrace)
    {
        foreach (var raw in (stackTrace ?? "").Split('\n'))
        {
            var line = raw.Trim();
            if (!line.StartsWith("at Sts2Emulator.", StringComparison.Ordinal))
            {
                continue;
            }

            var method = line[3..];
            int args = method.IndexOf('(');
            if (args >= 0)
            {
                method = method[..args];
            }

            var parts = method.Split('.');
            return parts.Length >= 2 ? $"{parts[^2]}.{parts[^1]}" : method;
        }

        return "unknown";
    }
}
