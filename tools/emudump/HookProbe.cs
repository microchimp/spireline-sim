// EmuDump hookprobe <scenarios.json> <out.json>: run synthetic combats on an engine that
// tools/fidelity/hook_order.py instrumented, and write down the order its listener probes fired in.
// This file reads the scenarios, hands them to the engine (IEngine.HookProbe) and writes what it answers; the
// run itself is the engine's (Engines/Old/OldHookProbe.cs). Exit 2, with the reason on stderr, for a bad call: too few
// arguments, a scenarios file that is missing, unreadable or not a JSON object, or an engine that is not instrumented.
using System.Text.Json;
using System.Text.Json.Nodes;
using EmuDump.Engines;

internal static class HookProbeMode
{
    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: EmuDump hookprobe <scenarios.json> <out.json>");
            return 2;
        }

        JsonObject scenarios;
        try
        {
            scenarios = JsonNode.Parse(File.ReadAllText(args[0])) as JsonObject
                ?? throw new JsonException("the document is not a JSON object");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException or JsonException)
        {
            Console.Error.WriteLine($"hookprobe: cannot read the scenarios file {args[0]}: {ex.Message}");
            return 2;
        }

        JsonObject log;
        try
        {
            log = EngineFactory.Create().HookProbe(scenarios);
        }
        catch (NotSupportedException ex)
        {
            Console.Error.WriteLine("hookprobe: " + ex.Message);
            return 2;
        }

        File.WriteAllText(args[1], log.ToJsonString());
        return 0;
    }
}
