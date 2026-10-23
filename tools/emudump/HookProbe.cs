// EmuDump hookprobe <scenarios.json> <out.json>: run synthetic combats on an engine that
// tools/fidelity/hook_order.py instrumented, and write down the order its listener probes fired in.
// This file reads the scenarios, hands them to the engine (IEngine.HookProbe) and writes what it answers; the
// run itself is the engine's (Engines/Old/OldHookProbe.cs). Exit 2 for a bad call or an engine that is not instrumented.
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

        JsonObject log;
        try
        {
            log = EngineFactory.Create().HookProbe(JsonNode.Parse(File.ReadAllText(args[0]))!.AsObject());
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
