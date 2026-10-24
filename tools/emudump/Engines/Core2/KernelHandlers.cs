// EmuDump kernel-handlers [out.json]: every (game model, hook) pair registered in the Tier 1 kernel's handler table with all
// switches on, as {"handlers": [{"kind", "model", "hook", "stub"}]}; stdout when no path is given. tools/kernel/port_list.py
// reads it to know which pairs have a Core2 handler: the registry itself, not the source text (a comment or a model file
// that does not register does not count; a handler marked [StubHandler] is listed with "stub": true and does not count).
using System.Text.Json.Nodes;
using Sts2Emulator.Core2;

namespace EmuDump.Engines.Core2;

internal static class KernelHandlersMode
{
    /// <summary>The table's registered pairs once every model has registered into a table with every switch on.</summary>
    public static JsonObject Dump(IModelIds ids, IEnumerable<Action<HandlerTable>> models)
    {
        var table = new HandlerTable(ids, _ => true);
        foreach (var register in models)
        {
            register(table);
        }

        var rows = new JsonArray();
        foreach (var h in table.Registered())
        {
            rows.Add(new JsonObject
            {
                ["kind"] = h.Kind.ToString(),
                ["model"] = h.GameId,
                ["hook"] = h.Hook.ToString(),
                ["stub"] = h.Stub,
            });
        }

        return new JsonObject { ["handlers"] = rows };
    }

    public static int Run(string[] args)
    {
        if (args.Length > 1)
        {
            Console.Error.WriteLine("usage: EmuDump kernel-handlers [out.json]");
            return 2;
        }

        string json = Dump(Core2Engine.ModelIds, Sts2Emulator.Core2.Models.Registry.All).ToJsonString();
        if (args.Length == 1)
        {
            File.WriteAllText(args[0], json);
        }
        else
        {
            Console.Out.Write(json);
        }

        return 0;
    }
}
