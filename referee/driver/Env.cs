// Training environment (prototype, proto/real-game-env): one game process plays whole runs back to back, and every
// choice of a run goes to an outside agent program over the process's own stdin/stdout (no socket, no file).
//
// Job: {"env": true, "out": "<path>", ...the usual speed keys}. After boot the driver writes
//   ENV {"t":"ready","pid":N}
// then reads one command per line on stdin:
//   {"cmd":"reset","spec":{run spec: seed, potions, ...},"hash":false,"record":"<path>"?}   start a run
//   {"a":K}                                                                                   answer the open decision
//   {"cmd":"quit"}                                                                            exit 0
// and writes, per pending choice with two or more legal actions,
//   ENV {"t":"decide","kind":k,"obs":{...},"actions":[...],"ctx":...,"h":"<state hash>"?}
// and at the run's end (its reason: death, victory, or what stopped it)
//   ENV {"t":"end","end":reason,"floor":F,"decisions":D,"ms":M}
// A choice with one legal action is taken without asking. ENV is the line prefix "\u001eENV "; everything else on
// stdout is the game's own log. An answer out of range gets ENV {"t":"bad_action"} and the same decision again.
//
// Without a channel the same decision points pick with the run's seeded rng exactly as before (RunPlay.cs); with
// job "env_trace": true each pick is also recorded as [kind, n, index, hash] in the run's "env_trace", so a recorded
// run's choices can be replayed through the channel and its per-decision state hashes compared (envproto.py).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

internal static class Env
{
    internal const string Prefix = "\u001eENV ";

    /// <summary>A channel to an agent is open (job "env").</summary>
    internal static bool Live;

    /// <summary>Record every pick and its state hash (job "env_trace", or a reset's "hash").</summary>
    internal static bool Hashing;

    internal static bool Tracing;

    /// <summary>A whole run is being played: only then do decision points ask or trace.</summary>
    internal static RunState? Run;

    internal static JsonArray? Trace;
    internal static int Asked;

    private static Stream? _out;
    private static TextReader? _in;

    internal static void Open()
    {
        _out = Console.OpenStandardOutput();
        _in = new StreamReader(Console.OpenStandardInput(), new UTF8Encoding(false));
        Live = true;
    }

    internal static void Send(JsonObject o)
    {
        var bytes = Encoding.UTF8.GetBytes("\n" + Prefix + o.ToJsonString() + "\n");
        _out!.Write(bytes, 0, bytes.Length);
        _out.Flush();
    }

    internal static JsonObject? Read()
    {
        while (true)
        {
            string? line = _in!.ReadLine();
            if (line == null) return null;  // the agent closed the pipe
            if (line.Trim().Length == 0) continue;
            try
            {
                return JsonNode.Parse(line)!.AsObject();
            }
            catch (Exception ex)
            {
                Send(new JsonObject { ["t"] = "bad_command", ["error"] = ex.Message });
            }
        }
    }

    /// <summary>Thrown when the agent closes the channel mid-run: the session ends.</summary>
    internal sealed class Closed : Exception { }

    /// <summary>
    /// One decision point with n legal actions. Without a channel: fallback() (the seeded random pick, which may draw
    /// from the rng even when n is 1, exactly as the driver always did), traced when n &gt; 1. With one: the agent's
    /// answer, asked only when n &gt; 1.
    /// </summary>
    internal static int Decide(string kind, int n, Func<JsonArray> actions, Func<int> fallback, Func<JsonNode?>? ctx = null)
    {
        if (Run == null || !(Live || Tracing))
        {
            return fallback();
        }

        if (!Live)
        {
            int idx = fallback();
            if (n > 1) Trace?.Add(new JsonArray(kind, n, idx, Hashing ? Hash() : null));
            return idx;
        }

        if (n <= 1) return n == 1 ? 0 : fallback();
        var msg = new JsonObject { ["t"] = "decide", ["kind"] = kind, ["obs"] = Obs(), ["actions"] = actions() };
        if (ctx != null) msg["ctx"] = ctx();
        if (Hashing) msg["h"] = Hash();
        Asked++;
        while (true)
        {
            Send(msg);
            var reply = Read() ?? throw new Closed();
            if ((string?)reply["cmd"] == "quit") throw new Closed();
            int a = reply["a"] is JsonValue v && v.TryGetValue(out int k) ? k : -1;
            if (a >= 0 && a < n) return a;
            Send(new JsonObject { ["t"] = "bad_action", ["n"] = n, ["got"] = reply.ToJsonString() });
            msg.Remove("h");
            if (Hashing) msg["h"] = Hash();
        }
    }

    /// <summary>
    /// Choose `count` of `labels.Count` items as `count` single picks from what is left (ascending indices). The
    /// seeded fallback's ascending indices map to picks at index[k] - k of the remaining list.
    /// </summary>
    internal static List<int> DecideMany(string kind, IReadOnlyList<JsonNode> labels, int count, Func<List<int>> fallback)
    {
        if (Run == null || !(Live || Tracing)) return fallback();
        if (!Live)
        {
            var idx = fallback();
            for (int k = 0; k < idx.Count; k++)
                if (labels.Count - k > 1) Trace?.Add(new JsonArray(kind, labels.Count - k, idx[k] - k, Hashing ? Hash() : null));
            return idx;
        }

        var left = Enumerable.Range(0, labels.Count).ToList();
        var picked = new List<int>();
        for (int k = 0; k < count && left.Count > 0; k++)
        {
            var now = left.ToList();
            int a = Decide(kind, now.Count, () => new JsonArray(now.Select(i => labels[i].DeepClone()).ToArray()), () => 0,
                () => new JsonObject { ["pick"] = k + 1, ["of"] = count });
            picked.Add(now[a]);
            left.Remove(now[a]);
        }

        picked.Sort();
        return picked;
    }

    /// <summary>The full state behind a decision: the run's save and, in a fight, the step snapshot.</summary>
    internal static string Hash()
    {
        var s = Run!;
        string text = Driver.RunSnapText(s) + (CombatManager.Instance.IsInProgress ? Driver.CombatSnapText(s) : "");
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)))[..16].ToLowerInvariant();
    }

    private static JsonArray Card(CardModel c) => new(c.Id.Entry, c.IsUpgraded ? 1 : 0);

    /// <summary>The compact observation: the run, and in a fight the board (draw pile as a sorted multiset).</summary>
    internal static JsonObject Obs()
    {
        var s = Run!;
        var p = s.Players[0];
        var o = new JsonObject
        {
            ["act"] = s.CurrentActIndex, ["floor"] = s.TotalFloor, ["hp"] = p.Creature.CurrentHp, ["max_hp"] = p.Creature.MaxHp,
            ["gold"] = p.Gold,
            ["deck"] = new JsonArray(p.Deck.Cards.Select(c => (JsonNode)Card(c)).ToArray()),
            ["relics"] = new JsonArray(p.Relics.Select(r => (JsonNode)r.Id.Entry).ToArray()),
            ["potions"] = new JsonArray(p.PotionSlots.Select(x => (JsonNode?)x?.Id.Entry).ToArray()),
        };
        if (CombatManager.Instance.IsInProgress)
        {
            var c = Driver.CombatSnap(s, p);
            c.Remove("rng");
            if (c["draw"] is JsonArray draw)
                c["draw"] = new JsonArray(draw.Select(x => x!.ToJsonString()).OrderBy(x => x, StringComparer.Ordinal)
                    .Select(x => JsonNode.Parse(x)).ToArray());
            if (c["hand"] is JsonArray hand && p.PlayerCombatState is { } pcs)
            {
                var cards = pcs.Hand.Cards;
                for (int i = 0; i < hand.Count && i < cards.Count; i++)
                {
                    var e = hand[i]!.AsArray();
                    e.Add(SafeCost(cards[i]));
                    e.Add(cards[i].CanPlay());
                }
            }

            o["combat"] = c;
        }

        return o;
    }

    private static int SafeCost(CardModel c)
    {
        try { return c.EnergyCost.GetResolved(); }
        catch (Exception) { return -99; }
    }
}

public static partial class Driver
{
    internal static string RunSnapText(RunState s) => RunSnap(s).ToJsonString();
    internal static string CombatSnapText(RunState s) => Snapshot(s, s.Players[0]).ToJsonString();
    internal static JsonObject CombatSnap(RunState s, Player p) => Snapshot(s, p);

    /// <summary>
    /// Job "env": runs back to back in this process, each started by the agent's reset and every choice its answer.
    /// </summary>
    private static async Task<int> ServeEnv(JsonObject job, bool fastReset)
    {
        Env.Open();
        int pid = System.Environment.ProcessId;
        Env.Send(new JsonObject { ["t"] = "ready", ["pid"] = pid });
        int runs = 0;
        while (true)
        {
            Heartbeat();
            var cmd = Env.Read();
            if (cmd == null || (string?)cmd["cmd"] == "quit") return 0;
            if ((string?)cmd["cmd"] != "reset")
            {
                Env.Send(new JsonObject { ["t"] = "bad_command", ["error"] = "expected reset or quit" });
                continue;
            }

            var spec = cmd["spec"]?.DeepClone().AsObject() ?? new JsonObject();
            spec["kind"] = "run";
            spec["seed"] ??= "ENV" + runs;
            Env.Hashing = (bool?)cmd["hash"] == true;
            Env.Asked = 0;
            var clock = System.Diagnostics.Stopwatch.StartNew();
            JsonObject result;
            try
            {
                result = await PlaySpec(spec, fastReset);
            }
            catch (Env.Closed)
            {
                return 0;
            }

            runs++;
            _done++;
            string end = (string?)result["end"] ?? "error";
            if (result["error"] is { } err && end == "error") end = "error: " + ((string?)err ?? "").Split('\n')[0];
            var msg = new JsonObject
            {
                ["t"] = "end", ["end"] = end, ["floor"] = result["floor"]?.DeepClone(), ["asked"] = Env.Asked,
                ["decisions"] = (result["decisions"] as JsonArray)?.Count, ["ms"] = clock.ElapsedMilliseconds, ["runs"] = runs,
                ["mem_mb"] = System.Diagnostics.Process.GetCurrentProcess().PrivateMemorySize64 >> 20,
            };
            var errs = new JsonArray();
            foreach (var d in (result["decisions"] as JsonArray) ?? new JsonArray())
                if (d?["fight"]?["error"] is { } fe) errs.Add(fe.DeepClone());
            if (errs.Count > 0) msg["fight_errors"] = errs;
            if (cmd["record"] is JsonValue rec && rec.TryGetValue(out string? path) && path != null)
                File.WriteAllText(path, result.ToJsonString());
            if (StaleState() is { } stale) msg["stale"] = stale;
            Env.Send(msg);
            if (msg["stale"] != null) return RetireExit;  // Server.cs: a fresh process for the next run
        }
    }
}

public static partial class Driver
{
    /// <summary>FirstChoiceSelector.PickHook inside a run: the first n (the seeded chooser) or the agent's picks.</summary>
    private static List<int> FightPick(List<CardModel> list, int n) =>
        Env.DecideMany("select_cards_combat", list.Select(c => (JsonNode)new JsonArray(c.Id.Entry, c.IsUpgraded ? 1 : 0)).ToList(), n,
            () => Enumerable.Range(0, n).ToList());

    private static int FightCardReward(IReadOnlyList<MegaCrit.Sts2.Core.Entities.Cards.CardCreationResult> options) =>
        Env.Decide("card_reward_combat", options.Count,
            () => new JsonArray(options.Select(o => (JsonNode)new JsonArray(o.Card.Id.Entry, o.Card.IsUpgraded ? 1 : 0)).ToArray()), () => 0);
}
