// EmuDump fight --rng-log: every draw from a combat stream, per step, as [stream, index, caller].
// tools/fidelity/rng_diff.py diffs two builds' logs fight by fight, so a refactor that keeps every
// stream's draw COUNT but moves a draw to another caller (or another step) is caught; a count-only
// diff cannot see that.
//
// The engine logs the draws (IEngine.StartRngLog, TakeRngLog: the game's stream name, the stream's counter after the
// draw, a label for the caller); this file shapes each step's draws into arrays. A draw the engine takes back later
// (a screen's answer replays a step and draws its draws again) is removed from the array it was written to, so the
// log holds each draw once, as the game made it. The callers are named from the stack, which tiered JIT inlining
// changes from run to run: run EmuDump with rng_diff.NO_INLINING when the labels matter.
using System.Text.Json.Nodes;
using EmuDump.Engine;

static class RngLog
{
    /// <summary>Logs every engine's draws and writes each step's into its record as "rng_log".</summary>
    public static void Install()
    {
        Fight.OnEngine += e => e.StartRngLog();  // a new fight starts a new log: what one that threw before its first step drew is not the next one's
        Fight.OnStep += (e, step) => step["rng_log"] = Take(e);
    }

    /// <summary>The draws since the last call as [stream, index, caller] arrays, each taken back when the engine drops its draw.</summary>
    internal static JsonArray Take(IEngine e)
    {
        var arr = new JsonArray();
        foreach (var d in e.TakeRngLog())
        {
            var node = new JsonArray(d.Stream, d.Index, d.Caller);
            arr.Add(node);
            d.TakenBack = () => arr.Remove(node);
        }

        return arr;
    }
}
