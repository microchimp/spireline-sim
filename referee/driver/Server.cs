// Server mode: a long-lived game process that plays fight specs from an inbox folder.
//
// Job: {"out": "<path>.jsonl", "inbox": "<dir>", "fast_reset": true, ...}. After the usual startup the
// driver loops:
//   1. take the oldest "<id>.json" in the inbox, claiming it atomically by renaming it to
//      "<id>.json.taken.<pid>" (a second server racing for the same file loses the rename);
//   2. play it exactly as a "runs" entry (PlaySpec);
//   3. append ONE JSON line to "out" (the result, plus "_id", "_pid", "_t" = unix ms);
//   4. delete the claimed file, heartbeat, repeat.
// It exits cleanly (code 0) when a file named STOP is in the inbox and no "*.json" is left.
// A crash or watchdog exit leaves the claimed file behind, so the dispatcher can see which spec was
// in flight and re-queue it. "hold.<pid>" in the inbox makes this process wait without claiming
// (a hot spare). hb/<pid> holds a status line, rewritten at most once a second by Heartbeat();
// hb/<pid>.ready holds the unix ms at which the server first became ready.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>Frame cap while waiting for work, so an idle server does not spin a core.</summary>
    private const int IdleFps = 20;

    private static string? _hbPath;
    private static string _status = "boot";
    private static long _lastHbWrite;

    private static long UnixMs() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    /// <summary>Called from <see cref="Heartbeat" />: rewrites this process's status file at most once a second.</summary>
    private static void WriteStatus(long tick, bool force = false)
    {
        if (_hbPath == null || (!force && tick - _lastHbWrite < 1000))
        {
            return;
        }

        _lastHbWrite = tick;
        try
        {
            File.WriteAllText(_hbPath, _status + " " + UnixMs());
        }
        catch (IOException)
        {
            // The dispatcher may be reading it; the next heartbeat rewrites it.
        }
    }

    /// <summary>Appends one result as a single line: the out file is only ever appended to.</summary>
    internal static void AppendLine(string path, JsonObject o) => File.AppendAllText(path, o.ToJsonString() + "\n");

    private static IEnumerable<FileInfo> PendingSpecs(string inbox) =>
        new DirectoryInfo(inbox).GetFiles("*.json").Where(f => f.Name.EndsWith(".json", StringComparison.Ordinal));

    /// <summary>The oldest pending spec, claimed by an atomic rename; null when there is none.</summary>
    private static int _lostClaims;

    private static (string taken, string id)? Claim(string inbox, int pid)
    {
        foreach (FileInfo f in PendingSpecs(inbox).OrderBy(f => f.LastWriteTimeUtc).ThenBy(f => f.Name, StringComparer.Ordinal))
        {
            string taken = f.FullName + ".taken." + pid;
            try
            {
                File.Move(f.FullName, taken);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                continue;  // another server took it first
            }

            // Seen 4 times in 600 fights: two servers claimed the same spec at the same moment, the
            // rename reported success here and the file was gone a moment later while the other
            // server played it. A claim counts only once its file is really ours to read.
            if (!File.Exists(taken))
            {
                _lostClaims++;
                GD.Print($"[referee] lost the claim on {f.Name} to another server");
                continue;
            }

            return (taken, f.Name[..^".json".Length]);
        }

        return null;
    }

    private static async Task<int> ServeInbox(JsonObject job, string outPath, bool fastReset)
    {
        string inbox = (string)job["inbox"]!;
        int pid = System.Environment.ProcessId;
        string hbDir = Path.Combine(inbox, "hb");
        Directory.CreateDirectory(hbDir);
        _hbPath = Path.Combine(hbDir, pid.ToString());
        string hold = Path.Combine(inbox, "hold." + pid);
        string stop = Path.Combine(inbox, "STOP");
        var tree = (SceneTree)Engine.GetMainLoop();
        _status = "idle";
        File.WriteAllText(_hbPath + ".ready", UnixMs().ToString());
        GD.Print($"[referee] server ready, pid {pid}");
        while (true)
        {
            bool held = File.Exists(hold);
            _status = held ? "hold" : "idle";
            Heartbeat();  // every poll: an idle server is alive, a hung main thread stops beating
            (string taken, string id)? claim = held ? null : Claim(inbox, pid);
            if (claim == null)
            {
                if (File.Exists(stop) && !PendingSpecs(inbox).Any())
                {
                    GD.Print($"[referee] STOP: inbox empty, exiting after {_done} fights ({_lostClaims} claims lost to other servers)");
                    return 0;
                }

                Engine.MaxFps = IdleFps;
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
                continue;
            }

            Engine.MaxFps = 0;
            var (taken, id) = claim.Value;
            _status = "busy " + id;
            Heartbeat();
            JsonObject? spec = null;
            string badSpec = "";
            try
            {
                spec = JsonNode.Parse(File.ReadAllText(taken))!.AsObject();
            }
            catch (FileNotFoundException)
            {
                _lostClaims++;  // as above, the other server noticed first: not ours, no result
                GD.Print($"[referee] lost the claim on {id} to another server");
                continue;
            }
            catch (Exception ex)
            {
                badSpec = ex.Message;
            }

            JsonObject result;
            bool fatal = false;
            if (spec == null)
            {
                result = new JsonObject { ["error"] = "bad spec file: " + badSpec };
            }
            else
            {
                try
                {
                    result = await PlaySpec(spec, fastReset);
                }
                catch (Exception ex)
                {
                    // The run may be left half-built. This is the harness failing, not a fight result:
                    // log it without an _id and leave the spec claimed, so the dispatcher restarts us and
                    // re-queues the spec once like a crash (seen: a map-generation NullRef 1 in 240 starts).
                    result = new JsonObject { ["_fatal"] = id, ["error"] = ex.ToString() };
                    fatal = true;
                }
            }

            if (fatal)
            {
                AppendLine(outPath, result);
                return 1;
            }

            result["_id"] = id;
            result["_pid"] = pid;
            result["_t"] = UnixMs();
            AppendLine(outPath, result);
            File.Delete(taken);
            _done++;

            _status = "idle";
            Heartbeat();
        }
    }
}
