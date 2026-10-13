// Per-frame and per-process cost of the game while it plays fights (SPEED_PLAN step 0, CPU side).
//
// Job key "profile": true turns on a frame hook that, while a fight is in progress, records the main
// thread's CPU cycles per frame (split into quiet frames, where the driver's Idle loop saw nothing
// change, and busy ones), every thread's CPU by start module and name (FMOD mixer, GC, JIT, Godot
// workers), GC counts and pauses, the fight's QPC window (to cut EventPipe traces), and in the first
// three fights a census of the scene tree (nodes by type and by branch, which process, tweens). Each
// fight prints one "[referee] prof" line to the game's log. "sample": true adds LeanSampler.cs;
// "exp": [...] adds the profile-only experiments below. Without "exp" nothing here changes the game.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Godot;
using MegaCrit.Sts2.Core.Combat;

namespace Spireline.Referee;

internal static partial class Lean
{
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    [DllImport("kernel32.dll")]
    private static extern bool QueryThreadCycleTime(IntPtr thread, out ulong cycles);

    /// <summary>Cycles the calling thread has run (exact, unlike GetThreadTimes' 15.6 ms ticks).</summary>
    internal static ulong ThreadCycles()
    {
        QueryThreadCycleTime(GetCurrentThread(), out ulong c);
        return c;
    }

    private static double _cyclesPerMs;

    /// <summary>
    /// Thread cycles per millisecond of busy time, measured once by spinning: the best of ten 5 ms
    /// spins, since a spin the scheduler interrupts only reads low.
    /// </summary>
    internal static double CyclesPerMs()
    {
        if (_cyclesPerMs > 0) return _cyclesPerMs;
        double best = 0, x = 0;
        for (int i = 0; i < 10; i++)
        {
            var w = System.Diagnostics.Stopwatch.StartNew();
            ulong c0 = ThreadCycles();
            while (w.Elapsed.TotalMilliseconds < 5) x += Math.Sqrt(x + 1);
            best = Math.Max(best, (ThreadCycles() - c0) / w.Elapsed.TotalMilliseconds);
        }

        _cyclesPerMs = best;
        return _cyclesPerMs;
    }

    private static bool _profile;
    private static bool _inFight;
    private static long _frames;
    private static ulong _lastCycles;
    private static double _mainMs;
    private static long _census, _qpc0;
    private static (int g0, int g1, int g2, double pauseMs, long bytes) _gc0;
    private static readonly List<(double ms, bool quiet)> _frameMs = new();

    /// <summary>Set by the driver's Idle loop: frames in a row with nothing running and the state unchanged.</summary>
    internal static int QuietFrames;
    private static JsonObject? _tree;

    internal static void Configure(JsonObject job, SceneTree tree)
    {
        _profile = (bool?)job["profile"] == true;
        foreach (var e in job["exp"]?.AsArray() ?? new JsonArray()) _exp.Add((string)e!);
        if (_profile)
        {
            GD.Print($"[referee] prof cycles/ms {CyclesPerMs():F0}");
            tree.ProcessFrame += OnFrame;
            if ((bool?)job["sample"] == true) StartSampler();
        }
    }

    private static void OnFrame()
    {
        bool fight = CombatManager.Instance?.IsInProgress == true;
        ulong now = ThreadCycles();
        if (fight && _inFight)
        {
            _frames++;
            double ms = (now - _lastCycles) / CyclesPerMs();
            _mainMs += ms;
            _frameMs.Add((ms, QuietFrames >= 2));
            if (_exp.Count > 0 && _frames % 5 == 1)
            {
                Experiment((SceneTree)Engine.GetMainLoop());
            }

            if (_frames == 40 && _census < 3)
            {
                _tree = Census((SceneTree)Engine.GetMainLoop());
            }
        }
        else if (!fight && _inFight)
        {
            Report();
        }

        if (fight && !_inFight)
        {
            _frames = 0;
            _mainMs = 0;
            _tree = null;
            _frameMs.Clear();
            _threads0 = ThreadCycleTable();
            _qpc0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _gc0 = (GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2), GC.GetTotalPauseDuration().TotalMilliseconds, GC.GetTotalAllocatedBytes());
            if (_mainThread != IntPtr.Zero) TakeModuleSplit();
        }

        _inFight = fight;
        _sampling = fight && _mainThread != IntPtr.Zero;
        _lastCycles = ThreadCycles();  // the census and the report are not frame cost
    }

    /// <summary>Count, mean and median of a set of frame times (ms).</summary>
    private static JsonArray FrameStats(IEnumerable<double> ms)
    {
        var a = ms.OrderBy(x => x).ToArray();
        return a.Length == 0 ? new JsonArray(0, 0, 0)
            : new JsonArray(a.Length, Math.Round(a.Average(), 3), Math.Round(a[a.Length / 2], 3));
    }

    private static readonly HashSet<string> _exp = new();

    /// <summary>
    /// Profile-only experiments (job key "exp": [...]): switch per-frame processing off for one
    /// class of visual node, to measure what that class costs. Not gate-checked; never used for results.
    /// </summary>
    private static void Experiment(SceneTree tree)
    {
        var stack = new Stack<Node>();
        stack.Push(tree.Root);
        while (stack.Count > 0)
        {
            Node n = stack.Pop();
            string cls = n.GetClass();
            string type = n.GetType().Name;
            bool off = (_exp.Contains("spine") && (cls == "SpineSprite" || cls == "SpineMesh2D"))
                || (_exp.Contains("cpuparticles") && cls == "CPUParticles2D")
                || (_exp.Contains("gpuparticles") && cls == "GPUParticles2D")
                || (_exp.Contains("richfx") && n is RichTextLabel)
                || (_exp.Contains("audio") && cls == "AudioStreamPlayer")
                || (_exp.Contains("anim") && cls == "AnimatedSprite2D")
                || (_exp.Contains("trail") && (type == "NCardTrail" || type == "NCardTrailVfx"));
            if (off)
            {
                n.SetProcessInternal(false);
                n.SetPhysicsProcessInternal(false);
                n.SetProcess(false);
            }

            foreach (Node ch in n.GetChildren(includeInternal: true)) stack.Push(ch);
        }
    }

    private static void Report()
    {
        if (_frames == 0) return;
        var o = new JsonObject
        {
            ["frames"] = _frames,
            ["main_ms_per_frame"] = Math.Round(_mainMs / _frames, 3),
            ["quiet"] = FrameStats(_frameMs.Where(f => f.quiet).Select(f => f.ms)),
            ["busy"] = FrameStats(_frameMs.Where(f => !f.quiet).Select(f => f.ms)),
            ["threads"] = ThreadDelta(_threads0),
            ["gc"] = new JsonArray(GC.CollectionCount(0) - _gc0.g0, GC.CollectionCount(1) - _gc0.g1, GC.CollectionCount(2) - _gc0.g2,
                Math.Round(GC.GetTotalPauseDuration().TotalMilliseconds - _gc0.pauseMs, 1), (GC.GetTotalAllocatedBytes() - _gc0.bytes) >> 20),  // gen0, gen1, gen2, pause ms, MB allocated
            ["main_by_module"] = _mainThread != IntPtr.Zero ? TakeModuleSplit() : null,
            ["qpc"] = new JsonArray(_qpc0, System.Diagnostics.Stopwatch.GetTimestamp()),  // fight window, for EventPipe traces
            ["tree"] = _tree,
        };
        _census++;
        GD.Print("[referee] prof " + o.ToJsonString());
    }

    // ---- Threads: who else in the process burns CPU (audio mixer, GC, thread pool, Godot workers) ----

    [DllImport("kernel32.dll")]
    private static extern IntPtr OpenThread(uint access, bool inherit, uint id);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr h);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetThreadDescription(IntPtr thread, out IntPtr desc);

    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr p);

    [DllImport("ntdll.dll")]
    private static extern int NtQueryInformationThread(IntPtr thread, int cls, out IntPtr info, int len, IntPtr retLen);

    private const uint ThreadQueryInformation = 0x0040;
    private static Dictionary<int, (string key, ulong cycles)>? _threads0;
    private static List<(long b, long e, string name)>? _modules;

    /// <summary>Thread id -> (start module + description, cycles so far) for every live thread.</summary>
    private static Dictionary<int, (string key, ulong cycles)> ThreadCycleTable()
    {
        var p = System.Diagnostics.Process.GetCurrentProcess();
        _modules ??= p.Modules.Cast<System.Diagnostics.ProcessModule>()
            .Select(m => ((long)m.BaseAddress, (long)m.BaseAddress + m.ModuleMemorySize, m.ModuleName)).ToList();
        var t = new Dictionary<int, (string, ulong)>();
        foreach (System.Diagnostics.ProcessThread th in p.Threads)
        {
            IntPtr h = OpenThread(ThreadQueryInformation, false, (uint)th.Id);
            if (h == IntPtr.Zero) continue;
            try
            {
                QueryThreadCycleTime(h, out ulong c);
                string mod = "?";
                if (NtQueryInformationThread(h, 9, out IntPtr start, IntPtr.Size, IntPtr.Zero) == 0)
                {
                    long a = (long)start;
                    mod = _modules.FirstOrDefault(m => a >= m.b && a < m.e).name ?? "?";
                }

                string desc = "";
                if (GetThreadDescription(h, out IntPtr d) >= 0 && d != IntPtr.Zero)
                {
                    desc = Marshal.PtrToStringUni(d) ?? "";
                    LocalFree(d);
                }

                t[th.Id] = ($"{mod}|{desc}", c);
            }
            finally
            {
                CloseHandle(h);
            }
        }

        return t;
    }

    /// <summary>CPU ms per thread group (start module | description) since <paramref name="t0" />, and the count.</summary>
    private static JsonObject ThreadDelta(Dictionary<int, (string key, ulong cycles)>? t0)
    {
        var now = ThreadCycleTable();
        var groups = new Dictionary<string, (double ms, int n)>();
        foreach (var (id, (key, c)) in now)
        {
            ulong c0 = t0 != null && t0.TryGetValue(id, out var o) ? o.cycles : 0;
            var g = groups.GetValueOrDefault(key);
            groups[key] = (g.ms + (c - c0) / CyclesPerMs(), g.n + 1);
        }

        var j = new JsonObject { ["count"] = now.Count };
        foreach (var (k, (ms, n)) in groups.OrderByDescending(kv => kv.Value.ms))
        {
            j[k] = new JsonArray(Math.Round(ms, 1), n);
        }

        return j;
    }

    /// <summary>Every node under the root: how many, and which have per-frame callbacks switched on.</summary>
    private static JsonObject Census(SceneTree tree)
    {
        var byType = new Dictionary<string, int[]>();  // total, process, physics, internal process, internal physics, visible canvas item
        int total = 0, proc = 0, phys = 0, iproc = 0, iphys = 0;
        var branches = new Dictionary<string, int[]>();  // nodes under each depth-5 path: total, visible, any processing
        var stack = new Stack<(Node n, int depth, string branch)>();
        stack.Push((tree.Root, 0, ""));
        while (stack.Count > 0)
        {
            var (n, depth, branch) = stack.Pop();
            if (depth <= 8) branch = branch + "/" + n.Name;
            total++;
            if (!branches.TryGetValue(branch, out var b)) branches[branch] = b = new int[3];
            b[0]++;
            if (n is CanvasItem bci && bci.IsVisibleInTree()) b[1]++;
            if (n.CanProcess() && (n.IsProcessing() || n.IsProcessingInternal() || n.IsPhysicsProcessingInternal())) b[2]++;
            string key = n.GetType().Namespace?.StartsWith("Godot", StringComparison.Ordinal) == true
                ? n.GetClass()
                : n.GetType().Name;
            if (!byType.TryGetValue(key, out var c)) byType[key] = c = new int[6];
            c[0]++;
            bool can = n.CanProcess();
            if (can && n.IsProcessing()) { c[1]++; proc++; }
            if (can && n.IsPhysicsProcessing()) { c[2]++; phys++; }
            if (can && n.IsProcessingInternal()) { c[3]++; iproc++; }
            if (can && n.IsPhysicsProcessingInternal()) { c[4]++; iphys++; }
            if (n is CanvasItem ci && ci.IsVisibleInTree()) c[5]++;
            foreach (Node ch in n.GetChildren(includeInternal: true)) stack.Push((ch, depth + 1, branch));
        }

        var br = new JsonObject();
        foreach (var (k, v) in branches.OrderByDescending(kv => kv.Value[2]).ThenByDescending(kv => kv.Value[0]).Take(30))
        {
            br[k] = new JsonArray(v[0], v[1], v[2]);
        }

        var tweens = tree.GetProcessedTweens();
        var top = new JsonObject();
        foreach (var (k, c) in byType.OrderByDescending(kv => kv.Value[1] + kv.Value[2] + kv.Value[3] + kv.Value[4])
                     .ThenByDescending(kv => kv.Value[0]).Take(40))
        {
            top[k] = new JsonArray(c.Select(v => (JsonNode)v).ToArray());
        }

        return new JsonObject
        {
            ["nodes"] = total, ["process"] = proc, ["physics"] = phys, ["internal"] = iproc, ["internal_physics"] = iphys,
            ["tweens"] = tweens.Count,
            ["tweens_looping_forever"] = tweens.Count(t => t.IsValid() && t.GetLoopsLeft() == -1),
            ["branches"] = br,
            ["objects"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectCount),
            ["resources"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectResourceCount),
            ["orphans"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectOrphanNodeCount),
            ["by_type"] = top,
        };
    }
}
