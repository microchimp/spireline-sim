// Where an EmuDump call spends its time: EMUDUMP_PROFILE=1 adds up the wall time and the main
// thread's CPU cycles of each phase (Prof.Scope; cycles do not count time the thread waited for a
// core, so they hold on a busy PC) and prints the totals, the process's CPU time and the time to
// Main on stderr at exit. Off by default; stdout is the same either way.
using System.Diagnostics;
using System.Runtime.InteropServices;

static class Prof
{
    [DllImport("kernel32.dll")]
    static extern bool QueryThreadCycleTime(nint thread, out ulong cycles);

    [DllImport("kernel32.dll")]
    static extern nint GetCurrentThread();

    static ulong Cycles() => OperatingSystem.IsWindows() && QueryThreadCycleTime(GetCurrentThread(), out ulong c) ? c : 0;

    internal static readonly bool On = Environment.GetEnvironmentVariable("EMUDUMP_PROFILE") == "1";
    static readonly Dictionary<string, (long Ticks, ulong Cycles, int N)> Acc = new();
    static TimeSpan _cpuAtMain;
    static double _msToMain;

    internal readonly struct Span : IDisposable
    {
        readonly string? _name;
        readonly long _t0;
        readonly ulong _c0;

        internal Span(string name)
        {
            _name = name;
            _t0 = Stopwatch.GetTimestamp();
            _c0 = Cycles();
        }

        public void Dispose()
        {
            if (_name == null) return;
            ulong dc = Cycles() - _c0;
            long dt = Stopwatch.GetTimestamp() - _t0;
            var (t, c, n) = Acc.GetValueOrDefault(_name);
            Acc[_name] = (t + dt, c + dc, n + 1);
        }
    }

    /// <summary>Times the enclosing <c>using</c> block under <paramref name="name"/> (nested names add up separately).</summary>
    internal static Span Scope(string name) => On ? new Span(name) : default;

    internal static void Start()
    {
        if (!On) return;
        var p = Process.GetCurrentProcess();
        _cpuAtMain = p.TotalProcessorTime;
        _msToMain = (DateTime.Now - p.StartTime).TotalMilliseconds;
    }

    internal static void Report(string mode, int fights)
    {
        if (!On) return;
        var p = Process.GetCurrentProcess();
        double cpu = p.TotalProcessorTime.TotalMilliseconds;
        var err = Console.Error;
        err.WriteLine($"profile {mode}: {fights} fights, process cpu {cpu:F0} ms ({_cpuAtMain.TotalMilliseconds:F0} before Main, "
            + $"{_msToMain:F0} ms wall to Main), gc {GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}");
        err.WriteLine($"  {"phase",-24} {"wall ms",9} {"Mcycles",9} {"calls",7} {"Mcyc/fight",10}");
        foreach (var (name, (ticks, cycles, n)) in Acc.OrderByDescending(kv => kv.Value.Cycles))
        {
            double ms = ticks * 1000.0 / Stopwatch.Frequency, mc = cycles / 1e6;
            err.WriteLine($"  {name,-24} {ms,9:F1} {mc,9:F1} {n,7} {mc / Math.Max(1, fights),10:F2}");
        }
    }
}
