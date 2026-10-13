// Profile only (job key "sample": true, with "profile"): a pool thread that, about once a millisecond
// while a fight runs, suspends the main thread just long enough to read its instruction pointer, and
// charges the main thread's CPU cycles since the previous sample to the module that pointer is in
// (the engine exe, a GDExtension, the .NET runtime, JIT-compiled managed code, ...). Nothing is read
// from the main thread's memory and nothing is called while it is suspended.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Threading;

namespace Spireline.Referee;

internal static partial class Lean
{
    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    [DllImport("kernel32.dll")]
    private static extern uint SuspendThread(IntPtr h);

    [DllImport("kernel32.dll")]
    private static extern uint ResumeThread(IntPtr h);

    [DllImport("kernel32.dll")]
    private static extern bool GetThreadContext(IntPtr h, IntPtr ctx);

    private static volatile bool _sampling;
    private static IntPtr _mainThread;
    private static readonly Dictionary<string, double> _byModule = new();
    private static readonly object _sampleLock = new();
    private static long _samples;

    private static void StartSampler()
    {
        // THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION
        _mainThread = OpenThread(0x0002 | 0x0008 | 0x0040, false, GetCurrentThreadId());
        _modules ??= System.Diagnostics.Process.GetCurrentProcess().Modules.Cast<System.Diagnostics.ProcessModule>()
            .Select(m => ((long)m.BaseAddress, (long)m.BaseAddress + m.ModuleMemorySize, m.ModuleName)).ToList();
        var mods = _modules.OrderBy(m => m.b).ToArray();
        var t = new Thread(() => SampleLoop(mods)) { IsBackground = true, Name = "spireline-sampler" };
        t.Start();
    }

    private static void SampleLoop((long b, long e, string name)[] mods)
    {
        // CONTEXT is 1232 bytes and must be 16-byte aligned; ContextFlags at 0x30, Rip at 0xF8.
        IntPtr raw = Marshal.AllocHGlobal(1232 + 16);
        IntPtr ctx = (IntPtr)(((long)raw + 15) & ~15L);
        ulong lastCycles = 0;
        long lastQpc = 0;
        var starts = mods.Select(m => m.b).ToArray();
        // Bind the P/Invokes once on harmless handles, so no stub is compiled while the main thread is held.
        SuspendThread(IntPtr.Zero);
        GetThreadContext(IntPtr.Zero, ctx);
        ResumeThread(IntPtr.Zero);
        while (true)
        {
            Thread.Sleep(1);
            if (!_sampling)
            {
                lastCycles = 0;
                continue;
            }

            Marshal.WriteInt32(ctx, 0x30, 0x100001);  // CONTEXT_CONTROL
            if (SuspendThread(_mainThread) == uint.MaxValue) continue;
            bool ok = GetThreadContext(_mainThread, ctx);
            ResumeThread(_mainThread);
            QueryThreadCycleTime(_mainThread, out ulong cycles);
            long qpc = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!ok) continue;
            long rip = Marshal.ReadInt64(ctx, 0xF8);
            // Count the sample only if the main thread ran for most of the interval since the last
            // one: then the module it is in now stands for that CPU time (a thread that just went to
            // sleep would otherwise charge its whole frame to the wait in ntdll).
            double ranMs = lastCycles == 0 ? 0 : (cycles - lastCycles) / CyclesPerMs();
            double wallMs = (qpc - lastQpc) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            lastCycles = cycles;
            lastQpc = qpc;
            if (ranMs < 0.8 * wallMs) continue;
            int i = Array.BinarySearch(starts, rip);
            if (i < 0) i = ~i - 1;
            string key = i >= 0 && rip < mods[i].e ? mods[i].name : "(jit code)";
            lock (_sampleLock)
            {
                _byModule[key] = _byModule.GetValueOrDefault(key) + 1;
                _samples++;
            }
        }
    }

    /// <summary>Share (%) of the main thread's busy samples by module since the last call, and the sample count.</summary>
    private static JsonObject TakeModuleSplit()
    {
        lock (_sampleLock)
        {
            var j = new JsonObject { ["samples"] = _samples };
            foreach (var (k, ms) in _byModule.OrderByDescending(kv => kv.Value).Take(14))
            {
                j[k] = Math.Round(100 * ms / Math.Max(1, _samples), 1);
            }

            _byModule.Clear();
            _samples = 0;
            return j;
        }
    }
}
