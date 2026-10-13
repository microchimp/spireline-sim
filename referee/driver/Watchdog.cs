// Stall detection that holds on a starved machine. The servers run at BelowNormal priority on a shared
// PC, and under load a fight's wall time stretches far beyond its CPU time (n04b canary: 6 s of CPU in
// 45 s, one run start alone 21 s; under a full test suite one frame took over 40 s). Wall-clock limits
// alone then kill healthy servers. So a limit on progress is spent only when BOTH its seconds and a number
// of game frames have passed: a game that waits on something that never comes keeps spinning frames
// (Engine.MaxFps 0) and is caught on time, while a slow game advances few frames (a run start takes about
// 30, a fight step about 15) and is left alone. And a frame that does not end counts as stuck only while
// the main thread is blocked (scheduler state Wait) or keeps burning CPU, not while it is merely waiting
// for a CPU (Ready): starvation is not a hang.

using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>Frames a budget needs on top of its seconds: ten times the most any phase was seen to take.</summary>
    internal const ulong BudgetFrames = 600;

    private static long _lastProgress = System.Environment.TickCount64;
    private static long _progressFrame;
    private static long _frame;
    private static volatile bool _armed;

    /// <summary>Marks progress for the watchdog: called on every action, not just every fight.</summary>
    internal static void Heartbeat()
    {
        long now = System.Environment.TickCount64;
        Interlocked.Exchange(ref _lastProgress, now);
        Interlocked.Exchange(ref _progressFrame, (long)Engine.GetProcessFrames());
        WriteStatus(now);
    }

    /// <summary>
    /// Every process frame (SceneTree.ProcessFrame): the main loop is alive. Also refreshes the server's
    /// status file (at most once a second), so the dispatcher's heartbeat age means "main loop stopped",
    /// not "a phase is slow"; whether a running game is stuck is this watchdog's call.
    /// </summary>
    private static void OnFrame()
    {
        long now = System.Environment.TickCount64;
        Interlocked.Exchange(ref _frame, (long)Engine.GetProcessFrames());
        WriteStatus(now);
    }

    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    private static int _mainThreadId;

    /// <summary>The main thread's scheduler state and CPU time (System.Diagnostics.ProcessThread), or null if unreadable.</summary>
    private static (bool blocked, double cpuMs)? MainThread()
    {
        try
        {
            using var p = System.Diagnostics.Process.GetCurrentProcess();
            foreach (System.Diagnostics.ProcessThread t in p.Threads)
            {
                if (t.Id == _mainThreadId)
                {
                    return (t.ThreadState == System.Diagnostics.ThreadState.Wait, t.TotalProcessorTime.TotalMilliseconds);
                }
            }
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or System.ComponentModel.Win32Exception)
        {
            // judged as blocked below, as before this check existed
        }

        return null;
    }

    /// <summary>
    /// Watchdog on a pool thread, armed by <see cref="ArmWatchdog" /> once startup is done (boot has no
    /// progress to report; serve.py's --boot-timeout and run_referee's --timeout cover it). Called on the
    /// main thread. It ends the process when the game runs without progress (no heartbeat for stall
    /// seconds and <see cref="BudgetFrames" /> frames: a screen nothing answers), or when no frame ends
    /// while the main thread is blocked for 2 x stall seconds or burns 2 x stall seconds of CPU (a hang).
    /// </summary>
    private static void StartWatchdog(int stallMs)
    {
        _mainThreadId = (int)GetCurrentThreadId();
        ((SceneTree)Engine.GetMainLoop()).ProcessFrame += OnFrame;
        _ = Task.Run(async () =>
        {
            long seenFrame = -1, blockedMs = 0, prev = System.Environment.TickCount64;
            double cpuAtFrame = 0;
            while (true)
            {
                await Task.Delay(1000);
                long now = System.Environment.TickCount64;
                long step = now - prev;
                prev = now;
                if (!_armed) continue;
                long frame = Interlocked.Read(ref _frame);
                var main = MainThread();
                if (frame != seenFrame)
                {
                    seenFrame = frame;
                    blockedMs = 0;
                    cpuAtFrame = main?.cpuMs ?? 0;
                }
                else if (main is not { blocked: false })
                {
                    blockedMs += step;
                }

                double spunMs = (main?.cpuMs ?? 0) - cpuAtFrame;
                long sinceProgress = now - Interlocked.Read(ref _lastProgress);
                long frames = frame - Interlocked.Read(ref _progressFrame);
                string? why = blockedMs > 2L * stallMs ? $"no frame, main thread blocked for {blockedMs / 1000} s"
                    : spunMs > 2.0 * stallMs ? $"no frame, main thread ran {spunMs / 1000:F0} s of CPU"
                    : sinceProgress > stallMs && frames > (long)BudgetFrames ? $"no progress for {sinceProgress / 1000} s and {frames} frames"
                    : null;
                if (why != null)
                {
                    GD.Print($"[referee] watchdog: {why}, exiting after {_done} fights");
                    System.Environment.Exit(3);
                }
            }
        });
    }

    /// <summary>Startup is over: from here on the watchdog judges progress.</summary>
    private static void ArmWatchdog()
    {
        Heartbeat();
        _armed = true;
    }

    /// <summary>
    /// A wait limit on the main thread, spent only when both <paramref name="seconds" /> and
    /// <see cref="BudgetFrames" /> frames have passed since it was made (see the file comment).
    /// </summary>
    internal readonly struct Budget
    {
        private readonly long _t0;
        private readonly ulong _f0;
        private readonly long _ms;

        internal Budget(double seconds)
        {
            _t0 = System.Environment.TickCount64;
            _f0 = Engine.GetProcessFrames();
            _ms = (long)(seconds * 1000);
        }

        internal bool Spent => System.Environment.TickCount64 - _t0 > _ms && Engine.GetProcessFrames() - _f0 > BudgetFrames;
    }
}
