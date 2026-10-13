// Stall detection that holds on a starved machine. The servers run at BelowNormal priority on a shared
// PC, and under load a fight's wall time stretches far beyond its CPU time (n04b canary: 6 s of CPU in
// 45 s, one run start alone 21 s). Wall-clock limits alone then kill healthy servers. So every limit here
// is spent only when BOTH its seconds and a number of game frames have passed: a game that waits on
// something that never comes keeps spinning frames (Engine.MaxFps 0) and is caught on time, while a slow
// game advances few frames (a run start takes about 30, a fight step about 15) and is left alone.

using System;
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
    private static long _lastFrameTick = System.Environment.TickCount64;
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
        Interlocked.Exchange(ref _lastFrameTick, now);
        WriteStatus(now);
    }

    /// <summary>
    /// Watchdog on a pool thread, armed by <see cref="ArmWatchdog" /> once startup is done (boot has no
    /// progress to report; serve.py's --boot-timeout and run_referee's --timeout cover it). It ends the
    /// process when the main loop has stopped (no frame for 2 x stall) or runs without progress (no
    /// heartbeat for stall seconds and <see cref="BudgetFrames" /> frames): a load or a screen nothing answers.
    /// </summary>
    private static void StartWatchdog(int stallMs)
    {
        ((SceneTree)Engine.GetMainLoop()).ProcessFrame += OnFrame;
        _ = Task.Run(async () =>
        {
            while (true)
            {
                await Task.Delay(1000);
                if (!_armed) continue;
                long now = System.Environment.TickCount64;
                long sinceFrame = now - Interlocked.Read(ref _lastFrameTick);
                long sinceProgress = now - Interlocked.Read(ref _lastProgress);
                long frames = Interlocked.Read(ref _frame) - Interlocked.Read(ref _progressFrame);
                string? why = sinceFrame > 2L * stallMs ? $"no frame for {sinceFrame / 1000} s"
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
