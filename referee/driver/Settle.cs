// Settling (SPEED_PLAN.md step 3): when is the game ready for the next action and its state safe to
// snapshot? The old rule (Idle) waits for six frames in a row with nothing running and the same
// state signature. The "flags" rule reads the game's own turn bookkeeping instead and then waits
// one frame for deferred work (CombatStateTracker.cs ~:205-226 recalculates enchantment values one
// frame after any state change). A job picks the rule ("settle": "frames" | "flags") and can audit
// a share of fights ("settle_audit": 0..1): after the flag settle, the old wait runs too and the
// two snapshots (and the playable-card list) must agree; disagreements are written to the result.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>"frames": six stable frames (the old rule); "flags": the game's turn flags plus one frame.</summary>
    internal static string SettleMode = "frames";

    /// <summary>Share of fights (0..1, picked by seed hash) whose every flag settle is checked against the old rule.</summary>
    internal static double SettleAudit;

    /// <summary>Record a per-frame trace of each fight step (diagnostics only).</summary>
    internal static bool TraceFrames;

    /// <summary>Godot's default low-processor sleep, which pads every headless frame.</summary>
    private const int StockSleepUsec = 6900;

    private static bool _auditFight;
    private static JsonArray? _auditLog;
    private static JsonArray? _stepTrace;
    private static bool _combatEndedSeen;
    private static bool _subscribed;

    // CombatManager._inPlayerTurnSetup (CombatManager.cs ~:54): true while StartTurn sets up the player
    // turn, until it moves to the Play phase (~:583).
    private static System.Reflection.FieldInfo? _inPlayerTurnSetupField;

    private static bool InPlayerTurnSetup(CombatManager cm)
    {
        _inPlayerTurnSetupField ??= AccessTools.Field(typeof(CombatManager), "_inPlayerTurnSetup")
            ?? throw new MissingFieldException(nameof(CombatManager), "_inPlayerTurnSetup");
        return (bool)_inPlayerTurnSetupField.GetValue(cm)!;
    }

    /// <summary>Per fight: reset the end-of-combat latch and decide whether this fight is audited.</summary>
    private static void BeginSettleFight(string seed, JsonObject dump)
    {
        if (!_subscribed)
        {
            // Fired at the very end of EndCombatInternal (CombatManager.cs ~:1033) and by
            // ProcessPendingLoss (~:965): the combat's own wrap-up has finished.
            CombatManager.Instance.CombatEnded += _ => _combatEndedSeen = true;
            _subscribed = true;
        }

        _combatEndedSeen = false;
        uint h = unchecked((uint)StringHelper.GetDeterministicHashCode(seed));
        _auditFight = SettleMode == "flags" && SettleAudit > 0 && (h % 10000) < SettleAudit * 10000;
        _auditLog = null;
        if (_auditFight)
        {
            _auditLog = new JsonArray();
            dump["settle_audit"] = _auditLog;
        }
    }

    /// <summary>The game's own flags say the player may act now (or the combat is over and wrapped up).</summary>
    private static bool FlagsSettled(Player me)
    {
        var run = RunManager.Instance;
        if (run.ActionExecutor.IsRunning || !run.ActionQueueSet.IsEmpty)
        {
            return false;
        }

        var cm = CombatManager.Instance;
        if (!cm.IsInProgress)
        {
            // Either not started yet (StartCombatInternal ~:388-403) or over: only "over" counts.
            return _combatEndedSeen;
        }

        var combat = cm.DebugOnlyGetState();
        var pcs = me.PlayerCombatState;
        return combat != null && pcs != null
            && run.ActionQueueSynchronizer.CombatState == ActionSynchronizerCombatState.PlayPhase
            && combat.CurrentSide == CombatSide.Player
            && pcs.Phase == PlayerTurnPhase.Play
            && !InPlayerTurnSetup(cm)
            && !cm.IsPlayerReadyToEndTurn(me)
            && !cm.EndingPlayerTurnPhaseOne
            && !cm.EndingPlayerTurnPhaseTwo
            && !cm.PlayerActionsDisabled
            && !cm.IsPaused
            && !run.ActionExecutor.IsPaused;
    }

    /// <summary>Settle after a fight action (or room entry) by the job's rule; audit if this fight is sampled.</summary>
    private static async Task SettleStep(RunState state, Player me, int step)
    {
        if (TraceFrames && _stepTrace == null)
        {
            _stepTrace = new JsonArray();
        }

        if (SettleMode != "flags")
        {
            await Idle();
            return;
        }

        await SettleFlags(me);
        if (_auditFight)
        {
            string snap = Snapshot(state, me).ToJsonString() + "|" + PlayableSig(me);
            // The old rule as it was validated: six quiet frames at the stock headless frame pacing
            // (~6.9 ms each), so the audit also covers anything that waits on time, not frames.
            int sleep = OS.LowProcessorUsageModeSleepUsec;
            OS.LowProcessorUsageModeSleepUsec = Math.Max(sleep, StockSleepUsec);
            try
            {
                await Idle();
            }
            finally
            {
                OS.LowProcessorUsageModeSleepUsec = sleep;
            }

            string late = Snapshot(state, me).ToJsonString() + "|" + PlayableSig(me);
            if (snap != late)
            {
                _auditLog!.Add(new JsonObject { ["step"] = step, ["flags"] = snap, ["frames"] = late });
                GD.Print($"[referee] settle audit: step {step} differs after the six-frame wait");
            }
            else
            {
                _auditLog!.Add(step);
            }
        }
    }

    /// <summary>
    /// Flag settle: the flags must hold on two consecutive checks one frame apart with the same state
    /// signature (the frame lets deferred work run, and any change restarts the wait).
    /// </summary>
    private static async Task SettleFlags(Player me, double maxSeconds = 30)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        string? lastSig = null;
        bool first = true;
        int waited = 0, busyRun = 0;
        double nextReport = 5;
        int pacing = OS.LowProcessorUsageModeSleepUsec;
        try
        {
            while (clock.Elapsed.TotalSeconds < maxSeconds)
            {
                waited++;
                if (clock.Elapsed.TotalSeconds > nextReport)
                {
                    nextReport += 5;
                    GD.Print($"[referee] settle still waiting after {waited} frames: {TraceLine(me, FlagsSettled(me), "")}");
                }

                // With awaited tweens finished at once, a long busy run is game code waiting on time
                // (e.g. the Rolling Boulder visual rolls, then waits 1 s: NRollingBoulderVfx.cs ~:240-293).
                // Spinning unpadded frames through it burns CPU, so fall back to the stock frame
                // pacing until the step settles. Pacing never decides the settle, only its cost.
                if (busyRun == SlowPacingAfter && pacing < StockSleepUsec)
                {
                    OS.LowProcessorUsageModeSleepUsec = StockSleepUsec;
                }

                if (!first)
                {
                    await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
                    AnswerOpenScreens();
                }

                first = false;
                System.Threading.Interlocked.Exchange(ref _lastDriverFrame, (long)Engine.GetProcessFrames());
                bool ok = FlagsSettled(me);
                string sig = ok ? Signature(CombatManager.Instance.DebugOnlyGetState()) : "busy";
                TraceFrame(me, ok, sig);
                if (ok && sig == lastSig)
                {
                    return;
                }

                busyRun = ok ? 0 : busyRun + 1;
                lastSig = ok ? sig : null;
            }
        }
        finally
        {
            OS.LowProcessorUsageModeSleepUsec = pacing;
        }
    }

    /// <summary>Busy frames in a row after which a settle waits at stock frame pacing.</summary>
    private const int SlowPacingAfter = 20;

    private static string PlayableSig(Player me)
    {
        var hand = me.PlayerCombatState?.Hand.Cards;
        return hand == null ? "-" : string.Concat(hand.Select(c => c.CanPlay() ? '1' : '0'));
    }

    internal static void TraceNote(string note) => _stepTrace?.Add(note);

    private static void TraceFrame(Player me, bool ok, string sig)
    {
        _stepTrace?.Add(TraceLine(me, ok, sig));
    }

    private static string TraceLine(Player me, bool ok, string sig)
    {
        var run = RunManager.Instance;
        var cm = CombatManager.Instance;
        var pcs = me.PlayerCombatState;
        return ($"{Engine.GetProcessFrames()} {(ok ? "OK" : "..")} x={(run.ActionExecutor.IsRunning ? 1 : 0)}{(run.ActionQueueSet.IsEmpty ? 0 : 1)} "
            + $"a={run.ActionExecutor.CurrentlyRunningAction?.GetType().Name} s={run.ActionQueueSynchronizer.CombatState} ph={pcs?.Phase} "
            + $"side={cm.DebugOnlyGetState()?.CurrentSide} rte={(cm.IsInProgress && cm.DebugOnlyGetState() != null && cm.IsPlayerReadyToEndTurn(me) ? 1 : 0)} "
            + $"e1={(cm.EndingPlayerTurnPhaseOne ? 1 : 0)} e2={(cm.EndingPlayerTurnPhaseTwo ? 1 : 0)} dis={(cm.PlayerActionsDisabled ? 1 : 0)} "
            + $"setup={(InPlayerTurnSetup(cm) ? 1 : 0)} inp={(cm.IsInProgress ? 1 : 0)} end={(_combatEndedSeen ? 1 : 0)} sig={(uint)sig.GetHashCode():x8} ap={(run.ActionExecutor.IsPaused ? 1 : 0)} cp={(cm.IsPaused ? 1 : 0)}");
    }
}
