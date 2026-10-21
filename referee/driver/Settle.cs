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

    /// <summary>
    /// The combat's own turn flow is between the player's phases: the player's turn start is still being
    /// set up (CombatManager.StartTurn: draw, AfterPlayerTurnStart, auto pre-play; the flag is cleared only
    /// after the synchronizer reaches PlayPhase), or the player's end of turn has begun (ready to end,
    /// EndingPlayerTurnPhaseOne/Two). The six-frame Idle shares this with the flag settle: without it a
    /// record could land while a turn-start hook still ran, or in the gap between an end-turn action
    /// finishing and the end-of-turn flow marking its phases (the "GAME RECORD UNSETTLED" records).
    /// Reads a missing flag field (an older build) as false, so the frames rule still runs there.
    /// </summary>
    private static bool TurnFlowBusy(CombatManager cm, Player? me)
    {
        _inPlayerTurnSetupField ??= AccessTools.Field(typeof(CombatManager), "_inPlayerTurnSetup");
        return cm.EndingPlayerTurnPhaseOne || cm.EndingPlayerTurnPhaseTwo
            || (_inPlayerTurnSetupField != null && InPlayerTurnSetup(cm))
            || (me != null && cm.IsPlayerReadyToEndTurn(me));
    }

    /// <summary>
    /// Once at job start: the flag settle and the frame trace read CombatManager._inPlayerTurnSetup. A game
    /// without it ends the job here (exit 1, one error line) instead of failing every fight.
    /// </summary>
    private static void CheckSettleFields()
    {
        if (SettleMode != "flags" && !TraceFrames)
        {
            return;
        }

        var f = AccessTools.Field(typeof(CombatManager), "_inPlayerTurnSetup");
        if (f == null || f.FieldType != typeof(bool))
        {
            throw new InvalidOperationException("settle \"flags\" (and trace_frames) need the bool field CombatManager._inPlayerTurnSetup, "
                + "which this game build does not have; run with settle \"frames\" (--old-settle) or update Settle.cs for this build");
        }

        _inPlayerTurnSetupField = f;
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
        // InPlayPhase (Combat.cs) is the one definition of "the synchronizer takes plays and the turn
        // is in its Play phase"; the flag settle adds the rest of the turn flags on top of it.
        return combat != null && pcs != null
            && InPlayPhase(combat)
            && combat.CurrentSide == CombatSide.Player
            && !TurnFlowBusy(cm, me)
            && !cm.PlayerActionsDisabled
            && !cm.IsPaused
            && !run.ActionExecutor.IsPaused;
    }

    /// <summary>Settle after a fight action (or room entry) by the job's rule; audit if this fight is sampled.</summary>
    /// <returns>false if the game did not settle in time (Idle's contract: the fight ends with idle_timeout).</returns>
    private static async Task<bool> SettleStep(RunState state, Player me, int step)
    {
        if (TraceFrames && _stepTrace == null)
        {
            _stepTrace = new JsonArray();
        }

        if (SettleMode != "flags")
        {
            return await Idle();
        }

        if (!await SettleFlags(me))
        {
            return false;
        }

        if (_auditFight)
        {
            string snap = Snapshot(state, me).ToJsonString() + "|" + PlayableSig(me);
            // The old rule as it was validated: six quiet frames at the stock headless frame pacing
            // (~6.9 ms each), so the audit also covers anything that waits on time, not frames.
            int sleep = OS.LowProcessorUsageModeSleepUsec;
            OS.LowProcessorUsageModeSleepUsec = Math.Max(sleep, StockSleepUsec);
            bool ok;
            try
            {
                ok = await Idle();
            }
            finally
            {
                OS.LowProcessorUsageModeSleepUsec = sleep;
            }

            string late = Snapshot(state, me).ToJsonString() + "|" + PlayableSig(me);
            if (!ok)
            {
                // The old rule never settled where the flags did: a disagreement too.
                _auditLog!.Add(new JsonObject { ["step"] = step, ["flags"] = snap, ["frames"] = "idle_timeout" });
                GD.Print($"[referee] settle audit: step {step} timed out in the six-frame wait");
            }
            else if (snap != late)
            {
                _auditLog!.Add(new JsonObject { ["step"] = step, ["flags"] = snap, ["frames"] = late });
                GD.Print($"[referee] settle audit: step {step} differs after the six-frame wait");
            }
            else
            {
                _auditLog!.Add(step);
            }
        }

        return true;
    }

    /// <summary>
    /// Flag settle: the flags must hold on two consecutive checks one frame apart with the same state
    /// signature (the frame lets deferred work run, and any change restarts the wait).
    /// </summary>
    /// <returns>true once the flags hold; false if they do not within maxSeconds (default <see cref="IdleTimeoutS" />,
    /// job key idle_timeout_s) and <see cref="BudgetFrames" /> frames (Watchdog.cs: a starved server is slow,
    /// not stuck; kept below the watchdog's stall_s).</returns>
    private static async Task<bool> SettleFlags(Player me, double? maxSeconds = null)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();  // progress reports only; the limit is the budget
        var budget = new Budget(maxSeconds ?? IdleTimeoutS);
        int quiet = 0;  // as Idle's stable count, for the frame-cost profile (Lean.QuietFrames)
        string? lastSig = null;
        bool first = true;
        int waited = 0, busyRun = 0;
        double nextReport = 5;
        int pacing = OS.LowProcessorUsageModeSleepUsec;
        try
        {
            while (!budget.Spent)
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
                // A fallback model seen (Nodeless.MarkFallback) ends a nodeless attempt here, as in Idle():
                // played on, its node-needing code (e.g. RollingBoulderPower.AfterPlayerTurnStart) throws
                // out of the combat's turn flow and leaves CombatManager.EndingPlayerTurnPhaseTwo set.
                Nodeless.ThrowIfFallback();
                System.Threading.Interlocked.Exchange(ref _lastDriverFrame, (long)Engine.GetProcessFrames());
                bool ok = FlagsSettled(me);
                string sig = ok ? Signature(CombatManager.Instance.DebugOnlyGetState()) : "busy";
                TraceFrame(me, ok, sig);
                quiet = ok && sig == lastSig ? quiet + 1 : 0;
                Lean.QuietFrames = quiet;
                if (ok && sig == lastSig)
                {
                    return true;
                }

                busyRun = ok ? 0 : busyRun + 1;
                lastSig = ok ? sig : null;
            }
        }
        finally
        {
            OS.LowProcessorUsageModeSleepUsec = pacing;
        }

        return false;
    }

    /// <summary>
    /// What a fight left behind after its reset that a freshly started game does not have, or null.
    /// The flags the settle reads (FlagsSettled) are set and cleared by the combat's own turn flow:
    /// EndingPlayerTurnPhaseOne/Two (CombatManager EndPlayerTurnPhaseOne, AfterAllPlayersReadyToBeginEnemyTurn).
    /// A flow cut short (an exception thrown out of it, or its combat cancelled mid-transition when the
    /// driver breaks a fight off) never clears them, and CombatManager.Reset does not either, so every
    /// later fight in this process would wait for them until "idle_timeout entering the room". Only
    /// NMultiplayerTest reads them in the game; the process is retired instead of patched (Server.cs).
    /// </summary>
    internal static string? StaleState()
    {
        var cm = CombatManager.Instance;
        var stale = new List<string>();
        if (cm.IsInProgress) stale.Add("combat still in progress");
        if (cm.EndingPlayerTurnPhaseOne) stale.Add("EndingPlayerTurnPhaseOne");
        if (cm.EndingPlayerTurnPhaseTwo) stale.Add("EndingPlayerTurnPhaseTwo");
        if (_inPlayerTurnSetupField != null && InPlayerTurnSetup(cm)) stale.Add("_inPlayerTurnSetup");
        return stale.Count == 0 ? null : string.Join(", ", stale);
    }

    /// <summary>Busy frames in a row after which a settle waits at stock frame pacing.</summary>
    private const int SlowPacingAfter = 20;

    /// <summary>One line of the game's turn flags for an idle_timeout message: what the wait was stuck on.</summary>
    internal static string WaitingOn(Player? me)
    {
        try
        {
            return TraceLine(me!, false, "");
        }
        catch (Exception ex)
        {
            return "state unavailable: " + ex.GetType().Name;
        }
    }

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
            + $"setup={(_inPlayerTurnSetupField != null && InPlayerTurnSetup(cm) ? 1 : 0)} inp={(cm.IsInProgress ? 1 : 0)} end={(_combatEndedSeen ? 1 : 0)} sig={(uint)sig.GetHashCode():x8} ap={(run.ActionExecutor.IsPaused ? 1 : 0)} cp={(cm.IsPaused ? 1 : 0)}");
    }
}
