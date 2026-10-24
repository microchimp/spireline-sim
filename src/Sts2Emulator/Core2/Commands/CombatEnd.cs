// game: CombatManager.CheckWinCondition, IsEnding, IsOverOrEnding, IsInProgress, IsAboutToLose, LoseCombat,
// ProcessPendingLoss and EndCombatInternal's start, over the kernel's per-combat progress (docs/rebuild/MIGRATION_F08.md
// section 3 item 7 and section 2's F17 hand-off). Round R3b: the check is called at the game's points that exist in Core2
// (the executor's boundary through IActionHost.CheckWin, the turn start's two points through ITurnStartFlow.CheckWin) and
// exposed for the turn-flow points the old CombatEngine.EndTurn still owns (R4 reroutes them). F17 (AfterCombatEnd,
// AfterCombatVictoryEarly, AfterCombatVictory) is not migrated: the combat's end hands off to the old engine's
// RunEngine.FinishCombat through ICombatEndHost, once. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>Where the combat's end goes once Core2 decided it (the F17 hand-off).</summary>
public interface ICombatEndHost
{
    /// <summary>
    /// game: EndCombatInternal after IsInProgress went false (a victory), or ProcessPendingLoss (a loss). The old engine's
    /// end of a fight runs here, once per combat, with the arguments the old in-step caller passed: the run engine's
    /// FinishCombat (victory heal, rewards, run counters, SyncAfterCombat) for a victory and, as the old engine does, for a
    /// loss too.
    /// </summary>
    void EndCombat(KernelState k, bool playerWon);
}

/// <summary>The combat's progress as CombatManager keeps it: per combat, copied with the kernel state.</summary>
public sealed class CombatProgress
{
    /// <summary>game: CombatManager.IsInProgress (true from the combat's start until its end begins).</summary>
    public bool InProgress { get; internal set; } = true;

    /// <summary>game: CombatManager._pendingLoss is set (IsAboutToLose).</summary>
    public bool PendingLoss { get; internal set; }

    /// <summary>The combat's result once it ended: true won, false lost, null not ended.</summary>
    public bool? PlayerWon { get; internal set; }

    /// <summary>How many times the end was handed off (ICombatEndHost.EndCombat). At most one.</summary>
    public int HandOffs { get; internal set; }

    /// <summary>How many CheckWinCondition calls ran, at every point.</summary>
    public int Checks { get; internal set; }

    internal CombatProgress Copy() => new()
    {
        InProgress = InProgress, PendingLoss = PendingLoss, PlayerWon = PlayerWon, HandOffs = HandOffs, Checks = Checks,
    };
}

public static class CombatEnd
{
    /// <summary>game: CombatManager.IsInProgress.</summary>
    public static bool IsInProgress(KernelState k) => k.Progress.InProgress;

    /// <summary>game: CombatManager.IsAboutToLose.</summary>
    public static bool IsAboutToLose(KernelState k) => k.Progress.PendingLoss;

    /// <summary>
    /// game: CombatManager.IsEnding. Not in progress: false. A pending loss: true. A living primary enemy in the combat:
    /// false. ShouldStopCombatFromEnding (any listener holds it open): false. Else true.
    /// </summary>
    public static bool IsEnding(KernelState k)
    {
        var p = k.Progress;
        if (!p.InProgress)
            return false;
        if (p.PendingLoss)
            return true;
        var w = k.World ?? throw new InvalidOperationException("CombatEnd needs a switched-on kernel state");
        var s = w.Combat ?? throw new InvalidOperationException("CombatEnd outside a combat");
        if (OldCreatures.Enemies(w).Any(e => !OldCreatures.IsDead(s, e) && OldCreatures.IsPrimaryEnemy(w, e)))
            return false;
        if (Hooks.ShouldStopCombatFromEnding(k, new ShouldStopCombatFromEndingArgs(), out _))
            return false;
        return true;
    }

    /// <summary>game: CombatManager.IsOverOrEnding.</summary>
    public static bool IsOverOrEnding(KernelState k) => IsEnding(k) || !IsInProgress(k);

    /// <summary>
    /// game: CombatManager.LoseCombat: marks the loss pending (once); it is processed at the next CheckWinCondition. The
    /// player-death path reaches it from CreatureCmd.Kill only while the combat is in progress. Nothing else runs here: no
    /// hook, no end of the combat (IsInProgress stays true, IsEnding and IsOverOrEnding turn true, so the gated commands
    /// stop), no multiplayer HandlePlayerDeath.
    /// </summary>
    public static void LoseCombat(KernelState k) => k.Progress.PendingLoss = true;

    // construct: game CombatManager.CheckWinCondition
    //   if                 ProcessPendingLoss
    //   if-cond/if-cond    Hook.ShouldStopCombatFromEnding
    //   if                 EndCombatInternal
    // end construct
    /// <summary>
    /// game: CombatManager.CheckWinCondition. A pending loss is processed first (the loss wins over a victory at the same
    /// point); else, the combat ending (IsEnding), EndCombatInternal. True when the combat ended here.
    /// </summary>
    public static bool CheckWinCondition(KernelState k, ICombatEndHost host)
    {
        k.Progress.Checks++;
        if (k.Progress.PendingLoss)
        {
            ProcessPendingLoss(k, host);
            return true;
        }

        if (IsEnding(k))
        {
            EndCombatInternal(k, host);
            return true;
        }

        return false;
    }

    /// <summary>
    /// game: CombatManager.ProcessPendingLoss: the pending loss cleared, IsInProgress false, the CombatEnded event. No
    /// AfterCombatEnd or AfterCombatVictory runs in the game. The old engine finishes a lost fight through FinishCombat
    /// (its AfterCombatEnd bodies, the sync back to the run), so the hand-off runs it, as now; F17 decides. The queue's
    /// CombatEnded (the game's at the run's reset after the loss) follows: no combat action runs after it.
    /// </summary>
    private static void ProcessPendingLoss(KernelState k, ICombatEndHost host)
    {
        var p = k.Progress;
        p.PendingLoss = false;
        p.InProgress = false;
        HandOff(k, host, playerWon: false);
    }

    /// <summary>
    /// game: CombatManager.EndCombatInternal. IsInProgress goes false first, so no action and no win check runs after it;
    /// then the F17 part (the revive before the end, AfterCombatEnd, AfterCombatVictory, the rewards) is the old engine's
    /// FinishCombat, handed off once; then ActionQueueSynchronizer's NotInCombat: ActionQueueSet.CombatEnded cancels every
    /// combat action that has not started (hook actions waiting or gathering a choice).
    /// </summary>
    private static void EndCombatInternal(KernelState k, ICombatEndHost host)
    {
        k.Progress.InProgress = false;
        HandOff(k, host, playerWon: true);
    }

    private static void HandOff(KernelState k, ICombatEndHost host, bool playerWon)
    {
        var p = k.Progress;
        if (p.HandOffs > 0)
            throw new InvalidOperationException("the combat's end was handed off twice (FinishCombat runs once per combat)");
        p.HandOffs++;
        p.PlayerWon = playerWon;
        host.EndCombat(k, playerWon);
        k.Queue?.CombatEnded();
    }

    // ------------------------------------------------------------------------------- the game's check points

    /// <summary>game: ActionExecutor.ExecuteActions after an action (not EndPlayerTurnAction or
    /// ReadyToBeginEnemyTurnAction; the executor asks IsInProgress first).</summary>
    public static bool AfterAction(KernelState k, ICombatEndHost host) => CheckWinCondition(k, host);

    /// <summary>game: CombatManager.StartTurn, the player side, after the auto-pre-play phase (and, a setup paused,
    /// before the executor runs its choice): true when the combat goes on (the executor is unpaused only then).</summary>
    public static bool StartPlayerTurn(KernelState k, ICombatEndHost host)
    {
        CheckWinCondition(k, host);
        return IsInProgress(k);
    }

    /// <summary>game: CombatManager.StartTurn, the enemy side: true when ExecuteEnemyTurn runs.</summary>
    public static bool StartEnemyTurn(KernelState k, ICombatEndHost host)
    {
        CheckWinCondition(k, host);
        return IsInProgress(k);
    }

    /// <summary>game: CombatManager.ExecuteEnemyTurn after each enemy's turn: true when the next enemy takes its turn.</summary>
    public static bool AfterEnemyTakesTurn(KernelState k, ICombatEndHost host)
    {
        CheckWinCondition(k, host);
        return IsInProgress(k);
    }

    /// <summary>game: CombatManager.EndEnemyTurn after EndEnemyTurnInternal: true when the sides switch and the player's
    /// turn starts (the game asks !IsEnding, not IsInProgress; a combat ended here is not ending any more, and StartTurn
    /// then returns at once on !IsInProgress).</summary>
    public static bool EndEnemyTurn(KernelState k, ICombatEndHost host)
    {
        CheckWinCondition(k, host);
        return !IsEnding(k);
    }

    /// <summary>game: CombatManager.EndPlayerTurnPhaseOneInternal after BeforeTurnEnd: true when the combat ended (the
    /// turn's end, DoTurnEnd and BeforeFlush, do not run).</summary>
    public static bool AfterBeforeTurnEnd(KernelState k, ICombatEndHost host) => CheckWinCondition(k, host);

    /// <summary>game: CombatManager.EndPlayerTurnPhaseOneInternal at its end, after BeforeFlush.</summary>
    public static bool EndPlayerTurnPhaseOne(KernelState k, ICombatEndHost host) => CheckWinCondition(k, host);

    // ------------------------------------------------------------------------------- R4's bridge

    /// <summary>
    /// The old step's in-step win check (CombatEngine.CheckWinCondition's callers) found the fight over, with the F10
    /// switch on: until R4 reroutes those sites to the points above and the player's death through CreatureCmd.Kill, the
    /// old check point stands for the game's point it copies. A loss is the old death answered (LoseCombat), then
    /// CheckWinCondition runs there. R4 deletes this.
    /// </summary>
    public static void OldStepEnded(KernelState k, bool playerWon, ICombatEndHost host)
    {
        if (!IsInProgress(k))
            return;
        if (!playerWon)
            LoseCombat(k);
        CheckWinCondition(k, host);
    }
}
