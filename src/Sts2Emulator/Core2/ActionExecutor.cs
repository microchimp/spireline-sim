// The executor and the pauses (docs/rebuild/TIER1_KERNEL.md section 5), in the shape of the game's ActionExecutor and
// HookPlayerChoiceContext. The executor takes the ready action from the queue set (ActionQueueSet.cs), runs its body,
// checks the win condition at the boundary, and goes on until nothing is ready. A body is the old engine's step for a
// player action, or a hook action's screen and resume record. What persists between steps is plain data in the
// combat's KernelState (queue, recorded hook actions); a body and its host live for one call.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>A kernel screen: a handler's choice request that the old engine can show (its PendingSelection).</summary>
public interface IKernelScreen
{
    PendingCardSelection Open(CombatState state);
}

/// <summary>The key of an answer to a kernel screen (HookChoiceReplay's keyed answers): the dispatcher, the listener,
/// the resume site, and how many screens of that triple the step had reached before it.</summary>
public readonly record struct HookAnswerKey(HookId Dispatcher, ListenerRef Who, string Site, int Seq);

/// <summary>A keyed answer and the hidden state of the combat that gave it (put back where the replay reaches it).</summary>
public sealed record KeyedAnswer(HookAnswerKey Key, int Answer, ScreenHiddenState Hidden);

/// <summary>Where a step's answers to kernel screens come from (the running step's replay record by default).</summary>
public interface IKernelAnswers
{
    bool TryAnswer(in HookAnswerKey key, out int answer);
}

/// <summary>What the executor runs a player action's body on (the run engine's step, or a test's).</summary>
public interface IActionHost
{
    /// <summary>The body of a player action: its step from the start, or (with <see cref="GameActionRec.Answer" />)
    /// the answer to the old screen it stopped on. True when it stopped on an old screen again.</summary>
    bool Run(KernelState k, GameActionRec a);

    /// <summary>game: CombatManager.IsInProgress.</summary>
    bool InProgress { get; }

    /// <summary>game: CombatManager.CheckWinCondition.</summary>
    void CheckWin(KernelState k);
}

/// <summary>
/// game: ActionExecutor.ExecuteActions. The game runs the ready action until it finishes or pauses for a choice, checks
/// the win condition after it unless it ends the turn or readies the enemy turn (F10's switch), and takes the next ready
/// action. A paused executor (game: Pause, CombatManager's) runs nothing.
/// </summary>
public static class ActionExecutor
{
    public static void Drain(KernelState k, IActionHost host)
    {
        var q = k.Queue!;
        while (!k.ExecutorPaused && q.GetReadyAction() is { } a)
        {
            Execute(k, host, a);
            if (host.InProgress && k.On(Switch.F10) && a.Kind is not (GameActionKind.EndTurn or GameActionKind.ReadyToBeginEnemyTurn))
            {
                k.WinChecks++;
                host.CheckWin(k);
            }
        }
    }

    /// <summary>game: GameAction.Execute (from waiting or ready to resume) to its finish or its pause.</summary>
    private static void Execute(KernelState k, IActionHost host, GameActionRec a)
    {
        var q = k.Queue!;
        q.Start(a);
        if (a.Kind == GameActionKind.Hook)
        {
            k.RunHookAction(a);
            return;
        }

        // game: EndPlayerTurnAction only marks the player ready; the turn's end runs after it, outside any action
        // (CombatManager.WaitForActionThenEndTurn). So it finishes before the old engine's step runs the turn end.
        if (a.Kind == GameActionKind.EndTurn)
        {
            q.Finish(a);
            host.Run(k, a);
            return;
        }

        uint id = a.Id!.Value;
        uint? first = a.FirstId;
        int owner = a.Owner;
        bool stopped = host.Run(k, a);
        // A replay restored the combat's kernel state to the step's start (HookChoiceReplay, Before.CopyTo): the action
        // there is the same one, executing; it keeps the id it was resumed under.
        var live = k.Queue!.Front(owner, GameActionState.Executing) ?? a;
        k.Queue.Carry(live, id, first);
        if (stopped)
            k.Queue.PauseForChoice(live);
        else
            k.Queue.Finish(live);
    }
}

/// <summary>
/// The turn start's flow (game: CombatManager.StartTurn's player part): the setup (energy reset, BeforeHandDraw, the
/// draw, AfterPlayerTurnStart) runs until it completes or pauses at a choice, WITHOUT a task assigned to its context
/// (WaitForPauseOrCompletionWithoutAssigningTask), so AfterSideTurnStart and the orb queue's turn start run next; the
/// auto-pre-play phase then assigns the setup's task to the context, which enqueues its hook action (type
/// CombatPlayPhaseOnly): the choice, the rest of the setup, then AfterAutoPrePlayPhaseEntered.
/// </summary>
public interface ITurnStartFlow
{
    /// <summary>SetupPlayerTurn up to its end or its first pause.</summary>
    HookStatus Setup(KernelState k);

    /// <summary>The rest of the setup after the paused listener's resume record ran (its dispatcher's Resume, then the
    /// setup's later parts); may pause again.</summary>
    HookStatus ResumeSetup(KernelState k, in HookStatus paused);

    /// <summary>Hook.AfterSideTurnStart.</summary>
    void SideStart(KernelState k);

    /// <summary>OrbQueue.AfterTurnStart, in its own context.</summary>
    void OrbStart(KernelState k);

    /// <summary>RunAutoPrePlayPhase's Hook.AfterAutoPrePlayPhaseEntered.</summary>
    void AutoPrePlay(KernelState k);
}
