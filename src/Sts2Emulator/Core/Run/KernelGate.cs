namespace Sts2Emulator.Core.Run;

/// <summary>
/// The kernel's entry gate for one combat step (TIER1_KERNEL.md section 5, NATIVE_KERNEL.md section 5): with a switch on, a
/// player action is enqueued and the executor drains the queue, the action's body the old engine's step; an answer to an old
/// screen resumes the action gathering it (a new id; its body replays the step with its keyed answers); an answer to a
/// kernel screen between actions runs that hook action's resume record. <see cref="RunEngine.Step" /> (the run's fight),
/// <see cref="RunEngine.StepCombat" /> (EmuDump's re-anchor) and the native combat handles all step through
/// <see cref="Run" />, so they are one code path. With no switch on, <see cref="Step" /> is <see cref="CombatEngine.Step" />.
/// </summary>
public static class KernelGate
{
    /// <summary>The guard's message: a switch is on and the combat has no KernelState.</summary>
    public const string Unbound =
        "kernel guard: a switch is on and the combat about to be stepped has no KernelState (bind it where the combat was built)";

    /// <summary>
    /// A combat stepped on its own, outside a run's active combat (a native combat handle, EmuDump's re-anchor), on the
    /// caller's stream. With a switch on the combat must carry its KernelState (throws otherwise) and a combat that ends
    /// here ends as the step's terminal result, there being no run to finish.
    /// </summary>
    public static StepResult Step(Core2.Kernel? kernel, CombatState combat, Random rng, int action, int target = -1)
    {
        if (kernel is not { AnyOn: true })
        {
            return CombatEngine.Step(combat, action, rng, target);
        }

        var k = combat.Kernel ?? throw new InvalidOperationException(Unbound);
        var host = new BareHost(combat, rng);
        Run(combat, k, host, action, target);
        return host.Result ?? StepResult.Invalid;
    }

    /// <summary>
    /// A combat read from a document stopped in the middle of a step (its <c>stopped_step</c>), with a fresh KernelState
    /// bound (a switch is on). A document carries no queue, so what the queue held at the screen is put back as the game
    /// holds it: the step's start gets a KernelState of its own (the replay restores the step from there), and for a card
    /// play or a potion that action is enqueued, executing in the step's start and gathering the player's choice at the
    /// screen (game: PauseActionForPlayerChoice), as the executor leaves it (<see cref="Core2.ActionExecutor" />). The turn's
    /// end holds none (EndPlayerTurnAction finishes before the turn end runs), so its screen's answer is the old step's, as
    /// on a live combat. A combat that is not on a stopped step is left as it is.
    /// </summary>
    internal static void PausedFromDocument(CombatState combat)
    {
        if (combat.Kernel is not { AnyOn: true } k || combat.PendingSelection?.Replay is not { } stopped)
        {
            return;
        }

        var before = stopped.Before;
        var start = before.Kernel ?? k.Shared.Bind(before);
        if (!k.Queue!.IsEmpty || !start.Queue!.IsEmpty)
        {
            throw new InvalidOperationException("a document's fight holds a kernel queue before its paused action is put back");
        }

        int action = stopped.Action;
        if (action == before.Hand.Count)
        {
            return;
        }

        var kind = action < before.Hand.Count ? Core2.GameActionKind.PlayCard : Core2.GameActionKind.UsePotion;
        var executing = start.Queue.Enqueue(new Core2.GameActionRec(kind, Core2.GameActionRec.TypeOf(kind)) { Action = action, Target = stopped.TargetEnemyIndex });
        start.Queue.Start(executing);
        var paused = k.Queue.Enqueue(new Core2.GameActionRec(kind, Core2.GameActionRec.TypeOf(kind)) { Action = action, Target = stopped.TargetEnemyIndex });
        k.Queue.Start(paused);
        k.Queue.PauseForChoice(paused);
    }

    /// <summary>The queue run of one step on <paramref name="host" /> (only with a switch on).</summary>
    /// <remarks>
    /// An answer to a screen whose replay is refused (<see cref="StoppedStepMismatch" />) gave no answer: the combat goes back
    /// on its screen (<c>CombatEngine.Step</c>), and its kernel state goes back with it, as it stood at the screen -- the
    /// action still gathering the player's choice under its own id, the id counter where it was. The old engine's restore
    /// alone put back the copy it took inside the action's body, where the executor had already resumed the action under a
    /// new id and started it: the queue's front stayed Executing, and the next step's executor refused it ("... at the front
    /// of queue 0 is in an invalid state"), a crash in every caller that went on after the refusal.
    /// </remarks>
    internal static void Run(CombatState combat, Core2.KernelState k, Host host, int action, int target)
    {
        var atScreen = combat.PendingSelection is not null ? k.CopyTo(combat, null, null) : null;
        try
        {
            RunQueue(combat, k, host, action, target);
        }
        catch (StoppedStepMismatch) when (atScreen is not null)
        {
            combat.Kernel = atScreen.CopyTo(combat, null, combat.Kernel);
            throw;
        }
    }

    private static void RunQueue(CombatState combat, Core2.KernelState k, Host host, int action, int target)
    {
        var q = k.Queue!;
        Core2.GameActionRec? fresh = null;
        if (combat.PendingSelection is { KernelAction: > 0 } screen)
        {
            combat.PendingSelection = null;
            k.Answer(screen.KernelAction, action);
        }
        else if (combat.PendingSelection is not null)
        {
            if (q.Front(0, Core2.GameActionState.GatheringPlayerChoice) is { Kind: not Core2.GameActionKind.Hook } gathering)
            {
                q.Resume(gathering.Id!.Value, action);
            }
            else
            {
                // A screen outside any action (the turn's end, the combat's start): its answer is the old step's.
                host.Body(action, target);
            }
        }
        else
        {
            var kind = action < combat.Hand.Count ? Core2.GameActionKind.PlayCard
                : action == combat.Hand.Count ? Core2.GameActionKind.EndTurn
                : Core2.GameActionKind.UsePotion;
            fresh = q.Enqueue(new Core2.GameActionRec(kind, Core2.GameActionRec.TypeOf(kind)) { Action = action, Target = target });
        }

        Core2.ActionExecutor.Drain(k, host);
        // game: a canceled action does nothing. When the queue ran no body (the action was canceled: its queue cancels play
        // cards or player-driven actions, or the combat is over; or the executor is paused and the action waits; or the step
        // was the answer to a kernel screen and the hook action's resume record ran) the old step does not run in its place.
        // A canceled action is refused: status -1, nothing changed.
        if (!host.Ran && fresh is { State: Core2.GameActionState.Canceled })
        {
            host.Status = -1;
        }

        if (!host.InProgress)
        {
            k.Queue!.CombatEnded();  // not `q`: a replayed step restored the kernel state and its queue (see ActionExecutor.Drain)
        }
    }

    /// <summary>The executor's host for one step: the body of a player action, the combat's progress and its end.</summary>
    internal abstract class Host(CombatState combat) : Core2.IActionHost, Core2.ICombatEndHost
    {
        protected readonly CombatState Combat = combat;
        public bool Ran;
        public int Status;
        public float Reward;
        public bool Terminal;
        public bool Truncated;

        /// <summary>The old engine's step of a player action (or of its answer); sets <see cref="Ran" />.</summary>
        public abstract void Body(int action, int target);

        public bool Run(Core2.KernelState k, Core2.GameActionRec a)
        {
            if (a.Answer is { } answer)
            {
                a.Answer = null;
                Body(answer, -1);
            }
            else
            {
                Body(a.Action, a.Target);
            }

            return Combat.PendingSelection is { Replay: not null, KernelAction: 0 };
        }

        public abstract bool InProgress { get; }

        public abstract void CheckWin(Core2.KernelState k);

        public abstract void EndCombat(Core2.KernelState k, bool playerWon);

        // game: CombatManager.IsInProgress: Core2's with F10 on (CombatEnd), else the old gate.
        protected bool Going =>
            Combat.Kernel is { } k && k.On(Core2.Switch.F10)
                ? Core2.CombatEnd.IsInProgress(k)
                : !CombatEngine.IsOverOrEnding(Combat);
    }

    /// <summary>The host of a combat stepped on its own: the body is the combat engine's step on the caller's stream.</summary>
    private sealed class BareHost(CombatState combat, Random rng) : Host(combat)
    {
        public StepResult? Result;

        public override void Body(int action, int target)
        {
            Ran = true;
            if (Combat.Kernel is { } kEnd)
            {
                kEnd.EndHost = this;
            }

            var r = CombatEngine.Step(Combat, action, rng, target);
            Result = r;
            (Reward, Terminal) = (r.Reward, r.Terminal);
        }

        public override bool InProgress => Going;

        // game: CheckWinCondition at the executor's boundary (called with F10 on only); the end is this host's.
        public override void CheckWin(Core2.KernelState k) => Core2.CombatEnd.AfterAction(k, this);

        // A combat stepped on its own has no run to finish: its end is the step's terminal result.
        public override void EndCombat(Core2.KernelState k, bool playerWon)
        {
            Terminal = true;
        }
    }
}
