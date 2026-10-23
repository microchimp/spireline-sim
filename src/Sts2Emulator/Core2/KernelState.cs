// The kernel's per-combat state (docs/rebuild/TIER1_KERNEL.md section 4): what one combat (or one run, outside a combat)
// owns of the kernel, as the HookRuntime Hooks.g.cs calls. Held by CombatState.Kernel and RunState.Kernel and copied with
// the state by Clone and CopyTo, so a clone or a replayed step (HookChoiceReplay) carries its own marks, binding and
// recorded actions and never writes into another state's. The shared part (switches, handler table, ids) is Kernel.
// With no switch on every member a site calls is a no-op that writes nothing.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Core2;

public sealed class KernelState : HookRuntime
{
    private readonly List<RecordedHookAction> _actions = [];
    private readonly List<PauseSite> _stops = [];
    private readonly List<WaitPoint> _waits = [];
    private long _nextAction;
    private CombatState? _combat;
    private RunState? _run;
    private OldWorld? _world;
    private OldListeners? _listeners;
    private ActionQueueSet? _queue;
    private readonly Dictionary<(HookId, ListenerRef, string), int> _seq = [];

    internal KernelState(Kernel shared, CombatState? combat, RunState? run)
    {
        Shared = shared;
        _combat = combat;
        _run = run;
    }

    /// <summary>The kernel's shared, immutable part.</summary>
    public Kernel Shared { get; }

    public KernelOptions Options => Shared.Options;

    /// <summary>Whether any switch is on. When false this state builds no world and records nothing.</summary>
    public bool AnyOn => Shared.AnyOn;

    /// <summary>Whether the switch of <paramref name="family" />'s group is on.</summary>
    public bool On(Switch family) => Shared.On(family);

    /// <summary>The shared handler table; null with no switch on.</summary>
    public HandlerTable? Handlers => Shared.Handlers;

    /// <summary>The old state the listeners are read from, with this state's marks; null with no switch on.</summary>
    public OldWorld? World => AnyOn ? _world ??= new OldWorld(_combat, _run, Shared.Ids!) : null;

    /// <summary>The listener cursor of the passes; null with no switch on.</summary>
    public OldListeners? Passes => AnyOn ? _listeners ??= new OldListeners(World!, Handlers!) : null;

    /// <summary>Hook actions per-listener passes recorded since this state began (K4's queue runs them).</summary>
    public IReadOnlyList<RecordedHookAction> HookActions => _actions;

    /// <summary>Pauses of dispatchers whose context awaits each listener (the dispatch stopped there).</summary>
    public IReadOnlyList<PauseSite> Stops => _stops;

    public IReadOnlyList<WaitPoint> Waits => _waits;

    /// <summary>
    /// The action queues (K4, ActionQueueSet.cs), built on first use and only with a switch on (null otherwise): the
    /// queues of a combat's state start in combat (game: ActionQueueSet.CombatStarted).
    /// </summary>
    public ActionQueueSet? Queue
    {
        get
        {
            if (!AnyOn)
                return null;
            if (_queue is null)
            {
                _queue = new ActionQueueSet();
                if (_combat is not null)
                    _queue.CombatStarted();
            }

            return _queue;
        }
    }

    /// <summary>game: ActionExecutor.IsPaused (CombatManager.Pause): a paused executor runs no action.</summary>
    public bool ExecutorPaused { get; set; }

    /// <summary>How many win checks the executor made at action boundaries (F10's switch).</summary>
    public int WinChecks { get; internal set; }

    /// <summary>
    /// Where answers to kernel screens raised inside a step come from; null: the running step's replay record
    /// (HookChoiceReplay's keyed answers). Transient (a driver's or a test's), never copied with the state.
    /// </summary>
    public IKernelAnswers? Answers { get; set; }

    /// <summary>Binds the world to the combat (and run) a site runs on; returns this.</summary>
    public KernelState At(CombatState? combat, RunState? run = null)
    {
        _combat = combat;
        if (run is not null)
            _run = run;
        if (_world is { } w)
        {
            w.Combat = combat;
            if (run is not null)
                w.Run = run;
        }

        return this;
    }

    // ------------------------------------------------------------------------------------- copying

    /// <summary>
    /// This state as a copy of the state it is on carries it: the same switches, marks, recorded actions, stops, waits
    /// and action counter, bound to <paramref name="combat" /> (and <paramref name="run" />; null keeps this state's
    /// run). The copy fills <paramref name="into" /> when that is a state of the same kernel (the in-place restore of
    /// <c>CopyTo</c>, whose holder is mid-step and keeps its object), else a new one. Nothing is shared with this state:
    /// the lists and pause contexts are copied, and a scope open on this state restores this state only.
    /// </summary>
    internal KernelState CopyTo(CombatState? combat, RunState? run, KernelState? into)
    {
        if (ReferenceEquals(into, this))
            return this;
        var k = into is not null && ReferenceEquals(into.Shared, Shared) ? into : new KernelState(Shared, combat, run);
        k._combat = combat;
        k._run = run ?? _run;
        k._actions.Clear();
        k._actions.AddRange(_actions.Select(a => a with { Site = Detach(a.Site) }));
        k._stops.Clear();
        k._stops.AddRange(_stops.Select(Detach));
        k._waits.Clear();
        k._waits.AddRange(_waits);
        k._nextAction = _nextAction;
        k._queue = _queue?.Copy();
        k._seq.Clear();
        foreach (var kv in _seq)
            k._seq[kv.Key] = kv.Value;
        k.ExecutorPaused = ExecutorPaused;
        k.WinChecks = WinChecks;
        k._listeners = null;
        if (_world is not null || k._world is not null)
            (k._world ??= new OldWorld(null, null, Shared.Ids!)).CopyFrom(_world, k._combat, k._run);
        return k;
    }

    private static PauseSite Detach(PauseSite site) => site with { Ctx = site.Ctx?.Copy() };

    // ------------------------------------------------------------------------------------- HookRuntime

    public override bool HasAny(HookId hook) => AnyOn && Handlers!.HasAny(hook);

    public override HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook) =>
        AnyOn ? Passes!.Cursor<THandler>(iterator, hook) : HookCursor<THandler>.Empty();

    public override ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who) => new(dispatcher, who);

    /// <summary>
    /// Per-listener contexts: the listener's choice becomes a recorded hook action (K4's queue runs it) and the dispatch
    /// moves on. Every other context: the dispatch stops; the stop is recorded for the action that owns the context.
    /// With no switch on nothing is recorded.
    /// </summary>
    public override void OnPaused(in PauseSite site, HookStep step)
    {
        if (!AnyOn)
            return;
        if (site.Context == ContextMode.PerListener)
        {
            // game: HookPlayerChoiceContext.SignalPlayerChoiceBegun: the listener's choice becomes a hook action of type
            // Combat on its owner's queue (single player: seat 0), run when the executor or a wait point reaches it.
            _actions.Add(new RecordedHookAction(++_nextAction, site, step));
            Queue!.Enqueue(new GameActionRec(GameActionKind.Hook, GameActionType.Combat) { HookAction = _nextAction });
        }
        else
        {
            _stops.Add(site);
        }
    }

    public override long HookActionMark() => _nextAction;

    /// <summary>K3 records the wait point and leaves the hook actions pending; K4's executor drains them here. With no
    /// switch on nothing is recorded.</summary>
    public override void WaitAll(HookId dispatcher, long mark)
    {
        if (!AnyOn)
            return;
        _waits.Add(new WaitPoint(dispatcher, mark, _actions.Count(a => a.Id > mark)));
        // game: the dispatcher awaits every listener's task; each paused listener's task ends when its hook action has run.
        // Here they run now, in id order, after every later listener of the dispatch ran (the screens come after them).
        foreach (var rec in Queue!.All.Where(r => r.Kind == GameActionKind.Hook && r.HookAction > mark
                                                    && r.State == GameActionState.WaitingForExecution)
                     .OrderBy(r => r.Id).ToList())
            RunAtWait(rec);
    }

    // ------------------------------------------------------------------------------------- hook actions and answers

    private int ActionIndex(long id) =>
        _actions.FindIndex(a => a.Id == id) is var i and >= 0 ? i : throw new InvalidOperationException($"no hook action {id}");

    private HookAnswerKey NextKey(in PauseSite site, HookStep step)
    {
        var triple = (site.Dispatcher, site.Who, step.Resume!.ResumeSite);
        int seq = _seq.GetValueOrDefault(triple);
        _seq[triple] = seq + 1;
        return new HookAnswerKey(site.Dispatcher, site.Who, step.Resume.ResumeSite, seq);
    }

    /// <summary>
    /// The answer to a kernel screen raised inside old code, which cannot stop and resume: the step's keyed answer when it
    /// has one (a replay, HookChoiceReplay), else the step stops with the screen open (PendingSelection, keyed) and its
    /// answer replays the step.
    /// </summary>
    private int AnswerOrStop(in HookAnswerKey key, object? request)
    {
        var source = Answers ?? (CombatEngine.Running is { } run && ReferenceEquals(run.State, _combat) ? run : null);
        if (source is not null && source.TryAnswer(key, out int answer))
            return answer;
        if (_combat is { } c && request is IKernelScreen screen)
            c.PendingSelection = screen.Open(c).Clone(replay: null, key: key);
        throw HookChoicePending.Instance;
    }

    /// <summary>A hook action drained at a wait point inside a step: its screen, then its resume record with the answer.</summary>
    private void RunAtWait(GameActionRec rec)
    {
        var q = Queue!;
        q.Start(rec);
        int i = ActionIndex(rec.HookAction);
        while (true)
        {
            var ha = _actions[i];
            q.Gather(rec);
            int answer = AnswerOrStop(NextKey(ha.Site, ha.Step), ha.Step.Request);
            q.Resume(rec.Id!.Value, answer);
            rec.Answer = null;
            q.Start(rec);
            var next = ha.Step.Resume!.Continue(this, answer);
            if (!next.IsPaused)
            {
                q.Remove(rec);
                return;
            }

            _actions[i] = ha with { Step = next };
        }
    }

    /// <summary>
    /// A hook action the executor reached between actions (ActionExecutor): there the caller can stop and resume, so its
    /// screen opens over the combat (PendingSelection naming the action) and the answer (<see cref="Answer" />) runs its
    /// resume record directly, no replay.
    /// </summary>
    internal void RunHookAction(GameActionRec rec)
    {
        var q = Queue!;
        int i = ActionIndex(rec.HookAction);
        var ha = _actions[i];
        if (rec.Answer is { } answer)
        {
            rec.Answer = null;
            var next = ha.Step.Resume!.Continue(this, answer);
            if (!next.IsPaused)
            {
                q.Finish(rec);
                return;
            }

            _actions[i] = ha = ha with { Step = next };
        }

        q.PauseForChoice(rec);
        if (_combat is { } c && ha.Step.Request is IKernelScreen screen)
            c.PendingSelection = screen.Open(c).Clone(replay: null, kernelAction: rec.Id!.Value);
    }

    /// <summary>The player's answer to the screen of action <paramref name="actionId" /> (game:
    /// ActionQueueSet.ResumeActionWithoutSynchronizing: it resumes under a new id when the executor next runs).</summary>
    public void Answer(uint actionId, int answer) => Queue!.Resume(actionId, answer);

    /// <summary>
    /// A dispatch whose context awaits each listener (none, passed, passed-unpushed, caller-hook) stopped at a choice
    /// inside old code: the choice is answered (keyed, or the step stops and replays), the listener's resume record runs
    /// (it may ask again), then <paramref name="resume" /> (the dispatcher's Resume and what follows it in the caller)
    /// continues; until the dispatch completes. With <paramref name="flow" /> the choice is the turn start's: its hook
    /// action (type CombatPlayPhaseOnly) is enqueued when the auto-pre-play phase assigns the task, and runs the choice;
    /// <paramref name="first" /> is that action when the caller enqueued it already (TurnStart, which checks the win
    /// condition between the enqueue and the answer).
    /// </summary>
    public HookStatus Settle(in HookStatus paused, Func<HookStatus, HookStatus> resume, bool flow = false,
        GameActionRec? first = null)
    {
        var st = paused;
        var q = Queue!;
        while (st.IsPaused)
        {
            var site = st.Site!.Value;
            var step = st.Step;
            var rec = first ?? (flow ? q.Enqueue(new GameActionRec(GameActionKind.Hook, GameActionType.CombatPlayPhaseOnly)) : null);
            first = null;
            if (rec is { State: GameActionState.WaitingForExecution })
            {
                q.Start(rec);
                q.Gather(rec);
            }
            else
            {
                rec = null;
            }

            while (true)
            {
                int answer = AnswerOrStop(NextKey(site, step), step.Request);
                if (rec is not null)
                {
                    q.Resume(rec.Id!.Value, answer);
                    rec.Answer = null;
                    q.Start(rec);
                }

                step = step.Resume!.Continue(this, answer);
                if (!step.IsPaused)
                    break;
                if (rec is not null)
                    q.Gather(rec);
            }

            if (rec is not null)
                q.Remove(rec);
            st = resume(st);
        }

        return st;
    }

    /// <summary>
    /// game: CombatManager.StartTurn's player part, as <see cref="ITurnStartFlow" /> says: the setup to its pause,
    /// AfterSideTurnStart, the orb queue's turn start; the hook action is enqueued when the auto-pre-play phase assigns the
    /// task; then CheckWinCondition (F10's behaviour); only then the executor runs the choice, the rest of the setup and
    /// AfterAutoPrePlayPhaseEntered, followed by the boundary win check (F10). A combat a death before or during
    /// AfterSideTurnStart or the orb start ended is over (CombatEnded cancels the pending choice): nothing after runs.
    /// </summary>
    public void TurnStart(ITurnStartFlow f)
    {
        var st = f.Setup(this);
        f.SideStart(this);
        f.OrbStart(this);
        var q = Queue!;
        var rec = st.IsPaused ? q.Enqueue(new GameActionRec(GameActionKind.Hook, GameActionType.CombatPlayPhaseOnly)) : null;
        if (WinCheck(f))
            return;
        Settle(st, s => f.ResumeSetup(this, s), flow: true, first: rec);
        f.AutoPrePlay(this);
        WinCheck(f);
    }

    /// <summary>game: CombatManager.CheckWinCondition at the turn start's two points, only with F10 on; true when it ended
    /// the combat, which cancels the combat's actions that have not started (ActionQueueSet.CombatEnded).</summary>
    private bool WinCheck(ITurnStartFlow f)
    {
        if (!On(Switch.F10))
            return false;
        WinChecks++;
        if (!f.CheckWin(this))
            return false;
        Queue!.CombatEnded();
        return true;
    }
}
