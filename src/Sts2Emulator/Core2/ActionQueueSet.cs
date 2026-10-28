// The action queues (docs/rebuild/TIER1_KERNEL.md section 5), in the shape of the game's ActionQueueSet: one queue per
// player of plain-data action records, ids from one counter, the ready action chosen by the game's rules, and the game's
// cancellations. All of it is plain data held by the combat's KernelState and copied with it: no task, closure or live
// cursor, so a copy of the combat mid-pause resumes as the source would. Built only when a switch is on.
namespace Sts2Emulator.Core2;

/// <summary>What an action is (the game's GameAction subclasses the kernel models).</summary>
public enum GameActionKind : byte
{
    /// <summary>game: PlayCardAction (CombatPlayPhaseOnly).</summary>
    PlayCard,
    /// <summary>game: UsePotionAction (CombatPlayPhaseOnly in combat, NonCombat outside).</summary>
    UsePotion,
    /// <summary>game: EndPlayerTurnAction (CombatPlayPhaseOnly).</summary>
    EndTurn,
    /// <summary>game: ReadyToBeginEnemyTurnAction (Combat).</summary>
    ReadyToBeginEnemyTurn,
    /// <summary>game: GenericHookGameAction (Combat for a per-listener hook context, CombatPlayPhaseOnly for the turn
    /// start's flow contexts).</summary>
    Hook,
}

/// <summary>game: GameActionType, same members and order.</summary>
public enum GameActionType : byte { None, Combat, CombatPlayPhaseOnly, NonCombat, Any }

/// <summary>game: GameActionState, same members and order.</summary>
public enum GameActionState : byte
{
    None,
    WaitingForExecution,
    Executing,
    GatheringPlayerChoice,
    ReadyToResumeExecuting,
    Finished,
    Canceled,
}

/// <summary>
/// One action in a queue, as plain data: its kind, type, owner, id and state; a player action's step encoding (the old
/// engine's body); a hook action's recorded hook action; the answer a paused action resumes with.
/// </summary>
public sealed class GameActionRec
{
    public GameActionRec(GameActionKind kind, GameActionType type, int owner = 0)
    {
        Kind = kind;
        Type = type;
        Owner = owner;
    }

    public GameActionKind Kind { get; }
    public GameActionType Type { get; }
    public int Owner { get; }
    /// <summary>game: GameAction.Id; null until enqueued, a new id when resumed after a choice.</summary>
    public uint? Id { get; internal set; }
    public GameActionState State { get; internal set; }
    /// <summary>A player action's step encoding (the old engine's action index and target).</summary>
    public int Action { get; init; } = -1;
    public int Target { get; init; } = -1;
    /// <summary>A hook action's <see cref="RecordedHookAction.Id" />; 0 for other kinds.</summary>
    public long HookAction { get; init; }
    /// <summary>The answer the action resumes with once its choice is answered; null before.</summary>
    public int? Answer { get; set; }
    /// <summary>The id a resumed action ran under before (diagnostics and tests).</summary>
    public uint? FirstId { get; internal set; }

    /// <summary>game: ActionQueueSet.IsGameActionPlayerDriven: every action but hook actions and ready-to-begin-enemy-turn.</summary>
    public bool PlayerDriven => Kind is not (GameActionKind.Hook or GameActionKind.ReadyToBeginEnemyTurn);

    /// <summary>The game's type of a player action of <paramref name="kind" /> (UsePotionAction: by whether in combat).</summary>
    public static GameActionType TypeOf(GameActionKind kind, bool inCombat = true) => kind switch
    {
        GameActionKind.PlayCard or GameActionKind.EndTurn => GameActionType.CombatPlayPhaseOnly,
        GameActionKind.UsePotion => inCombat ? GameActionType.CombatPlayPhaseOnly : GameActionType.NonCombat,
        GameActionKind.ReadyToBeginEnemyTurn => GameActionType.Combat,
        _ => throw new ArgumentException($"{kind}: a hook action's type is its context's"),
    };

    internal GameActionRec Copy() => (GameActionRec)MemberwiseClone();

    public override string ToString() => $"{Kind}#{Id} {State}";
}

/// <summary>game: ActionQueueSet (one queue per player; single-player runs have one).</summary>
public sealed class ActionQueueSet
{
    private sealed class Queue
    {
        public readonly List<GameActionRec> Actions = [];
        public int Owner;
        public bool CancellingPlayCard;
        public bool CancellingPlayerDriven;
        public bool CancellingCombat = true;
        public bool Paused;

        public Queue Copy(Dictionary<GameActionRec, GameActionRec> map)
        {
            var q = new Queue
            {
                Owner = Owner, CancellingPlayCard = CancellingPlayCard, CancellingPlayerDriven = CancellingPlayerDriven,
                CancellingCombat = CancellingCombat, Paused = Paused,
            };
            q.Actions.AddRange(Actions.Select(a => map[a]));
            return q;
        }
    }

    private readonly List<Queue> _queues = [];
    private readonly List<(uint Old, uint New)> _waitingForResumption = [];
    private readonly List<GameActionRec> _all = [];

    public ActionQueueSet(int players = 1)
    {
        for (int i = 0; i < players; i++)
            _queues.Add(new Queue { Owner = i });
    }

    private ActionQueueSet()
    {
    }

    /// <summary>game: ActionQueueSet.NextActionId.</summary>
    public uint NextId { get; private set; }

    /// <summary>game: _isInCombat (CombatStarted sets it, CombatEnded clears it).</summary>
    public bool InCombat { get; private set; }

    /// <summary>Every action ever enqueued, in enqueue order (finished and canceled ones too): what a test or a writer reads.</summary>
    public IReadOnlyList<GameActionRec> All => _all;

    /// <summary>The queues (one per player), for the writer.</summary>
    public int QueueCount => _queues.Count;

    /// <summary>
    /// The id the last action to finish ran under (a resumed action's new id); null before any finished. It is the queue's own
    /// count, not the game's lastExecutedActionId: the game's synchronizer also numbers actions the emulator does not run, so a
    /// record's value cannot be matched (the comparer lists it as surplus).
    /// </summary>
    public uint? LastExecutedActionId { get; private set; }

    /// <summary>The <see cref="RecordedHookAction.Id" /> of the last action to finish when it was a hook action (1-based; 0 for the
    /// turn start's flow action, which has no recorded hook action), else null. Not the game's lastExecutedHookId: the game writes
    /// the HookId from ActionQueueSynchronizer._nextHookId (0-based, which also numbers the turn start flow action), so this value
    /// cannot be matched against a record (the comparer lists it as surplus).</summary>
    public long? LastExecutedHookId { get; private set; }

    /// <summary>One queue's cancel flags, as the game keeps them (the writer's).</summary>
    public (bool CancellingPlayCard, bool CancellingPlayerDriven, bool CancellingCombat) Cancelling(int owner = 0) =>
        (Q(owner).CancellingPlayCard, Q(owner).CancellingPlayerDriven, Q(owner).CancellingCombat);

    private void Executed(GameActionRec a)
    {
        LastExecutedActionId = a.Id;
        LastExecutedHookId = a.Kind == GameActionKind.Hook ? a.HookAction : null;
    }

    /// <summary>The actions of <paramref name="owner" />'s queue, front first.</summary>
    public IReadOnlyList<GameActionRec> QueueOf(int owner = 0) => Q(owner).Actions;

    public bool IsPaused(int owner = 0) => Q(owner).Paused;

    public bool IsEmpty => _queues.All(q => q.Actions.Count == 0);

    /// <summary>Resumptions that arrived before their action paused (the solver's key refuses a state holding one).</summary>
    internal int PendingResumptions => _waitingForResumption.Count;

    private Queue Q(int owner) =>
        _queues.FirstOrDefault(q => q.Owner == owner) ?? throw new InvalidOperationException($"no action queue for player {owner}");

    /// <summary>
    /// game: ActionQueueSet.EnqueueWithoutSynchronizing. The action takes the next id and waits; it is canceled instead
    /// when its queue cancels play-card actions and it plays a card, cancels player-driven combat actions and it is one,
    /// or cancels combat actions (outside a combat) and it is of a combat type.
    /// </summary>
    public GameActionRec Enqueue(GameActionRec a)
    {
        if (a.Id is not null || a.State != GameActionState.None)
            throw new InvalidOperationException($"{a} was enqueued before");
        a.Id = NextId++;
        a.State = GameActionState.WaitingForExecution;
        _all.Add(a);
        var q = Q(a.Owner);
        if ((q.CancellingPlayCard && a.Kind == GameActionKind.PlayCard)
            || (q.CancellingPlayerDriven && a.PlayerDriven && a.Type is not (GameActionType.NonCombat or GameActionType.Any))
            || (q.CancellingCombat && a.Type is GameActionType.Combat or GameActionType.CombatPlayPhaseOnly))
        {
            a.State = GameActionState.Canceled;
            return a;
        }

        q.Actions.Add(a);
        return a;
    }

    /// <summary>
    /// game: ActionQueueSet.GetReadyAction. Per queue, canceled fronts are dropped; a queue is skipped when its front is a
    /// non-combat action during a combat, a combat-play-phase-only action of a paused queue, or gathering a choice. Of the
    /// fronts left, the lowest id.
    /// </summary>
    public GameActionRec? GetReadyAction()
    {
        GameActionRec? best = null;
        foreach (var q in _queues)
        {
            while (q.Actions.Count > 0 && q.Actions[0].State == GameActionState.Canceled)
                q.Actions.RemoveAt(0);
            if (q.Actions.Count == 0)
                continue;
            var front = q.Actions[0];
            if (InCombat && front.Type == GameActionType.NonCombat)
                continue;
            if (q.Paused && front.Type == GameActionType.CombatPlayPhaseOnly)
                continue;
            if (front.State == GameActionState.GatheringPlayerChoice)
                continue;
            if (front.State is not (GameActionState.WaitingForExecution or GameActionState.ReadyToResumeExecuting))
                throw new InvalidOperationException($"{front} at the front of queue {q.Owner} is in an invalid state");
            if (best is null || front.Id < best.Id)
                best = front;
        }

        return best;
    }

    /// <summary>game: GameAction.Execute's start: a waiting or ready-to-resume action is executing.</summary>
    internal void Start(GameActionRec a)
    {
        if (a.State is not (GameActionState.WaitingForExecution or GameActionState.ReadyToResumeExecuting))
            throw new InvalidOperationException($"executing {a} from an invalid state");
        a.State = GameActionState.Executing;
    }

    /// <summary>game: GameAction.Execute's end and ActionQueueSet.PopAction: the finished action leaves the front.</summary>
    internal void Finish(GameActionRec a)
    {
        a.State = GameActionState.Finished;
        Executed(a);
        var q = Q(a.Owner);
        if (q.Actions.Count == 0 || !ReferenceEquals(q.Actions[0], a))
            throw new InvalidOperationException($"popped {a}, which is not the front of queue {a.Owner}");
        q.Actions.RemoveAt(0);
    }

    /// <summary>
    /// game: ActionQueueSet.PauseActionForPlayerChoice. The executing action at its queue's front gathers a choice; with
    /// <paramref name="cancelPlayCards" /> (PlayerChoiceOptions.CancelPlayCardActions) its queue's waiting play-card
    /// actions are canceled and later ones too until it resumes. A resumption that arrived first applies now.
    /// </summary>
    public void PauseForChoice(GameActionRec a, bool cancelPlayCards = false)
    {
        var q = Q(a.Owner);
        if (q.Actions.Count == 0 || !ReferenceEquals(q.Actions[0], a))
            throw new InvalidOperationException($"paused {a}, which is not the front of its queue");
        if (a.State != GameActionState.Executing)
            throw new InvalidOperationException($"paused {a}, which is not executing");
        a.State = GameActionState.GatheringPlayerChoice;
        int early = _waitingForResumption.FindIndex(w => w.Old == a.Id);
        (uint Old, uint New)? waiting = early >= 0 ? _waitingForResumption[early] : null;
        if (early >= 0)
            _waitingForResumption.RemoveAt(early);
        if (cancelPlayCards)
        {
            CancelWaiting(a.Owner, x => x.Kind == GameActionKind.PlayCard && (waiting is null || x.Id < waiting.Value.New));
            q.CancellingPlayCard = true;
        }

        if (waiting is { } w)
        {
            Ready(a, w.New);
            q.CancellingPlayCard = false;
        }
    }

    /// <summary>
    /// game: ActionQueueSet.ResumeActionWithoutSynchronizing. The answered action takes a NEW id (from the one counter)
    /// and is ready to resume; an action not gathering yet keeps the new id until it pauses.
    /// </summary>
    public void Resume(uint id, int answer)
    {
        uint newId = NextId++;
        var a = _queues.SelectMany(q => q.Actions).FirstOrDefault(x => x.Id == id);
        if (a is { State: GameActionState.GatheringPlayerChoice })
        {
            Q(a.Owner).CancellingPlayCard = false;
            a.Answer = answer;
            Ready(a, newId);
        }
        else
        {
            _waitingForResumption.Add((id, newId));
        }
    }

    private static void Ready(GameActionRec a, uint newId)
    {
        a.FirstId ??= a.Id;
        a.Id = newId;
        a.State = GameActionState.ReadyToResumeExecuting;
    }

    /// <summary>game: ActionQueueSet.PauseAllPlayerQueues.</summary>
    public void PauseAll()
    {
        foreach (var q in _queues)
        {
            q.Paused = true;
            q.CancellingPlayerDriven = false;
        }
    }

    /// <summary>game: ActionQueueSet.UnpauseAllPlayerQueues.</summary>
    public void UnpauseAll()
    {
        foreach (var q in _queues)
        {
            q.Paused = false;
            q.CancellingPlayerDriven = false;
        }
    }

    /// <summary>
    /// game: ActionQueueSet.StartCancellingAllPlayerDrivenCombatActions (the player-driven cancel): every queue cancels
    /// its waiting player-driven actions of a combat type and every such action enqueued later.
    /// </summary>
    public void StartCancellingPlayerDriven()
    {
        foreach (var q in _queues)
        {
            q.CancellingPlayerDriven = true;
            Drop(q, x => x.PlayerDriven && x.Type is not (GameActionType.NonCombat or GameActionType.Any)
                         && x.State == GameActionState.WaitingForExecution);
        }
    }

    /// <summary>game: ActionQueueSet.CombatStarted.</summary>
    public void CombatStarted()
    {
        InCombat = true;
        foreach (var q in _queues)
            q.CancellingCombat = false;
    }

    /// <summary>
    /// game: ActionQueueSet.CombatEnded. Every action of a combat type that is not executing is canceled and removed (a
    /// hook action not started yet, or one gathering a choice); the queues then cancel combat actions.
    /// </summary>
    public void CombatEnded()
    {
        InCombat = false;
        foreach (var q in _queues)
        {
            Drop(q, x => x.Type is GameActionType.Combat or GameActionType.CombatPlayPhaseOnly
                         && x.State != GameActionState.Executing);
            q.CancellingPlayCard = false;
            q.CancellingPlayerDriven = false;
            q.CancellingCombat = true;
        }
    }

    /// <summary>game: ActionQueueSet.CancelNonExecutingActionsForPlayer (its waiting actions).</summary>
    public void CancelWaitingFor(int owner) => CancelWaiting(owner, _ => true);

    private void CancelWaiting(int owner, Func<GameActionRec, bool> which) =>
        Drop(Q(owner), x => x.State == GameActionState.WaitingForExecution && which(x));

    private static void Drop(Queue q, Func<GameActionRec, bool> which)
    {
        for (int i = 0; i < q.Actions.Count; i++)
        {
            if (!which(q.Actions[i]))
                continue;
            q.Actions[i].State = GameActionState.Canceled;
            q.Actions.RemoveAt(i--);
        }
    }

    /// <summary>A copy sharing nothing with this set (KernelState.CopyTo).</summary>
    public ActionQueueSet Copy()
    {
        var map = _all.ToDictionary(a => a, a => a.Copy());
        var c = new ActionQueueSet
        {
            NextId = NextId, InCombat = InCombat, LastExecutedActionId = LastExecutedActionId, LastExecutedHookId = LastExecutedHookId,
        };
        c._all.AddRange(_all.Select(a => map[a]));
        foreach (var q in _queues)
            c._queues.Add(q.Copy(map));
        c._waitingForResumption.AddRange(_waitingForResumption);
        return c;
    }

    /// <summary>The front of <paramref name="owner" />'s queue when it is in <paramref name="state" />, else null.</summary>
    public GameActionRec? Front(int owner, GameActionState state) =>
        Q(owner).Actions is { Count: > 0 } l && l[0].State == state ? l[0] : null;

    /// <summary>The action keeps <paramref name="id" /> (the id it was resumed under) after a replay restored the queue to
    /// the step's start, and the counter stays past it.</summary>
    internal void Carry(GameActionRec a, uint id, uint? first)
    {
        if (a.Id == id)
            return;
        a.Id = id;
        a.FirstId = first;
        NextId = Math.Max(NextId, id + 1);
    }

    /// <summary>A hook action drained at a wait point gathers its choice wherever it stands in its queue (the step's own
    /// action can be ahead of it, executing).</summary>
    internal void Gather(GameActionRec a)
    {
        if (a.State != GameActionState.Executing)
            throw new InvalidOperationException($"{a} gathers a choice from an invalid state");
        a.State = GameActionState.GatheringPlayerChoice;
    }

    /// <summary>A hook action drained at a wait point finishes and leaves its queue wherever it stands.</summary>
    internal void Remove(GameActionRec a)
    {
        a.State = GameActionState.Finished;
        Executed(a);
        if (!Q(a.Owner).Actions.Remove(a))
            throw new InvalidOperationException($"{a} is in no queue");
    }

    /// <summary>The action with <paramref name="id" /> in a queue, or null.</summary>
    public GameActionRec? Find(uint id) => _queues.SelectMany(q => q.Actions).FirstOrDefault(x => x.Id == id);
}
