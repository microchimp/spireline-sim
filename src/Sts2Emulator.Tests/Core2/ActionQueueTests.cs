// The action queue, the executor and the pauses (docs/rebuild/TIER1_KERNEL.md section 5). Each test is a rule of the
// game's classes, cited `game: <Class>.<Method>`; the handlers, screens and hosts are test-only, and the switches are test
// groups. No model is migrated by them.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class ActionQueueTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly Switch[] Every = Enum.GetValues<Switch>().Where(s => s != Switch.None).ToArray();
    private static readonly Switch[][] TestGroups = [[Switch.F10], [Switch.F05]];

    // One group per family, the turn-start flow kept whole: lets a test switch F10 off and the others on.
    private static readonly Switch[][] FineGroups =
        [.. Every.Where(s => s is not (Switch.F02 or Switch.F03 or Switch.F13)).Select(s => new[] { s }),
            [Switch.F02, Switch.F03, Switch.F13]];

    private static KernelState K(Action<HandlerTable>? models = null, CombatState? combat = null, bool f10 = true) =>
        new Kernel(new KernelOptions(f10 ? Every : Every.Where(s => s != Switch.F10), FineGroups), () => Ids,
            [models ?? (_ => { })]).Bind(combat ?? Combat());

    private static CombatState Combat()
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = [new EnemyState { DefId = 16, Hp = 40, MaxHp = 40 }, new EnemyState { DefId = 16, Hp = 30, MaxHp = 30 }];
        s.Relics = [new RelicInstance(3), new RelicInstance(1), new RelicInstance(2)];
        return s;
    }

    private static GameActionRec Act(GameActionKind kind, int owner = 0) =>
        new(kind, kind == GameActionKind.Hook ? GameActionType.Combat : GameActionRec.TypeOf(kind), owner);

    /// <summary>A test host: logs each body; a body listed in <c>Stops</c> stops on an old screen once.</summary>
    private sealed class Host(List<string> log) : IActionHost
    {
        public readonly HashSet<GameActionKind> Stops = [];
        public Action<KernelState, GameActionRec>? During;
        public bool InProgress { get; set; } = true;

        public bool Run(KernelState k, GameActionRec a)
        {
            log.Add($"{a.Kind}#{a.Id}{(a.Answer is { } x ? $"={x}" : "")}");
            During?.Invoke(k, a);
            bool stop = a.Answer is null && Stops.Contains(a.Kind);
            a.Answer = null;
            return stop;
        }

        public void CheckWin(KernelState k) => log.Add("win");
    }

    /// <summary>A kernel screen: the old engine's selection over the hand.</summary>
    private sealed record Screen(string Who) : IKernelScreen
    {
        public PendingCardSelection Open(CombatState state) =>
            new() { Kind = CardSelectionKind.None, Candidates = [0, 1], SourceCardDefId = 0 };
    }

    /// <summary>A resume record that logs, and takes HP so a copy's run can be compared with its source's.</summary>
    private sealed record Log(List<string>? Into, string Who) : HookResume
    {
        public override HookStep Continue(HookRuntime rt, int answer)
        {
            Into?.Add($"resume:{Who}={answer}");
            ((KernelState)rt).World!.Combat!.PlayerHp -= answer + 1;
            return HookStep.Done;
        }
    }

    private sealed class Answers(List<string> log, int answer = 1) : IKernelAnswers
    {
        public bool TryAnswer(in HookAnswerKey key, out int a)
        {
            log.Add($"screen:{key.Who.ModelId}");
            a = answer;
            return true;
        }
    }

    // ------------------------------------------------------------------------------------- the queue's rules

    [Fact]
    public void WinIsCheckedAfterEveryActionButEndTurnAndReadyToBeginEnemyTurn()
    {
        // game: ActionExecutor.ExecuteActions (CheckWinCondition unless EndPlayerTurnAction or ReadyToBeginEnemyTurnAction)
        var log = new List<string>();
        var k = K();
        foreach (var kind in new[] { GameActionKind.PlayCard, GameActionKind.EndTurn, GameActionKind.UsePotion, GameActionKind.ReadyToBeginEnemyTurn })
            k.Queue!.Enqueue(Act(kind));
        ActionExecutor.Drain(k, new Host(log));
        Assert.Equal(["PlayCard#0", "win", "EndTurn#1", "UsePotion#2", "win", "ReadyToBeginEnemyTurn#3"], log);
        Assert.Equal(2, k.WinChecks);

        // Off F10's switch the boundary check is not the kernel's.
        var off = K(f10: false);
        off.Queue!.Enqueue(Act(GameActionKind.PlayCard));
        log.Clear();
        ActionExecutor.Drain(off, new Host(log));
        Assert.Equal(["PlayCard#0"], log);
    }

    [Fact]
    public void CanceledFrontsAreSkippedAndTheLowestFrontIdIsReady()
    {
        // game: ActionQueueSet.GetReadyAction
        var q = new ActionQueueSet(2);
        q.CombatStarted();
        var a = q.Enqueue(Act(GameActionKind.PlayCard, 0));
        var b = q.Enqueue(Act(GameActionKind.PlayCard, 1));
        var c = q.Enqueue(Act(GameActionKind.UsePotion, 0));
        Assert.Same(a, q.GetReadyAction());
        q.CancelWaitingFor(0);
        Assert.Equal(GameActionState.Canceled, a.State);
        Assert.Same(b, q.GetReadyAction());
        Assert.Equal(GameActionState.Canceled, c.State);
    }

    [Fact]
    public void APausedActionYieldsToTheNextReadyActionAndResumesUnderANewId()
    {
        // game: ActionQueueSet.PauseActionForPlayerChoice, GetReadyAction (a gathering front is skipped),
        // ResumeActionWithoutSynchronizing (a new id from the one counter), GameAction.ResumeAfterGatheringPlayerChoice
        var q = new ActionQueueSet(2);
        q.CombatStarted();
        var a = q.Enqueue(Act(GameActionKind.PlayCard, 0));
        var b = q.Enqueue(Act(GameActionKind.PlayCard, 1));
        q.Start(a);
        q.PauseForChoice(a);
        Assert.Same(b, q.GetReadyAction());
        q.Resume(a.Id!.Value, 1);
        Assert.Equal(2u, a.Id);
        Assert.Equal(0u, a.FirstId);
        Assert.Equal(GameActionState.ReadyToResumeExecuting, a.State);
        Assert.Same(b, q.GetReadyAction());
        q.Start(b);
        q.Finish(b);
        Assert.Same(a, q.GetReadyAction());

        // Through the executor: the body that stops leaves its queue gathering; the other queue's action runs.
        var log = new List<string>();
        var k = K();
        var p = k.Queue!.Enqueue(Act(GameActionKind.PlayCard));
        k.Queue.Enqueue(Act(GameActionKind.UsePotion));
        var host = new Host(log) { Stops = { GameActionKind.PlayCard } };
        ActionExecutor.Drain(k, host);
        Assert.Equal(["PlayCard#0", "win"], log);
        Assert.Equal(GameActionState.GatheringPlayerChoice, p.State);
        k.Answer(p.Id!.Value, 0);
        ActionExecutor.Drain(k, host);
        Assert.Equal(["PlayCard#0", "win", "PlayCard#2=0", "win", "UsePotion#1", "win"], log);
    }

    [Fact]
    public void TypeRulesSkipAndCancelAsTheGameDoes()
    {
        // game: ActionQueueSet.EnqueueWithoutSynchronizing, GetReadyAction, PauseAllPlayerQueues,
        // StartCancellingAllPlayerDrivenCombatActions; GenericHookGameAction (Combat or CombatPlayPhaseOnly only)
        var q = new ActionQueueSet();
        Assert.Equal(GameActionState.Canceled, q.Enqueue(Act(GameActionKind.PlayCard)).State); // not in combat yet
        q.CombatStarted();
        var potion = q.Enqueue(new GameActionRec(GameActionKind.UsePotion, GameActionRec.TypeOf(GameActionKind.UsePotion, inCombat: false)));
        Assert.Null(q.GetReadyAction()); // a non-combat action at the front during a combat
        q.CancelWaitingFor(0);
        Assert.Equal(GameActionState.Canceled, potion.State);
        var play = q.Enqueue(Act(GameActionKind.PlayCard));
        q.PauseAll();
        Assert.Null(q.GetReadyAction()); // combat-play-phase-only in a paused queue
        q.UnpauseAll();
        Assert.Same(play, q.GetReadyAction());
        var hook = q.Enqueue(Act(GameActionKind.Hook));
        q.StartCancellingPlayerDriven();
        Assert.Equal(GameActionState.Canceled, play.State);
        Assert.Equal(GameActionState.WaitingForExecution, hook.State); // a hook action is not player-driven
        Assert.Equal(GameActionState.Canceled, q.Enqueue(Act(GameActionKind.EndTurn)).State);
        Assert.False(Act(GameActionKind.ReadyToBeginEnemyTurn).PlayerDriven);
        Assert.Equal(GameActionType.Combat, GameActionRec.TypeOf(GameActionKind.ReadyToBeginEnemyTurn));
        Assert.Equal(GameActionType.CombatPlayPhaseOnly, GameActionRec.TypeOf(GameActionKind.EndTurn));
    }

    [Fact]
    public void CombatEndedCancelsTheHookActionsNotStarted()
    {
        // game: ActionQueueSet.CombatEnded (every combat-typed action not executing is canceled and removed)
        var q = new ActionQueueSet();
        q.CombatStarted();
        var running = q.Enqueue(Act(GameActionKind.PlayCard));
        var gathering = q.Enqueue(Act(GameActionKind.Hook));
        var waiting = q.Enqueue(Act(GameActionKind.Hook));
        q.Start(running);
        q.Start(gathering);
        q.Gather(gathering);
        q.CombatEnded();
        Assert.Equal(GameActionState.Executing, running.State);
        Assert.Equal(GameActionState.Canceled, gathering.State);
        Assert.Equal(GameActionState.Canceled, waiting.State);
        Assert.Equal([running], q.QueueOf());
        Assert.Equal(GameActionState.Canceled, q.Enqueue(Act(GameActionKind.Hook)).State);
    }

    // ------------------------------------------------------------------------------------- pauses

    private static KernelState Relics<TArgs>(List<string> log, string pauses, params HookId[] hooks) where TArgs : struct =>
        Relics<TArgs>(true, log, pauses, hooks);

    private static KernelState Relics<TArgs>(bool f10, List<string> log, string pauses, params HookId[] hooks) where TArgs : struct
    {
        HookEvent<TArgs> ev = (HookCtx _, ListenerRef self, in TArgs _, ChoiceCtx? _) =>
        {
            string who = Name(self.ModelId);
            log.Add($"run:{who}");
            return who == pauses ? HookStep.Paused(new Screen(who), new Log(log, who)) : HookStep.Done;
        };
        return K(t =>
        {
            foreach (int r in new[] { 3, 1, 2 })
            {
                var m = t.Model(ListenerKind.Relic, RelicNames[r]);
                foreach (var h in hooks)
                    m.On(h, ev);
            }
        }, f10: f10);
    }

    private static readonly Dictionary<int, string> RelicNames = LoadRelicNames();

    private static Dictionary<int, string> LoadRelicNames()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        var map = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!["relics"]!.AsObject();
        return map.Where(kv => (int)kv.Value! is 1 or 2 or 3).ToDictionary(kv => (int)kv.Value!, kv => kv.Key);
    }

    private static string Name(int relic) => RelicNames[relic];

    [Theory]
    [InlineData("BeforeTurnEnd")]
    [InlineData("AfterTurnEnd")]
    [InlineData("BeforeFlush")]
    public void LaterListenersRunBeforeAPausedListenersScreen(string dispatcher)
    {
        // game: Hook.BeforeTurnEnd, Hook.AfterTurnEnd, Hook.BeforeFlush (a HookPlayerChoiceContext per listener; the
        // listener that asks becomes a GenericHookGameAction and the loop moves on; Task.WhenAll after the passes)
        var log = new List<string>();
        string first = Name(3), second = Name(1), third = Name(2);
        var k = dispatcher switch
        {
            "BeforeTurnEnd" => Relics<BeforeTurnEndArgs>(log, first, HookId.BeforeSideTurnEnd),
            "AfterTurnEnd" => Relics<AfterTurnEndArgs>(log, first, HookId.AfterSideTurnEnd),
            _ => Relics<BeforeFlushArgs>(log, first, HookId.BeforeFlush),
        };
        k.Answers = new Answers(log);
        Dispatch(k, dispatcher);
        Assert.Equal([$"run:{first}", $"run:{second}", $"run:{third}", $"screen:3", $"resume:{first}=1"], log);
        var hook = Assert.Single(k.Queue!.All);
        Assert.Equal((GameActionKind.Hook, GameActionType.Combat, GameActionState.Finished, 1u, 0u),
            (hook.Kind, hook.Type, hook.State, hook.Id!.Value, hook.FirstId!.Value));
        Assert.Equal(48, k.World!.Combat!.PlayerHp);

        // With no answer yet the step stops with the kernel's screen open, keyed, after the later listeners ran.
        log.Clear();
        var stopped = dispatcher switch
        {
            "BeforeTurnEnd" => Relics<BeforeTurnEndArgs>(log, first, HookId.BeforeSideTurnEnd),
            "AfterTurnEnd" => Relics<AfterTurnEndArgs>(log, first, HookId.AfterSideTurnEnd),
            _ => Relics<BeforeFlushArgs>(log, first, HookId.BeforeFlush),
        };
        Assert.Throws<HookChoicePending>(() => Dispatch(stopped, dispatcher));
        Assert.Equal([$"run:{first}", $"run:{second}", $"run:{third}"], log);
        var key = stopped.World!.Combat!.PendingSelection!.HookKey!.Value;
        Assert.Equal((first, 0), (Name(key.Who.ModelId), key.Seq));
        Assert.Equal(nameof(Log), key.Site);
    }

    private static void Dispatch(KernelState k, string dispatcher)
    {
        var who = new List<CreatureRef> { new(0) };
        switch (dispatcher)
        {
            case "BeforeTurnEnd":
                Hooks.BeforeTurnEndPasses(k, new BeforeTurnEndArgs(Side.Player, who));
                break;
            case "AfterTurnEnd":
                Hooks.AfterTurnEndPasses(k, new AfterTurnEndArgs(Side.Player, who));
                break;
            default:
                Hooks.BeforeFlushPasses(k, new BeforeFlushArgs(new PlayerRef(0)));
                break;
        }
    }

    private sealed class Flow(List<string> log) : ITurnStartFlow
    {
        /// <summary>A death during AfterSideTurnStart that ends the combat (the flow's win check then says so).</summary>
        public bool DiesInSideStart;
        private bool _over;

        private readonly AfterPlayerTurnStartArgs _a = new(new ChoiceCtx(), new PlayerRef(0));

        public HookStatus Setup(KernelState k) => Hooks.AfterPlayerTurnStart(k, _a);

        public HookStatus ResumeSetup(KernelState k, in HookStatus paused) => Hooks.AfterPlayerTurnStartResume(k, _a, paused);

        public void SideStart(KernelState k)
        {
            log.Add("AfterSideTurnStart");
            _over = DiesInSideStart;
        }

        public void OrbStart(KernelState k) => log.Add("OrbQueue.AfterTurnStart");

        public void AutoPrePlay(KernelState k) => log.Add("AfterAutoPrePlayPhaseEntered");

        public bool CheckWin(KernelState k)
        {
            log.Add("CheckWin");
            return _over;
        }
    }

    [Fact]
    public void TheTurnStartsChoiceComesAfterAfterSideTurnStartAndTheOrbQueue()
    {
        // game: CombatManager.StartTurn (SetupPlayerTurn awaited by WaitForPauseOrCompletionWithoutAssigningTask, then
        // AfterSideTurnStart, the orb queue's AfterTurnStart, then RunAutoPrePlayPhase assigned to the context:
        // HookPlayerChoiceContext.SignalPlayerChoiceBegun enqueues the hook action only once the task is assigned)
        var log = new List<string>();
        var k = Relics<AfterPlayerTurnStartArgs>(log, Name(1), HookId.AfterPlayerTurnStart);
        k.Answers = new Answers(log, 0);
        k.TurnStart(new Flow(log));
        // The win condition is checked once the hook action is enqueued, before the executor runs it, and after the
        // phase's last hook (the boundary check of the hook action).
        Assert.Equal([$"run:{Name(3)}", $"run:{Name(1)}", "AfterSideTurnStart", "OrbQueue.AfterTurnStart", "CheckWin", "screen:1",
            $"resume:{Name(1)}=0", $"run:{Name(2)}", "AfterAutoPrePlayPhaseEntered", "CheckWin"], log);
        var hook = Assert.Single(k.Queue!.All);
        Assert.Equal((GameActionType.CombatPlayPhaseOnly, GameActionState.Finished), (hook.Type, hook.State));
        Assert.Single(k.Stops);
        Assert.Equal(2, k.WinChecks);
    }

    [Fact]
    public void ADeathInAfterSideTurnStartEndsTheCombatAndCancelsTheTurnStartsChoice()
    {
        // game: CombatManager.StartTurn: the hook action is enqueued, CheckWinCondition runs, and only then the executor is
        // unpaused; a death in AfterSideTurnStart (or before it, or in the orb queue's turn start) ends the combat there and
        // CombatEnded cancels the pending hook action: the choice, the rest of the setup and AfterAutoPrePlayPhaseEntered
        // never run.
        var log = new List<string>();
        var k = Relics<AfterPlayerTurnStartArgs>(log, Name(1), HookId.AfterPlayerTurnStart);
        k.Answers = new Answers(log, 0);
        k.TurnStart(new Flow(log) { DiesInSideStart = true });
        Assert.Equal([$"run:{Name(3)}", $"run:{Name(1)}", "AfterSideTurnStart", "OrbQueue.AfterTurnStart", "CheckWin"], log);
        Assert.Equal(GameActionState.Canceled, Assert.Single(k.Queue!.All).State);
        Assert.Equal(1, k.WinChecks);
        Assert.Null(k.World!.Combat!.PendingSelection);

        // Off F10's switch the turn start makes no win check of the kernel's: the flow goes on to the choice.
        log.Clear();
        var off = Relics<AfterPlayerTurnStartArgs>(false, log, Name(1), HookId.AfterPlayerTurnStart);
        off.Answers = new Answers(log, 0);
        off.TurnStart(new Flow(log) { DiesInSideStart = true });
        Assert.Equal([$"run:{Name(3)}", $"run:{Name(1)}", "AfterSideTurnStart", "OrbQueue.AfterTurnStart", "screen:1",
            $"resume:{Name(1)}=0", $"run:{Name(2)}", "AfterAutoPrePlayPhaseEntered"], log);
        Assert.Equal(0, off.WinChecks);
    }

    [Fact]
    public void ASetupThatDoesNotPauseChecksTheWinOnceAfterTheAutoPrePlayPhase()
    {
        // game: CombatManager.StartTurn awaits the auto-pre-play phase (AfterAutoPrePlayPhaseEntered) when the setup did not
        // pause, then calls CheckWinCondition once: a death in AfterSideTurnStart does not end the combat before it.
        var log = new List<string>();
        var k = Relics<AfterPlayerTurnStartArgs>(log, "none", HookId.AfterPlayerTurnStart);
        k.Answers = new Answers(log, 0);
        k.TurnStart(new Flow(log) { DiesInSideStart = true });
        Assert.Equal([$"run:{Name(3)}", $"run:{Name(1)}", $"run:{Name(2)}", "AfterSideTurnStart", "OrbQueue.AfterTurnStart",
            "AfterAutoPrePlayPhaseEntered", "CheckWin"], log);
        Assert.Empty(k.Queue!.All);
        Assert.Equal(1, k.WinChecks);

        // Without a death the same single check runs and the combat goes on.
        log.Clear();
        var alive = Relics<AfterPlayerTurnStartArgs>(log, "none", HookId.AfterPlayerTurnStart);
        alive.Answers = new Answers(log, 0);
        alive.TurnStart(new Flow(log));
        Assert.Equal([$"run:{Name(3)}", $"run:{Name(1)}", $"run:{Name(2)}", "AfterSideTurnStart", "OrbQueue.AfterTurnStart",
            "AfterAutoPrePlayPhaseEntered", "CheckWin"], log);
        Assert.Equal(1, alive.WinChecks);
    }

    private static (KernelState K, CombatState C) WithHookAction(List<string>? log)
    {
        var c = Combat();
        var k = K(combat: c);
        var who = new ListenerRef((int)ListenerKind.Relic, 1, 1);
        var ctx = k.NewHookContext(HookId.Dispatch_AfterDeath, who);
        k.OnPaused(new PauseSite(HookId.Dispatch_AfterDeath, HookId.AfterDeath, 0, 1, who, ContextMode.PerListener, ctx),
            HookStep.Paused(new Screen("relic"), new Log(log, "relic")));
        return (k, c);
    }

    [Fact]
    public void ACopyMidPauseResumesAsItsSource()
    {
        // game: GameAction / ActionQueueSet state is the combat's: a copy on the hook action's screen, answered the same,
        // ends the same, and answering one leaves the other on its screen.
        var (k, c) = WithHookAction(null);
        ActionExecutor.Drain(k, new Host([]));
        var hook = k.Queue!.All.Single();
        Assert.Equal(GameActionState.GatheringPlayerChoice, hook.State);
        Assert.Equal(hook.Id, c.PendingSelection!.KernelAction);
        var copy = c.Clone();
        Assert.NotSame(c.Kernel, copy.Kernel);
        Assert.Equal(hook.Id, copy.PendingSelection!.KernelAction);

        foreach (var s in new[] { copy, c })
        {
            s.PendingSelection = null;
            s.Kernel!.Answer(hook.Id!.Value, 1);
            ActionExecutor.Drain(s.Kernel, new Host([]));
        }

        Assert.Equal(48, c.PlayerHp);
        Assert.Equal(c.PlayerHp, copy.PlayerHp);
        Assert.Equal(k.Queue.All.Select(a => (a.Id, a.FirstId, a.State)), copy.Kernel!.Queue!.All.Select(a => (a.Id, a.FirstId, a.State)));
        Assert.Equal(GameActionState.Finished, k.Queue.All.Single().State);

        var (k2, c2) = WithHookAction(null);
        ActionExecutor.Drain(k2, new Host([]));
        var copy2 = c2.Clone();
        copy2.Kernel!.Answer(k2.Queue!.All.Single().Id!.Value, 0);
        ActionExecutor.Drain(copy2.Kernel, new Host([]));
        Assert.Equal(GameActionState.GatheringPlayerChoice, k2.Queue.All.Single().State);
        Assert.Equal(50, c2.PlayerHp);
    }

    [Fact]
    public void ADeadEnemyLeavesTheCombatBeforeAHookActionsScreenOpens()
    {
        // game: CreatureCmd.KillWithoutCheckingWinCondition: Hook.AfterDeath (a per-listener context: a listener that asks
        // becomes a hook action and the loop moves on), then CombatManager.RemoveCreature and CombatState.RemoveCreature,
        // all inside the action that killed; HookPlayerChoiceContext.SignalPlayerChoiceBegun awaits the hook action's
        // ExecutionStartedTask, so the screen opens only when the executor reaches it, after that action finished.
        // The removal has two conditions in the game, and the kill-site fix must keep both: it happens only when
        // ShouldCreatureBeRemovedFromCombatAfterDeath says so (the model's own answer), and CombatState.RemoveCreature is
        // skipped while the monster IsPerformingMove. This test's enemy is removable and not mid-move.
        var log = new List<string>();
        var c = Combat();
        var k = K(combat: c);
        var dead = c.Enemies[0];
        k.Queue!.Enqueue(Act(GameActionKind.PlayCard));
        var host = new Host(log)
        {
            During = (kk, a) =>
            {
                if (a.Kind != GameActionKind.PlayCard)
                    return;
                dead.Hp = 0;
                var who = new ListenerRef((int)ListenerKind.Relic, 1, 1);
                kk.OnPaused(new PauseSite(HookId.Dispatch_AfterDeath, HookId.AfterDeath, 0, 1, who, ContextMode.PerListener,
                    kk.NewHookContext(HookId.Dispatch_AfterDeath, who)), HookStep.Paused(new Screen("relic"), new Log(log, "relic")));
                log.Add("AfterDeath done");
                c.Enemies.Remove(dead);
                log.Add("removed");
            },
        };
        ActionExecutor.Drain(k, host);
        // The hook action is an action too: the executor checks the win condition after it.
        Assert.Equal(["PlayCard#0", "AfterDeath done", "removed", "win", "win"], log);
        Assert.NotNull(c.PendingSelection);
        Assert.DoesNotContain(dead, c.Enemies);
        Assert.Equal(GameActionState.GatheringPlayerChoice, k.Queue.All[1].State);
    }

    // ------------------------------------------------------------------------------------- the entry gate

    [Fact]
    public void WithASwitchOnEveryCombatSteppedCarriesItsKernelStateAndPlaysAsTheOldEngine()
    {
        // The queue path with nothing migrated is the old engine's step; the guard refuses a combat without a state.
        var kernel = new Kernel(new KernelOptions([Switch.F10], TestGroups), () => Ids, []);
        var old = new RunEngine();
        var core2 = new RunEngine { Kernel = kernel };
        old.Reset("PRB00058", ascension: 8);
        core2.Reset("PRB00058", ascension: 8);
        var mask = new int[RunConstants.MaxActions];
        var mask2 = new int[RunConstants.MaxActions];
        bool terminal = false;
        int combatSteps = 0;
        for (int step = 0; !terminal && step < 20_000; step++)
        {
            old.WriteActionMask(mask);
            core2.WriteActionMask(mask2);
            Assert.Equal(mask, mask2);
            int action = Array.IndexOf(mask, 1);
            if (core2.State.Phase == RunPhase.Combat)
            {
                Assert.NotNull(core2.State.ActiveCombat!.Kernel);
                Assert.Null(old.State.ActiveCombat!.Kernel);
                combatSteps++;
            }

            int s1 = old.Step(action, -1, out float r1, out terminal, out _);
            int s2 = core2.Step(action, -1, out float r2, out bool t2, out _);
            Assert.Equal((s1, r1, terminal), (s2, r2, t2));
            Assert.Equal(old.State.PlayerHp, core2.State.PlayerHp);
            Assert.Equal(old.State.Phase, core2.State.Phase);
        }

        Assert.True(terminal);
        Assert.True(combatSteps > 50, $"{combatSteps} combat steps");

        var guarded = new RunEngine { Kernel = kernel };
        guarded.Reset("PRB00058", ascension: 8);
        for (int step = 0; guarded.State.Phase != RunPhase.Combat && step < 200; step++)
        {
            guarded.WriteActionMask(mask);
            guarded.Step(Array.IndexOf(mask, 1), -1, out _, out _, out _);
        }

        guarded.State.ActiveCombat!.Kernel = null;
        guarded.WriteActionMask(mask);
        Assert.Throws<InvalidOperationException>(() => guarded.Step(Array.IndexOf(mask, 1), -1, out _, out _, out _));
        guarded.BindActiveCombat();
        Assert.NotNull(guarded.State.ActiveCombat!.Kernel);
    }

    private static RunEngine InCombat(Kernel kernel)
    {
        var e = new RunEngine { Kernel = kernel };
        e.Reset("PRB00058", ascension: 8);
        var mask = new int[RunConstants.MaxActions];
        for (int step = 0; e.State.Phase != RunPhase.Combat && step < 200; step++)
        {
            e.WriteActionMask(mask);
            e.Step(Array.IndexOf(mask, 1), -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Combat, e.State.Phase);
        return e;
    }

    private static (int, int, int, int, int) Shape(CombatState c) =>
        (c.Hand.Count, c.DiscardPile.Count, c.DrawPile.Count, c.Energy, c.CardPlaysThisTurn);

    [Fact]
    public void ACanceledActionDoesNothing()
    {
        // game: ActionQueueSet.Enqueue cancels an action its queue refuses (one cancelling player-driven actions), and
        // ActionExecutor never runs a canceled action's body. The old step does not run in its place: the action is
        // refused (status -1) and nothing changes.
        var kernel = new Kernel(new KernelOptions([Switch.F10], TestGroups), () => Ids, []);
        var mask = new int[RunConstants.MaxActions];
        var control = InCombat(kernel);
        var e = InCombat(kernel);
        var before = Shape(e.State.ActiveCombat!);
        Assert.Equal(before, Shape(control.State.ActiveCombat!));
        control.WriteActionMask(mask);
        int action = Array.IndexOf(mask, 1);
        Assert.Equal(0, control.Step(action, 0, out _, out _, out _));
        Assert.NotEqual(before, Shape(control.State.ActiveCombat!));

        e.State.ActiveCombat!.Kernel!.Queue!.StartCancellingPlayerDriven();
        int status = e.Step(action, 0, out float reward, out bool terminal, out _);
        Assert.Equal((-1, 0f, false), (status, reward, terminal));
        Assert.Equal(before, Shape(e.State.ActiveCombat!));
        Assert.Equal(GameActionState.Canceled, e.State.ActiveCombat.Kernel!.Queue!.All[^1].State);
    }
}
