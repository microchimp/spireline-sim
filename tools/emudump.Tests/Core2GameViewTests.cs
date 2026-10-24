// The core2 engine's game view with the kernel's own state (Engines/Core2/Core2GameView.cs, TIER1_KERNEL.md section 6): with a
// test switch on and a test handler that pauses, the view shows the pending hook action and its screen; a copy of the state
// writes the same kernel block; with no switch on the view is the old engine's, byte for byte.
using System.Text.Json.Nodes;
using EmuDump.Engines.Core2;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

public class Core2GameViewTests
{
    static readonly OldModelIds Ids = OldModelIds.Load(Path.Combine(AppContext.BaseDirectory, "id_map.json"),
        Path.Combine(AppContext.BaseDirectory, "power_aliases.json"));

    static readonly Sts2Emulator.Core2.Switch[] Every = Enum.GetValues<Sts2Emulator.Core2.Switch>().Where(s => s != Sts2Emulator.Core2.Switch.None).ToArray();

    /// <summary>A kernel screen: the old engine's selection over the hand.</summary>
    sealed record Screen(string Who) : IKernelScreen
    {
        public PendingCardSelection Open(CombatState state) =>
            new() { Kind = CardSelectionKind.None, Candidates = [0, 1], SourceCardDefId = 0 };
    }

    /// <summary>A resume record that takes HP, so a copy's run can be compared with its source's.</summary>
    sealed record Take(string Who) : HookResume
    {
        public override HookStep Continue(HookRuntime rt, int answer)
        {
            ((KernelState)rt).World!.Combat!.PlayerHp -= answer + 1;
            return HookStep.Done;
        }
    }

    sealed class Host : IActionHost
    {
        public bool InProgress => true;

        public bool Run(KernelState k, GameActionRec a) => false;

        public void CheckWin(KernelState k)
        {
        }
    }

    static (KernelState K, CombatState C) WithHookAction()
    {
        var c = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        c.Enemies = [new EnemyState { DefId = 16, Hp = 40, MaxHp = 40 }];
        c.Relics = [new RelicInstance(1)];
        var k = new Kernel(new KernelOptions(Every), () => Ids).Bind(c);
        var who = new ListenerRef((int)ListenerKind.Relic, 1, 1);
        var ctx = k.NewHookContext(HookId.Dispatch_AfterDeath, who);
        // The test handler's pause: a per-listener listener asked a question (the game's HookPlayerChoiceContext).
        k.OnPaused(new PauseSite(HookId.Dispatch_AfterDeath, HookId.AfterDeath, 0, 1, who, ContextMode.PerListener, ctx),
            HookStep.Paused(new Screen("relic"), new Take("relic")));
        return (k, c);
    }

    [Fact]
    public void TheViewShowsThePendingHookActionAndItsScreen()
    {
        var (k, c) = WithHookAction();
        ActionExecutor.Drain(k, new Host());
        var b = Core2GameView.Block(k, c);

        var queued = Assert.Single(b["queue"]!["actions"]!.AsArray())!;
        Assert.Equal(("Hook", "Combat", "GatheringPlayerChoice", 0), ((string)queued["kind"]!, (string)queued["type"]!, (string)queued["state"]!, (int)queued["owner"]!));
        var pending = Assert.Single(b["hook_actions"]!.AsArray())!;
        Assert.Equal(("Dispatch_AfterDeath", "AfterDeath", "PerListener"), ((string)pending["dispatcher"]!, (string)pending["hook"]!, (string)pending["context"]!));
        Assert.Equal(("Screen", "relic"), ((string)pending["screen"]!["type"]!, (string)pending["screen"]!["Who"]!));
        Assert.Equal("Take", (string)pending["resume"]!["type"]!);
        Assert.Equal((long)queued["hook_action"]!, (long)pending["id"]!);
        var screen = b["screen"]!;
        Assert.Equal((uint)queued["id"]!, (uint)screen["hook_action_id"]!);
        Assert.Equal(2, (int)screen["candidates"]!);
        Assert.False((bool)b["turn_start_flow"]!["paused"]!);
        Assert.Equal([0u], b["queue"]!["queues"]![0]!["front_first"]!.AsArray().Select(n => (uint)n!));

        // The flow pause is the turn start's own hook action, CombatPlayPhaseOnly, with no recorded hook action behind it.
        var flow = k.Queue!.Enqueue(new GameActionRec(GameActionKind.Hook, GameActionType.CombatPlayPhaseOnly));
        var f = Core2GameView.Block(k, c)["turn_start_flow"]!;
        Assert.Equal((true, flow.Id, "WaitingForExecution"), ((bool)f["paused"]!, (uint?)(uint)f["action"]!, (string)f["state"]!));
    }

    [Fact]
    public void ACopyOfTheStateWritesTheSameKernelBlockBeforeAndAfterTheAnswer()
    {
        var (k, c) = WithHookAction();
        ActionExecutor.Drain(k, new Host());
        var copy = c.Clone();
        Assert.NotSame(c.Kernel, copy.Kernel);
        Assert.Equal(Core2GameView.Block(k, c).ToJsonString(), Core2GameView.Block(copy.Kernel!, copy).ToJsonString());

        uint id = k.Queue!.All.Single().Id!.Value;
        foreach (var s in new[] { copy, c })
        {
            s.PendingSelection = null;
            s.Kernel!.Answer(id, 1);
            ActionExecutor.Drain(s.Kernel, new Host());
        }

        var done = Core2GameView.Block(k, c);
        Assert.Equal(done.ToJsonString(), Core2GameView.Block(copy.Kernel!, copy).ToJsonString());
        Assert.Null(done["screen"]);
        Assert.Equal("Finished", (string)done["queue"]!["actions"]![0]!["state"]!);
        Assert.NotNull(done["queue"]!["actions"]![0]!["first_id"]);  // resumed under a new id

        // The kernel's sequencing counters under the game's field names, in the net shape; the copy writes the same.
        var net = new JsonObject();
        Core2GameView.Sequencing(k, net);
        var net2 = new JsonObject();
        Core2GameView.Sequencing(copy.Kernel!, net2);
        Assert.Equal(net.ToJsonString(), net2.ToJsonString());
        Assert.Equal((uint)done["queue"]!["actions"]![0]!["id"]!, (uint)net["last_action"]!);
        // last_hook is RecordedHookAction.Id (1-based), not the game's HookId (0-based, from _nextHookId): the first hook action
        // is 1 here, so the value cannot match a record and the comparer lists the path as surplus.
        Assert.Equal((long)done["queue"]!["actions"]![0]!["hook_action"]!, (long)net["last_hook"]!);
        Assert.Equal(1L, (long)net["last_hook"]!);
        Assert.Equal(48, copy.PlayerHp);
    }

    [Fact]
    public void WithNoSwitchOnTheViewIsTheOldEnginesAndHoldsNoKernelBlock()
    {
        var core2 = new Core2Engine(KernelOptions.None);
        var old = new OldEngine();
        core2.Run.Reset("K5VIEW", 0);
        old.Run.Reset("K5VIEW", 0);
        Assert.Equal(old.GameView().ToJsonString(), core2.GameView().ToJsonString());
        Assert.Null(core2.GameView()["private"]?["kernel"]);

        // A switch on adds the kernel block to the same view and changes nothing else.
        var on = new Core2Engine(new KernelOptions(Every));
        on.Run.Reset("K5VIEW", 0);
        var view = on.GameView();
        Assert.NotNull(view["private"]!["kernel"]);
        var plain = (JsonObject)JsonNode.Parse(view.ToJsonString())!;
        plain["private"]!.AsObject().Remove("kernel");
        Assert.Equal(old.GameView().ToJsonString(), plain.ToJsonString());
    }
}
