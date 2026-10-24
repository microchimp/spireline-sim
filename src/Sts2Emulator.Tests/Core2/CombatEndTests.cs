// Core2's combat end (docs/rebuild/MIGRATION_F08.md section 3 item 7 and section 2's F17 hand-off, round R3b):
// CombatManager.CheckWinCondition at the executor's boundary and the turn-flow points, IsEnding with
// ShouldStopCombatFromEnding, the pending loss, ActionQueueSet.CombatEnded, and the hand-off to the old FinishCombat once.
// Test-only handlers stand in for the models they name (registered under the F08+F10 switch); each test cites the game
// method it is read from.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class CombatEndTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];

    private static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = 16, Hp = hp, MaxHp = hp }).ToList();
        s.Relics = [new RelicInstance(1)];
        return s;
    }

    private static KernelState K(CombatState s, Action<HandlerTable>? models = null) =>
        new Kernel(new KernelOptions(DamageOnly), () => Ids, models is null ? [] : [models]).Bind(s);

    private static readonly string Relic = RelicName();

    private static string RelicName()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        var map = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!;
        return map["relics"]!.AsObject().First(kv => (int)kv.Value! == 1).Key;
    }

    /// <summary>The executor's host and the end's: a play body runs the test's action; the end is logged.</summary>
    private sealed class Host(List<string> log) : IActionHost, ICombatEndHost
    {
        public Action<KernelState>? Body;

        public bool Run(KernelState k, GameActionRec a)
        {
            log.Add($"run {a.Kind}");
            Body?.Invoke(k);
            log.Add($"done {a.Kind}");
            return false;
        }

        public bool InProgress => true;

        public void CheckWin(KernelState k)
        {
            log.Add("check");
            CombatEnd.AfterAction(k, this);
        }

        public void EndCombat(KernelState k, bool playerWon)
        {
            Assert.False(CombatEnd.IsInProgress(k));
            log.Add($"end {(playerWon ? "won" : "lost")}");
        }
    }

    private static GameActionRec Play(KernelState k) =>
        k.Queue!.Enqueue(new GameActionRec(GameActionKind.PlayCard, GameActionRec.TypeOf(GameActionKind.PlayCard)));

    [Fact]
    public void AVictoryEndsTheCombatAtTheBoundaryNotMidAction()
    {
        // game: ActionExecutor.ExecuteActions runs the action to its end, then CheckWinCondition: IsEnding (no living
        // primary enemy), EndCombatInternal with IsInProgress false first. The kill inside the action checks nothing.
        var s = Combat(5);
        var k = K(s);
        var log = new List<string>();
        var host = new Host(log)
        {
            Body = kk =>
            {
                CreatureCmd.Kill(kk, [OldCreature.Of(s.Enemies[0])]);
                log.Add($"after kill: in progress {CombatEnd.IsInProgress(kk)}, ending {CombatEnd.IsEnding(kk)}");
            },
        };
        Play(k);
        ActionExecutor.Drain(k, host);
        Assert.Equal(["run PlayCard", "after kill: in progress True, ending True", "done PlayCard", "check", "end won"], log);
        Assert.False(CombatEnd.IsInProgress(k));
        Assert.False(CombatEnd.IsEnding(k));
        Assert.True(CombatEnd.IsOverOrEnding(k));
        Assert.Equal((1, true), (k.Progress.HandOffs, k.Progress.PlayerWon!.Value));
        // A later check finds the combat over, not ending: no second end.
        Assert.False(CombatEnd.CheckWinCondition(k, host));
        Assert.Equal(1, k.Progress.HandOffs);
    }

    [Fact]
    public void ALivingPrimaryEnemyOrAMinionLeftDecideTheCombat()
    {
        // game: CombatManager.IsEnding asks for a living primary enemy; a minion (MinionPower: OwnerIsSecondaryEnemy) left
        // alive does not hold the combat.
        var s = Combat(5, 5);
        var k = K(s);
        var log = new List<string>();
        var host = new Host(log);
        Assert.False(CombatEnd.CheckWinCondition(k, host));
        BuffSystem.Apply(s.Enemies[1].Buffs, BuffId.Minion, 1);
        CreatureCmd.Kill(k, [OldCreature.Of(s.Enemies[0])]);
        Assert.True(CombatEnd.CheckWinCondition(k, host));
        Assert.Equal(["end won"], log);
    }

    [Fact]
    public void ShouldStopCombatFromEndingHoldsTheCombatOpen()
    {
        // game: CombatManager.IsEnding: no living primary enemy, but a listener of ShouldStopCombatFromEnding says so: not
        // ending; once none does, the next check ends it.
        bool hold = true;
        int asked = 0;
        var s = Combat(5);
        var k = K(s, t => t.Model(ListenerKind.Relic, Relic).On(HookId.ShouldStopCombatFromEnding,
            (HookPredicate<ShouldStopCombatFromEndingArgs>)((ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
            {
                asked++;
                return hold;
            })));
        var log = new List<string>();
        var host = new Host(log);
        Assert.False(CombatEnd.CheckWinCondition(k, host));
        Assert.Equal(0, asked);  // a living primary enemy: the hook is not asked
        CreatureCmd.Kill(k, [OldCreature.Of(s.Enemies[0])]);
        Assert.False(CombatEnd.IsEnding(k));
        Assert.False(CombatEnd.CheckWinCondition(k, host));
        Assert.True(CombatEnd.IsInProgress(k));
        Assert.Empty(log);
        hold = false;
        Assert.True(CombatEnd.CheckWinCondition(k, host));
        Assert.Equal(["end won"], log);
        Assert.True(asked >= 3);
    }

    [Fact]
    public void ThePlayersDeathIsAPendingLossProcessedAtTheNextCheckAndBeatsAVictory()
    {
        // game: CreatureCmd.Kill: every player dead and the combat in progress: LoseCombat marks the loss pending (nothing
        // else runs; IsEnding and IsOverOrEnding turn true, IsInProgress stays true). CheckWinCondition takes the pending
        // loss first (ProcessPendingLoss: IsInProgress false), even with the last enemy dead at the same point.
        var s = Combat(5);
        var k = K(s);
        var log = new List<string>();
        var host = new Host(log)
        {
            Body = kk =>
            {
                CreatureCmd.Kill(kk, [OldCreature.Player]);
                log.Add($"pending {CombatEnd.IsAboutToLose(kk)}, in progress {CombatEnd.IsInProgress(kk)}, ending {CombatEnd.IsEnding(kk)}");
                CreatureCmd.Kill(kk, [OldCreature.Of(s.Enemies[0])]);
            },
        };
        Play(k);
        ActionExecutor.Drain(k, host);
        Assert.Equal(["run PlayCard", "pending True, in progress True, ending True", "done PlayCard", "check", "end lost"], log);
        Assert.Equal((false, false, 1, false), (CombatEnd.IsInProgress(k), CombatEnd.IsAboutToLose(k), k.Progress.HandOffs,
            k.Progress.PlayerWon!.Value));
        Assert.Equal(1, k.Commands.LossCalls);
        // A kill of the dead player after the end reaches the loss branch but marks nothing (not in progress).
        CreatureCmd.Kill(k, [OldCreature.Player]);
        Assert.Equal((2, false), (k.Commands.LossCalls, CombatEnd.IsAboutToLose(k)));
    }

    [Fact]
    public void CombatEndedCancelsTheHookActionsThatHaveNotStartedAfterAVictory()
    {
        // game: EndCombatInternal ends with ActionQueueSynchronizer's NotInCombat: ActionQueueSet.CombatEnded cancels every
        // combat action not executing (a hook action an AfterDeath listener enqueued during the killing action), so the
        // executor runs none of them; an action enqueued after the end is cancelled as it is enqueued.
        var s = Combat(5);
        var k = K(s);
        var log = new List<string>();
        GameActionRec? hook = null;
        var host = new Host(log)
        {
            Body = kk =>
            {
                CreatureCmd.Kill(kk, [OldCreature.Of(s.Enemies[0])]);
                hook = kk.Queue!.Enqueue(new GameActionRec(GameActionKind.Hook, GameActionType.Combat));
            },
        };
        Play(k);
        ActionExecutor.Drain(k, host);
        Assert.Equal(["run PlayCard", "done PlayCard", "check", "end won"], log);
        Assert.Equal(GameActionState.Canceled, hook!.State);
        Assert.Null(k.Queue!.GetReadyAction());
        Assert.Equal(GameActionState.Canceled, Play(k).State);
    }

    [Fact]
    public void TheTurnFlowPointsAnswerAsTheGamesCallers()
    {
        // game: CombatManager: StartTurn and ExecuteEnemyTurn go on while IsInProgress; EndEnemyTurn switches sides on
        // !IsEnding (true once the combat is over); EndPlayerTurnPhaseOneInternal stops after BeforeTurnEnd when the check
        // ended the combat.
        var s = Combat(5);
        var k = K(s);
        var log = new List<string>();
        var host = new Host(log);
        Assert.True(CombatEnd.StartPlayerTurn(k, host));
        Assert.True(CombatEnd.StartEnemyTurn(k, host));
        Assert.True(CombatEnd.AfterEnemyTakesTurn(k, host));
        Assert.True(CombatEnd.EndEnemyTurn(k, host));
        Assert.False(CombatEnd.AfterBeforeTurnEnd(k, host));
        Assert.False(CombatEnd.EndPlayerTurnPhaseOne(k, host));
        CreatureCmd.Kill(k, [OldCreature.Of(s.Enemies[0])]);
        Assert.False(CombatEnd.AfterEnemyTakesTurn(k, host));
        Assert.True(CombatEnd.EndEnemyTurn(k, host));
        Assert.False(CombatEnd.StartPlayerTurn(k, host));
        Assert.Equal(["end won"], log);
        Assert.Equal(9, k.Progress.Checks);
    }

    [Fact]
    public void TheProgressIsCopiedWithTheKernelState()
    {
        var s = Combat(5);
        var k = K(s);
        CreatureCmd.Kill(k, [OldCreature.Player]);
        var copy = s.Clone();
        var k2 = copy.Kernel!;
        Assert.NotSame(k.Progress, k2.Progress);
        Assert.True(CombatEnd.IsAboutToLose(k2));
        CombatEnd.CheckWinCondition(k2, new Host([]));
        Assert.True(CombatEnd.IsAboutToLose(k));
        Assert.True(CombatEnd.IsInProgress(k));
    }

    [Fact]
    public void FinishCombatRunsOncePerCombatThroughTheHandOffAndTheRunPlaysAsTheOldEngine()
    {
        // The F17 hand-off: with F08+F10 on, the old step's end of a fight goes to Core2's CheckWinCondition, which calls
        // the run engine's FinishCombat once, after IsInProgress went false: a victory (victory heal, rewards, counters)
        // and, as the old engine does, a loss. The whole run is the old engine's: same masks, results, HP and phase.
        var kernel = new Kernel(new KernelOptions(DamageOnly), () => Ids, []);
        var old = new RunEngine();
        var core2 = new RunEngine { Kernel = kernel };
        old.Reset("PRB00058", ascension: 8);
        core2.Reset("PRB00058", ascension: 8);
        var mask = new int[RunConstants.MaxActions];
        var mask2 = new int[RunConstants.MaxActions];
        bool terminal = false;
        CombatState? fight = null;
        int won = 0, lost = 0;
        var seen = new List<string>();
        for (int step = 0; !terminal && step < 20_000; step++)
        {
            old.WriteActionMask(mask);
            core2.WriteActionMask(mask2);
            Assert.Equal(mask, mask2);
            int action = Array.IndexOf(mask, 1);
            fight = core2.State.Phase == RunPhase.Combat ? core2.State.ActiveCombat : null;
            int s1 = old.Step(action, -1, out float r1, out terminal, out _);
            int s2 = core2.Step(action, -1, out float r2, out bool t2, out _);
            Assert.Equal((s1, r1, terminal), (s2, r2, t2));
            Assert.Equal((old.State.PlayerHp, old.State.Gold, old.State.Phase), (core2.State.PlayerHp, core2.State.Gold, core2.State.Phase));
            Assert.Equal(old.FinishCombatCalls, core2.FinishCombatCalls);
            if (fight is not null && (core2.State.Phase != RunPhase.Combat || !ReferenceEquals(core2.State.ActiveCombat, fight)))
            {
                var p = fight.Kernel!.Progress;
                Assert.Equal((1, false), (p.HandOffs, p.InProgress));
                Assert.Equal(old.State.LastPlayerWon, p.PlayerWon);
                if (p.PlayerWon == true)
                    won++;
                else
                {
                    lost++;
                    seen.Add($"hp {fight.PlayerHp} enemies {string.Join(",", fight.Enemies.Select(e => e.Hp))} esc {fight.EscapedEnemies} phase {core2.State.Phase} t {t2}");
                }
            }
        }

        Assert.True(terminal);
        Assert.Equal(won + lost, core2.FinishCombatCalls);
        Assert.True(won > 0, "no combat won");
        Assert.True(core2.State.PlayerHp <= 0 ? lost == 1 : lost == 0, string.Join(" | ", seen));
    }
}
