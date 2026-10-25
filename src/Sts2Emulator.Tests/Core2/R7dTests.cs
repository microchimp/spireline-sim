// Round R7d, the R7 proof's blockers (docs/rebuild/MIGRATION_F08.md, "R7 proof on a42c41a3"). With F08+F10 on the combat's
// end is Core2's CheckWinCondition, called at the game's points only: the old step's own HP reading (Settle) is not the end.
// The old turn start had no such point, so a kill or a loss there (Inferno at the player's turn start, a power's damage)
// left the combat in progress with nothing standing: the run went on in combat where the game had ended it. The game checks
// in CombatManager.StartTurn, the player side after the auto-pre-play phase and the enemy side before ExecuteEnemyTurn.
// Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class R7dTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;

    /// <summary>A run with F08+F10 on, stepped (first legal action) to its first combat's player turn.</summary>
    private static RunEngine IntoCombat()
    {
        var e = new RunEngine { Kernel = new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => Ids, []) };
        e.Reset("PRB00058", ascension: 8);
        var mask = new int[RunConstants.MaxActions];
        for (int step = 0; e.State.Phase != RunPhase.Combat && step < 200; step++)
        {
            e.WriteActionMask(mask);
            e.Step(Array.IndexOf(mask, 1), -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Combat, e.State.Phase);
        Assert.NotNull(e.State.ActiveCombat!.Kernel);
        return e;
    }

    /// <summary>Ends the turn and checks Core2 ended the combat once (won or lost) and handed it off to FinishCombat.</summary>
    private static void EndTurnAndAssertTheEnd(RunEngine e, bool won, int turns)
    {
        var c = e.State.ActiveCombat!;
        int calls = e.FinishCombatCalls, turn = c.Turn;
        e.Step(c.Hand.Count, -1, out _, out _, out _);  // end the turn
        var p = c.Kernel!.Progress;
        Assert.Equal((false, won, 1), (p.InProgress, p.PlayerWon == true, p.HandOffs));
        Assert.Equal(calls + 1, e.FinishCombatCalls);  // a loss too: the hand-off runs FinishCombat, as the old engine does
        Assert.Equal(turn + turns, c.Turn);
        Assert.NotEqual(RunPhase.Combat, e.State.Phase);
    }

    [Fact]
    public void AKillAtThePlayersTurnStartEndsTheCombatThere()
    {
        var e = IntoCombat();
        var c = e.State.ActiveCombat!;
        // Every enemy at 1 HP; Rolling Boulder's turn-start hit (AfterPlayerTurnStart) kills them all after their turn,
        // through any block their moves add. The player side's check (after the auto-pre-play phase) is the end.
        foreach (var en in c.Enemies)
        {
            en.Hp = 1;
            en.Block = 0;
        }

        c.PlayerHp = c.PlayerMaxHp = 500;
        BuffSystem.Apply(c.PlayerBuffs, BuffId.RollingBoulderPower, 999);
        EndTurnAndAssertTheEnd(e, won: true, turns: 1);
    }

    [Fact]
    public void ALossAtThePlayersTurnStartEndsTheCombatThere()
    {
        var e = IntoCombat();
        var c = e.State.ActiveCombat!;
        // The enemies' hits land on the block; the turn start clears it and Inferno's self-damage takes the last HP: the
        // player side's check at the turn start's player-death return ends the combat, lost.
        c.PlayerHp = 1;
        c.PlayerBlock = 999;
        BuffSystem.Apply(c.PlayerBuffs, BuffId.Inferno, 1);
        BuffSystem.Apply(c.PlayerBuffs, BuffId.InfernoSelfDamage, 1);
        EndTurnAndAssertTheEnd(e, won: false, turns: 1);
    }

    [Fact]
    public void ADoomKillAtThePlayersTurnEndEndsTheCombatBeforeTheEnemiesAct()
    {
        var e = IntoCombat();
        var c = e.State.ActiveCombat!;
        // Doom at least each enemy's HP: the turn end's Doom kills them all, and the combat ends before the enemy side acts.
        // The turn end's own check ends it (the test passes with the enemy side's StartTurnWinCheck off): in the old order
        // nothing between that check and the enemy side's deals damage (MIGRATION_F08, R7e), so this guards the turn end.
        foreach (var en in c.Enemies)
        {
            BuffSystem.Apply(en.Buffs, BuffId.Doom, en.Hp);
        }

        c.PlayerHp = c.PlayerMaxHp = 500;
        EndTurnAndAssertTheEnd(e, won: true, turns: 0);
    }

    [Fact]
    public void ADemiseKillAtTheEnemyTurnsEndLeavesThePassOverTheRosterItBegan()
    {
        var e = IntoCombat();
        var c = e.State.ActiveCombat!;
        // Demise kills its owner at the enemy side's turn end; under the switch the kill removes it from the roster inside
        // the pass (it threw "Collection was modified"), and the combat then ends at the game's next point.
        foreach (var en in c.Enemies)
        {
            en.Hp = 2;
            en.Block = 0;
            BuffSystem.Apply(en.Buffs, BuffId.Demise, 5);
        }

        c.PlayerHp = c.PlayerMaxHp = 500;
        c.PlayerBlock = 999;
        e.Step(c.Hand.Count, -1, out _, out _, out _);
        var p = c.Kernel!.Progress;
        Assert.Equal((false, true), (p.InProgress, p.PlayerWon == true));
        Assert.NotEqual(RunPhase.Combat, e.State.Phase);
    }
}
