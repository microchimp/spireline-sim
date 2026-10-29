// A screen inside a card play under core2 (the kernel-queue-busy refusals of the 2026-10-29 coverage run): the game pauses
// the play's action for the player's choice, so the screen's queue holds it, gathering, and the stopped step's start holds
// it executing. The key wrote no queue and refused any that was not idle, so every such screen was refused, on both sides
// of the equal-value harness alike. The key now writes the queued player actions as plain data (FIGHT_SOLVER.md section 2).
// Reduced from g5-emu HUR7D64TNM k0046 (an Armaments with upgradable cards in hand, refused at its first move).
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class KernelQueueKeyTests
{
    static int Card(string name) => Cards.FindId(name)!.Value;

    /// <summary>Turn one with Armaments, two Strikes and two Defends in hand, the draw pile empty.</summary>
    static (CombatState State, CountingRandom Rng) Armaments(string engine)
    {
        var (s, rng) = SolverPositions.Fight(engine, 46,
            [Card("Armaments"), Card("StrikeIronclad"), Card("StrikeIronclad"), Card("DefendIronclad"), Card("DefendIronclad")], [], []);
        Assert.Empty(s.DrawPile);
        Assert.Contains(s.Hand, c => c.DefId == Card("Armaments"));
        foreach (var e in s.Enemies)
        {
            e.Hp = e.MaxHp = 200;
        }

        return (s, rng);
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void AScreenInsideACardPlayIsSolvedAndEqualsTheReference(string engine)
    {
        var (s, rng) = Armaments(engine);
        var r = new Expectimax(new SolveOptions { MaxHorizon = 0, UseEstimate = false }).SolveExpected(s, rng);
        Assert.True(r.Status != SolveStatus.Refused, r.Reason);
        Assert.True(r.Counters.Screens > 0, "the Armaments screen was not reached");
        var reference = Reference.Solve(s, rng, 0, tightBound: true);
        Assert.Null(reference.Refused);
        Assert.Equal(reference.Value, r.Value);
        Assert.Equal(reference.Exact, r.Exact);
    }

    [Fact]
    public void TheScreensQueueHoldsThePlayAndTheKeyWritesIt()
    {
        var (s, rng) = Armaments("core2");
        int play = s.Hand.FindIndex(c => c.DefId == Card("Armaments"));
        var screen = s.Clone();
        KernelGate.Step(screen.Kernel!.Shared, screen, rng.Copy(), play);
        Assert.NotNull(screen.PendingSelection?.Replay);
        var front = Assert.Single(screen.Kernel!.Queue!.QueueOf(0));
        Assert.Equal((GameActionKind.PlayCard, GameActionState.GatheringPlayerChoice, play), (front.Kind, front.State, front.Action));
        var started = Assert.Single(screen.PendingSelection!.Replay!.Before.Kernel!.Queue!.QueueOf(0));
        Assert.Equal(GameActionState.Executing, started.State);
        // The key of the screen is written (it was refused), and it is a key of the queue: the same screen with its
        // action no longer paused keys apart.
        var key = StateKey.Encode(screen);
        Assert.Equal(key, StateKey.Encode(screen.Clone()));
        var idle = screen.Clone();
        idle.Kernel!.Queue!.Finish(idle.Kernel.Queue.QueueOf(0)[0]);
        Assert.Empty(idle.Kernel.Queue.QueueOf(0));
        Assert.NotEqual(key, StateKey.Encode(idle));
    }
}
