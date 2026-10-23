// The re-anchor steps the fight it anchored through the engine's entry gate (docs/rebuild/TIER1_KERNEL.md section 5): with
// a switch on (a core2 engine) every combat it steps carries its KernelState and goes through the action queue; a combat
// without one is refused. With no switch the step is the combat engine's own.
using EmuDump.Engines.Core2;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

public class ReanchorKernelGateTests
{
    static Core2Engine Switched() =>
        new(new Sts2Emulator.Core2.KernelOptions([Sts2Emulator.Core2.Switch.F10], [[Sts2Emulator.Core2.Switch.F10]]));

    /// <summary>Headbutt in hand, two cards in the discard: its play stops on the pick.</summary>
    static CombatState Headbutt()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.Headbutt, false)];
        state.DiscardPile = [new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.Bash, false)];
        state.Energy = 3;
        return state;
    }

    [Fact]
    public void WithASwitchOnReanchorStepsOnlyBoundCombats()
    {
        var engine = Switched();
        var run = engine.Run;

        // An anchored combat nobody bound is refused, at a step and at the finish of a stopped step.
        var unbound = Headbutt();
        Assert.Null(unbound.Kernel);
        Assert.Throws<InvalidOperationException>(() => run.StepCombat(unbound, 0, new CountingRandom(1), 0));
        Assert.Equal(0, unbound.CardPlaysThisTurn);
        var stopped = Headbutt();
        var stoppedRng = new CountingRandom(1);
        CombatEngine.Step(stopped, 0, stoppedRng);
        Assert.NotNull(stopped.PendingSelection?.Replay);
        Assert.Throws<InvalidOperationException>(() => OldReanchor.Finish(run, stopped, stoppedRng));

        // Bound where it was anchored, it is stepped through the queue: the action is enqueued, runs, and finishes; the
        // answer to the screen it stopped on resumes that action (a new id) and the step is replayed.
        var state = Headbutt();
        var rng = new CountingRandom(1);
        run.BindCombat(state);
        Assert.NotNull(state.Kernel);
        run.StepCombat(state, 0, rng, 0);
        var action = Assert.Single(state.Kernel!.Queue!.All);
        uint firstId = action.Id!.Value;
        Assert.Equal(Sts2Emulator.Core2.GameActionState.GatheringPlayerChoice, action.State);
        Assert.False(OldReanchor.Finish(run, state, rng));
        Assert.Null(state.PendingSelection);
        Assert.Equal(1, state.CardPlaysThisTurn);
        Assert.Contains(state.DiscardPile, c => c.DefId == IC.Headbutt);
        // (The replay restored the kernel state of the step's start, queue included: the queue is read again.)
        var done = Assert.Single(state.Kernel!.Queue!.All);
        Assert.Equal(Sts2Emulator.Core2.GameActionState.Finished, done.State);
        Assert.NotEqual(firstId, done.Id);
        Assert.Equal(firstId, done.FirstId);

        // A clone of a bound combat is bound (the step after the anchor clones it).
        Assert.NotNull(state.Clone().Kernel);
        Assert.NotSame(state.Kernel, state.Clone().Kernel);
    }

    [Fact]
    public void WithNoSwitchAReanchorStepIsTheCombatEnginesAndBindsNothing()
    {
        var run = new OldEngine().Run;
        var state = Headbutt();
        run.BindCombat(state);
        Assert.Null(state.Kernel);
        run.StepCombat(state, 0, new CountingRandom(1), 0);
        Assert.NotNull(state.PendingSelection?.Replay);
        Assert.Null(state.Kernel);
    }
}
