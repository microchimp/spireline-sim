using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Battleworn Dummy's clock runs out in the turn end, and with the kernel on (F08) the dummy ESCAPES: it leaves
/// the roster in the middle of the pass that ticks the clock. The pass has to survive that. It once walked the live
/// roster, so the walk's next step threw "Collection was modified" and, in a native build, took the whole process
/// down (real-arm workers, 38 deaths from 2026-10-26; seeds E4EY632BRX and FR9V3LENXE reach the dummy fight).
/// game: the hook dispatch (`CombatState.IterateHookListeners`) first copies every listener of every creature into a
/// list and walks that, so a creature that leaves while a hook runs does not change the walk.
/// </summary>
public class EndTurnEscapeWalkTests
{
    private static StepResult StepWithKernel(CombatState state, int action)
    {
        // Its own kernel, as TestSwitches makes one, so the case does not depend on STS2_TEST_SWITCHES.
        var kernel = new Sts2Emulator.Core2.Kernel(
            Sts2Emulator.Core2.KernelOptions.FromFamilies(["F08", "F10"]),
            () => Core2.ListenerOrderTests.Ids,
            Sts2Emulator.Core2.Models.Registry.All
        );
        var engine = new RunEngine { Kernel = kernel };
        CardUids.NumberNew(state);
        kernel.Attach(state, engine.State);
        return engine.StepCombat(state, action, new CountingRandom(0));
    }

    [Fact]
    public void Dummy1EscapingAtTurnEndDoesNotBreakTheRosterWalk() =>
        DummyEscapes(CombatFactory.ActOneEncounter.BattlewornDummy1);

    [Fact]
    public void Dummy2EscapingAtTurnEndDoesNotBreakTheRosterWalk() =>
        DummyEscapes(CombatFactory.ActOneEncounter.BattlewornDummy2);

    [Fact]
    public void Dummy3EscapingAtTurnEndDoesNotBreakTheRosterWalk() =>
        DummyEscapes(CombatFactory.ActOneEncounter.BattlewornDummy3);

    private static void DummyEscapes(CombatFactory.ActOneEncounter encounter)
    {
        var fight = Fight.Encounter(encounter);
        var dummy = Assert.Single(fight.State.Enemies);

        // One turn left on the clock: this turn end is the one that ends it.
        BuffSystem.Remove(dummy.Buffs, BuffId.BattlewornDummyTimeLimit);
        BuffSystem.Apply(dummy.Buffs, BuffId.BattlewornDummyTimeLimit, 1);

        StepWithKernel(fight.State, fight.State.Hand.Count);

        Assert.True(dummy.Escaped);
        Assert.True(fight.State.BattlewornDummyRanOutOfTime);
        Assert.Empty(fight.State.Enemies);
    }
}
