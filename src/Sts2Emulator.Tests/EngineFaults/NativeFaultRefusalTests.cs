// The engine fault refusal (docs/ENGINE_FAULTS.md): an exception out of the engine's step at a native entry point is a
// counted refusal with its class and top frame, where it used to end the process. The hazard is engine fault class 2's
// state, whose GameHangs is the one fault the copy is known to meet after the four fixes.
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.NativeKernelHandleTests;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.EngineFaults;

[Collection(nameof(NativeEngineCollection))]
public unsafe class NativeFaultRefusalTests : IDisposable
{
    public NativeFaultRefusalTests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");

    /// <summary>Pillage and a Defend in hand, Hellraiser and Ringing on, only Strikes in the draw and discard piles.</summary>
    private static void Hazard(CombatState s)
    {
        s.PendingSelection = null;
        s.Hand = [Card(IC.Pillage), Card(IC.DefendIronclad)];
        s.DrawPile.Clear();
        s.DrawPile.AddRange([Card(IC.StrikeIronclad), Card(IC.StrikeIronclad)]);
        s.DiscardPile.Clear();
        s.DiscardPile.Add(Card(IC.PommelStrike));
        s.Energy = 3;
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Hellraiser, 1);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Ringing, 1);
        CardUids.NumberNew(s);
    }

    private static int Faults()
    {
        delegate* unmanaged<int> count = &NativeExports.Sts2_EngineFaults;
        return count();
    }

    [Fact]
    public void ACombatStepThatThrowsIsRefusedWithItsClassAndFrameAndTheHandleStaysRefused()
    {
        int h = Create(11);
        int copy = -1;
        try
        {
            Hazard(NativeExports.StateOf(h)!);
            int before = Faults();

            Assert.Equal(EngineFault.Code, Step(h, 0));
            string fault = LastError();
            var parts = fault.Split('\t');
            Assert.Equal("GameHangs", parts[0]);
            Assert.Contains("EndlessDraw", parts[1]);
            Assert.Contains("Pillage draws forever", parts[2]);
            Assert.Equal(before + 1, Faults());

            // The combat is part way through the Pillage: every later step is refused with the same fault, a clone of it
            // too, and none of them counts again.
            Assert.Equal(EngineFault.Code, Step(h, 1));
            Assert.Equal(fault, LastError());
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2_Clone;
            var obs = new int[NativeExports.OBS_SIZE];
            fixed (int* o = obs)
            {
                copy = clone(h, 0, 0, o);
            }

            Assert.True(copy >= 0);
            Assert.Equal(EngineFault.Code, Step(copy, 0));
            Assert.Equal(before + 1, Faults());

            // A reset deals a new fight on the handle: it steps again.
            delegate* unmanaged<int, int*, void> reset = &NativeExports.Sts2_Reset;
            fixed (int* o = obs)
            {
                reset(h, o);
            }

            Assert.True(Step(h, NativeExports.StateOf(h)!.Hand.Count) >= 0, LastError());
        }
        finally
        {
            Destroy(h);
            if (copy >= 0)
            {
                Destroy(copy);
            }
        }
    }

    [Fact]
    public void ARunStepThatThrowsIsRefusedWithItsClassAndTheRunStaysRefused()
    {
        int run = RunInCombat();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(run, out var e));
            Hazard(e!.State.ActiveCombat!);
            var obs = new int[Sts2Emulator.Core.Run.RunConstants.RunObsSize];
            float reward;
            int terminal, truncated;
            fixed (int* o = obs)
            {
                Assert.Equal(EngineFault.Code, RunNativeExports.Sts2Run_Step(run, 0, -1, o, &reward, &terminal, &truncated));
                string fault = LastError();
                Assert.StartsWith("GameHangs\t", fault);
                Assert.Equal((0f, 0, 0), (reward, terminal, truncated));
                Assert.Equal(EngineFault.Code, RunNativeExports.Sts2Run_Step(run, 1, -1, o, &reward, &terminal, &truncated));
                Assert.Equal(fault, LastError());
            }
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(run);
        }
    }

    [Theory]
    [InlineData("   at System.Collections.Generic.List`1.AddWithResize(T) + 0x2f\n   at Sts2Emulator.Core.StepInProgress.OnTake(Int32) + 0x1c\n", "StepInProgress.OnTake")]
    [InlineData("   at Sts2Emulator.Core.Effects.CardEffects.DrawCards(CombatState, Int32, Random, List`1, Boolean) + 0xb69\n", "CardEffects.DrawCards")]
    [InlineData("   at System.Linq.Enumerable.Count[TSource](IEnumerable`1)\n", "unknown")]
    public void TheFrameIsTheFirstInTheEngine(string trace, string frame) =>
        Assert.Equal(frame, EngineFault.TopEngineFrame(trace));
}
