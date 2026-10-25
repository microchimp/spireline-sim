using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.NativeKernelHandleTests;

namespace Sts2Emulator.Tests;

/// <summary>
/// Snapshots under core2 (NATIVE_KERNEL.md piece 3): a loaded document binds a fresh KernelState and plays as the fight it
/// was taken from; a document stopped in the middle of a step is refused, with the reason, and loads on the old engine.
/// </summary>
[Collection(nameof(NativeEngineCollection))]
public unsafe class NativeKernelSnapshotTests : IDisposable
{
    private const int ObsSize = NativeExports.OBS_SIZE;

    public NativeKernelSnapshotTests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");

    [Fact]
    public void ALoadedDocumentBindsAFreshKernelAndPlaysAsTheFightItWasTakenFrom()
    {
        int h = Create(7);
        int loaded = -1;
        try
        {
            byte[] doc = CombatSnapshot.Write(NativeExports.StateOf(h)!);
            loaded = Load(doc, 7);
            Assert.True(loaded >= 0, LastError());
            var a = NativeExports.StateOf(h)!;
            var b = NativeExports.StateOf(loaded)!;
            Assert.NotNull(b.Kernel);
            Assert.NotSame(a.Kernel, b.Kernel);
            Assert.Empty(b.Kernel!.Queue!.All);
            // The plays of turn one roll nothing, so the two fights go the same on any streams.
            for (int i = 0; i < 6; i++)
            {
                int[] valid = ValidActions(h);
                Assert.Equal(valid, ValidActions(loaded));
                int play = valid[0];
                if (play >= a.Hand.Count)
                {
                    break;
                }

                Assert.Equal(StepObs(h, play), StepObs(loaded, play));
                Assert.Equal(Encoding.UTF8.GetString(CombatSnapshot.Write(a)), Encoding.UTF8.GetString(CombatSnapshot.Write(b)));
                Assert.NotEmpty(b.Kernel!.Queue!.All);
            }
        }
        finally
        {
            Destroy(h);
            if (loaded >= 0)
            {
                Destroy(loaded);
            }
        }
    }

    [Fact]
    public void ADocumentStoppedInAStepIsRefusedUnderCore2AndLoadsOnTheOldEngine()
    {
        int h = Create(7);
        try
        {
            var state = NativeExports.StateOf(h)!;
            state.PlayerBuffs.Add(new BuffState(BuffId.Tyranny, 1));
            StepObs(h, state.Hand.Count);
            Assert.NotNull(state.PendingSelection?.Replay);
            Assert.False(NativeExports.OnHookPause(state));  // an old screen in a player action: written
            byte[] doc = CombatSnapshot.Write(state);
            Assert.Equal(-1, Load(doc, 7));
            Assert.Equal(NativeExports.StoppedStepRefusal, LastError());

            Assert.Null(RunNativeExports.SetEngine("old"));
            int old = Load(doc, 7);
            Assert.True(old >= 0, LastError());
            Destroy(old);
        }
        finally
        {
            Destroy(h);
        }
    }
}
