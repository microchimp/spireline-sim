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
/// was taken from; a document stopped in the middle of a step loads with what the queue held at its screen put back.
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
    public void ADocumentStoppedInATurnEndLoadsUnderCore2WithAnIdleQueue()
    {
        int h = Create(7);
        int loaded = -1;
        try
        {
            var state = NativeExports.StateOf(h)!;
            state.PlayerBuffs.Add(new BuffState(BuffId.Tyranny, 1));
            StepObs(h, state.Hand.Count);
            Assert.NotNull(state.PendingSelection?.Replay);
            Assert.False(NativeExports.OnHookPause(state));  // an old screen outside any action: written
            Assert.Empty(state.Kernel!.Queue!.QueueOf(0));    // the turn end's action finished before the turn end ran
            byte[] doc = CombatSnapshot.Write(state);
            loaded = Load(doc, 7);
            Assert.True(loaded >= 0, LastError());
            var b = NativeExports.StateOf(loaded)!;
            Assert.Empty(b.Kernel!.Queue!.QueueOf(0));
            Assert.NotNull(b.PendingSelection!.Replay!.Before.Kernel);
            Assert.Equal(ValidActions(h), ValidActions(loaded));
            StepObs(loaded, ValidActions(loaded)[0]);
            Assert.Null(b.PendingSelection);
            Assert.Empty(b.Kernel!.Queue!.QueueOf(0));
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

    /// <summary>
    /// A document stopped on a screen inside a card play (Armaments) loads under core2 with the play's action put back as the
    /// game holds it (gathering at the screen, executing in the step's start), and its answer plays as the live fight's: the
    /// 17 snapshot-load refusals of the 2026-10-29 coverage run were such documents (plays and potions).
    /// </summary>
    [Fact]
    public void ADocumentStoppedInACardPlayLoadsUnderCore2WithItsPausedActionAndAnswersAsTheLiveFight()
    {
        int h = Create(7);
        int loaded = -1;
        try
        {
            var state = NativeExports.StateOf(h)!;
            state.Hand.Insert(0, new CardInstance(Sts2Emulator.Core.Effects.IC.Armaments, false));
            state.Energy = Math.Max(state.Energy, 1);
            StepObs(h, 0);
            Assert.NotNull(state.PendingSelection?.Replay);
            var live = Assert.Single(state.Kernel!.Queue!.QueueOf(0));
            byte[] doc = CombatSnapshot.Write(state);
            loaded = Load(doc, 7);
            Assert.True(loaded >= 0, LastError());
            var b = NativeExports.StateOf(loaded)!;
            var paused = Assert.Single(b.Kernel!.Queue!.QueueOf(0));
            Assert.Equal((live.Kind, live.Type, live.State, live.Action, live.Target), (paused.Kind, paused.Type, paused.State, paused.Action, paused.Target));
            var started = Assert.Single(b.PendingSelection!.Replay!.Before.Kernel!.Queue!.QueueOf(0));
            Assert.Equal((GameActionKind.PlayCard, GameActionState.Executing), (started.Kind, started.State));
            Assert.Equal(ValidActions(h), ValidActions(loaded));
            int answer = ValidActions(h)[0];
            Assert.Equal(StepObs(h, answer), StepObs(loaded, answer));
            Assert.Null(b.PendingSelection);
            Assert.Empty(b.Kernel!.Queue!.QueueOf(0));
            Assert.Equal(Encoding.UTF8.GetString(CombatSnapshot.Write(state)), Encoding.UTF8.GetString(CombatSnapshot.Write(b)));
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
}
