using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.NativeKernelHandleTests;

namespace Sts2Emulator.Tests;

/// <inheritdoc cref="RunForkCombatCore2Tests" />
[Collection(nameof(NativeEngineCollection))]
public sealed class DeterminizedStopCore2Tests : DeterminizedStopTests, IDisposable
{
    public DeterminizedStopCore2Tests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");
}

/// <summary>
/// Screen answers under core2 (NATIVE_KERNEL.md piece 4): an answer on a combat handle reaches the kernel's queue (the
/// action gathering the choice is resumed, its replay holding the keyed answers), as the run's answer does.
/// </summary>
[Collection(nameof(NativeEngineCollection))]
public unsafe class NativeKernelAnswerTests : IDisposable
{
    private const int ObsSize = NativeExports.OBS_SIZE;

    public NativeKernelAnswerTests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");

    /// <summary>
    /// A hook's choice in the middle of a step (Tyranny's pick at the turn start an end of turn runs into), answered through
    /// <c>Sts2_Step</c> on a fork, reaches the kernel's queue as the run's answer through <c>Sts2Run_Step</c> does (the same queue,
    /// an action gathering it resumed with a new id and its first id kept) and leaves the same fight.
    /// </summary>
    [Fact]
    public void AHookChoiceAnsweredOnAForkGoesThroughTheQueueAsTheRunsAnswer()
    {
        int run = RunInCombat();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(run, out var e));
            e!.State.ActiveCombat!.PlayerBuffs.Add(new BuffState(BuffId.Tyranny, 1));
            int fork = Fork(run, out _);
            try
            {
                var forked = NativeExports.StateOf(fork)!;
                int end = forked.Hand.Count;
                StepObs(fork, end);
                RunStep(run, end);
                Assert.NotNull(forked.PendingSelection);
                Assert.NotNull(e.State.ActiveCombat!.PendingSelection);
                // The pick comes in the turn start the end of turn ran into: inside the end-turn action when the queue still
                // holds it gathering, else between actions (the old step's answer), on the fork as on the run.
                var gathering = forked.Kernel!.Queue!.All.SingleOrDefault(a => a.State == GameActionState.GatheringPlayerChoice);
                var liveGathering = e.State.ActiveCombat.Kernel!.Queue!.All.SingleOrDefault(a => a.State == GameActionState.GatheringPlayerChoice);
                Assert.Equal(liveGathering?.Kind, gathering?.Kind);
                Assert.Equal(liveGathering?.Id, gathering?.Id);

                int[] forkObs = StepObs(fork, 0);
                int[] runObs = RunStep(run, 0);
                Assert.Null(forked.PendingSelection);
                Assert.Equal(runObs[..ObsSize], forkObs);
                if (gathering is { Id: { } first })
                {
                    var done = Assert.Single(forked.Kernel!.Queue!.All, a => a.FirstId == first);
                    Assert.Equal(GameActionState.Finished, done.State);
                    Assert.NotEqual(first, done.Id);
                }

                Assert.Equal(
                    e.State.ActiveCombat.Kernel!.Queue!.All.Select(a => (a.Kind, a.State, a.Id, a.FirstId)),
                    forked.Kernel!.Queue!.All.Select(a => (a.Kind, a.State, a.Id, a.FirstId)));
            }
            finally
            {
                Destroy(fork);
            }
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(run);
        }
    }
}
