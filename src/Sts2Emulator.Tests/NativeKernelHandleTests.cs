using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The fork tests with the engine on core2 (NATIVE_KERNEL.md piece 2): every handle the suite makes is made after
/// <c>Sts2_SetEngine("core2")</c>, so a run's clone and fork, a combat's determinized clone and their answers to a stopped
/// step go through the kernel's entry gate and must still be the fights the old engine's tests hold them to.
/// </summary>
[Collection(nameof(NativeEngineCollection))]
public sealed class RunForkCombatCore2Tests : RunForkCombatTests, IDisposable
{
    public RunForkCombatCore2Tests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");
}

/// <summary>
/// The combat handles under core2 (NATIVE_KERNEL.md piece 2): bound where they are built, stepped through the kernel's
/// entry gate, carried by every copy (a run's clone, a fork, a determinized clone).
/// </summary>
[Collection(nameof(NativeEngineCollection))]
public unsafe class NativeKernelHandleTests : IDisposable
{
    private const int ObsSize = NativeExports.OBS_SIZE;

    public NativeKernelHandleTests() => Assert.Null(RunNativeExports.SetEngine("core2"));

    public void Dispose() => RunNativeExports.SetEngine("old");

    [Fact]
    public void ACore2HandleIsBoundAndStepsThroughTheQueueAndAResetBindsAgain()
    {
        int h = Create(7);
        try
        {
            var state = NativeExports.StateOf(h)!;
            var k = state.Kernel;
            Assert.NotNull(k);
            Assert.True(k!.On(Switch.F10));
            Assert.Empty(k.Queue!.All);
            Assert.True(Step(h, state.Hand.Count) >= 0);  // the turn's end, enqueued and drained
            Assert.Contains(state.Kernel!.Queue!.All, a => a is { Kind: GameActionKind.EndTurn, State: GameActionState.Finished });

            delegate* unmanaged<int, int*, void> reset = &NativeExports.Sts2_Reset;
            var obs = new int[ObsSize];
            fixed (int* o = obs)
            {
                reset(h, o);
            }

            Assert.NotNull(state.Kernel);
            Assert.NotSame(k, state.Kernel);
            Assert.Empty(state.Kernel!.Queue!.All);
        }
        finally
        {
            Destroy(h);
        }
    }

    [Fact]
    public void AForkedStepEqualsTheSameStepUnforkedAndLeavesTheLiveRunAlone()
    {
        int run = RunInCombat();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(run, out var e));
            for (int step = 0; step < 40 && e!.State.Phase == RunPhase.Combat; step++)
            {
                var live = e.State.ActiveCombat!;
                int liveRng = e.State.ActiveCombatRng!.CallCount;
                string liveDoc = Encoding.UTF8.GetString(CombatSnapshot.WriteSave(live));
                int fork = Fork(run, out _);
                try
                {
                    var forked = NativeExports.StateOf(fork)!;
                    Assert.NotNull(forked.Kernel);
                    Assert.NotSame(live.Kernel, forked.Kernel);
                    Assert.Null(forked.Kernel!.World?.Run);  // a fork reads no live run
                    int action = ValidActions(fork)[0];
                    int[] forkObs = StepObs(fork, action);

                    Assert.Equal(liveRng, e.State.ActiveCombatRng.CallCount);
                    Assert.Equal(liveDoc, Encoding.UTF8.GetString(CombatSnapshot.WriteSave(live)));
                    int[] runObs = RunStep(run, action);
                    if (e.State.Phase != RunPhase.Combat)
                    {
                        break;  // the run finished the fight; the fork has no run to finish
                    }

                    Assert.Equal(runObs[..ObsSize], forkObs);
                }
                finally
                {
                    Destroy(fork);
                }
            }
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(run);
        }
    }

    [Fact]
    public void ARunCloneCarriesTheKernelAndItsFightIsBoundToTheCopy()
    {
        int run = RunInCombat();
        delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2Run_Clone;
        var obs = new int[RunConstants.RunObsSize];
        int copy;
        fixed (int* o = obs)
        {
            copy = clone(run, 1, 11, o);
        }

        try
        {
            Assert.True(copy >= 0, $"Sts2Run_Clone returned {copy}");
            Assert.True(RunNativeExports.TryGetRun(run, out var e));
            Assert.True(RunNativeExports.TryGetRun(copy, out var c));
            Assert.Same(e!.Kernel, c!.Kernel);
            var fight = c.State.ActiveCombat!;
            Assert.NotSame(e.State.ActiveCombat!.Kernel, fight.Kernel);
            Assert.Same(c.State, fight.Kernel!.World!.Run);
            string liveDoc = Encoding.UTF8.GetString(CombatSnapshot.WriteSave(e.State.ActiveCombat));
            RunStep(copy, fight.Hand.Count);
            Assert.Contains(fight.Kernel!.Queue!.All, a => a.Kind == GameActionKind.EndTurn);
            Assert.Equal(liveDoc, Encoding.UTF8.GetString(CombatSnapshot.WriteSave(e.State.ActiveCombat)));
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(copy);
            RunNativeExports.Sts2Run_Destroy(run);
        }
    }

    internal static int Create(int seed)
    {
        delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
        int h = create(seed);
        Assert.True(h >= 0, LastError());
        return h;
    }

    internal static void Destroy(int h)
    {
        delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
        destroy(h);
    }

    internal static int Load(byte[] doc, int seed)
    {
        delegate* unmanaged<byte*, int, int, int*, int> load = &NativeExports.LoadCombatSnapshot;
        fixed (byte* d = doc)
        {
            return load(d, doc.Length, seed, null);
        }
    }

    internal static int Step(int h, int action)
    {
        delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
        var obs = new int[ObsSize];
        float reward;
        fixed (int* o = obs)
        {
            return step(h, action, o, &reward);
        }
    }

    internal static int[] StepObs(int h, int action)
    {
        delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
        var obs = new int[ObsSize];
        float reward;
        fixed (int* o = obs)
        {
            Assert.True(step(h, action, o, &reward) >= 0, LastError());
        }

        return obs;
    }

    internal static int[] ValidActions(int h)
    {
        delegate* unmanaged<int, int*, int, void> valid = &NativeExports.Sts2_ValidActions;
        var mask = new int[32];
        fixed (int* m = mask)
        {
            valid(h, m, mask.Length);
        }

        return Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).ToArray();
    }

    internal static int Fork(int run, out int[] obs)
    {
        delegate* unmanaged<int, int, int, int*, int> fork = &NativeExports.Sts2Run_ForkCombat;
        obs = new int[ObsSize];
        fixed (int* o = obs)
        {
            int f = fork(run, 0, 0, o);
            Assert.True(f >= 0, $"Sts2Run_ForkCombat returned {f}: {LastError()}");
            return f;
        }
    }

    internal static int[] RunStep(int run, int action)
    {
        var obs = new int[RunConstants.RunObsSize];
        float reward;
        int terminal, truncated;
        fixed (int* o = obs)
        {
            Assert.Equal(0, RunNativeExports.Sts2Run_Step(run, action, -1, o, &reward, &terminal, &truncated));
        }

        return obs;
    }

    internal static int RunInCombat()
    {
        int run = RunNativeExports.Sts2Run_Create();
        Assert.True(RunNativeExports.TryGetRun(run, out var e));
        Assert.True(RunNativeExports.KernelOn(e!));
        int[] deck = [.. RunConstants.StarterDeckIds];
        var obs = new int[RunConstants.RunObsSize];
        fixed (byte* s = Encoding.UTF8.GetBytes("QS2GYXRKWN"))
        fixed (int* d = deck)
        fixed (int* o = obs)
        {
            Assert.Equal(0, RunNativeExports.Sts2Run_Reset(run, s, 10, 10, o));
            Assert.Equal(0, RunNativeExports.Sts2Run_StartCombat(run, d, deck.Length,
                RunConstants.SlitheringStranglerEncounterId, null, 0, 64, 80, null, 0, 99, 0, o));
        }

        return run;
    }

    internal static string LastError()
    {
        delegate* unmanaged<byte*, int, int> get = &NativeExports.GetLastError;
        var buf = new byte[512];
        fixed (byte* p = buf)
        {
            return Encoding.UTF8.GetString(buf, 0, get(p, buf.Length));
        }
    }
}
