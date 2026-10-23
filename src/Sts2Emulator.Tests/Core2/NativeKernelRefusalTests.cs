// The native run API carries no kernel state (docs/rebuild/TIER1_KERNEL.md section 5): every export that builds or steps
// a combat refuses a run whose engine has a switch on, and leaves the run as it was.
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class NativeKernelRefusalTests
{
    private static readonly Switch[][] TestGroups = [[Switch.F10]];

    private static (int Handle, RunEngine Run) NewRun(bool switchedOn)
    {
        int handle = RunNativeExports.Sts2Run_Create();
        Assert.True(handle >= 0);
        Assert.True(RunNativeExports.TryGetRun(handle, out var run));
        run.Reset("PRB00058", ascension: 8);
        if (switchedOn)
        {
            run.Kernel = new Kernel(new KernelOptions([Switch.F10], TestGroups), () => ListenerOrderTests.Ids, []);
        }

        return (handle, run);
    }

    [Fact]
    public unsafe void WithASwitchOnStepStartCombatAndReanchorAreRefusedAndTheRunIsUntouched()
    {
        var (handle, run) = NewRun(switchedOn: true);
        try
        {
            var phase = run.State.Phase;
            var obs = new int[RunConstants.RunObsSize];
            var rec = Encoding.UTF8.GetBytes("{}");
            var msg = new byte[64];
            float reward;
            int terminal, truncated;
            var deck = new[] { 1, 1, 1 };
            fixed (int* o = obs)
            fixed (int* d = deck)
            fixed (byte* r = rec)
            fixed (byte* m = msg)
            {
                Assert.Equal(-1, RunNativeExports.Sts2Run_Step(handle, 0, -1, o, &reward, &terminal, &truncated));
                Assert.Equal(-1, RunNativeExports.Sts2Run_StartCombat(handle, d, deck.Length, 0, null, 0, 50, 80, null, 0, 0, 0, o));
                Assert.Equal(-1, RunNativeExports.Sts2Run_ReanchorFight(handle, r, rec.Length, m, msg.Length, o));
                Assert.Equal("kernel_on", Encoding.UTF8.GetString(msg, 0, Array.IndexOf(msg, (byte)0)));
                Assert.Equal(-1, RunNativeExports.Sts2Run_ReanchorFightHidden(handle, r, rec.Length, 3, m, msg.Length, o));
            }

            Assert.Equal(phase, run.State.Phase);
            Assert.Null(run.State.ActiveCombat);
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public unsafe void WithNoSwitchOnTheSameCallsAreNotRefusedForTheKernel()
    {
        var (handle, run) = NewRun(switchedOn: false);
        try
        {
            var obs = new int[RunConstants.RunObsSize];
            var rec = Encoding.UTF8.GetBytes("{}");
            var msg = new byte[64];
            fixed (int* o = obs)
            fixed (byte* r = rec)
            fixed (byte* m = msg)
            {
                // No fight in progress, and no tables handed over in this process or an unreadable record: its own codes,
                // never the kernel's.
                RunNativeExports.Sts2Run_ReanchorFight(handle, r, rec.Length, m, msg.Length, o);
                Assert.NotEqual("kernel_on", Encoding.UTF8.GetString(msg, 0, Array.IndexOf(msg, (byte)0)));
            }
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }
}
