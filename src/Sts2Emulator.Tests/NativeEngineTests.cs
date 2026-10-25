using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>The engine option changes a process-wide default, so its tests run alone, after the parallel ones.</summary>
[CollectionDefinition(nameof(NativeEngineCollection), DisableParallelization = true)]
public sealed class NativeEngineCollection;

/// <summary>
/// The native library's engine option (docs/rebuild/NATIVE_KERNEL.md piece 1): old by default, byte for byte; core2 builds
/// the kernel of the shipped migrated_families.json (F08+F10) as EmuDump's Core2Engine does, and the run API steps through
/// it; every path a later piece covers still refuses a switched-on run, with its reason.
/// </summary>
[Collection(nameof(NativeEngineCollection))]
public unsafe class NativeEngineTests
{
    private const string Seed = "QS2GYXRKWN";
    private const int StepCap = 400;

    [Fact]
    public void TheDefaultEngineIsOldAndItsRunsAreThePlainRunEngine()
    {
        Assert.Equal("old", RunNativeExports.Engine.Name);
        Assert.Null(RunNativeExports.Engine.Kernel);
        Assert.Equal("old", GetEngine());
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var native));
            Assert.Null(native!.Kernel);
            var plain = new RunEngine();
            int combatSteps = PlayBoth(native, handle, plain, oldEngine: true);
            Assert.True(combatSteps > 0, "the walk never reached a fight");
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public void Core2BuildsTheShippedKernelWithF08AndF10On()
    {
        var shipped = JsonNode.Parse(File.ReadAllText(Path.Combine(RepoRoot(), "tools", "fidelity", "migrated_families.json")))!
            ["migrated"]!.AsArray().Select(f => (string)f!).ToList();
        Assert.Equal(["F08", "F10"], shipped);  // this test pins today's list; a new family updates both
        var core2 = NativeEngine.Parse("core2");
        Assert.Equal("core2:F08,F10", core2.Name);
        Assert.True(core2.KernelOn);
        Assert.True(core2.Kernel!.On(Switch.F08));
        Assert.True(core2.Kernel.On(Switch.F10));
        Assert.Equal(shipped.Order(), core2.Kernel.Options.Switches.Select(s => s.ToString()).Order());
        Assert.NotNull(core2.Kernel.Handlers);  // the ids and Registry.All, from the embedded files
        Assert.Same(core2, NativeEngine.Parse("core2"));  // built once
        Assert.Equal("core2:F08,F10", NativeEngine.Parse("core2:F10,F08").Name);
    }

    [Fact]
    public void ACore2RunStepsThroughTheKernelAsEmuDumpsCore2EngineDoes()
    {
        // The reference is EmuDump's recipe (Core2Engine.Make): the families, the ids read from the repo's files, Registry.All.
        var ids = OldModelIds.Load(Path.Combine(RepoRoot(), "data", "id_map.json"),
            Path.Combine(RepoRoot(), "tools", "fidelity", "power_aliases.json"));
        var reference = new RunEngine
        {
            Kernel = new Kernel(KernelOptions.FromFamilies(["F08", "F10"]), () => ids, Sts2Emulator.Core2.Models.Registry.All),
        };
        int handle = RunNativeExports.CreateWith(NativeEngine.Parse("core2"));
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var native));
            Assert.True(RunNativeExports.KernelOn(native!));
            int combatSteps = PlayBoth(native, handle, reference, oldEngine: false);
            Assert.True(combatSteps > 0, "the walk never reached a fight");
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public void ACore2RunStartsItsCombatBound()
    {
        int handle = RunNativeExports.CreateWith(NativeEngine.Parse("core2"));
        try
        {
            Assert.Equal(0, StartCombat(handle));
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            Assert.Equal(RunPhase.Combat, run!.State.Phase);
            Assert.NotNull(run.State.ActiveCombat!.Kernel);
            Assert.Equal(0, Step(handle, run.State.ActiveCombat.Hand.Count));  // end the turn, through the queue
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public void ThePathsLaterPiecesCoverStillRefuseACore2Run()
    {
        int handle = RunNativeExports.CreateWith(NativeEngine.Parse("core2"));
        try
        {
            Assert.Equal(0, StartCombat(handle));
            var obs = new int[RunConstants.RunObsSize + 64];
            fixed (int* o = obs)
            {
                Assert.Equal(-1, RunNativeExports.Sts2Run_Clone(handle, 0, 0, o));
                Assert.StartsWith("kernel_on: Sts2Run_Clone", LastError());
                delegate* unmanaged<int, int, int, int*, int> fork = &NativeExports.Sts2Run_ForkCombat;
                Assert.Equal(-3, fork(handle, 0, 0, o));
                Assert.StartsWith("kernel_on: combat handles", LastError());
                byte[] record = Encoding.UTF8.GetBytes("{}");
                var msg = new byte[256];
                fixed (byte* r = record)
                fixed (byte* m = msg)
                {
                    Assert.Equal(-1, RunNativeExports.Sts2Run_ReanchorFight(handle, r, record.Length, m, msg.Length, o));
                    Assert.Equal(-1, RunNativeExports.Sts2Run_ReanchorFightHidden(handle, r, record.Length, 1, m, msg.Length, o));
                }

                Assert.StartsWith("kernel_on: the re-anchor", Encoding.UTF8.GetString(msg).TrimEnd('\0'));
            }

            // The process-wide option: with core2 set, a combat handle and a snapshot load refuse; back on old they work.
            Assert.Equal(0, SetEngine("core2"));
            Assert.Equal("core2:F08,F10", GetEngine());
            delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
            Assert.Equal(-1, create(7));
            Assert.StartsWith("kernel_on: combat handles", LastError());
            byte[] doc = Encoding.UTF8.GetBytes("{}");
            delegate* unmanaged<byte*, int, int, int*, int> load = &NativeExports.LoadCombatSnapshot;
            fixed (byte* d = doc)
            {
                Assert.Equal(-1, load(d, doc.Length, 7, null));
            }

            Assert.StartsWith("kernel_on: a loaded snapshot", LastError());
            Assert.Equal(0, SetEngine("old"));
            int combat = create(7);
            Assert.True(combat >= 0);
            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
            destroy(combat);
        }
        finally
        {
            RunNativeExports.SetEngine("old");
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public void ARefusedSpecLeavesTheEngineAsItWas()
    {
        Assert.Equal(-1, SetEngine("core3"));
        Assert.Contains("unknown engine 'core3'", LastError());
        Assert.Equal(-1, SetEngine("core2:F99"));
        Assert.Contains("F99", LastError());
        Assert.Equal("old", GetEngine());
        Assert.Equal(0, SetEngine("old"));
    }

    /// <summary>
    /// Plays <paramref name="native" /> through the export and <paramref name="plain" /> directly, the first legal action
    /// each step, and asserts the same status, observation and phase at every step; returns the combat steps taken.
    /// </summary>
    private static int PlayBoth(RunEngine native, int handle, RunEngine plain, bool oldEngine)
    {
        fixed (byte* s = Encoding.UTF8.GetBytes(Seed))
        {
            var o = new int[RunConstants.RunObsSize];
            fixed (int* op = o)
            {
                Assert.Equal(0, RunNativeExports.Sts2Run_Reset(handle, s, Seed.Length, 10, op));
            }
        }

        plain.Reset(Seed, 10);
        int combatSteps = 0;
        var mask = new int[RunConstants.MaxActions];
        var a = new int[RunConstants.RunObsSize];
        var b = new int[RunConstants.RunObsSize];
        for (int step = 0; step < StepCap && plain.State.Phase != RunPhase.Complete; step++)
        {
            plain.WriteActionMask(mask);
            int action = Array.IndexOf(mask, 1);
            Assert.True(action >= 0, $"no legal action at step {step}");
            bool inCombat = plain.State.Phase == RunPhase.Combat;
            if (inCombat)
            {
                combatSteps++;
                Assert.Equal(oldEngine, native.State.ActiveCombat!.Kernel is null);
            }

            int sb = plain.Step(action, -1, out _, out _, out _);
            int sa = Step(handle, action, a);
            plain.WriteObservation(b);
            Assert.Equal(sb, sa);
            Assert.True(b.AsSpan().SequenceEqual(a), $"the observations differ after step {step}");
            Assert.Equal(plain.State.Phase, native.State.Phase);
        }

        return combatSteps;
    }

    private static int Step(int handle, int action) => Step(handle, action, new int[RunConstants.RunObsSize]);

    private static int Step(int handle, int action, int[] obs)
    {
        float reward;
        int terminal, truncated;
        fixed (int* o = obs)
        {
            return RunNativeExports.Sts2Run_Step(handle, action, -1, o, &reward, &terminal, &truncated);
        }
    }

    private static int StartCombat(int handle)
    {
        int[] deck = [.. RunConstants.StarterDeckIds];
        var obs = new int[RunConstants.RunObsSize];
        fixed (byte* s = Encoding.UTF8.GetBytes(Seed))
        fixed (int* d = deck)
        fixed (int* o = obs)
        {
            Assert.Equal(0, RunNativeExports.Sts2Run_Reset(handle, s, Seed.Length, 10, o));
            return RunNativeExports.Sts2Run_StartCombat(handle, d, deck.Length, RunConstants.SlitheringStranglerEncounterId,
                null, 0, 64, 80, null, 0, 99, 0, o);
        }
    }

    private static int SetEngine(string spec)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(spec);
        delegate* unmanaged<byte*, int, int> set = &NativeExports.Sts2_SetEngine;
        fixed (byte* p = bytes)
        {
            return set(p, bytes.Length);
        }
    }

    private static string GetEngine()
    {
        delegate* unmanaged<byte*, int, int> get = &NativeExports.Sts2_GetEngine;
        var buf = new byte[64];
        fixed (byte* p = buf)
        {
            return Encoding.UTF8.GetString(buf, 0, get(p, buf.Length));
        }
    }

    private static string LastError()
    {
        delegate* unmanaged<byte*, int, int> get = &NativeExports.GetLastError;
        var buf = new byte[512];
        fixed (byte* p = buf)
        {
            return Encoding.UTF8.GetString(buf, 0, get(p, buf.Length));
        }
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "HANDOFF.md")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        return dir!.FullName;
    }
}
