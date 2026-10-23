// EmuDump --engine old|core2 (Engines/EngineFactory.cs) and the core2 engine with no switch on (TIER1_KERNEL.md 1, 6).
using EmuDump.Engines;
using EmuDump.Engines.Core2;
using EmuDump.Engines.Old;
using Sts2Emulator.Core2;
using Xunit;

public class EngineFlagTests
{
    [Theory]
    [InlineData("fight in.json --compact --engine core2", "fight in.json --compact", "core2")]
    [InlineData("reanchor --engine old in.json --seed S", "reanchor in.json --seed S", "old")]
    [InlineData("fight in.json --compact", "fight in.json --compact", null)]
    [InlineData("hookprobe --engine core2 a.json b.json", "hookprobe a.json b.json", "core2")]
    public void TheFlagIsTakenOutAndNamesTheEngine(string args, string rest, string? engine)
    {
        var left = EngineFactory.TakeFlag(args.Split(' '), out var name, out var error);
        Assert.Null(error);
        Assert.Equal(rest.Split(' '), left);
        Assert.Equal(engine, name);
    }

    [Theory]
    [InlineData("fight in.json --engine")]
    [InlineData("fight in.json --engine core3")]
    [InlineData("codex runs.jsonl out.jsonl --engine core2")]
    [InlineData("--engine core2 fight in.json")]
    public void ABrokenFlagIsRefused(string args)
    {
        Assert.Null(EngineFactory.TakeFlag(args.Split(' '), out _, out var error));
        Assert.NotNull(error);
    }

    [Fact]
    public void Core2WithNoSwitchOnIsTheOldEngineWithAnIdleKernel()
    {
        var e = (Core2Engine)EngineFactory.Create("core2");
        Assert.Equal("core2", e.Name);
        Assert.False(e.Kernel.AnyOn);
        Assert.Empty(e.Kernel.Options.Switches);
        Assert.Null(e.Kernel.Handlers);
        Assert.Null(e.Run.State.Kernel);
        Assert.Same(e.Run, OldEngine.Of(e).Run);
        Assert.Equal("old", EngineFactory.Create("old").Name);
    }

    [Fact]
    public void TheKernelIsBoundToTheStatesOfItsEngineAndNoOtherOnTheSameThread()
    {
        var on = new KernelOptions(Enum.GetValues<Sts2Emulator.Core2.Switch>().Where(s => s != Sts2Emulator.Core2.Switch.None));
        var core2 = new Core2Engine(on);
        core2.Run.Reset("KERNELSTATE", 0);
        var old = new OldEngine();
        old.Run.Reset("KERNELSTATE", 0);
        Assert.NotNull(core2.Run.State.Kernel);
        Assert.Same(core2.Kernel, core2.Run.State.Kernel!.Shared);
        Assert.Null(old.Run.State.Kernel);
        var copy = Sts2Emulator.Core.StateCloning.Clone(core2.Run.State);
        Assert.NotNull(copy.Kernel);
        Assert.NotSame(core2.Run.State.Kernel, copy.Kernel);
        Assert.Null(Sts2Emulator.Core.StateCloning.Clone(old.Run.State).Kernel);
        Assert.Null(((Core2Engine)EngineFactory.Create("core2")).Run.State.Kernel);
    }
}
