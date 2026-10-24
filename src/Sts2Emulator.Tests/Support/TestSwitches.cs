using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The switched-on old suites (docs/rebuild/MIGRATION_F08.md, "The old suites under the switch"; the F08 review's
/// recommendation 2). With <c>STS2_TEST_SWITCHES</c> set to a family list (<c>F08,F10</c>) there is one Core2 kernel
/// with those switches and <c>Registry.All</c>, and:
/// <list type="bullet">
/// <item>a combat a suite steps itself (the <c>Fight</c> helper's steps, <see cref="Step(CombatState,int,Random,int)" />)
/// is bound at its FIRST step, after all of the test's setup (the factory, the fluent <c>Enemy</c> / <c>PlayerBuff</c> /
/// <c>Seed</c>, direct writes), as <c>RunEngine.StartCombat</c> binds a run's combat after its setup and pre-play phase
/// and before its first step: <c>Attach(combat, run)</c>, the run being the state of the run engine (one per combat)
/// whose <c>StepCombat</c>, the kernel's entry gate, takes this and every later step. An engine call a test makes on the
/// combat before its first step runs unbound (old). A combat already bound (a run engine's) keeps its binding.</item>
/// <item>a run engine a suite makes gets the kernel before its <c>Reset</c> (<see cref="Run" />), so it binds the
/// combats it starts itself; its steps go through <see cref="Step(RunEngine,int,int,out float,out bool,out bool)" />.</item>
/// </list>
/// Coverage: with <c>STS2_TEST_SWITCHES_COVERAGE</c> also set to a file, each kernel-bound step appends one line, the
/// test method that made it (<c>Namespace.Class.Method</c>, found on the stack; a theory's cases share the line), so the
/// runner can tell a test that never reached the kernel from one that passed with it.
/// Unset, empty or <c>none</c>: no kernel, nothing bound, counted or written; every step is <c>CombatEngine.Step</c>
/// or <c>RunEngine.Step</c>, so the suites run exactly as before.
/// </summary>
internal static class TestSwitches
{
    public const string Variable = "STS2_TEST_SWITCHES";
    public const string CoverageVariable = "STS2_TEST_SWITCHES_COVERAGE";

    private static readonly Lazy<Kernel?> Shared = new(Make);
    private static readonly ConditionalWeakTable<CombatState, RunEngine> Engines = new();
    private static readonly object CoverageLock = new();

    /// <summary>The test kernel, or null (the variable unset): the old engine.</summary>
    public static Kernel? Kernel => Shared.Value;

    /// <summary>
    /// True when the test kernel holds every one of <paramref name="families" /> (the switched-on run), false with no
    /// kernel. A case that pins an old-engine behaviour the game does not have asserts the game's under it (with a
    /// <c>game:</c> cite) and keeps the old assertion when it is false.
    /// </summary>
    public static bool On(params Sts2Emulator.Core2.Switch[] families) => Kernel is { } k && families.All(k.On);

    private static Kernel? Make()
    {
        string? v = Environment.GetEnvironmentVariable(Variable);
        if (string.IsNullOrWhiteSpace(v) || v.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
            return null;
        var options = KernelOptions.FromFamilies(
            v.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries));
        return options.Switches.Count == 0
            ? null
            : new Kernel(options, () => Core2.ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All);
    }

    /// <summary>Gives a suite's run engine the test kernel; call before its <c>Reset</c>. With none, untouched.</summary>
    public static RunEngine Run(RunEngine engine)
    {
        if (Kernel is { } k)
            engine.Kernel = k;
        return engine;
    }

    /// <summary>
    /// A combat step: <c>CombatEngine.Step</c> with no test kernel. With one, the combat's run engine (made with its
    /// first step, holding the kernel) binds it if it is not bound yet, <c>Attach(combat, run)</c> as RunEngine does,
    /// and steps it through <c>RunEngine.StepCombat</c>.
    /// </summary>
    public static StepResult Step(CombatState state, int action, Random rng, int targetEnemyIndex = -1)
    {
        if (Kernel is not { } k)
            return CombatEngine.Step(state, action, rng, targetEnemyIndex);
        var engine = Engines.GetValue(state, _ => new RunEngine { Kernel = k });
        if (state.Kernel is null)
        {
            // game: every card of a combat is a model with its own identity. A test's hand-made cards carry no uid (0);
            // the kernel resolves a card source by uid (CARD_UID.md), so they get fresh ones here as CombatFactory
            // numbers a combat's cards, keeping every uid already given.
            CardUids.NumberNew(state);
            k.Attach(state, engine.State);
        }
        Counted();
        return engine.StepCombat(state, action, rng, targetEnemyIndex);
    }

    /// <summary>A run engine's step; counted as kernel-bound when the engine holds the kernel and steps a combat.</summary>
    public static int Step(RunEngine engine, int action, int targetEnemyIndex, out float reward, out bool terminal,
        out bool truncated)
    {
        if (engine.Kernel is { AnyOn: true } && engine.State is { Phase: RunPhase.Combat, ActiveCombat: not null })
            Counted();
        return engine.Step(action, targetEnemyIndex, out reward, out terminal, out truncated);
    }

    private static void Counted()
    {
        string? path = Environment.GetEnvironmentVariable(CoverageVariable);
        if (string.IsNullOrEmpty(path))
            return;
        string line = CurrentTest() + "\n";
        lock (CoverageLock)
            File.AppendAllText(path, line);
    }

    /// <summary>The [Fact] / [Theory] method on the stack (a theory derives from Fact), or "?" when none is found.</summary>
    private static string CurrentTest()
    {
        foreach (var frame in new StackTrace().GetFrames())
        {
            if (frame.GetMethod() is { } m && m.IsDefined(typeof(FactAttribute), inherit: true))
                return $"{m.DeclaringType?.FullName?.Replace('+', '.')}.{m.Name}";
        }

        return "?";
    }
}
