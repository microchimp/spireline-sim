using System.Runtime.CompilerServices;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;

namespace Sts2Emulator.Tests;

/// <summary>
/// The switched-on old suites (docs/rebuild/MIGRATION_F08.md, "The old suites under the switch"; the F08 review's
/// recommendation 2). With <c>STS2_TEST_SWITCHES</c> set to a family list (<c>F08,F10</c>), every combat the old
/// behaviour suites build is bound to one Core2 kernel with those switches and <c>Registry.All</c>, as
/// <c>RunEngine.StartCombat</c> binds a run's combat (after its setup), and its steps go through the kernel's entry gate,
/// <c>RunEngine.StepCombat</c> (the queue, its body the old step). A run engine a suite makes gets the kernel before its
/// <c>Reset</c>. Unset, empty or <c>none</c>, there is no kernel: <see cref="Bind" /> and <see cref="Run" /> write
/// nothing and <see cref="Step" /> is <c>CombatEngine.Step</c>, so the suites run exactly as before.
/// </summary>
internal static class TestSwitches
{
    public const string Variable = "STS2_TEST_SWITCHES";

    private static readonly Lazy<Kernel?> Shared = new(Make);
    private static readonly ConditionalWeakTable<CombatState, RunEngine> Engines = new();

    /// <summary>The test kernel, or null (the variable unset): the old engine.</summary>
    public static Kernel? Kernel => Shared.Value;

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

    /// <summary>Binds <paramref name="combat" /> to the test kernel (once); with none, returns it untouched.</summary>
    public static CombatState Bind(CombatState combat)
    {
        if (Kernel is { } k && combat.Kernel is null)
            k.Bind(combat);
        return combat;
    }

    /// <summary>Gives a suite's run engine the test kernel; call before its <c>Reset</c>. With none, untouched.</summary>
    public static RunEngine Run(RunEngine engine)
    {
        if (Kernel is { } k)
            engine.Kernel = k;
        return engine;
    }

    /// <summary>
    /// A combat step: <c>CombatEngine.Step</c> with no test kernel; with one, the combat (bound if it is not) is stepped
    /// through <c>RunEngine.StepCombat</c> of an engine holding the kernel, one engine per combat.
    /// </summary>
    public static StepResult Step(CombatState state, int action, Random rng, int targetEnemyIndex = -1)
    {
        if (Kernel is not { } k)
            return CombatEngine.Step(state, action, rng, targetEnemyIndex);
        Bind(state);
        var engine = Engines.GetValue(state, _ => new RunEngine { Kernel = k });
        return engine.StepCombat(state, action, rng, targetEnemyIndex);
    }
}
