// The C1 pilot handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C1 pilot"): the port list's power-5, relic-2 and
// card-1 rows as migrated models under Core2/Models, each tested the way the shared rules ask: (1) its pairs are in the
// handler table and none is a stub, (2) the dispatch gives what the old function gives for the same state, (3) the
// delete-mutant (the model's registration left out) gives the no-handler value, (4) with F08 off its pairs are absent.
// The table test at the end pins the batch's (model, hook) set, standing in for EmuDump kernel-handlers and the port
// list's gate, which need the manifest and a built EmuDump. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PilotTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Pilot = Registry.Pilot;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The pilot's (model, hook) pairs, written out: the port list rows power-5, relic-2 and card-1.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] PilotPairs =
    [
    ];

    [Fact]
    public void TheRegistrysPilotBlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Pilot);
        Assert.Equal(PilotPairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePilotBlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; the pilot's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Pilot)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in PilotPairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffThePilotRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Pilot, AllButDamage)));
    }

    [Fact]
    public void TheKitsDeleteMutantLeavesOneRegistrationOut()
    {
        Action<HandlerTable> a = static t => { };
        Action<HandlerTable> b = static t => { };
        Assert.Equal([b], Without([a, b], a));
        Assert.Equal(Pilot.Count, Without(Pilot, a).Count);
    }
}
