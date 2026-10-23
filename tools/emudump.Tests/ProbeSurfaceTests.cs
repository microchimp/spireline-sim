// The probe surfaces behind IEngine: the full-state hash and its per-field hashes (EmuDump fight --state-hash), and the
// listener-order probe (EmuDump hookprobe), which only an engine instrumented by tools/fidelity/hook_order.py can run.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

public class ProbeSurfaceTests
{
    static OldEngine InCombat()
    {
        var run = new RunEngine();
        run.Reset("PROBE001");
        var engine = new OldEngine(run);
        run.StartCombat(RunConstants.StarterDeckIds, RunConstants.SlitheringStranglerEncounterId,
            [RunConstants.StartingRelicFor(PlayerCharacter.Ironclad)], playerHp: 64, playerMaxHp: 80, potionIds: [], playerGold: 99);
        return engine;
    }

    [Fact]
    public void TheStateHashIsSixteenHexDigitsThatMoveWithTheState()
    {
        IEngine e = InCombat();
        var before = e.StateHash();
        Assert.Matches("^[0-9a-f]{16}$", before);
        Assert.Equal(before, e.StateHash());
        e.EndTurn();
        Assert.NotEqual(before, e.StateHash());
    }

    [Fact]
    public void TheFieldHashesListTheRunsFieldsThenTheCombatsWithTheirPrefix()
    {
        IEngine e = InCombat();
        var names = e.StateFieldHashes().Select(kv => kv.Key).ToList();
        var split = names.FindIndex(n => n.StartsWith("ActiveCombat.", StringComparison.Ordinal));
        Assert.True(split > 0, "the combat's fields follow the run's");
        Assert.All(names.Skip(split), n => Assert.StartsWith("ActiveCombat.", n));
        Assert.Contains("ActiveCombat", names.Take(split));  // the combat as a field of the run is there too
        Assert.Equal(names.Take(split).OrderBy(n => n, StringComparer.Ordinal), names.Take(split));
        Assert.All(e.StateFieldHashes(), kv => Assert.Matches("^[0-9a-f]{8}$", kv.Value));
    }

    [Fact]
    public void HookProbeOnAPlainEngineSaysItIsNotInstrumented()
    {
        // The test build has no Sts2Emulator.HookProbe: only the scratch copy hook_order.py instruments does.
        var ex = Assert.Throws<NotSupportedException>(() => new OldEngine().HookProbe(new JsonObject { ["scenarios"] = new JsonArray() }));
        Assert.Contains("hook_table order", ex.Message);
    }
}
