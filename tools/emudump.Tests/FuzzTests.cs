// EmuDump fuzz (tools/emudump/Fuzz.cs, tools/fidelity/fuzz_core2.py): a spec's seed names its fight, the engine's legal
// actions are the ones ValidActions offers with the targets each takes, a record without a room meets its encounter at the
// first map option and says which, and the same spec on core2 with no switch on plays the same fight.
using System.Text.Json.Nodes;
using EmuDump.Engines;
using EmuDump.Engines.Core2;
using Sts2Emulator.Core2;
using Xunit;

public class FuzzTests
{
    static JsonObject Spec(long seed, int maxSteps = 40) => new()
    {
        ["seed"] = "FUZZTEST01", ["ascension"] = 10, ["neow_pick"] = 0, ["use_potions"] = true,
        ["setup"] = new JsonArray(
            new JsonObject { ["cmd"] = "relics_set", ["ids"] = new JsonArray("BURNING_BLOOD") },
            new JsonObject
            {
                ["cmd"] = "deck_set", ["cards"] = new JsonArray(
                    Card("STRIKE_IRONCLAD"), Card("STRIKE_IRONCLAD"), Card("STRIKE_IRONCLAD"), Card("DEFEND_IRONCLAD"), Card("DEFEND_IRONCLAD"),
                    Card("BASH"), Card("HEADBUTT"), Card("BURNING_PACT"), Card("TRUE_GRIT"), Card("ARMAMENTS")),
            },
            new JsonObject { ["cmd"] = "potions_set", ["ids"] = new JsonArray("FIRE_POTION") },
            new JsonObject { ["cmd"] = "fight", ["id"] = "NibbitsWeak" }),
        ["fight"] = new JsonObject { ["id"] = "NibbitsWeak", ["act"] = "OVERGROWTH", ["kind"] = "weak", ["index"] = 1 },
        ["fuzz"] = new JsonObject { ["seed"] = seed, ["max_steps"] = maxSteps },
    };

    static JsonObject Card(string id) => new() { ["id"] = id, ["upgraded"] = false };

    static JsonObject Play(string engine, JsonObject spec)
    {
        EngineFactory.Selected = engine;
        try { return Fuzz.Play(spec, false); }
        finally { EngineFactory.Selected = "old"; }
    }

    // core2 with no switch chosen here, whatever tools/fidelity/migrated_families.json lists.
    static JsonObject PlayCore2NoSwitch(JsonObject spec) =>
        Fuzz.Play(spec, false, () => new Core2Engine(KernelOptions.None));

    [Fact]
    public void TheSeedNamesTheFight()
    {
        var a = Play("old", Spec(5));
        Assert.Null(a["error"]);
        Assert.True(a["steps"]!.AsArray().Count > 3);
        Assert.Equal(a.ToJsonString(), Play("old", Spec(5)).ToJsonString());
        Assert.NotEqual(a["steps"]!.ToJsonString(), Play("old", Spec(6))["steps"]!.ToJsonString());
    }

    [Fact]
    public void ARecordWithoutARoomMeetsItsFightAtTheFirstOptionAndSaysSo()
    {
        var a = Play("old", Spec(1, 0));
        Assert.Null(a["error"]);
        Assert.Equal(2, a["room"]!.AsArray().Count);
        Assert.Single(a["steps"]!.AsArray());
    }

    [Fact]
    public void Core2WithNoSwitchPlaysTheSameFight()
    {
        for (long seed = 1; seed <= 3; seed++)
        {
            var old = Play("old", Spec(seed));
            var core2 = PlayCore2NoSwitch(Spec(seed));
            Assert.Null(core2["error"]);
            Assert.Equal(old["steps"]!.ToJsonString(), core2["steps"]!.ToJsonString());
            Assert.Equal(old["choices"]!.ToJsonString(), core2["choices"]!.ToJsonString());
        }
    }

    [Fact]
    public void TheLegalActionsAreEndTurnPlaysAndPotionsWithTheirTargets()
    {
        var e = EngineFactory.Create("old");
        e.StartFight(new EmuDump.Engine.FightSpec(Spec(1)), new JsonObject());
        var legal = e.LegalActions();
        Assert.Single(legal, a => a.Kind == "end_turn");
        Assert.Contains(legal, a => a.Kind == "play");
        var potion = Assert.Single(legal, a => a.Kind == "potion");  // Fire Potion aims at an enemy
        Assert.DoesNotContain(-1, potion.Targets);
        Assert.All(legal.Where(a => a.Kind == "end_turn"), a => Assert.Equal([-1], a.Targets));
    }
}
