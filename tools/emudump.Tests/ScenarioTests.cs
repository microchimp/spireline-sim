// EmuDump scenario (Scenario.cs): the path language, the expectation check, the format check, the setup-to-record translation,
// and a scenario run through both engines. The checked-in directory (tests/scenarios) is run by tests/python/test_scenario_runner.py.
using System.Text.Json.Nodes;
using Xunit;

public class ScenarioTests
{
    static JsonObject Obj(string json) => JsonNode.Parse(json)!.AsObject();

    static readonly string Minimal = """
        {"name": "t/min", "source": "game: Hook.ModifyDamage",
         "setup": {"character": "IRONCLAD", "ascension": 10, "seed": "SCENARIO01",
                   "encounter": {"id": "HUNTER_KILLER", "act": "HIVE", "kind": "normal", "index": 3},
                   "relics": [], "deck": ["STRIKE_IRONCLAD"]},
         "actions": [{"do": "play", "card": "STRIKE_IRONCLAD", "target": 0, "name": "hit"}],
         "expect": [{"path": "snapshot.enemies[0].hp", "delta_from": "start", "equals": -6}]}
        """;

    // ---------------------------------------------------------------------------------------------- paths

    [Fact]
    public void ResolveReadsKeysIndicesPairListsAndLengths()
    {
        var v = Obj("""{"snapshot": {"player": {"powers": [["STRENGTH_POWER", 3], ["WEAK_POWER", 1]]}, "hand": [["A", false], ["B", true]], "rng": {"X": {"counter": 4}}}}""");
        Assert.Equal(3, (int)Scenarios.Resolve(v, "snapshot.player.powers{STRENGTH_POWER}")!);
        Assert.Null(Scenarios.Resolve(v, "snapshot.player.powers{VULNERABLE_POWER}"));
        Assert.Equal("B", (string)Scenarios.Resolve(v, "snapshot.hand[1][0]")!);
        Assert.Equal(2, (int)Scenarios.Resolve(v, "snapshot.hand#len")!);
        Assert.Equal(4, (int)Scenarios.Resolve(v, "snapshot.rng.X.counter")!);
        Assert.Null(Scenarios.Resolve(v, "snapshot.hand[5]"));
        Assert.Null(Scenarios.Resolve(v, "snapshot.nothing.deeper"));
        Assert.Null(Scenarios.Resolve(v, "snapshot.nothing#len"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("a[x]")]
    [InlineData("a[1")]
    [InlineData("a{")]
    [InlineData("a#size")]
    [InlineData("a{}")]
    public void ResolveRejectsAMalformedPath(string path) =>
        Assert.Throws<FormatException>(() => Scenarios.Resolve(new JsonObject(), path));

    [Fact]
    public void SameComparesNumbersByValueAndStructuresDeeply()
    {
        Assert.True(Scenarios.Same(JsonValue.Create(6), Obj("""{"v": 6.0}""")["v"]));
        Assert.True(Scenarios.Same(JsonNode.Parse("""[["A", false], 3]"""), JsonNode.Parse("""[["A", false], 3.0]""")));
        Assert.False(Scenarios.Same(JsonNode.Parse("""[1, 2]"""), JsonNode.Parse("""[1, 2, 3]""")));
        Assert.False(Scenarios.Same(JsonNode.Parse("""{"a": 1}"""), JsonNode.Parse("""{"a": 1, "b": 2}""")));
        Assert.False(Scenarios.Same(JsonValue.Create("6"), JsonValue.Create(6)));
        Assert.True(Scenarios.Same(null, null));
        Assert.False(Scenarios.Same(null, JsonValue.Create(0)));
    }

    // ---------------------------------------------------------------------------------------------- checks

    static Dictionary<string, JsonObject> Points() => new()
    {
        ["start"] = Obj("""{"snapshot": {"player": {"hp": 80, "block": 0}, "enemies": [{"hp": 44}]}}"""),
        ["end"] = Obj("""{"snapshot": {"player": {"hp": 67, "block": 5}, "enemies": [{"hp": 38}]}}"""),
    };

    [Fact]
    public void CheckPassesWhatHoldsAndNamesWhatDoesNot()
    {
        var ok = JsonNode.Parse("""
            [{"path": "snapshot.player.hp", "equals": 67},
             {"at": "start", "path": "snapshot.player.hp", "equals": 80},
             {"path": "snapshot.enemies[0].hp", "delta_from": "start", "equals": -6},
             {"path": "snapshot.player.block", "one_of": [0, 5]},
             {"path": "snapshot.player.hp", "at_least": 67}, {"path": "snapshot.player.hp", "at_most": 67},
             {"path": "snapshot.player.gold", "absent": true}]
            """)!.AsArray();
        Assert.Empty(Scenarios.Check(Points(), ok));

        var bad = JsonNode.Parse("""
            [{"path": "snapshot.player.hp", "equals": 58},
             {"path": "snapshot.enemies[0].hp", "delta_from": "start", "equals": -7},
             {"path": "snapshot.player.block", "one_of": [1, 2]},
             {"path": "snapshot.player.hp", "at_least": 68},
             {"path": "snapshot.player.gold", "equals": 1},
             {"path": "snapshot.player.hp", "absent": true},
             {"at": "after:nothing", "path": "snapshot.player.hp", "equals": 1}]
            """)!.AsArray();
        var diffs = Scenarios.Check(Points(), bad);
        Assert.Equal(7, diffs.Count);
        Assert.Equal("at end, snapshot.player.hp: expected 58, got 67", diffs[0]);
        Assert.Equal("at end, snapshot.enemies[0].hp (change since start): expected -7, got -6", diffs[1]);
        Assert.Contains("expected one of [1,2], got 5", diffs[2]);
        Assert.Contains("expected at least 68, got 67", diffs[3]);
        Assert.Contains("got <absent>", diffs[4]);
        Assert.Contains("expected absent, got 67", diffs[5]);
        Assert.Contains("never reached", diffs[6]);
    }

    // ---------------------------------------------------------------------------------------------- format

    [Fact]
    public void AMinimalScenarioIsWellFormed() => Assert.Empty(Scenarios.Validate(JsonNode.Parse(Minimal)));

    [Theory]
    [InlineData("\"hp\"", "\"hpp\"", "unknown key")]
    [InlineData("\"source\": \"game: Hook.ModifyDamage\"", "\"source\": \"somewhere\"", "source")]
    [InlineData("\"character\": \"IRONCLAD\"", "\"character\": \"SILENT\"", "setup.character")]
    [InlineData("\"act\": \"HIVE\"", "\"act\": \"MARS\"", "setup.encounter.act")]
    [InlineData("\"do\": \"play\"", "\"do\": \"dance\"", "actions[0].do")]
    [InlineData("\"delta_from\": \"start\"", "\"delta_from\": \"after:nope\"", "expect[0].delta_from")]
    [InlineData("\"equals\": -6", "\"equals\": -6, \"at_least\": 1", "exactly one of")]
    [InlineData("\"name\": \"hit\"", "\"name\": \"hit\", \"engine\": \"old\"", "unknown key 'engine'")]
    public void ABrokenScenarioIsNamedByItsField(string from, string to, string expected)
    {
        string text = (from == "\"hp\"" ? Minimal.Replace("\"relics\": []", "\"relics\": [], \"hpp\": 1") : Minimal.Replace(from, to));
        var problems = Scenarios.Validate(JsonNode.Parse(text));
        Assert.Contains(problems, p => p.Contains(expected));
    }

    [Fact]
    public void AnXfailForAnEngineTheScenarioDoesNotRunOnIsRejected()
    {
        var sc = Obj(Minimal);
        sc["engines"] = new JsonArray("core2");
        sc["xfail_engines"] = new JsonObject { ["old"] = "a reason the row waits on" };
        Assert.Contains(Scenarios.Validate(sc), p => p.Contains("not run on old"));
        sc["engines"] = new JsonArray("old", "core2");
        sc["xfail_engines"] = new JsonObject { ["old"] = "no" };
        Assert.Contains(Scenarios.Validate(sc), p => p.Contains("a reason"));
    }

    // ---------------------------------------------------------------------------------------------- setup to record

    [Fact]
    public void TheSetupBecomesTheItemsSetupCsRuns()
    {
        var sc = Obj(Minimal);
        var setup = sc["setup"]!.AsObject();
        setup["relics"] = new JsonArray("BURNING_BLOOD", "MEAT_ON_THE_BONE");
        setup["potions"] = new JsonArray("FIRE_POTION");
        setup["hp"] = Obj("""{"max": 80, "current": 40}""");
        setup["powers"] = JsonNode.Parse("""[{"id": "STRENGTH_POWER", "target": "player", "amount": 2}, {"id": "VULNERABLE_POWER", "target": "enemy:1", "amount": 1}]""");
        setup["energy"] = 2;
        setup["hand"] = JsonNode.Parse("""["DEFEND_IRONCLAD"]""");
        setup["draw"] = JsonNode.Parse("""[{"id": "BASH", "upgraded": true, "top": true}]""");
        setup["discard"] = JsonNode.Parse("""["STRIKE_IRONCLAD"]""");
        var rec = Scenarios.BuildRecord(setup);
        var items = rec["setup"]!.AsArray().Select(i => (string)i!["cmd"]!).ToList();
        Assert.Equal(["act", "relics_set", "deck_set", "potions_set", "hp", "fight", "power", "power", "energy", "card", "card", "card"], items);
        Assert.Equal(2, (int)rec["setup"]![0]!["index"]!);
        Assert.Equal("Hand", (string)rec["setup"]![9]!["pile"]!);
        Assert.Equal("top", (string)rec["setup"]![10]!["pos"]!);
        Assert.True((bool)rec["setup"]![10]!["upgraded"]!);
        Assert.Equal("Discard", (string)rec["setup"]![11]!["pile"]!);
        Assert.Equal("normal", (string)rec["fight"]!["kind"]!);
        Assert.Equal(3, (int)rec["fight"]!["index"]!);
        Assert.Equal("SCENARIO01", (string)rec["seed"]!);
    }

    // ---------------------------------------------------------------------------------------------- a run

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void AScenarioRunsThroughTheEngineAndItsExpectationsAreChecked(string engine)
    {
        var pass = Scenarios.Run("t.json", JsonNode.Parse(Minimal), engine, keepPoints: true);
        Assert.Equal("pass", pass.Status);
        Assert.NotNull(pass.Points);
        Assert.NotNull(pass.Points!["after:hit"]);

        var wrong = Scenarios.Run("t.json", JsonNode.Parse(Minimal.Replace("\"equals\": -6", "\"equals\": -7")), engine, keepPoints: false);
        Assert.Equal("fail", wrong.Status);
        Assert.Contains("expected -7, got -6", wrong.Diffs[0]);
        Assert.True(wrong.Unexpected);

        var listed = JsonNode.Parse(Minimal.Replace("\"equals\": -6", "\"equals\": -7"))!.AsObject();
        listed["xfail_engines"] = new JsonObject { [engine] = "a reason the row waits on" };
        Assert.Equal("xfail", Scenarios.Run("t.json", listed, engine, keepPoints: false).Status);

        var illegal = JsonNode.Parse(Minimal.Replace("\"target\": 0", "\"target\": 3"));
        var res = Scenarios.Run("t.json", illegal, engine, keepPoints: false);
        Assert.Equal("fail", res.Status);
        Assert.Contains("not legal", res.Diffs[0]);
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void ARunThatStoppedIsAFailEvenWhenTheEngineIsListedXfail(string engine)
    {
        var sc = Obj(Minimal.Replace("\"target\": 0", "\"target\": 3"));
        sc["xfail_engines"] = new JsonObject { [engine] = "a reason the row waits on" };
        var res = Scenarios.Run("t.json", sc, engine, keepPoints: false);
        Assert.Equal("fail", res.Status);
        Assert.True(res.Unexpected);
        Assert.Contains("run stopped", res.Diffs[0]);
    }

    [Theory]
    [InlineData("at_least", "\"5\"")]
    [InlineData("at_least", "true")]
    [InlineData("at_most", "[1]")]
    [InlineData("at_most", "null")]
    [InlineData("absent", "\"yes\"")]
    [InlineData("absent", "1")]
    public void AnOperandOfTheWrongTypeIsInvalidNotACrash(string op, string operand)
    {
        var sc = Obj(Minimal);
        sc["expect"] = JsonNode.Parse("[{\"path\": \"snapshot.enemies[0].hp\", \"" + op + "\": " + operand + "}]");
        Assert.Contains(Scenarios.Validate(sc), p => p.StartsWith("expect[0]." + op, StringComparison.Ordinal));
        var res = Scenarios.Run("t.json", sc, "old", keepPoints: false);
        Assert.Equal("invalid", res.Status);
        Assert.True(res.Unexpected);

        // Check on its own, for a caller that skipped Validate: a named fault, no exception.
        var diffs = Scenarios.Check(Points(), sc["expect"]!.AsArray());
        Assert.StartsWith(Scenarios.InvalidMark, Assert.Single(diffs));
    }

    [Fact]
    public void AbsentOnAPathWhoseParentDoesNotExistIsInvalid()
    {
        var sc = Obj(Minimal);
        sc["expect"] = JsonNode.Parse("""[{"path": "snapshot.enemis[0].gold", "absent": true}]""");
        var diffs = Scenarios.Check(Points(), sc["expect"]!.AsArray());
        Assert.StartsWith(Scenarios.InvalidMark, Assert.Single(diffs));
        Assert.Contains("parent does not exist", diffs[0]);
        var res = Scenarios.Run("t.json", sc, "old", keepPoints: false);
        Assert.Equal("invalid", res.Status);
        Assert.Contains("parent does not exist", res.Reason ?? string.Join(";", res.Diffs));

        // Where the parent exists the check still means what it says, and a scalar is no parent.
        Assert.Empty(Scenarios.Check(Points(), JsonNode.Parse("""[{"path": "snapshot.player.gold", "absent": true}, {"path": "snapshot.enemies[0].gold", "absent": true}]""")!.AsArray()));
        Assert.StartsWith(Scenarios.InvalidMark, Assert.Single(Scenarios.Check(Points(), JsonNode.Parse("""[{"path": "snapshot.player.hp.gold", "absent": true}]""")!.AsArray())));
    }

    [Fact]
    public void AnEngineTheScenarioLeavesOutIsSkipped()
    {
        var sc = Obj(Minimal);
        sc["engines"] = new JsonArray("core2");
        var r = Scenarios.Run("t.json", sc, "old", keepPoints: false);
        Assert.Equal("skip", r.Status);
        Assert.False(r.Unexpected);
    }
}
