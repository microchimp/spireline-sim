// EmuDump scenario (Scenario.cs): the path language, the expectation check, the format check, the setup-to-record translation,
// and a scenario run through both engines. The checked-in directory (tests/scenarios) is run by tests/python/test_scenario_runner.py.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines;
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

    // ---------------------------------------------------------------------------------------------- the setters' format

    /// <summary>Minimal with the player's block, the lone enemy's HP and its block set (docs/rebuild/SCENARIOS.md: `block`, `enemy_hp`, `enemy_block`).</summary>
    static JsonObject WithSetters(string block = "5", string enemyHp = """[{"slot": 0, "current": 8}]""", string enemyBlock = """[{"slot": 0, "amount": 4}]""")
    {
        var sc = Obj(Minimal);
        var setup = sc["setup"]!.AsObject();
        setup["block"] = JsonNode.Parse(block);
        setup["enemy_hp"] = JsonNode.Parse(enemyHp);
        setup["enemy_block"] = JsonNode.Parse(enemyBlock);
        return sc;
    }

    [Fact]
    public void ASetupWithTheSettersIsWellFormed()
    {
        Assert.Empty(Scenarios.Validate(WithSetters()));
        Assert.Empty(Scenarios.Validate(WithSetters("0", """[{"slot": 1, "max": 30}, {"slot": 0, "max": 20, "current": 20}]""", "[]")));
    }

    [Theory]
    [InlineData("-1", null, null, "setup.block: an integer, 0 or more")]
    [InlineData("\"5\"", null, null, "setup.block: an integer, 0 or more")]
    [InlineData(null, """[{"slot": 0}]""", null, "setup.enemy_hp[0]: `max` or `current`, at least one")]
    [InlineData(null, """[{"current": 5}]""", null, "setup.enemy_hp[0]: missing 'slot'")]
    [InlineData(null, """[{"slot": 0, "current": 0}]""", null, "setup.enemy_hp[0].current: an integer, 1 or more")]
    [InlineData(null, """[{"slot": 0, "max": -3}]""", null, "setup.enemy_hp[0].max: an integer, 1 or more")]
    [InlineData(null, """[{"slot": 0, "max": 5, "current": 9}]""", null, "setup.enemy_hp[0]: current 9 above max 5")]
    [InlineData(null, """[{"slot": -1, "current": 5}]""", null, "setup.enemy_hp[0].slot: an integer, 0 or more (a creature slot, as powers' enemy:<slot>)")]
    [InlineData(null, """[{"slot": 0, "current": 5}, {"slot": 0, "max": 9}]""", null, "setup.enemy_hp[1].slot: slot 0 is set twice")]
    [InlineData(null, """[{"slot": 0, "hp": 5}]""", null, "setup.enemy_hp[0]: unknown key 'hp'")]
    [InlineData(null, """{"slot": 0, "current": 5}""", null, "setup.enemy_hp: a list of objects {slot, max, current}")]
    [InlineData(null, null, """[{"slot": 0}]""", "setup.enemy_block[0]: missing 'amount'")]
    [InlineData(null, null, """[{"slot": 0, "amount": -2}]""", "setup.enemy_block[0].amount: an integer, 0 or more")]
    [InlineData(null, null, """[{"slot": 0, "amount": 2}, {"slot": 0, "amount": 3}]""", "setup.enemy_block[1].slot: slot 0 is set twice")]
    [InlineData(null, null, """[5]""", "setup.enemy_block[0]: an object {slot, amount}")]
    public void ABrokenSetterIsInvalidAndNamedByItsField(string? block, string? enemyHp, string? enemyBlock, string expected)
    {
        var sc = WithSetters(block ?? "5", enemyHp ?? """[{"slot": 0, "current": 8}]""", enemyBlock ?? """[{"slot": 0, "amount": 4}]""");
        var problems = Scenarios.Validate(sc);
        Assert.Contains(expected, problems);
        // The file never reaches an engine: the run is invalid, and so unexpected.
        var res = Scenarios.Run("t.json", sc, "old", keepPoints: false);
        Assert.Equal("invalid", res.Status);
        Assert.True(res.Unexpected);
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

    [Fact]
    public void TheSettersBecomeCombatItemsAfterTheFightAndBeforeThePowers()
    {
        var sc = WithSetters("5", """[{"slot": 0, "current": 8}, {"slot": 1, "max": 30}]""", """[{"slot": 0, "amount": 4}]""");
        var setup = sc["setup"]!.AsObject();
        setup["powers"] = JsonNode.Parse("""[{"id": "STRENGTH_POWER", "target": "player", "amount": 2}]""");
        setup["energy"] = 1;
        setup["hand"] = JsonNode.Parse("""["DEFEND_IRONCLAD"]""");
        var items = Scenarios.BuildRecord(setup)["setup"]!.AsArray();
        Assert.Equal(["act", "relics_set", "deck_set", "potions_set", "hp", "fight", "block", "enemy_hp", "enemy_hp", "enemy_block", "power", "energy", "card"],
            items.Select(i => (string)i!["cmd"]!).ToList());
        Assert.Equal(5, (int)items[6]!["amount"]!);
        Assert.Equal(0, (int)items[7]!["slot"]!);
        Assert.Equal(8, (int)items[7]!["current"]!);
        Assert.Null(items[7]!["max"]);  // left out: the rolled max stays
        Assert.Equal(1, (int)items[8]!["slot"]!);
        Assert.Equal(30, (int)items[8]!["max"]!);
        Assert.Null(items[8]!["current"]);
        Assert.Equal(0, (int)items[9]!["slot"]!);
        Assert.Equal(4, (int)items[9]!["amount"]!);

        // Without the keys, no item: the converted scenarios build the record they always did.
        var plain = Scenarios.BuildRecord(Obj(Minimal)["setup"]!.AsObject())["setup"]!.AsArray().Select(i => (string)i!["cmd"]!).ToList();
        Assert.DoesNotContain("block", plain);
        Assert.DoesNotContain("enemy_hp", plain);
        Assert.DoesNotContain("enemy_block", plain);
    }

    // ---------------------------------------------------------------------------------------------- the setters through the engine

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void TheSettersTakeEffectAtTheFightsStartAndTheHitReadsThem(string engine)
    {
        var sc = WithSetters();
        sc["expect"] = JsonNode.Parse("""
            [{"at": "start", "path": "snapshot.player.block", "equals": 5},
             {"at": "start", "path": "snapshot.enemies[0].hp", "equals": 8},
             {"at": "start", "path": "snapshot.enemies[0].block", "equals": 4},
             {"at": "start", "path": "snapshot.enemies[0].max_hp", "at_least": 9},
             {"path": "snapshot.enemies[0].block", "equals": 0},
             {"path": "snapshot.enemies[0].hp", "equals": 6}]
            """);
        var res = Scenarios.Run("t.json", sc, engine, keepPoints: true);
        Assert.Equal("pass", res.Status);
        Assert.Empty(res.Diffs);
        // The net view carries the same creature fields (the player is the first creature, the lone enemy the second).
        var creatures = res.Points!["start"]!["net"]!["creatures"]!.AsArray();
        Assert.Equal(2, creatures.Count);
        Assert.Equal(5, (int)creatures[0]!["block"]!);
        Assert.Equal(8, (int)creatures[1]!["hp"]!);
        Assert.Equal(4, (int)creatures[1]!["block"]!);
    }

    [Theory]
    [InlineData("old", """[{"slot": 3, "current": 8}]""", "no living enemy in slot 3")]
    [InlineData("core2", """[{"slot": 3, "current": 8}]""", "no living enemy in slot 3")]
    [InlineData("old", """[{"slot": 0, "current": 9999}]""", "current HP 9999 above max HP")]
    [InlineData("core2", """[{"slot": 0, "current": 9999}]""", "current HP 9999 above max HP")]
    public void ASetterTheFightCannotHoldFailsTheRunAtItsSetupItem(string engine, string enemyHp, string reason)
    {
        var sc = WithSetters(enemyHp: enemyHp);
        var res = Scenarios.Run("t.json", sc, engine, keepPoints: false);
        Assert.Equal("fail", res.Status);
        Assert.True(res.Unexpected);
        Assert.StartsWith("run stopped", res.Diffs[0]);
        Assert.Contains("enemy_hp", res.Diffs[0]);
        Assert.Contains("did not take", res.Diffs[0]);
        Assert.Contains(reason, res.Diffs[0]);

        // Refused, not capped: an xfail row cannot claim a run that stopped.
        sc["xfail_engines"] = new JsonObject { [engine] = "a reason the row waits on" };
        Assert.Equal("fail", Scenarios.Run("t.json", sc, engine, keepPoints: false).Status);
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void ASlotWithNoLivingEnemyRefusesTheBlockToo(string engine)
    {
        var res = Scenarios.Run("t.json", WithSetters(enemyBlock: """[{"slot": 1, "amount": 4}]"""), engine, keepPoints: false);
        Assert.Equal("fail", res.Status);
        Assert.Contains("enemy_block", res.Diffs[0]);
        Assert.Contains("no living enemy in slot 1", res.Diffs[0]);
    }

    /// <summary>The fight Minimal's setup opens, on an engine, reached through IEngine only.</summary>
    static IEngine Fight(string engine)
    {
        var e = EngineFactory.Create(engine);
        e.StartFight(new FightSpec(Scenarios.BuildRecord(Obj(Minimal)["setup"]!.AsObject())), new JsonObject());
        Assert.True(e.CombatOpen);
        return e;
    }

    static (int Hp, int MaxHp, int Block) Enemy(IEngine e)
    {
        var en = e.GameView()["snapshot"]!["enemies"]![0]!;
        return ((int)en["hp"]!, (int)en["max_hp"]!, (int)en["block"]!);
    }

    static int PlayerBlock(IEngine e) => (int)e.GameView()["snapshot"]!["player"]!["block"]!;

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void TheEngineRefusesABadValueAndLeavesTheStateAsItWas(string engine)
    {
        var e = Fight(engine);
        var before = Enemy(e);
        Assert.True(before.MaxHp >= before.Hp && before.Hp > 1);
        int max = before.MaxHp;

        void Refused(SetupResult r, string reason)
        {
            Assert.False(r.Ok);
            Assert.Contains(reason, r.Message);
            Assert.DoesNotContain("unsupported", r.Message);  // a bad value is no mirror gap
        }

        Refused(e.SetBlock(-1), "block -1 below 0");
        Refused(e.SetEnemyBlock(0, -1), "block -1 below 0");
        Refused(e.SetEnemyBlock(5, 2), "no living enemy in slot 5");
        Refused(e.SetEnemyBlock(-1, 2), "no living enemy in slot -1");
        Refused(e.SetEnemyHp(5, null, 2), "no living enemy in slot 5");
        Refused(e.SetEnemyHp(0, null, 0), "current HP 0 below 1");
        Refused(e.SetEnemyHp(0, null, -4), "current HP -4 below 1");
        Refused(e.SetEnemyHp(0, 0, null), "max HP 0 below 1");
        Refused(e.SetEnemyHp(0, null, max + 1), $"current HP {max + 1} above max HP {max}");
        Refused(e.SetEnemyHp(0, 10, 11), "current HP 11 above max HP 10");
        Assert.Equal(before, Enemy(e));
        Assert.Equal(0, PlayerBlock(e));
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void TheEngineSetsMaxHpThenCurrentHpAndAHeldCurrentComesDownToALoweredMax(string engine)
    {
        var e = Fight(engine);
        var (hp, max, _) = Enemy(e);

        Assert.True(e.SetEnemyHp(0, max + 100, null).Ok);  // a raised max leaves the held current alone
        Assert.Equal((hp, max + 100, 0), Enemy(e));
        Assert.True(e.SetEnemyHp(0, max + 100, max + 100).Ok);  // a current above the rolled max, with its max given
        Assert.Equal((max + 100, max + 100, 0), Enemy(e));
        Assert.True(e.SetEnemyHp(0, 10, null).Ok);  // a lowered max brings the held current down to it
        Assert.Equal((10, 10, 0), Enemy(e));
        Assert.True(e.SetEnemyHp(0, null, 3).Ok);  // a current alone keeps the max
        Assert.Equal((3, 10, 0), Enemy(e));
        Assert.True(e.SetEnemyBlock(0, 7).Ok);
        Assert.Equal((3, 10, 7), Enemy(e));
        Assert.True(e.SetEnemyBlock(0, 0).Ok);
        Assert.Equal((3, 10, 0), Enemy(e));
        Assert.True(e.SetBlock(9).Ok);
        Assert.Equal(9, PlayerBlock(e));
        Assert.True(e.SetBlock(0).Ok);
        Assert.Equal(0, PlayerBlock(e));
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void ASetterBeforeTheFightItemIsAMirrorGapNotADivergence(string engine)
    {
        // A record (not a scenario file: BuildRecord always puts them after the fight) with the items on the map, where no fight is open.
        var record = Scenarios.BuildRecord(Obj(Minimal)["setup"]!.AsObject());
        var items = record["setup"]!.AsArray();
        int fight = items.Select((it, i) => ((string)it!["cmd"]!, i)).First(x => x.Item1 == "fight").i;
        items.Insert(fight, new JsonObject { ["cmd"] = "block", ["amount"] = 3 });
        items.Insert(fight, new JsonObject { ["cmd"] = "enemy_hp", ["slot"] = 0, ["current"] = 3 });
        items.Insert(fight, new JsonObject { ["cmd"] = "enemy_block", ["slot"] = 0, ["amount"] = 3 });
        var e = EngineFactory.Create(engine);
        var report = new JsonObject();
        e.StartFight(new FightSpec(record), report);
        var log = report["setup_log"]!.AsArray().Where(x => (string?)x!["cmd"] is "block" or "enemy_hp" or "enemy_block").ToList();
        Assert.Equal(3, log.Count);
        Assert.All(log, x =>
        {
            Assert.False((bool)x!["ok"]!);
            Assert.True((bool?)x["unsupported"] == true);
            Assert.Contains("outside a fight", (string)x["msg"]!);
        });
        Assert.True(e.CombatOpen);
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
