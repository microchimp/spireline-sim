// The codex replay's walker must FAIL on a deliberately broken run history (see
// tests/python/test_*_broken.py for the other checkers). WalkerTests.Record() writes the
// emulator's own run (Neow, then the first fight) in the game's run-history shape, so the faithful
// record walks to the end; each case below breaks one field the walker compares, and the walk must
// stop with "mismatch", never "complete".
using System.Text.Json.Nodes;
using Xunit;

public class WalkerBrokenCaseTests
{
    static JsonObject Floor(JsonObject rec, int floor) =>
        rec["map_point_history"]![0]![floor - 1]!.AsObject();

    static JsonObject Stats(JsonObject rec, int floor) => Floor(rec, floor)["player_stats"]![0]!.AsObject();

    static JsonObject Room(JsonObject rec, int floor) => Floor(rec, floor)["rooms"]![0]!.AsObject();

    /// <summary>One field of the faithful record made wrong, by the component expected to catch it.</summary>
    public static TheoryData<string> Cases() => new(Breaks.Keys);

    static readonly Dictionary<string, Action<JsonObject>> Breaks = new()
    {
        ["acts"] = rec => rec["acts"]![1] = (string)rec["acts"]![1]! == "ACT.HIVE" ? "ACT.GLORY" : "ACT.HIVE",
        ["ancient: another option offered"] = rec =>
        {
            var opts = Stats(rec, 1)["ancient_choice"]!.AsArray();
            var other = opts.First(o => (bool)o!["was_chosen"]! == false)!;
            other["TextKey"] = "LEAD_PAPERWEIGHT";
            other["title"]!["key"] = "LEAD_PAPERWEIGHT.title";
        },
        ["ancient: another option chosen"] = rec =>
        {
            var opts = Stats(rec, 1)["ancient_choice"]!.AsArray();
            foreach (var o in opts) o!["was_chosen"] = !(bool)o["was_chosen"]!;
            opts[opts.Count - 1]!["was_chosen"] = false;
        },
        ["encounter"] = rec => Room(rec, 2)["model_id"] = "ENCOUNTER.NIBBITS_WEAK",
        ["monsters"] = rec => Room(rec, 2)["monster_ids"] = new JsonArray("MONSTER.NIBBIT", "MONSTER.NIBBIT"),
        ["map: another room type"] = rec => Floor(rec, 2)["map_point_type"] = "elite",
        ["card_reward"] = rec => Stats(rec, 2)["card_choices"]![1]!["card"]!["id"] = "CARD.ZAP",
        ["gold_reward"] = rec => Stats(rec, 2)["gold_gained"] = (int)Stats(rec, 2)["gold_gained"]! + 7,
        ["outcome: gold on a floor without a fight"] = rec => Stats(rec, 1)["current_gold"] = (int)Stats(rec, 1)["current_gold"]! + 7,
        ["outcome: hp on a floor without a fight"] = rec => Stats(rec, 1)["current_hp"] = (int)Stats(rec, 1)["current_hp"]! - 3,
        ["outcome: max hp on a floor without a fight"] = rec => Stats(rec, 1)["max_hp"] = (int)Stats(rec, 1)["max_hp"]! + 2,
        ["deck: another card gained"] = rec => Stats(rec, 2)["cards_gained"]![0]!["id"] = "CARD.ZAP",
        ["relics: a relic the run never had"] = rec =>
            Stats(rec, 2)["relic_choices"] = new JsonArray(new JsonObject { ["choice"] = "RELIC.ANCHOR", ["was_picked"] = true }),
    };

    /// <summary>
    /// What the walker takes from the history on purpose (Walker.cs: "Fights are not replayed: their HP
    /// and gold are the history's"; the fight floor's gold is compared as gold_gained instead). Changing
    /// it must NOT stop the walk, so the list of what is not compared stays explicit.
    /// </summary>
    public static TheoryData<string> NotComparedCases() => new(NotCompared.Keys);

    static readonly Dictionary<string, Action<JsonObject>> NotCompared = new()
    {
        ["hp after a fight"] = rec => Stats(rec, 2)["current_hp"] = (int)Stats(rec, 2)["current_hp"]! - 3,
        ["gold after a fight (gold_gained is compared)"] = rec => Stats(rec, 2)["current_gold"] = (int)Stats(rec, 2)["current_gold"]! + 7,
    };

    [Theory]
    [MemberData(nameof(NotComparedCases))]
    public void What_the_walker_takes_from_the_history_does_not_stop_it(string what)
    {
        var rec = WalkerTests.Record();
        NotCompared[what](rec);
        Assert.Equal("complete", (string?)WalkerTests.Walk(rec)["status"]);
    }

    [Fact]
    public void The_faithful_record_walks_to_the_end()
    {
        Assert.Equal("complete", (string?)WalkerTests.Walk(WalkerTests.Record())["status"]);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void A_broken_record_stops_the_walk(string what)
    {
        var rec = WalkerTests.Record();
        Breaks[what](rec);
        var result = WalkerTests.Walk(rec);
        Assert.True((string?)result["status"] == "mismatch",
            $"{what}: walk {(string?)result["status"]} ({result["stop"]?.ToJsonString() ?? result["error"]?.ToString() ?? "no stop"})");
    }
}
