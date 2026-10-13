// The codex replay's walker against synthetic run histories: records written in the game's
// run-history shape from the emulator's own run (Neow, then the first fight), so a faithful walk
// must match every floor, and a record with one offer changed must stop on exactly that offer.
using System.Text.Json;
using System.Text.Json.Nodes;
using EmuDump.Codex;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

public class WalkerTests
{
    const string Seed = "CODEXTEST1";
    const int Asc = 10;

    static JsonObject Stats(RunState s, int goldGained) => new()
    {
        ["current_gold"] = s.Gold,
        ["current_hp"] = s.PlayerHp,
        ["max_hp"] = s.PlayerMaxHp,
        ["gold_gained"] = goldGained,
        ["gold_spent"] = 0,
        ["gold_stolen"] = 0,
        ["stolen_loot"] = 0,
    };

    /// <summary>Neow and the first monster fight of the emulator's run, as the game records them.</summary>
    static JsonObject Record(Func<List<string>, List<string>>? tamperOffer = null)
    {
        var e = new RunEngine();
        e.Reset(Seed, Asc);
        var s = e.State;

        // Floor 1: the first blessing that is a relic and nothing else (no screen, no cards).
        int choice = -1;
        for (int i = 0; i < 3 && choice < 0; i++)
        {
            var c = e.Clone();
            c.Step(i, -1, out _, out _, out _);
            if (c.State.Phase == RunPhase.Ancient && c.State.NeowAwaitingProceed && c.State.Deck.Count == s.Deck.Count
                && c.State.Relics.Count == s.Relics.Count + 1 && c.State.Gold == s.Gold && c.State.PotionSlots.All(p => p == 0))
            {
                choice = i;
            }
        }

        Assert.True(choice >= 0, "no plain relic blessing on this seed");
        var options = s.NeowOptions.Select(Names.Relic).ToList();
        string chosen = options[choice];
        e.Step(choice, -1, out _, out _, out _);
        e.Step(0, -1, out _, out _, out _); // Neow's Proceed
        var neow = Stats(s, 0);
        neow["ancient_choice"] = new JsonArray(options.Select((o, i) => (JsonNode)new JsonObject
        {
            ["TextKey"] = o,
            ["title"] = new JsonObject { ["key"] = o + ".title", ["table"] = "relics" },
            ["was_chosen"] = i == choice,
        }).ToArray());
        neow["relic_choices"] = new JsonArray(new JsonObject { ["choice"] = "RELIC." + chosen, ["was_picked"] = true });
        var floor1 = new JsonObject
        {
            ["map_point_type"] = "ancient",
            ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = "EVENT.NEOW", ["room_type"] = "event" }),
            ["player_stats"] = new JsonArray(neow),
        };

        // Floor 2: the first monster node, its fight resolved, its gold and first card taken.
        int node = Array.IndexOf(s.MapNodeTypes, RunConstants.NodeNormal);
        Assert.True(node >= 0, "no monster node in the first row");
        e.Step(node, -1, out _, out _, out _);
        var fight = s.ActiveCombat!;
        string encounter = "ENCOUNTER." + Names.GameEntry(Names.EncounterName(fight.EncounterId));
        var monsters = fight.Enemies.Select(x => (JsonNode)("MONSTER." + Names.Enemy(x.DefId))).ToArray();
        e.DebugResolveCombat();
        int goldBefore = s.Gold;
        var offer = s.RewardCards.Where(c => c != 0).Select(c => "CARD." + Names.Card(c)).ToList();
        string? potion = s.RewardPotion != 0 ? "POTION." + Names.Potion(s.RewardPotion) : null;
        string picked = offer[0];
        if (s.RewardGold != 0)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        int gold = s.Gold - goldBefore;
        // The rest of the screen, claimed the way the walker will: the card reward, its first card.
        int cardItem = (s.RewardPotion != 0 ? 1 : 0) + (s.RelicReward != 0 ? 1 : 0);
        e.Step(cardItem, -1, out _, out _, out _);
        e.Step(0, -1, out _, out _, out _);
        e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);
        var shown = (tamperOffer ?? (x => x))(offer);
        var fightStats = Stats(s, gold);
        // The history lists the card taken first.
        fightStats["card_choices"] = new JsonArray(shown.Select((c, i) => (JsonNode)new JsonObject
        {
            ["card"] = new JsonObject { ["id"] = c },
            ["was_picked"] = i == 0,
        }).ToArray());
        fightStats["cards_gained"] = new JsonArray(new JsonObject { ["id"] = picked });
        if (potion is not null)
        {
            fightStats["potion_choices"] = new JsonArray(new JsonObject { ["choice"] = potion, ["was_picked"] = false });
        }

        var floor2 = new JsonObject
        {
            ["map_point_type"] = "monster",
            ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = encounter, ["monster_ids"] = new JsonArray(monsters), ["room_type"] = "monster" }),
            ["player_stats"] = new JsonArray(fightStats),
        };

        return new JsonObject
        {
            ["seed"] = Seed,
            ["ascension"] = Asc,
            ["game_mode"] = "standard",
            ["build_id"] = "v0.107.1",
            ["win"] = true,
            ["was_abandoned"] = false,
            ["modifiers"] = new JsonArray(),
            ["acts"] = new JsonArray(s.Acts.Select(a => (JsonNode)("ACT." + Names.Act(a.Act))).ToArray()),
            ["players"] = new JsonArray(new JsonObject { ["character"] = "CHARACTER.IRONCLAD" }),
            ["map_point_history"] = new JsonArray(new JsonArray(floor1, floor2)),
        };
    }

    static JsonObject Walk(JsonObject record)
    {
        using var doc = JsonDocument.Parse(record.ToJsonString());
        return new Walker(new EventOptionIndex(null)).Walk(doc.RootElement);
    }

    [Fact]
    public void A_faithful_record_walks_every_floor_and_matches()
    {
        var result = Walk(Record());
        Assert.Equal("complete", (string?)result["status"]);
        Assert.Equal(2, (int)result["floors_walked"]!);
        var checks = result["checks"]!.AsObject();
        foreach (string component in new[] { "acts", "ancient", "map", "encounter", "monsters", "card_reward", "gold_reward" })
        {
            var counts = checks[component]!.AsArray();
            Assert.True((int)counts[0]! > 0, component);
            Assert.Equal((int)counts[0]!, (int)counts[1]!);
        }
    }

    [Fact]
    public void A_changed_card_offer_stops_the_walk_on_that_offer()
    {
        var result = Walk(Record(offer => [offer[0], "CARD.ZAP", .. offer.Skip(2)]));
        Assert.Equal("mismatch", (string?)result["status"]);
        var stop = result["stop"]!.AsObject();
        Assert.Equal(2, (int)stop["floor"]!);
        Assert.Equal("card_reward", (string?)stop["component"]);
        Assert.Equal(Walker.Emulator, (string?)stop["class"]);
        Assert.Contains("ZAP", (string?)stop["expected"]);
    }

    [Fact]
    public void Card_rewards_are_matched_to_the_history_group_with_their_cards_in_any_claim_order()
    {
        // The history lists rewards in the order they were claimed: the second offer first.
        var history = new List<string> { "SLICE", "POKE", "BULWARK", "ANGER", "BASH", "CLASH" };
        Assert.Equal([3, 4, 5], Walker.CardGroup(history, [], ["BASH", "CLASH", "ANGER"]));
        Assert.Equal([0, 1, 2], Walker.CardGroup(history, [3, 4, 5], ["POKE", "SLICE", "BULWARK"]));
        // Nothing matches: the next entries are compared, and the mismatch reported there.
        Assert.Equal([0, 1, 2], Walker.CardGroup(history, [], ["ZAP", "BASH", "CLASH"]));
    }

    [Fact]
    public void A_larger_group_is_not_matched_by_its_tail()
    {
        // A fourth card the emulator did not add: compared as the group's first three, not
        // passed off as three entries from the middle of it.
        var history = new List<string> { "WHIRLWIND", "SECOND_WIND", "INFERNAL_BLADE", "FIGHT_ME" };
        Assert.Equal([0, 1, 2], Walker.CardGroup(history, [], ["SECOND_WIND", "INFERNAL_BLADE", "FIGHT_ME"]));
    }

    [Fact]
    public void An_ancient_option_is_named_by_its_title_key_with_the_sea_glass_character()
    {
        static JsonElement Option(string textKey, string titleKey) =>
            JsonDocument.Parse(new JsonObject
            {
                ["TextKey"] = textKey,
                ["title"] = new JsonObject { ["key"] = titleKey, ["table"] = "relics" },
            }.ToJsonString()).RootElement;

        Assert.Equal("SEA_GLASS.DEFECT", Walker.AncientOption(Option("DEFECT", "SEA_GLASS.DEFECT.title")));
        Assert.Equal("KALEIDOSCOPE", Walker.AncientOption(Option("KALEIDOSCOPE", "KALEIDOSCOPE.title")));
    }

    [Fact]
    public void Runs_with_a_mods_acts_or_ids_are_skipped_as_modded()
    {
        var rec = Record();
        rec["acts"] = new JsonArray("ACT.ACTSFROMTHEPAST-UNDERCITY", "ACT.HIVE", "ACT.GLORY");
        using var doc = JsonDocument.Parse(rec.ToJsonString());
        Assert.Equal("modded act (modded)", Driver.Skip(doc.RootElement, "IRONCLAD", null));
        Assert.Matches(Driver.ModdedId, "{\"model_id\": \"EVENT.ACTSFROMTHEPAST-DUPLICATOR\"}");
        Assert.DoesNotMatch(Driver.ModdedId, "{\"model_id\": \"EVENT.COLORFUL_PHILOSOPHERS\"}");
    }

    [Fact]
    public void Content_of_an_epoch_the_run_never_shows_is_taken_as_locked()
    {
        // An epoch class names what it unlocks; a synthetic one stands in for the decompile.
        string dir = Directory.CreateTempSubdirectory("codex-epochs").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "Ironclad2Epoch.cs"), "Card<Cruelty>(), Card<Dominate>(), Card<MoltenFist>()");
            var index = new ProfileIndex(dir);
            Assert.True(index.Available);
            Assert.Equal(("DOMINATE", "Ironclad2"), index.Unseen(["BASH", "DOMINATE"], ProfileIndex.Seen("{\"id\": \"CARD.BASH\"}")));
            // One card of the epoch anywhere in the run says it is unlocked.
            Assert.Null(index.Unseen(["DOMINATE"], ProfileIndex.Seen("{\"id\": \"CARD.CRUELTY\"}")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void A_fight_that_draws_a_stream_marks_it_for_later_offers()
    {
        // After a resolved fight its turns' streams are tainted, so an event that rolls on one
        // of them is classed fight-dependent; Rewards is not among them.
        Assert.DoesNotContain("Rewards", Streams.FightTurns);
        Assert.DoesNotContain("Niche", Streams.FightTurns);
        Assert.Contains("CombatCardGeneration", Streams.FightTurns);
    }
}
