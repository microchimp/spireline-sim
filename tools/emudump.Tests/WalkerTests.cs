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
    internal static JsonObject Record(Func<List<string>, List<string>>? tamperOffer = null)
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

    internal static JsonObject Walk(JsonObject record)
    {
        using var doc = JsonDocument.Parse(record.ToJsonString());
        return new Walker(new EventOptionIndex(null)).Walk(doc.RootElement);
    }

    static string[] Belt(RunState s) => [.. s.PotionSlots.Take(s.MaxPotionSlots).Select(x => x == 0 ? "-" : Names.Potion(x))];

    /// <summary>
    /// Neow's Phial Holster (a third slot and two potions), then the first monster fight, in which
    /// the history drank the SECOND potion and took the fight's potion. Also the belt the game
    /// ends the floor with. Null when the seed's Neow has no Phial Holster or the fight drops no
    /// potion.
    /// </summary>
    static (JsonObject Record, string[] Belt)? PotionRecord(string seed)
    {
        var e = new RunEngine();
        e.Reset(seed, Asc);
        var s = e.State;
        var options = s.NeowOptions.Select(Names.Relic).ToList();
        int choice = options.IndexOf("PHIAL_HOLSTER");
        if (choice < 0)
        {
            return null;
        }

        e.Step(choice, -1, out _, out _, out _);
        if (s.Phase != RunPhase.Ancient || !s.NeowAwaitingProceed)
        {
            return null;
        }

        var held = s.PotionSlots.Where(p => p != 0).Select(Names.Potion).ToList();
        e.Step(0, -1, out _, out _, out _); // Neow's Proceed
        var neow = Stats(s, 0);
        neow["ancient_choice"] = new JsonArray(options.Select((o, i) => (JsonNode)new JsonObject
        {
            ["TextKey"] = o,
            ["title"] = new JsonObject { ["key"] = o + ".title", ["table"] = "relics" },
            ["was_chosen"] = i == choice,
        }).ToArray());
        neow["relic_choices"] = new JsonArray(new JsonObject { ["choice"] = "RELIC.PHIAL_HOLSTER", ["was_picked"] = true });
        neow["potion_choices"] = new JsonArray(held.Select(p => (JsonNode)new JsonObject { ["choice"] = "POTION." + p, ["was_picked"] = true }).ToArray());
        var floor1 = new JsonObject
        {
            ["map_point_type"] = "ancient",
            ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = "EVENT.NEOW", ["room_type"] = "event" }),
            ["player_stats"] = new JsonArray(neow),
        };

        int node = Array.IndexOf(s.MapNodeTypes, RunConstants.NodeNormal);
        e.Step(node, -1, out _, out _, out _);
        var fight = s.ActiveCombat!;
        string encounter = "ENCOUNTER." + Names.GameEntry(Names.EncounterName(fight.EncounterId));
        var monsters = fight.Enemies.Select(x => (JsonNode)("MONSTER." + Names.Enemy(x.DefId))).ToArray();
        // The game's fight: the second potion drunk, gone from the belt when the fight ends.
        int second = Array.FindLastIndex(fight.PotionSlots, p => p != 0);
        string drunk = Names.Potion(fight.PotionSlots[second]);
        fight.PotionSlots[second] = 0;
        e.DebugResolveCombat();
        if (s.RewardPotion == 0)
        {
            return null;
        }

        string reward = Names.Potion(s.RewardPotion);
        int goldBefore = s.Gold;
        if (s.RewardGold != 0)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        int gold = s.Gold - goldBefore;
        e.Step(0, -1, out _, out _, out _); // the potion
        var offer = s.RewardCards.Where(c => c != 0).Select(c => "CARD." + Names.Card(c)).ToList();
        e.Step(s.RelicReward != 0 ? 1 : 0, -1, out _, out _, out _);
        e.Step(0, -1, out _, out _, out _);
        e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);
        var stats = Stats(s, gold);
        stats["card_choices"] = new JsonArray(offer.Select((c, i) => (JsonNode)new JsonObject
        {
            ["card"] = new JsonObject { ["id"] = c },
            ["was_picked"] = i == 0,
        }).ToArray());
        stats["cards_gained"] = new JsonArray(new JsonObject { ["id"] = offer[0] });
        stats["potion_choices"] = new JsonArray(new JsonObject { ["choice"] = "POTION." + reward, ["was_picked"] = true });
        stats["potion_used"] = new JsonArray("POTION." + drunk);
        var floor2 = new JsonObject
        {
            ["map_point_type"] = "monster",
            ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = encounter, ["monster_ids"] = new JsonArray(monsters), ["room_type"] = "monster" }),
            ["player_stats"] = new JsonArray(stats),
        };

        var rec = new JsonObject
        {
            ["seed"] = s.StringSeed,
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
        return (rec, Belt(s));
    }

    [Fact]
    public void A_potion_drunk_in_a_fight_frees_its_slot_before_the_rewards()
    {
        // The resolve copies the fight's own belt back, so the drunk potion has to leave THAT belt:
        // dropped from the run's before the resolve, it came back, the fight's potion took the
        // third slot, and the second was empty after the floor where the game holds the new one.
        var (record, belt) = PotionRecord("CODEXP0016") ?? throw new InvalidOperationException("no Phial Holster with a potion drop on this seed");
        Assert.Equal("-", belt[2]);
        using var doc = JsonDocument.Parse(record.ToJsonString());
        var walker = new Walker(new EventOptionIndex(null));
        var result = walker.Walk(doc.RootElement);
        Assert.Equal("complete", (string?)result["status"]);
        Assert.Equal(belt, Belt(walker.State));
    }

    [Fact]
    public void A_potion_a_step_traded_is_not_taken_off_the_belt_again()
    {
        // The Future of Potions traded one of two Strength Potions; the history lists it as
        // discarded. The floor's check used to take the whole list off the emulator's belt,
        // which then held none against the history's one.
        var left = new List<string> { "STRENGTH_POTION", "FIRE_POTION" };
        Walker.TakeOff(left, ["STRENGTH_POTION", "STRENGTH_POTION", "SWIFT_POTION"], ["STRENGTH_POTION", "SWIFT_POTION"]);
        Assert.Equal(["FIRE_POTION"], left);
        // A potion the history does not list (the emulator traded another one) stays a difference.
        Walker.TakeOff(left, ["SWIFT_POTION"], []);
        Assert.Equal(["FIRE_POTION"], left);
    }

    [Fact]
    public void At_the_merchant_a_foul_potion_is_thrown_before_a_brew_is_drunk()
    {
        var e = new RunEngine();
        e.Reset(Seed, Asc);
        var s = e.State;
        RunRewardGenerator.EnterShop(s);
        s.MaxPotionSlots = 3;
        Array.Clear(s.PotionSlots);
        s.PotionSlots[0] = RunNonCombatEffects.NamedPotion("EntropicBrew");
        s.PotionSlots[1] = RunNonCombatEffects.NamedPotion("FoulPotion");
        int gold = s.Gold;
        var actions = new List<int>();
        var (used, brewed) = Walker.UseAtTheMerchant(s, ["ENTROPIC_BREW", "FIRE_POTION", "FOUL_POTION"], a =>
        {
            actions.Add(a);
            Assert.Equal(0, e.Step(a, -1, out _, out _, out _));
        });

        // The throw first, so its gold is there for the buys; a potion not held is not used.
        Assert.Equal([RunConstants.UsePotionAction + 1, RunConstants.UsePotionAction + 0], actions);
        Assert.Equal(["FOUL_POTION", "ENTROPIC_BREW"], used);
        Assert.Equal(gold + 100, s.Gold);
        Assert.Equal(RunPhase.Shop, s.Phase);
        // The brew filled every open slot, its own and the thrown potion's included, in slot order.
        Assert.Equal(3, brewed.Count);
        Assert.Equal(Belt(s), brewed);
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
    [Fact]
    public void Each_result_names_its_input_line_whatever_the_shard()
    {
        // refset_score's gate pairs two codex walks run by run by this index: a walk split into other
        // shards must give every run the same index, and a broken run must keep its own.
        string dir = Directory.CreateTempSubdirectory("codex-index").FullName;
        try
        {
            var good = Record();
            var bad = Record();
            // The first fight's card reward made another card: the walk stops there (WalkerBrokenCaseTests' card_reward).
            bad["map_point_history"]![0]![1]!["player_stats"]![0]!["card_choices"]![1]!["card"]!["id"] = "CARD.ZAP";
            foreach (var r in new[] { good, bad })
            {
                // The deck and relics the codex holds at the run's end, which the driver checks for modded content.
                r["players"]![0]!["deck"] = new JsonArray(new JsonObject { ["id"] = "CARD.STRIKE_IRONCLAD" });
                r["players"]![0]!["relics"] = new JsonArray(new JsonObject { ["id"] = "RELIC.BURNING_BLOOD" });
            }

            string input = Path.Combine(dir, "runs.jsonl");
            File.WriteAllLines(input, [good.ToJsonString(), bad.ToJsonString(), good.ToJsonString()]);
            Dictionary<int, string> Walk(params string[] extra)
            {
                var got = new Dictionary<int, string>();
                for (int i = 0; i < (extra.Length == 0 ? 1 : 2); i++)
                {
                    string output = Path.Combine(dir, $"out{i}.jsonl");
                    string[] shard = extra.Length == 0 ? [] : ["--shard", $"{i}/2"];
                    Assert.Equal(0, EmuDump.Codex.Driver.Run([input, output, .. shard]));
                    foreach (var line in File.ReadAllLines(output))
                    {
                        var r = JsonNode.Parse(line)!;
                        got.Add((int)r["index"]!, (string)r["status"]!);
                    }
                }

                return got;
            }

            var whole = Walk();
            Assert.Equal(new Dictionary<int, string> { [0] = "complete", [1] = "mismatch", [2] = "complete" }, whole);
            Assert.Equal(whole, Walk("sharded"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
