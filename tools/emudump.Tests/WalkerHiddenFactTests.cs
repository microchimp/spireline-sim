// The codex walker against facts of a resolved fight that the history does not record. A Gremlin
// Merc that stole before it died leaves a Heist gold row (one more Rewards draw behind the room's
// own rewards); claimed in full, the history's gold_stolen and stolen_loot are 0, as if it never
// stole. The walker must not guess the fact from the fight's turn count: it settles it by which
// way the next rewards fit, and leaves it open (later Rewards stops FIGHT-DEPENDENT) when neither
// or both do. The records are the emulator's own runs, written in the game's run-history shape.
using System.Text.Json.Nodes;
using EmuDump.Codex;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

public class WalkerHiddenFactTests
{
    const int Asc = 10;

    /// <summary>A record of Neow then monster fights, and how its Merc fight went.</summary>
    internal sealed record MercRun(JsonObject Record, int MercFloor, bool LastFloorHasPotion, string LastOffer);

    /// <summary>
    /// Neow, then <paramref name="fights" /> monster floors along the path the walker takes (the
    /// first monster node from which the rest of the record's floors can be walked), each fight
    /// resolved, its gold and first card taken. The Gremlin Merc fight among them stole up to 20
    /// gold before it died when <paramref name="stole" />, and its Heist row is claimed back (the
    /// history then shows no theft); its room says it took <paramref name="turns" /> turns. Null
    /// when the seed's run has no Merc fight before its last floor.
    /// </summary>
    internal static MercRun? Run(string seed, int fights, bool stole, int turns)
    {
        var e = new RunEngine();
        e.Reset(seed, Asc, PlayerCharacter.Ironclad);
        var s = e.State;
        if (WalkerTests.NeowFloor(e) is not { } neow)
        {
            return null;
        }

        var floors = new List<JsonNode> { neow };
        int mercFloor = -1;
        bool lastPotion = false;
        string lastOffer = "";
        for (int f = 0; f < fights; f++)
        {
            int node = -1;
            for (int i = 0; i < s.MapNodeTypes.Length && node < 0; i++)
            {
                if (s.MapNodeTypes[i] == RunConstants.NodeNormal && s.MapOptionCoords[i] is { } c && MonsterPath(s, c, fights - f - 1))
                {
                    node = i;
                }
            }

            if (node < 0)
            {
                return null;
            }

            e.Step(node, -1, out _, out _, out _);
            if (s.Phase != RunPhase.Combat || s.ActiveCombat is not { } fight)
            {
                return null;
            }

            bool merc = fight.EncounterId == RunConstants.GremlinMercEncounterId;
            if (merc && (f == fights - 1 || mercFloor > 0))
            {
                return null;
            }

            string encounter = "ENCOUNTER." + Names.GameEntry(Names.EncounterName(fight.EncounterId));
            var monsters = fight.Enemies.Select(x => "MONSTER." + Names.Enemy(x.DefId)).ToList();
            int goldBefore = s.Gold;
            int heist = 0;
            if (merc)
            {
                mercFloor = floors.Count + 1;
                // The Merc's death spawns its gremlins, which the history lists too.
                monsters.AddRange(["MONSTER.SNEAKY_GREMLIN", "MONSTER.FAT_GREMLIN"]);
                heist = stole ? Math.Min(20, s.Gold) : 0;
                fight.PlayerGold = Math.Max(0, fight.PlayerGold - heist);
            }

            e.DebugResolveCombat(stolenBackGold: heist);
            var offer = s.RewardCards.Where(c => c != 0).Select(c => "CARD." + Names.Card(c)).ToList();
            string? potion = s.RewardPotion != 0 ? "POTION." + Names.Potion(s.RewardPotion) : null;
            if (offer.Count == 0 || s.RelicReward != 0)
            {
                return null;
            }

            // Every gold row, the Heist's too: the history's gold_gained is the room's own (a
            // stolen-back claim goes to gold_stolen), and the theft and claim net out.
            for (int guard = 0; s.RewardGold != 0 && guard < 4; guard++)
            {
                e.Step(0, -1, out _, out _, out _);
            }

            int gold = s.Gold - goldBefore;
            e.Step(potion is null ? 0 : 1, -1, out _, out _, out _);
            e.Step(0, -1, out _, out _, out _);
            e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);
            if (s.Phase != RunPhase.Map)
            {
                return null;
            }

            var stats = new JsonObject
            {
                ["current_gold"] = s.Gold,
                ["current_hp"] = s.PlayerHp,
                ["max_hp"] = s.PlayerMaxHp,
                ["gold_gained"] = gold,
                ["gold_spent"] = 0,
                ["gold_stolen"] = 0,
                ["stolen_loot"] = 0,
                ["card_choices"] = new JsonArray(offer.Select((c, i) => (JsonNode)new JsonObject
                {
                    ["card"] = new JsonObject { ["id"] = c },
                    ["was_picked"] = i == 0,
                }).ToArray()),
                ["cards_gained"] = new JsonArray(new JsonObject { ["id"] = offer[0] }),
            };
            if (potion is not null)
            {
                stats["potion_choices"] = new JsonArray(new JsonObject { ["choice"] = potion, ["was_picked"] = false });
            }

            floors.Add(new JsonObject
            {
                ["map_point_type"] = "monster",
                ["rooms"] = new JsonArray(new JsonObject
                {
                    ["model_id"] = encounter,
                    ["monster_ids"] = new JsonArray(monsters.Select(m => (JsonNode)m).ToArray()),
                    ["room_type"] = "monster",
                    ["turns_taken"] = merc ? turns : 3,
                }),
                ["player_stats"] = new JsonArray(stats),
            });
            lastPotion = potion is not null;
            lastOffer = string.Join(",", offer);
        }

        return mercFloor < 0 ? null : new MercRun(WalkerTests.RunRecord(s, seed, [.. floors]), mercFloor, lastPotion, lastOffer);
    }

    static bool MonsterPath(RunState s, (int Col, int Row) node, int rest) =>
        rest == 0 || (s.MapNodes.TryGetValue(node, out var n) && n.Children.Any(c =>
            s.MapNodes.TryGetValue(c, out var cn) && cn.NodeType == RunConstants.NodeNormal && MonsterPath(s, c, rest - 1)));

    // Neow, four fights with the Merc's fourth (floor 5), and one more: neither world drops a
    // potion on the last floor, and the two worlds' last card offers differ.
    const string Seed = "HIDDEN0003";
    const int Fights = 5;

    static MercRun World(bool stole, int turns)
    {
        var run = Run(Seed, Fights, stole, turns);
        Assert.NotNull(run);
        Assert.Equal(5, run!.MercFloor);
        Assert.False(run.LastFloorHasPotion);
        return run;
    }

    static JsonNode Fact(JsonObject result, string name) =>
        Assert.Single(result["hidden_facts"]!.AsArray(), f => (string?)f!["fact"] == name)!;

    static JsonObject MercStats(JsonObject record, int floor) =>
        record["map_point_history"]![0]![floor - 1]!["player_stats"]![0]!.AsObject();

    [Fact]
    public void The_two_worlds_differ_on_the_next_floor()
    {
        // Otherwise the tests below would not tell the walker's two ways apart.
        Assert.NotEqual(World(stole: false, turns: 3).LastOffer, World(stole: true, turns: 3).LastOffer);
    }

    [Fact]
    public void A_long_Merc_fight_that_stole_nothing_is_not_taken_for_a_Heist()
    {
        // The guess this replaces took a fight of three turns or more for a Heist: one Rewards
        // draw too many, and the next fight's offer mismatched as an EMULATOR stop.
        // The default way (no Heist) fits the whole history, so nothing needs settling.
        var r = WalkerTests.Walk(World(stole: false, turns: 3).Record);
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Null(r["hidden_facts"]);
    }

    [Fact]
    public void A_short_Merc_fight_that_stole_is_settled_by_the_next_rewards()
    {
        var r = WalkerTests.Walk(World(stole: true, turns: 2).Record);
        Assert.Equal("complete", (string?)r["status"]);
        var fact = Fact(r, "heist");
        Assert.Equal(5, (int)fact["floor"]!);
        Assert.Equal("heist", (string?)fact["fact"]);
        Assert.Equal("yes", (string?)fact["settled"]);
    }

    [Fact]
    public void When_neither_way_fits_the_next_rewards_the_stop_is_fight_dependent()
    {
        // The next fight's offer fits neither world: both ways stop on it, so the Heist stays
        // open and the stop (a Rewards draw after it) is the fight's, not the emulator's.
        var run = World(stole: false, turns: 3);
        var last = MercStats(run.Record, 6);
        last["card_choices"]![1]!["card"]!["id"] = "CARD.ZAP";
        var r = WalkerTests.Walk(run.Record);
        Assert.Equal("mismatch", (string?)r["status"]);
        var stop = r["stop"]!;
        Assert.Equal(6, (int)stop["floor"]!);
        Assert.Equal("card_reward", (string?)stop["component"]);
        Assert.Equal(Walker.FightDependent, (string?)stop["class"]);
        Assert.Contains("Gremlin Merc", (string?)stop["tainted_by"]);
        Assert.Equal("open", (string?)Fact(r, "heist")["settled"]);
    }

    [Fact]
    public void An_open_Heist_does_not_taint_its_own_floors_rewards()
    {
        // The Heist row is populated behind the room's own gold, potion and cards: an offer of the
        // Merc's own floor that fits no way is still the emulator's.
        var run = World(stole: false, turns: 3);
        MercStats(run.Record, 5)["card_choices"]![1]!["card"]!["id"] = "CARD.ZAP";
        var r = WalkerTests.Walk(run.Record);
        var stop = r["stop"]!;
        Assert.Equal(5, (int)stop["floor"]!);
        Assert.Equal("card_reward", (string?)stop["component"]);
        Assert.Equal(Walker.Emulator, (string?)stop["class"]);
    }

    [Fact]
    public void A_one_turn_Merc_fight_never_stole()
    {
        // The Merc's first move comes after the player's first turn: evidence, no second walk.
        var r = WalkerTests.Walk(World(stole: false, turns: 1).Record);
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Null(r["hidden_facts"]);
    }

    [Fact]
    public void The_joiners_a_fight_created_are_a_hidden_fact_drawn_on_Niche()
    {
        // The Merc's death creates its two gremlins, each rolling its HP on Niche; the history
        // lists them, which is the least the count can be. When the walk stops it settles the count
        // like any hidden fact; with nothing later reading Niche every count fits and it stays open.
        var run = World(stole: true, turns: 2);
        var r = WalkerTests.Walk(run.Record);
        Assert.Equal("complete", (string?)r["status"]);
        var joined = Fact(r, "joined");
        Assert.Equal(5, (int)joined["floor"]!);
        Assert.Equal("open", (string?)joined["settled"]); // no later check reads Niche here: every count fits
    }

    [Fact]
    public void A_fact_left_open_is_settled_again_once_the_facts_after_it_are_settled()
    {
        // The joiners count is met before the Heist and ties while the Heist is still at its
        // default; it is visited again after the Heist is settled, and the walk completes either way.
        var r = WalkerTests.Walk(World(stole: true, turns: 2).Record);
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Equal("yes", (string?)Fact(r, "heist")["settled"]);
        Assert.Equal("open", (string?)Fact(r, "joined")["settled"]);
        Assert.Equal(5, (int)Fact(r, "joined")["floor"]!);
    }

    static void GiveLamp(RunState s) => s.Relics.Add(new RelicInstance(RelicEffects.LavaLamp));

    /// <summary>
    /// Neow and the first fight of a run that holds Lava Lamp from the start, the fight hurting the
    /// player (damage_taken 6) only through unblockable damage or not: the lamp then upgrades the
    /// offer or not. The history's damage_taken is the same either way.
    /// </summary>
    static (JsonObject Record, bool AnyUpgraded) LampRun(bool unblockableOnly)
    {
        const string seed = "CODEXTEST1";
        var e = new RunEngine();
        e.Reset(seed, Asc, PlayerCharacter.Ironclad);
        var s = e.State;
        GiveLamp(s);
        var neow = WalkerTests.NeowFloor(e)!;
        e.Step(Array.IndexOf(s.MapNodeTypes, RunConstants.NodeNormal), -1, out _, out _, out _);
        var fight = s.ActiveCombat!;
        string encounter = "ENCOUNTER." + Names.GameEntry(Names.EncounterName(fight.EncounterId));
        var monsters = fight.Enemies.Select(x => (JsonNode)("MONSTER." + Names.Enemy(x.DefId))).ToArray();
        s.TookUnblockedDamageThisCombat = !unblockableOnly;
        e.DebugResolveCombat();
        var offer = Enumerable.Range(0, s.RewardCards.Length).Where(i => s.RewardCards[i] != 0)
            .Select(i => (Id: "CARD." + Names.Card(s.RewardCards[i]), Up: s.RewardUpgraded[i])).ToList();
        bool potion = s.RewardPotion != 0;
        string? potionId = potion ? "POTION." + Names.Potion(s.RewardPotion) : null;
        int goldBefore = s.Gold;
        if (s.RewardGold != 0)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        int gold = s.Gold - goldBefore;
        e.Step(potion ? 1 : 0, -1, out _, out _, out _);
        e.Step(0, -1, out _, out _, out _);
        e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);
        JsonObject Card((string Id, bool Up) c) => c.Up
            ? new JsonObject { ["id"] = c.Id, ["current_upgrade_level"] = 1 }
            : new JsonObject { ["id"] = c.Id };
        var stats = new JsonObject
        {
            ["current_gold"] = s.Gold,
            ["current_hp"] = s.PlayerHp,
            ["max_hp"] = s.PlayerMaxHp,
            ["gold_gained"] = gold,
            ["gold_spent"] = 0,
            ["gold_stolen"] = 0,
            ["stolen_loot"] = 0,
            ["damage_taken"] = 6,
            ["card_choices"] = new JsonArray(offer.Select((c, i) => (JsonNode)new JsonObject { ["card"] = Card(c), ["was_picked"] = i == 0 }).ToArray()),
            ["cards_gained"] = new JsonArray(Card(offer[0])),
        };
        if (potionId is not null)
        {
            stats["potion_choices"] = new JsonArray(new JsonObject { ["choice"] = potionId, ["was_picked"] = false });
        }

        var floor = new JsonObject
        {
            ["map_point_type"] = "monster",
            ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = encounter, ["monster_ids"] = new JsonArray(monsters), ["room_type"] = "monster" }),
            ["player_stats"] = new JsonArray(stats),
        };
        return (WalkerTests.RunRecord(s, seed, neow, floor), offer.Any(c => c.Up));
    }

    static JsonObject WalkWithLamp(JsonObject record)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(record.ToJsonString());
        return new Walker(new EventOptionIndex(null)) { AfterReset = GiveLamp }.Walk(doc.RootElement);
    }

    [Fact]
    public void A_hurt_fight_whose_offer_Lava_Lamp_upgraded_is_settled_by_the_offer()
    {
        // damage_taken counts unblockable damage too, so a hurt fight may still have lit the lamp.
        // That used to stop the walk as FIGHT-DEPENDENT; the offer itself says which it was.
        var (record, upgraded) = LampRun(unblockableOnly: true);
        Assert.True(upgraded, "the lamp upgrades nothing in this offer");
        var r = WalkWithLamp(record);
        Assert.Equal("complete", (string?)r["status"]);
        var fact = Assert.Single(r["hidden_facts"]!.AsArray())!;
        Assert.Equal("lava_lamp_unblockable", (string?)fact["fact"]);
        Assert.Equal("yes", (string?)fact["settled"]);
    }

    [Fact]
    public void A_hurt_fight_with_the_lamp_dark_needs_no_second_walk()
    {
        var (record, upgraded) = LampRun(unblockableOnly: false);
        Assert.False(upgraded);
        var r = WalkWithLamp(record);
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Null(r["hidden_facts"]);
    }

    [Fact]
    public void A_Heist_claimed_through_a_gold_bonus_is_read_from_the_history()
    {
        // GainGold takes what the claim paid (after Bowler Hat and the like) off gold_stolen, so a
        // negative gold_stolen is a Heist row that was there.
        var run = World(stole: true, turns: 2);
        MercStats(run.Record, 5)["gold_stolen"] = -5;
        var r = WalkerTests.Walk(run.Record);
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Null(r["hidden_facts"]);
    }

    // --- Thieving Hopper ---------------------------------------------------------------------

    const string HopperSeed = "CODEXTEST1";

    /// <summary>
    /// The run's first fight made a Thieving Hopper's (in the builder and in the walker alike), and
    /// the deck's third Strike Sharp 2, so one Strike can be told from the others.
    /// </summary>
    static void HopperFirst(RunState s)
    {
        s.NormalEncounterSequence = [HopperId, .. s.NormalEncounterSequence.Skip(1)];
        int third = s.Deck.Select((c, i) => (c, i)).Where(t => Names.Card(t.c.DefId) == "STRIKE_IRONCLAD").ElementAt(2).i;
        s.Deck[third] = s.Deck[third] with { Enchantment = Enchantment.Sharp, EnchantAmount = 2 };
    }

    static readonly int HopperId = Enumerable.Range(0, 400).First(i => Names.EncounterName(i) == "ThievingHopper");

    static int SharpStrike(RunState s) => s.Deck.FindIndex(c => c.Enchantment == Enchantment.Sharp);

    static JsonObject HistoryCard(CardInstance c) => c.Enchantment == Enchantment.None
        ? new JsonObject { ["id"] = "CARD." + Names.Card(c.DefId), ["floor_added_to_deck"] = 1 }
        : new JsonObject
        {
            ["id"] = "CARD." + Names.Card(c.DefId),
            ["enchantment"] = new JsonObject { ["id"] = "ENCHANTMENT." + c.Enchantment.ToString().ToUpperInvariant(), ["amount"] = c.EnchantAmount },
            ["floor_added_to_deck"] = 1,
        };

    /// <summary>
    /// Neow, a Thieving Hopper fight that stole the Sharp Strike, and (with <paramref name="next" />)
    /// one more monster fight. The Hopper left with the card (<paramref name="kept" />, its loot
    /// still missing) or died and the card was claimed back; its room had a gold row
    /// (<paramref name="goldRow" />), claimed unless <paramref name="leaveGold" />. Every card
    /// reward is left.
    /// </summary>
    static JsonObject HopperRun(bool kept, bool goldRow, bool leaveGold, bool next)
    {
        var e = new RunEngine();
        e.Reset(HopperSeed, Asc, PlayerCharacter.Ironclad);
        var s = e.State;
        HopperFirst(s);
        var floors = new List<JsonNode> { WalkerTests.NeowFloor(e)! };
        for (int f = 0; f < (next ? 2 : 1); f++)
        {
            int node = -1;
            for (int i = 0; i < s.MapNodeTypes.Length && node < 0; i++)
            {
                if (s.MapNodeTypes[i] == RunConstants.NodeNormal && s.MapOptionCoords[i] is { } c && MonsterPath(s, c, next ? 1 - f : 0))
                {
                    node = i;
                }
            }

            e.Step(node, -1, out _, out _, out _);
            var fight = s.ActiveCombat!;
            bool hopper = f == 0;
            Assert.Equal(hopper, fight.EncounterId == HopperId);
            string encounter = "ENCOUNTER." + Names.GameEntry(Names.EncounterName(fight.EncounterId));
            var monsters = fight.Enemies.Select(x => (JsonNode)("MONSTER." + Names.Enemy(x.DefId))).ToArray();
            int goldBefore = s.Gold;
            CardInstance stolen = default;
            if (hopper)
            {
                stolen = s.Deck[SharpStrike(s)];
                e.DebugResolveCombat(escapedEnemies: goldRow ? 0 : 1, stolenCards: [stolen], stolenBackCards: kept ? null : [stolen]);
            }
            else
            {
                e.DebugResolveCombat();
            }

            var offer = s.RewardCards.Where(c => c != 0).Select(c => "CARD." + Names.Card(c)).ToList();
            string? potion = s.RewardPotion != 0 ? "POTION." + Names.Potion(s.RewardPotion) : null;
            var gained = new JsonArray();
            if (s.RewardGold != 0 && !(hopper && leaveGold))
            {
                e.Step(0, -1, out _, out _, out _);
            }

            int gold = s.Gold - goldBefore;
            int index = (s.RewardGold != 0 ? 1 : 0) + (potion is null ? 0 : 1);
            if (s.RewardSpecialCards.Count > 0)
            {
                // The returned card's row sits after the card items; claimed before the card reward.
                e.Step(index + RunRewardGenerator.CardItemCount(s), -1, out _, out _, out _);
                gained.Add(HistoryCard(stolen));
            }

            e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);
            Assert.Equal(RunPhase.Map, s.Phase);
            var stats = new JsonObject
            {
                ["current_gold"] = s.Gold,
                ["current_hp"] = s.PlayerHp,
                ["max_hp"] = s.PlayerMaxHp,
                ["gold_gained"] = gold,
                ["gold_spent"] = 0,
                ["gold_stolen"] = 0,
                ["stolen_loot"] = hopper && kept ? 1 : 0,
                ["card_choices"] = new JsonArray(offer.Select(c => (JsonNode)new JsonObject { ["card"] = new JsonObject { ["id"] = c }, ["was_picked"] = false }).ToArray()),
                ["cards_gained"] = gained,
            };
            if (hopper)
            {
                stats["cards_removed"] = new JsonArray(HistoryCard(stolen));
            }

            if (potion is not null)
            {
                stats["potion_choices"] = new JsonArray(new JsonObject { ["choice"] = potion, ["was_picked"] = false });
            }

            floors.Add(new JsonObject
            {
                ["map_point_type"] = "monster",
                ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = encounter, ["monster_ids"] = new JsonArray(monsters), ["room_type"] = "monster", ["turns_taken"] = 5 }),
                ["player_stats"] = new JsonArray(stats),
            });
        }

        return WalkerTests.RunRecord(s, HopperSeed, [.. floors]);
    }

    static (JsonObject Result, RunState State) WalkHopper(JsonObject record)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(record.ToJsonString());
        var walker = new Walker(new EventOptionIndex(null)) { AfterReset = HopperFirst };
        return (walker.Walk(doc.RootElement), walker.State);
    }

    [Fact]
    public void The_Hopper_took_the_copy_the_history_names()
    {
        // cards_removed names the Sharp Strike: that copy leaves its place and comes back at the
        // deck's bottom, not the first Strike of the name (Fresnel Lens, Slippery Bridge and the
        // enchant events then read the deck as the game had it).
        var (r, s) = WalkHopper(HopperRun(kept: false, goldRow: true, leaveGold: false, next: false));
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Equal(s.Deck.Count - 1, SharpStrike(s));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_Hopper_room_that_kept_its_gold_row_with_the_loot_missing_is_settled_by_the_rewards(bool leaveGold)
    {
        // The loot still missing says the Hopper left with the card, and an escape leaves no gold
        // row; the Codex has such floors with the row, paid or left, and others whose purse came
        // from elsewhere, so the purse does not settle it: the next rewards do.
        var (r, _) = WalkHopper(HopperRun(kept: true, goldRow: true, leaveGold: leaveGold, next: true));
        Assert.Equal("complete", (string?)r["status"]);
        var fact = Assert.Single(r["hidden_facts"]!.AsArray())!;
        Assert.Equal("hopper_gold_row", (string?)fact["fact"]);
        Assert.Equal("yes", (string?)fact["settled"]);
    }

    [Fact]
    public void An_escaped_Hopper_leaves_no_gold_row()
    {
        var (r, _) = WalkHopper(HopperRun(kept: true, goldRow: false, leaveGold: false, next: true));
        Assert.Equal("complete", (string?)r["status"]);
        Assert.Null(r["hidden_facts"]);
    }
}
