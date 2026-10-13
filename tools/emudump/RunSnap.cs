// The emulator's run state as a run snapshot: the same canonical fields that
// tools/fidelity/run_snapshot.py reads out of the game's own save (RunManager.ToSave), in game ids.
// Read-only: it never changes the run.
using System.Reflection;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;

static class RunSnap
{
    internal static string Snake(string s) =>
        System.Text.RegularExpressions.Regex.Replace(s, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    internal static string CardEntry(int id) => id == 0 ? "" : Cards.Get(Math.Abs(id)).Entry;
    internal static string RelicEntry(int id) => id == 0 ? "" : Relics.Get(id).Entry;
    internal static string PotionEntry(int id) => id == 0 ? "" : Snake(Potions.Get(id).Name);

    /// <summary>The emulator's encounter enum name; run_snapshot.py normalises both sides.</summary>
    internal static string EncounterEntry(int id) => Snake(((CombatFactory.ActOneEncounter)id).ToString());

    static Dictionary<int, string>? _events;

    /// <summary>Event id to game entry, from RunConstants' Event* constants (EventSunkenTreasury -> SUNKEN_TREASURY).</summary>
    internal static string EventEntry(int id)
    {
        if (_events == null)
        {
            _events = new Dictionary<int, string>();
            string[] notEvents = ["EventSkipAction", "EventResultPending", "EventRandomOfferSize"];
            foreach (var f in typeof(RunConstants).GetFields(BindingFlags.Public | BindingFlags.Static)
                         .Where(f => f.IsLiteral && f.FieldType == typeof(int) && f.Name.StartsWith("Event")
                                     && f.Name.Length > 5 && char.IsUpper(f.Name[5]) && !notEvents.Contains(f.Name)))
            {
                int v = (int)f.GetRawConstantValue()!;
                if (v > 0) _events.TryAdd(v, Snake(f.Name[5..]));
            }
        }

        return _events.TryGetValue(id, out var n) ? n : $"EVENT_{id}";
    }

    static JsonObject Counters(object set) => new(set.GetType().GetProperties()
        .Where(p => p.PropertyType == typeof(GameRng))
        .Select(p => KeyValuePair.Create(Snake(p.Name).ToLowerInvariant(), (JsonNode?)((GameRng)p.GetValue(set)!).CallCount)));

    /// <summary>
    /// The run streams' counters. A fight draws on its own copies of the combat streams
    /// (CombatState's CountingRandoms) and syncs them back when it ends, so while one is open
    /// those copies are where the game's run streams stand.
    /// </summary>
    static JsonObject RunCounters(RunState s)
    {
        var o = Counters(s.Rng);
        if (s.Phase == RunPhase.Combat && s.ActiveCombat is { } c)
        {
            void Set(string k, CountingRandom? r) { if (r != null) o[k] = r.CallCount; }
            Set("shuffle", c.ShuffleRng); Set("combat_targets", c.TargetRng); Set("monster_ai", c.AiRng as CountingRandom);
            Set("combat_card_selection", c.CardSelectionRng); Set("combat_card_generation", c.CardGenerationRng);
            Set("combat_energy_costs", c.EnergyCostRng); Set("combat_potion_generation", c.PotionGenerationRng);
            Set("combat_orbs", c.OrbGenerationRng); Set("niche", c.NicheHpRng);
        }

        return o;
    }

    static JsonArray Arr<T>(IEnumerable<T> xs, Func<T, JsonNode?> f) => new(xs.Select(f).ToArray());

    static JsonObject Bag(RelicGrabBag bag) => new(bag.Lists
        .Select(l => KeyValuePair.Create(l.Rarity.ToString().ToLowerInvariant(), (JsonNode?)Arr(l.Relics, r => RelicEntry(r)))));

    static readonly string[] NodeNames = ["Unassigned", "Monster", "Elite", "RestSite", "Shop", "Treasure", "Boss", "Unknown", "Ancient"];

    /// <summary>The act's map in the driver's Map shape: every point with its type and children.</summary>
    public static JsonArray Map(RunState s) => new(s.MapNodes.Values.OrderBy(n => n.Row).ThenBy(n => n.Col)
        .Select(n => (JsonNode)new JsonObject
        {
            ["col"] = n.Col, ["row"] = n.Row, ["type"] = NodeNames[n.NodeType],
            ["children"] = new JsonArray(n.Children.OrderBy(c => c.Row).ThenBy(c => c.Col)
                .Select(c => (JsonNode)new JsonArray(c.Col, c.Row)).ToArray()),
        }).ToArray());

    public static JsonObject Export(RunEngine e)
    {
        var s = e.State;
        var o = new JsonObject
        {
            ["phase"] = s.Phase.ToString(),
            ["act"] = s.CurrentActIndex,
            ["floor"] = s.Floor,
            ["coord"] = new JsonArray(s.CurrentMapCoord.Col, s.CurrentMapCoord.Row),
            // A fight keeps the player's HP on its own state until it ends (Blood Vial's heal
            // at combat start lands there), as it does the combat streams.
            ["hp"] = s.Phase == RunPhase.Combat && s.ActiveCombat is { } fight ? fight.PlayerHp : s.PlayerHp,
            ["max_hp"] = s.Phase == RunPhase.Combat && s.ActiveCombat is { } fight2 ? fight2.PlayerMaxHp : s.PlayerMaxHp,
            ["gold"] = s.Gold,
            ["potion_slots"] = s.MaxPotionSlots,
            ["potions"] = Arr(s.UsablePotionSlots(), p => PotionEntry(p)),
            ["deck"] = Arr(s.Deck, c => new JsonArray(CardEntry(c.DefId), c.Upgraded ? 1 : 0,
                c.Enchantment == Enchantment.None ? null : Snake(c.Enchantment.ToString()),
                c.Enchantment == Enchantment.None ? 0 : c.EnchantAmount)),
            ["relics"] = Arr(s.Relics, r => RelicEntry(r.DefId)),
            ["odds"] = new JsonObject
            {
                ["card_rarity"] = Math.Round(s.CardRarityOffset, 6),
                ["potion_reward"] = Math.Round(s.PotionRewardOdds, 6),
                ["unknown_monster"] = Math.Round(s.UnknownMapPointMonsterOdds, 6),
                ["unknown_elite"] = Math.Round(s.UnknownMapPointEliteOdds, 6),
                ["unknown_treasure"] = Math.Round(s.UnknownMapPointTreasureOdds, 6),
                ["unknown_shop"] = Math.Round(s.UnknownMapPointShopOdds, 6),
            },
            ["rng"] = RunCounters(s),
            ["player_rng"] = Counters(s.PlayerRng),
            ["grab_bag"] = Bag(s.RelicBag),
            ["shared_grab_bag"] = Bag(s.SharedRelicBag),
            ["events_seen"] = Arr(s.VisitedEventIds, v => EventEntry(v)),
            ["shop_removals_used"] = s.ShopRemovalsUsed,
        };
        var acts = new JsonArray();
        for (int i = 0; i < s.Acts.Count; i++)
        {
            var a = s.Acts[i];
            var act = new JsonObject
            {
                ["events"] = Arr(a.Events, v => EventEntry(v)),
                ["normal"] = Arr(a.NormalEncounters, v => EncounterEntry(v)),
                ["elite"] = Arr(a.EliteEncounters, v => EncounterEntry(v)),
                ["boss"] = EncounterEntry(a.BossEncounterId),
                ["second_boss"] = a.SecondBossEncounterId == 0 ? null : EncounterEntry(a.SecondBossEncounterId),
                ["ancient"] = a.Ancient,
            };
            if (i == s.CurrentActIndex)
            {
                act["normal_visited"] = s.NormalEncountersVisited;
                act["elite_visited"] = s.EliteEncountersVisited;
                act["events_visited"] = s.EventSequenceIndex;
            }

            acts.Add(act);
        }

        o["acts"] = acts;
        return o;
    }
}
