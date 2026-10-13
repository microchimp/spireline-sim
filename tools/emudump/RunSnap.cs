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

    /// <summary>
    /// A relic's saved property (the game's <c>[SavedProperty]</c>, as its save writes it under
    /// <c>props</c>) in the game's units: read off the emulator's state, and written back by the
    /// replay's anchoring. Kind is the save's list ("ints" or "bools"; a bool is 0 or 1 here).
    /// Absent: the value a save that leaves the property out stands for (SaveIfNotTypeDefault;
    /// run_snapshot.SAVED_IF_NOT_DEFAULT on the Python side), null for an AlwaysSave property.
    /// Only these are compared; any other saved property is reported by run_snapshot.not_compared.
    /// </summary>
    internal sealed record RelicProp(string Relic, string Kind, string Name, Func<RunState, RelicInstance, int> Get,
        Action<RunState, int, int> Set, int? Absent = null);

    static int C(RunState s, RelicInstance r) => r.Counter;
    static void SetC(RunState s, int i, int v) => s.Relics[i] = s.Relics[i] with { Counter = v };
    static bool InFight(RunState s) => s.Phase == RunPhase.Combat && s.ActiveCombat != null;

    static readonly int LizardTailId = RunConstants.RelicLizardTail;

    internal static readonly RelicProp[] RelicProps =
    [
        // RelicInstance.Counter in the game's units: the fight's own copy while one is open (the
        // game has one relic object for the run), else the run's. A fight starts from the run's
        // Counter (CombatFactory.Reset), and only RelicEffects.RunCounterRelics carry theirs back.
        new("PEN_NIB", "ints", "AttacksPlayed", C, SetC),
        new("JOSS_PAPER", "ints", "CardsExhausted", C, SetC, Absent: 0),
        new("NUNCHAKU", "ints", "AttacksPlayed", C, SetC),
        new("TUNING_FORK", "ints", "SkillsPlayed", C, SetC),
        new("HAPPY_FLOWER", "ints", "TurnsSeen", C, SetC),
        new("PENDULUM", "ints", "TurnsSeen", C, SetC),
        new("FAKE_HAPPY_FLOWER", "ints", "TurnsSeen", C, SetC),
        // The game zeroes TurnsSeen as the bonus draw lands (AfterModifyingHandDraw); the
        // emulator holds the turn count until the next tick restarts it at 1.
        new("POLLINOUS_CORE", "ints", "TurnsSeen", (s, r) => r.Counter == Sts2Emulator.Core.Effects.RelicEffects.PollinousCoreTurns ? 0 : r.Counter, SetC),
        new("IRON_CLUB", "ints", "CardsPlayed", C, SetC),
        new("BOOK_OF_FIVE_RINGS", "ints", "CardsAdded", C, SetC),
        new("FISHING_ROD", "ints", "CombatsSeen", C, SetC),
        new("LASTING_CANDY", "ints", "CombatsSeen", C, SetC),
        new("TOY_BOX", "ints", "CombatsSeen", C, SetC),
        new("PUMPKIN_CANDLE", "ints", "KindleCount", C, SetC),
        new("SWORD_OF_STONE", "ints", "ElitesDefeated", C, SetC),
        new("PAELS_WING", "ints", "RewardsSacrificed", C, SetC),
        new("GIRYA", "ints", "TimesLifted", C, SetC),
        new("WINGED_BOOTS", "ints", "TimesUsed", C, SetC),
        new("BONE_TEA", "ints", "CombatsLeft", C, SetC),
        new("TEA_OF_DISCOURTESY", "ints", "CombatsLeft", C, SetC),
        new("EMBER_TEA", "ints", "CombatsLeft", C, SetC),
        // The crucible's Counter counts DOWN from its three uses (StartingRelicCounter).
        new("SILVER_CRUCIBLE", "ints", "TimesUsed", (s, r) => 3 - r.Counter, (s, i, v) => SetC(s, i, 3 - v)),
        new("SILVER_CRUCIBLE", "ints", "TreasureRoomsEntered", (s, _) => s.SilverCrucibleTreasureRoomsEntered,
            (s, _, v) => s.SilverCrucibleTreasureRoomsEntered = v),
        // Armed to 1 on entering a rest site (RelicEffects.ArmTeaSetsAtRestSite), spent by the next fight.
        new("VENERABLE_TEA_SET", "bools", "GainEnergyInNextCombat", (s, r) => r.Counter > 0 ? 1 : 0, SetC),
        new("FAKE_VENERABLE_TEA_SET", "bools", "GainEnergyInNextCombat", (s, r) => r.Counter > 0 ? 1 : 0, SetC),
        new("LAVA_LAMP", "bools", "TookDamageThisCombat", (s, _) => s.TookUnblockedDamageThisCombat ? 1 : 0,
            (s, _, v) => s.TookUnblockedDamageThisCombat = v != 0),
        new("LAVA_ROCK", "bools", "HasTriggered", (s, _) => s.LavaRockTriggered ? 1 : 0, (s, _, v) => s.LavaRockTriggered = v != 0),
        // Spent for the run in RunState.UsedUpRelics once a fight ends; inside the fight it is the
        // fight's copy that says so (RelicEffects.CollectUsedUpRelics).
        new("LIZARD_TAIL", "bools", "WasUsed", (s, r) => s.UsedUpRelics.Contains(LizardTailId) || (InFight(s) && r.Counter > 0) ? 1 : 0,
            (s, _, v) =>
            {
                s.UsedUpRelics.Remove(LizardTailId);
                if (v != 0) s.UsedUpRelics.Add(LizardTailId);
            }),
        // 1: the first card screen after pickup was built (and Glam-enchanted); 2: answered.
        // The game's IsUsed is set as that screen is built (AfterModifyingCardRewardOptions).
        new("SILKEN_TRESS", "bools", "IsUsed", (s, r) => r.Counter >= 1 ? 1 : 0, (s, i, v) => SetC(s, i, v != 0 ? 2 : 0)),
    ];

    /// <summary>The held relics' modelled saved properties: {RELIC: {Name: value}}.</summary>
    static JsonObject RelicPropsOf(RunState s)
    {
        var o = new JsonObject();
        foreach (var p in RelicProps)
        {
            int i = s.Relics.FindIndex(r => RelicEntry(r.DefId) == p.Relic);
            if (i < 0) continue;
            var live = s.Relics[i];
            if (InFight(s) && s.ActiveCombat!.Relics.FindIndex(r => r.DefId == live.DefId) is int k and >= 0)
                live = s.ActiveCombat.Relics[k];
            var props = (o[p.Relic] ??= new JsonObject()).AsObject();
            int v = p.Get(s, live);
            props[p.Name] = p.Kind == "bools" ? JsonValue.Create(v != 0) : JsonValue.Create(v);
        }

        return o;
    }

    /// <summary>
    /// The per-card saved properties the emulator keeps: Mad Science's Tinker Time type and
    /// rider (on the CardInstance), and Guilty's combats served (RunState.GuiltyCombatsServed,
    /// one count per copy, new copies at 0). Both as sorted multisets: the emulator does not
    /// tie a Guilty count to a deck position.
    /// </summary>
    static JsonObject CardPropsOf(RunState s)
    {
        var o = new JsonObject();
        var mad = s.Deck.Where(c => CardEntry(c.DefId) == "MAD_SCIENCE")
            .Select(c => $"{c.TinkerType}/{c.TinkerRider}").OrderBy(x => x, StringComparer.Ordinal).ToList();
        if (mad.Count > 0) o["MAD_SCIENCE"] = Arr(mad, x => x);
        int guilty = s.Deck.Count(c => CardEntry(c.DefId) == "GUILTY");
        if (guilty > 0)
        {
            var served = s.GuiltyCombatsServed.Take(guilty).ToList();
            while (served.Count < guilty) served.Add(0);
            o["GUILTY"] = Arr(served.OrderBy(x => x), x => x);
        }

        return o;
    }

    static int? _startingEnergy;

    /// <summary>
    /// The emulator has no run-level max energy: every fight starts at CombatFactory's
    /// StartingEnergy and relics add to it inside the fight. The game's save keeps a max_energy.
    /// </summary>
    static int StartingEnergy() => _startingEnergy ??= (int)typeof(CombatFactory)
        .GetField("StartingEnergy", BindingFlags.NonPublic | BindingFlags.Static)!.GetRawConstantValue()!;

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
            ["relic_props"] = RelicPropsOf(s),
            // Toy Box's wax relics still held, in relic order (a melted one is gone from Relics).
            ["wax"] = Arr(s.Relics.Where(r => s.WaxRelics.Contains(r.DefId)), r => RelicEntry(r.DefId)),
            ["card_props"] = CardPropsOf(s),
            ["max_energy"] = StartingEnergy(),
            ["ascension"] = s.AscensionLevel,
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
