// The old engine's combat state (IEngine.GameState) in the shape of the game's NetFullCombatState, as the referee's
// ChecksumTap writes it with "trace": true (referee/driver/ChecksumTap.cs), so that
// tools/fidelity/netstate_diff.py can compare the two field by field.
//
//   EmuDump fight <referee out.json> --netstate   adds step["net"] = [{"ctx": "step", "state": ...}]
//
// Ids are the game's (UPPER_SNAKE, through data/id_map.json; powers as <NAME>_POWER, orbs as
// <NAME>_ORB); the comparer applies fight_compare's power rules on top. Only fields the emulator
// models are written. Not modelled, so absent here and skipped by the comparer:
//   - choice_ids, reward_ids, last_action, last_hook: the game's network sequencing counters;
//   - players[].phase: the emulator has no turn phases between its steps;
//   - card and relic "floor" (FloorAddedToDeck) and "props" ([SavedProperty] values: relic
//     counters, The Scythe / Genetic Algorithm growth, ...); relics carry the emulator's own
//     "counter" instead, which has no common unit with the game's named props;
//   - card affliction amounts (only Bound exists in the emulator, as a flag);
//   - the RNG seeds (counters are compared), and the player's Rewards/Shops counters (the game
//     strips those too).
// The emulator's own hidden state (enemy AI, intents, per-turn counters) has no place in the
// game's state and is not written.

using System.Reflection;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

static class OldNetState
{
    static Dictionary<string, Dictionary<int, string>>? _names;

    /// <summary>Emulator def id to the game's id, per table of data/id_map.json.</summary>
    internal static string Name(string table, int id)
    {
        _names ??= JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!.AsObject()
            .ToDictionary(t => t.Key, t => t.Value!.AsObject().ToDictionary(kv => (int)kv.Value!, kv => Snake(kv.Key)));
        return _names[table].TryGetValue(id, out var n) ? n : $"#{id}";
    }

    internal static string Snake(string name) =>
        Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    /// <summary>The state after the engine's last step, or null outside a fight.</summary>
    public static JsonObject? From(RunEngine e)
    {
        var c = e.State.ActiveCombat;
        if (c == null || e.State.Phase != RunPhase.Combat) return null;
        var creatures = new JsonArray
        {
            new JsonObject
            {
                ["monster"] = null, ["player"] = 1,
                ["hp"] = c.PlayerHp, ["max_hp"] = c.PlayerMaxHp, ["block"] = c.PlayerBlock, ["powers"] = Powers(c.PlayerBuffs),
            },
        };
        // The game drops a dead creature from the combat; the emulator keeps it at 0 HP.
        foreach (var en in c.Enemies.Where(x => x.Hp > 0))
        {
            creatures.Add(new JsonObject
            {
                ["monster"] = Name("enemies", en.DefId), ["player"] = null,
                ["hp"] = en.Hp, ["max_hp"] = en.MaxHp, ["block"] = en.Block, ["powers"] = Powers(en.Buffs),
            });
        }

        var piles = new JsonObject
        {
            ["Hand"] = Cards(c.Hand, c, inHand: true), ["Draw"] = Cards(c.DrawPile, c), ["Discard"] = Cards(c.DiscardPile, c),
            ["Exhaust"] = Cards(c.ExhaustPile, c), ["Play"] = new JsonArray(),
        };
        var player = new JsonObject
        {
            ["id"] = 1, ["character"] = Snake(c.Character.ToString()),
            ["turn"] = c.Turn + 1,  // the game counts player turns from 1
            ["energy"] = c.Energy, ["stars"] = c.Stars, ["max_potions"] = c.MaxPotionSlots, ["gold"] = c.PlayerGold,
            ["piles"] = piles,
            ["potions"] = new JsonArray(c.PotionSlots.Take(c.MaxPotionSlots).Where(p => p > 0)
                .Select(p => (JsonNode)Name("potions", p)).ToArray()),
            ["relics"] = new JsonArray(c.Relics.Select(r => (JsonNode)new JsonObject
            {
                ["id"] = Name("relics", r.DefId), ["counter"] = r.Counter,
            }).ToArray()),
            ["orbs"] = new JsonArray(c.Orbs.Select(o => (JsonNode)new JsonArray(Snake(o.Type.ToString()) + "_ORB", o.PassiveValue, o.EvokeValue)).ToArray()),
            ["rng"] = new JsonObject { ["counters"] = new JsonObject { ["Transformations"] = e.State.PlayerRng.Transformations.CallCount } },
            ["grab_bag"] = GrabBag(e.State.RelicBag),
        };
        var run = e.State.Rng;
        // In a fight the combat streams count (they hand their counts back to the run's at the end).
        int N(CountingRandom? r, GameRng runStream) => r?.CallCount ?? runStream.CallCount;
        var counters = new JsonObject
        {
            ["CombatCardGeneration"] = N(c.CardGenerationRng, run.CombatCardGeneration),
            ["CombatCardSelection"] = N(c.CardSelectionRng, run.CombatCardSelection),
            ["CombatEnergyCosts"] = N(c.EnergyCostRng, run.CombatEnergyCosts),
            ["CombatOrbs"] = N(c.OrbGenerationRng, run.CombatOrbs),
            ["CombatPotionGeneration"] = N(c.PotionGenerationRng, run.CombatPotionGeneration),
            ["CombatTargets"] = N(c.TargetRng, run.CombatTargets),
            ["MonsterAi"] = N(c.AiRng as CountingRandom, run.MonsterAi),
            ["Niche"] = N(c.NicheHpRng, run.Niche),
            ["Shuffle"] = N(c.ShuffleRng, run.Shuffle),
            ["TreasureRoomRelics"] = run.TreasureRoomRelics.CallCount,
            ["UnknownMapPoint"] = run.UnknownMapPoint.CallCount,
            ["UpFront"] = run.UpFront.CallCount,
        };
        return new JsonObject
        {
            ["creatures"] = creatures,
            ["players"] = new JsonArray(player),
            ["rng"] = new JsonObject { ["counters"] = counters },
        };
    }

    static JsonArray Powers(IEnumerable<BuffState> buffs) =>
        new(buffs.Select(b => (JsonNode)new JsonArray(PowerId(b.Id), b.Magnitude)).ToArray());

    /// <summary>A BuffId as a game power id: some BuffIds already end in Power.</summary>
    static string PowerId(BuffId id)
    {
        string s = Snake(id.ToString());
        return s.EndsWith("_POWER") ? s : s + "_POWER";
    }

    static JsonArray Cards(IEnumerable<CardInstance> cards, CombatState c, bool inHand = false) =>
        new(cards.Select(x => (JsonNode)Card(x, c, inHand)).ToArray());

    /// <summary>
    /// One card as the game's CardState: id and upgrade level, the enchantment, the cost when it
    /// is not the printed one (the game's GetWithModifiers(All) against Canonical), and the
    /// keyword set (CardModel.Keywords: printed, upgrade- and enchantment-granted, and the
    /// combat-long grants; the single-turn Retain and Sly grants are flags, not keywords).
    /// The cost is written for cards in hand only: EffectiveCost prices a card about to be
    /// played, and some of the game's cost hooks look at the card's pile (FreeAttackPower
    /// .TryModifyEnergyCostInCombatLate: hand and play pile only).
    /// </summary>
    static JsonObject Card(CardInstance x, CombatState c, bool inHand)
    {
        var def = Sts2Emulator.GeneratedData.Cards.Get(x.DefId);
        var o = new JsonObject { ["id"] = Name("cards", x.DefId), ["upgrades"] = x.Upgraded ? 1 : 0 };
        if (x.Enchantment != Enchantment.None)
        {
            o["enchantment"] = new JsonObject { ["id"] = Snake(x.Enchantment.ToString()), ["amount"] = x.EnchantAmount };
        }

        if (inHand && def.Cost >= 0 && !def.HasEnergyCostX && !def.Unplayable)
        {
            int cost = Sts2Emulator.Core.CombatEngine.EffectiveCost(x, c);
            if (cost != def.Cost) o["cost"] = cost;
        }

        if (x.Bound) o["affliction"] = new JsonArray("BOUND", null);
        bool retain = def.Retain || x.RetainForCombat || (x.Upgraded && def.RetainWhenUpgraded)
            || x.Enchantment is Enchantment.Steady or Enchantment.RoyallyApproved;
        var kw = new (string name, bool on)[]
        {
            ("Exhaust", x.IsExhaust()), ("Ethereal", x.IsEthereal()), ("Innate", x.IsInnate()),
            ("Unplayable", def.Unplayable), ("Retain", retain), ("Sly", def.Sly || x.SlyForCombat), ("Eternal", def.Eternal),
        };
        if (kw.Any(k => k.on)) o["keywords"] = new JsonArray(kw.Where(k => k.on).Select(k => (JsonNode)k.name).ToArray());
        return o;
    }

    static readonly FieldInfo? Deques = typeof(RelicGrabBag).GetField("_deques", BindingFlags.NonPublic | BindingFlags.Instance);

    static JsonObject GrabBag(RelicGrabBag bag)
    {
        var o = new JsonObject();
        if (Deques?.GetValue(bag) is not Dictionary<RelicRarity, List<int>> lists) return o;
        foreach (var kv in lists.OrderBy(kv => kv.Key))
        {
            o[kv.Key.ToString()] = new JsonArray(kv.Value.Select(id => (JsonNode)Name("relics", id)).ToArray());
        }

        return o;
    }
}
