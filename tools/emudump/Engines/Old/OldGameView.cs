// The old engine's step in the game's own shapes (IEngine.GameView), for tools/fidelity/struct_compare.py:
//
//   snapshot  the referee's per-step snapshot (referee/driver/Combat.cs Snapshot): player HP, block, energy and
//             powers, the four piles, the living enemies with HP, block, powers and intents, the RNG counters, and
//             once a fight is over what the run holds;
//   net       the game's NetFullCombatState (OldNetState.From) with the game's power ids;
//   private   the engine's own block for what the game's schema has no field for (its split buffs, relic counters).
//             No comparer reads it.
//
// Every name is the game's: card, enemy, potion and relic ModelIds (UPPER_SNAKE through data/id_map.json), power ids
// (STRENGTH_POWER), pile and RNG stream names. The translation of the old engine's names was fight_compare.py's
// (emu_view, emu_intent, power, NOT_POWERS, EMU_RENAME) and lives here now, each rule cited where it is applied, so the
// comparer reads both sides with no alias table. What the old engine cannot say in the game's terms is not invented: it
// holds every subclass of the game's TemporaryStrengthPower in one buff, written as TEMPORARY_STRENGTH_POWER, and its
// five intent kinds are the coarse ones; tools/fidelity/allowed_diff.json names those rows.

using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

static class OldGameView
{
    /// <summary>
    /// Old buffs that are no game power (fight_compare.NOT_POWERS, with its reasons): the stun the game keeps as the
    /// monster's move (written as the Stun intent), and counters the game keeps as DynamicVars or internal data of another
    /// power. They go to the private block.
    /// </summary>
    static readonly HashSet<string> NotPowers =
    [
        "STUNNED", "INFERNO_SELF_DAMAGE", "CRIMSON_MANTLE_SELF_DAMAGE", "THE_BOMB_DAMAGE", "PANACHE_ARMED", "CLAW_DAMAGE",
        "SLOW_COUNT", "SKITTISH_SPENT",
    ];

    static Dictionary<string, string>? _rename;

    /// <summary>Old buff key to the game's power key (power_aliases.json emu_rename, as fight_compare.EMU_RENAME read it).</summary>
    static string Rename(string key)
    {
        _rename ??= JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "power_aliases.json")))!["emu_rename"]!
            .AsObject().ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
        return _rename.TryGetValue(key, out var g) ? g : key;
    }

    /// <summary>A power key (UPPER_SNAKE without _POWER, fight_compare.power) of a name with or without the suffix.</summary>
    static string Key(string name)
    {
        string n = OldNetState.Snake(name);
        return n.EndsWith("_POWER") ? n[..^"_POWER".Length] : n;
    }

    /// <summary>(game power id or null when it is no game power, the old key) of an old buff.</summary>
    static (string? Game, string Key) Power(string name)
    {
        string k = Key(name);
        return (NotPowers.Contains(k) ? null : Rename(k) + "_POWER", k);
    }

    /// <summary>Splits buffs into the game's power list and the private rest.</summary>
    static (JsonArray Game, JsonArray Private) Powers(IEnumerable<(string Name, int Amount)> buffs)
    {
        var game = new JsonArray();
        var rest = new JsonArray();
        foreach (var (name, amount) in buffs)
        {
            var (g, k) = Power(name);
            if (g != null) game.Add(new JsonArray(g, amount));
            else rest.Add(new JsonArray(k, amount));
        }

        return (game, rest);
    }

    static IEnumerable<(string, int)> Of(IEnumerable<BuffState> b) => b.Select(x => (x.Id.ToString(), x.Magnitude));

    static JsonArray Cards(IEnumerable<CardInstance> cards) =>
        new(cards.Select(x => (JsonNode)new JsonArray(OldIds.SnakeByCardId(x.DefId), x.Upgraded)).ToArray());

    /// <summary>
    /// The enemy's next move as the game's MonsterModel.NextMove intents ([type] or [type, base damage per hit, hits]).
    /// A stunned enemy's move is the game's Stun (fight_compare.emu_intent: the STUNNED buff). A move whose number is damage
    /// leads with its attack (Core/Enemy.cs Intent.CarriesDamage), then the intent of its kind; the second intent the move
    /// declares follows. The old engine's kinds are five (IntentType), Unknown standing for the game's others.
    /// </summary>
    static JsonArray Intents(EnemyState en)
    {
        if (en.Buffs.Any(b => Key(b.Id.ToString()) == "STUNNED")) return new JsonArray(new JsonArray("Stun"));
        var o = new JsonArray();
        void One(Intent it)
        {
            if (it.Type == IntentType.Attack || it.CarriesDamage) o.Add(new JsonArray("Attack", it.Magnitude, it.Hits));
            if (it.Type != IntentType.Attack) o.Add(new JsonArray(it.Type.ToString()));
        }

        One(en.CurrentIntent);
        if (en.SecondaryIntent is { } s) One(s);
        return o;
    }

    static JsonObject Rng(JsonObject counts) =>
        new(counts.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)new JsonObject { ["counter"] = kv.Value!.DeepClone() })));

    static JsonArray Relics(IEnumerable<int> defIds) => new(defIds.Select(id => (JsonNode)OldSetup.RelicEntry(id)).ToArray());

    /// <summary>The step in the game's shapes; see the file's head.</summary>
    internal static JsonObject View(RunEngine e)
    {
        var priv = new JsonObject();
        return new JsonObject { ["snapshot"] = Snapshot(e, priv), ["net"] = Net(e, priv), ["private"] = priv };
    }

    static JsonObject Snapshot(RunEngine e, JsonObject priv)
    {
        var s = e.State;
        var c = s.ActiveCombat;
        bool inCombat = c != null && s.Phase == RunPhase.Combat;
        if (c == null) return new JsonObject { ["in_combat"] = false };
        if (!inCombat && s.LastPlayerWon)
        {
            // A won fight is over inside its last step (Player.AfterCombatEnd): no block, powers or piles, no roster; what
            // the run holds stays (OldDump.Final writes the same values).
            priv["relic_counters"] = new JsonArray(s.Relics.Select(r => (JsonNode)r.Counter).ToArray());
            return new JsonObject
            {
                ["in_combat"] = false,
                ["player"] = new JsonObject
                {
                    ["hp"] = s.PlayerHp, ["max_hp"] = s.PlayerMaxHp, ["block"] = 0, ["powers"] = new JsonArray(), ["gold"] = s.Gold,
                },
                ["hand"] = new JsonArray(), ["draw"] = new JsonArray(), ["discard"] = new JsonArray(), ["exhaust"] = new JsonArray(),
                ["enemies"] = new JsonArray(),
                ["rng"] = Rng(OldSetup.RunRng(s)),
                ["potions"] = OldSetup.Belt(s.PotionSlots, s.MaxPotionSlots),
                ["relics"] = Relics(s.Relics.Select(r => r.DefId)),
            };
        }

        // In a fight, and a lost one as the engine left it (it plays the enemy phase out; the game stops at the killing hit).
        var (pp, ppriv) = Powers(Of(c.PlayerBuffs));
        var enemies = new JsonArray();
        var epriv = new JsonArray();
        foreach (var en in c.Enemies.Where(x => x.Hp > 0))  // the game's roster holds the living (fight_compare.emu_view)
        {
            var (ep, eprv) = Powers(Of(en.Buffs));
            epriv.Add(eprv);
            enemies.Add(new JsonObject
            {
                ["name"] = OldNetState.Name("enemies", en.DefId), ["hp"] = en.Hp, ["max_hp"] = en.MaxHp, ["block"] = en.Block,
                ["powers"] = ep, ["intents"] = Intents(en),
            });
        }

        priv["player_powers"] = ppriv;
        priv["enemy_powers"] = epriv;
        priv["relic_counters"] = new JsonArray(c.Relics.Select(r => (JsonNode)r.Counter).ToArray());
        int N(CountingRandom? r) => r?.CallCount ?? -1;
        var run = s.Rng;
        var rng = new JsonObject
        {
            ["Shuffle"] = N(c.ShuffleRng), ["CombatTargets"] = N(c.TargetRng), ["MonsterAi"] = N(c.AiRng as CountingRandom),
            ["CombatCardSelection"] = N(c.CardSelectionRng), ["CombatCardGeneration"] = N(c.CardGenerationRng),
            ["CombatEnergyCosts"] = N(c.EnergyCostRng), ["CombatPotionGeneration"] = N(c.PotionGenerationRng),
            ["CombatOrbGeneration"] = N(c.OrbGenerationRng), ["Niche"] = N(c.NicheHpRng),
            ["UpFront"] = run.UpFront.CallCount, ["TreasureRoomRelics"] = run.TreasureRoomRelics.CallCount,
            ["UnknownMapPoint"] = run.UnknownMapPoint.CallCount,
        };
        return new JsonObject
        {
            ["in_combat"] = inCombat,
            ["round"] = c.Turn + 1,  // the game counts rounds from 1
            ["player"] = new JsonObject
            {
                ["hp"] = c.PlayerHp, ["max_hp"] = c.PlayerMaxHp, ["block"] = c.PlayerBlock, ["energy"] = c.Energy, ["powers"] = pp,
                ["gold"] = c.PlayerGold,
            },
            ["hand"] = Cards(c.Hand), ["draw"] = Cards(c.DrawPile), ["discard"] = Cards(c.DiscardPile), ["exhaust"] = Cards(c.ExhaustPile),
            ["enemies"] = enemies,
            ["rng"] = Rng(rng),
            ["potions"] = OldSetup.Belt(c.PotionSlots, c.MaxPotionSlots),
            ["relics"] = Relics(c.Relics.Select(r => r.DefId)),
        };
    }

    /// <summary>OldNetState's NetFullCombatState with the game's power ids; relic counters move to the private block.</summary>
    static JsonObject? Net(RunEngine e, JsonObject priv)
    {
        var n = OldNetState.From(e);
        if (n == null) return null;
        var cpriv = new JsonArray();
        foreach (var cr in n["creatures"]!.AsArray())
        {
            var (g, p) = Powers(cr!["powers"]!.AsArray().Select(x => ((string)x![0]!, (int)x[1]!)).ToList());
            cr["powers"] = g;
            cpriv.Add(p);
        }

        priv["net_creature_powers"] = cpriv;
        foreach (var r in n["players"]![0]!["relics"]!.AsArray()) r!.AsObject().Remove("counter");
        return n;
    }
}
