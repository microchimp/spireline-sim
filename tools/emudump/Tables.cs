// `EmuDump tables`: the emulator's content tables (pool orders, rarities, costs, act encounter
// pools, Neow's option list) as JSON, for tools/fidelity/table_check.py. Read-only.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;

static class Tables
{
    static JsonArray Arr(IEnumerable<string> xs) => new(xs.Select(x => (JsonNode)x).ToArray());

    static string CardEntry(int id) => Cards.Get(Math.Abs(id)).Entry;
    static string RelicEntry(int id) => Relics.Get(id).Entry;

    static string Snake(string s) =>
        string.Concat(s.Select((c, i) => i > 0 && char.IsUpper(c) ? "_" + c : c.ToString())).ToUpperInvariant();

    static JsonArray Enc(ReadOnlySpan<int> ids) =>
        Arr(ids.ToArray().Select(i => ((CombatFactory.ActOneEncounter)i).ToString()));

    /// <summary>The act's encounter list of one kind (weak, normal, elite, boss), in the game's order.</summary>
    internal static ReadOnlySpan<int> EncounterList(int act, string kind) => (act, kind) switch
    {
        (RunConstants.ActOvergrowth, "weak") => RunConstants.OvergrowthWeakEncounters,
        (RunConstants.ActUnderdocks, "weak") => RunConstants.UnderdocksWeakEncounters,
        (RunConstants.ActHive, "weak") => RunConstants.HiveWeakEncounters,
        (RunConstants.ActOvergrowth, "normal") => RunConstants.OvergrowthNormalEncounters,
        (RunConstants.ActUnderdocks, "normal") => RunConstants.UnderdocksNormalEncounters,
        (RunConstants.ActHive, "normal") => RunConstants.HiveNormalEncounters,
        (RunConstants.ActOvergrowth, "elite") => RunConstants.OvergrowthEliteEncounters,
        (RunConstants.ActUnderdocks, "elite") => RunConstants.UnderdocksEliteEncounters,
        (RunConstants.ActHive, "elite") => RunConstants.HiveEliteEncounters,
        (RunConstants.ActOvergrowth, "boss") => RunConstants.OvergrowthBossEncounters,
        (RunConstants.ActUnderdocks, "boss") => RunConstants.UnderdocksBossEncounters,
        (RunConstants.ActHive, "boss") => RunConstants.HiveBossEncounters,
        (_, "weak") => EncounterTags.Pool(act, "Weak"),
        (_, "normal") => EncounterTags.Pool(act, "Normal"),
        (_, "elite") => EncounterTags.Pool(act, "Elite"),
        _ => EncounterTags.Pool(act, "Boss"),
    };

    /// <summary>Act ids by the game's act entry.</summary>
    internal static readonly (string Name, int Act)[] ActIds =
    [
        ("OVERGROWTH", RunConstants.ActOvergrowth), ("UNDERDOCKS", RunConstants.ActUnderdocks),
        ("HIVE", RunConstants.ActHive), ("GLORY", RunConstants.ActGlory),
    ];

    public static JsonObject Dump()
    {
        var o = new JsonObject();
        var cards = new JsonObject();
        foreach (var c in Cards.All)
        {
            cards[c.Entry] = new JsonObject
            {
                ["type"] = c.Type.ToString(), ["rarity"] = c.Rarity.ToString(), ["cost"] = c.Cost,
                ["cost_x"] = c.HasEnergyCostX, ["star_cost"] = c.StarCost, ["target"] = c.Target.ToString(),
                ["upgradable"] = c.Upgradable, ["multiplayer_only"] = c.MultiplayerOnly,
            };
        }

        o["cards"] = cards;
        o["card_pools"] = new JsonObject
        {
            ["IroncladCardPool"] = Arr(CardPools.Ironclad.ToArray().Select(CardEntry)),
            ["SilentCardPool"] = Arr(CardPools.Silent.ToArray().Select(CardEntry)),
            ["RegentCardPool"] = Arr(CardPools.Regent.ToArray().Select(CardEntry)),
            ["NecrobinderCardPool"] = Arr(CardPools.Necrobinder.ToArray().Select(CardEntry)),
            ["DefectCardPool"] = Arr(CardPools.Defect.ToArray().Select(CardEntry)),
            ["ColorlessCardPool"] = Arr(CardPools.Colorless.ToArray().Select(CardEntry)),
            ["CurseCardPool"] = Arr(CardPools.Curse.ToArray().Select(CardEntry)),
            ["StatusCardPool"] = Arr(CardPools.Status.ToArray().Select(CardEntry)),
        };
        var relics = new JsonObject();
        foreach (var r in Relics.All)
        {
            relics[r.Entry] = new JsonObject { ["rarity"] = r.Rarity.ToString() };
        }

        o["relics"] = relics;
        o["relic_pools"] = new JsonObject
        {
            ["IroncladRelicPool"] = Arr(RelicPools.Ironclad.ToArray().Select(RelicEntry)),
            ["SilentRelicPool"] = Arr(RelicPools.Silent.ToArray().Select(RelicEntry)),
            ["RegentRelicPool"] = Arr(RelicPools.Regent.ToArray().Select(RelicEntry)),
            ["NecrobinderRelicPool"] = Arr(RelicPools.Necrobinder.ToArray().Select(RelicEntry)),
            ["DefectRelicPool"] = Arr(RelicPools.Defect.ToArray().Select(RelicEntry)),
            ["SharedRelicPool"] = Arr(RelicPools.Shared.ToArray().Select(RelicEntry)),
        };
        var potions = new JsonObject();
        var seen = new HashSet<int>();
        string PotionEntry(int id) => Snake(Potions.Get(id).Name);
        foreach (int id in PotionPools.Shared.ToArray().Concat(PotionPools.Ironclad.ToArray())
                     .Concat(PotionPools.Silent.ToArray()).Concat(PotionPools.Regent.ToArray())
                     .Concat(PotionPools.Necrobinder.ToArray()).Concat(PotionPools.Defect.ToArray()))
        {
            if (!seen.Add(id)) continue;
            var p = Potions.Get(id);
            potions[PotionEntry(id)] = new JsonObject { ["rarity"] = p.Rarity.ToString(), ["target"] = p.TargetType.ToString() };
        }

        o["potions"] = potions;
        o["potion_pools"] = new JsonObject
        {
            ["IroncladPotionPool"] = Arr(PotionPools.Ironclad.ToArray().Select(PotionEntry)),
            ["SilentPotionPool"] = Arr(PotionPools.Silent.ToArray().Select(PotionEntry)),
            ["RegentPotionPool"] = Arr(PotionPools.Regent.ToArray().Select(PotionEntry)),
            ["NecrobinderPotionPool"] = Arr(PotionPools.Necrobinder.ToArray().Select(PotionEntry)),
            ["DefectPotionPool"] = Arr(PotionPools.Defect.ToArray().Select(PotionEntry)),
            ["SharedPotionPool"] = Arr(PotionPools.Shared.ToArray().Select(PotionEntry)),
        };
        // Encounter ids are the emulator's CombatFactory.ActOneEncounter names.
        var acts = new JsonObject();
        var spec = ActIds;
        foreach (var (name, act) in spec)
        {
            acts[name] = new JsonObject
            {
                ["weak"] = Enc(EncounterList(act, "weak")),
                ["normal"] = Enc(EncounterList(act, "normal")),
                ["elite"] = Enc(EncounterList(act, "elite")),
                ["boss"] = Enc(EncounterList(act, "boss")),
            };
        }

        o["acts"] = acts;
        o["acts_by_index"] = new JsonArray(RunConstants.ActCandidatesByIndex.Select(g =>
            (JsonNode)new JsonArray(g.Select(a => (JsonNode)spec.First(s => s.Act == a).Name).ToArray())).ToArray());
        o["neow_options"] = Arr(RunConstants.NeowCurseOptions.ToArray().Concat(RunConstants.NeowPositiveOptions.ToArray())
            .Concat(RunConstants.NeowPairedOptions.ToArray()).Select(RelicEntry));
        o["neow_groups"] = new JsonObject
        {
            ["curse"] = Arr(RunConstants.NeowCurseOptions.ToArray().Select(RelicEntry)),
            ["positive"] = Arr(RunConstants.NeowPositiveOptions.ToArray().Select(RelicEntry)),
            ["paired"] = Arr(RunConstants.NeowPairedOptions.ToArray().Select(RelicEntry)),
        };
        return o;
    }
}
