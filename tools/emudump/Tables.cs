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

    /// <summary>A card's keywords, in the game's names, as printed or after one upgrade.</summary>
    static string[] Keywords(CardDef c, bool upgraded)
    {
        var k = new List<string>();
        if (c.Exhaust && !(upgraded && c.ExhaustRemovedWhenUpgraded)) k.Add("Exhaust");
        if (c.Ethereal && !(upgraded && c.EtherealRemovedWhenUpgraded)) k.Add("Ethereal");
        if (c.Innate || (upgraded && c.InnateWhenUpgraded)) k.Add("Innate");
        if (c.Unplayable) k.Add("Unplayable");
        if (c.Retain || (upgraded && c.RetainWhenUpgraded)) k.Add("Retain");
        if (c.Sly) k.Add("Sly");
        if (c.Eternal) k.Add("Eternal");
        k.Sort(StringComparer.Ordinal);
        return k.ToArray();
    }

    /// <summary>Every power def, by the game's entry name: buff or debuff, stack type, ticks down.</summary>
    static JsonObject PowerTable()
    {
        var o = new JsonObject();
        for (int id = 1; id < 1000; id++)
        {
            try
            {
                var p = Powers.Get(id);
                o[Snake(p.Name)] = new JsonObject
                {
                    ["is_buff"] = p.IsBuff, ["stack_type"] = p.StackType, ["ticks_down"] = p.TicksDown,
                };
            }
            catch (ArgumentException) { }
        }

        return o;
    }

    /// <summary>
    /// Every enemy def: HP band at A0 and the distinct damage moves ("damage x hits") that the
    /// intent chooser yields over move indexes 0..11 and three seeds, at ascension 0. The emulator keeps
    /// monster moves in code, not in a table, so this is read by running the chooser.
    /// </summary>
    static JsonObject MonsterTable()
    {
        var o = new JsonObject();
        foreach (int id in Enumerable.Range(1, 300).Concat(Enumerable.Range(10000, 100)))
        {
            EnemyDef d;
            try { d = Enemies.Get(id); }
            catch (ArgumentException) { continue; }
            var moves = new SortedSet<string>(StringComparer.Ordinal);
            string? err = null;
            try
            {
                for (int idx = 0; idx < 12; idx++)
                {
                    for (int seed = 1; seed <= 3; seed++)
                    {
                        var e = new EnemyState { DefId = id, Hp = d.MinHpBelowToughEnemies > 0 ? d.MinHpBelowToughEnemies : d.MinHp, MoveIndex = idx };
                        e.MaxHp = e.Hp;
                        EnemyAI.ChooseIntents([e], idx + 1, new Random(seed), null, 0);
                        var it = e.CurrentIntent;
                        if ((it.Type == IntentType.Attack || it.CarriesDamage) && it.Magnitude > 0)
                            moves.Add($"{it.Magnitude}x{it.Hits}");
                    }
                }
            }
            catch (Exception ex) { err = ex.GetType().Name; }
            o[Snake(d.Name)] = new JsonObject
            {
                ["min_hp"] = d.MinHpBelowToughEnemies > 0 ? d.MinHpBelowToughEnemies : d.MinHp,
                ["max_hp"] = d.MaxHpBelowToughEnemies > 0 ? d.MaxHpBelowToughEnemies : d.MaxHp,
                ["attacks"] = Arr(moves), ["error"] = err,
            };
        }

        return o;
    }

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
                ["base_damage"] = c.BaseDamage, ["base_block"] = c.BaseBlock,
                ["upgrade_damage"] = c.UpgradeDamage, ["upgrade_block"] = c.UpgradeBlock, ["upgrade_cost"] = c.UpgradeCost,
                ["keywords"] = Arr(Keywords(c, false)), ["keywords_upgraded"] = Arr(Keywords(c, true)),
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
            relics[r.Entry] = new JsonObject
            {
                ["rarity"] = r.Rarity.ToString(), ["upon_pickup"] = r.HasUponPickupEffect,
                ["allowed_in_shops"] = r.IsAllowedInShops,
            };
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
        var allPotions = new JsonObject();
        for (int id = 1; id < 400; id++)
        {
            try
            {
                var p = Potions.Get(id);
                allPotions[PotionEntry(id)] = new JsonObject { ["rarity"] = p.Rarity.ToString(), ["target"] = p.TargetType.ToString() };
            }
            catch (ArgumentException) { }
        }

        o["all_potions"] = allPotions;
        o["powers"] = PowerTable();
        o["monsters"] = MonsterTable();
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
