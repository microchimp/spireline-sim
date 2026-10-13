// The emulator's mirror of a setup fight (referee/driver/Setup.cs): the same items, in the same order,
// at the same moments. Map items run in the run's map phase after Neow (Fight.EnterRoom, which
// re-anchoring shares), the fight item walks onto the referee's row-1 coordinate but meets the named
// encounter, and combat items run before the step-0 dump, so a wrong mirror shows at step 0.
// Each item is logged with the run's RNG counters (tools/fidelity/setup_compare.py lines them up
// with the game's). An item the emulator cannot build is logged "unsupported": a mirror gap, not an
// emulator divergence. An item the game refused (ok false) is skipped, so both sides stay alike.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

static class Setup
{
    /// <summary>Whether the fight being replayed records the setup extras (potions in each snapshot).</summary>
    [ThreadStatic] internal static bool Extended;

    internal static bool IsExtended(JsonObject game) =>
        game["setup"] is JsonArray || (bool?)game["use_potions"] == true || game["script"] is JsonArray;

    static JsonArray Items(JsonObject game) => game["setup"] as JsonArray ?? new JsonArray();

    static int FightItem(JsonArray items)
    {
        for (int i = 0; i < items.Count; i++)
        {
            if ((string?)items[i]?["cmd"] == "fight") return i;
        }

        return -1;
    }

    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    /// <summary>Emulator relic id for a game relic entry, or null.</summary>
    internal static int? RelicId(string entry)
    {
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            if (r.Entry == entry) return r.Id;
        }

        return null;
    }

    static Dictionary<string, int>? _potions;
    static Dictionary<int, string>? _potionNames;

    /// <summary>Emulator potion id for a game potion entry (FIRE_POTION), or null; through data/id_map.json.</summary>
    internal static int? PotionId(string entry)
    {
        LoadPotions();
        return _potions!.TryGetValue(entry, out int v) ? v : null;
    }

    static void LoadPotions()
    {
        if (_potions != null) return;
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!["potions"]!.AsObject();
        _potions = map.ToDictionary(kv => Snake(kv.Key), kv => (int)kv.Value!);
        _potionNames = _potions.ToDictionary(kv => kv.Value, kv => kv.Key);
    }

    /// <summary>Game potion entry of an emulator potion id.</summary>
    internal static string PotionEntry(int id)
    {
        LoadPotions();
        return _potionNames!.TryGetValue(id, out var n) ? n : id.ToString();
    }

    /// <summary>The potion belt as the game's snapshot writes it: one entry per slot, null when empty.</summary>
    internal static JsonArray Belt(int[] slots, int max) =>
        new(slots.Take(Math.Min(max, slots.Length)).Select(p => p == 0 ? null : (JsonNode)PotionEntry(p)).ToArray());

    static JsonObject Counters(RunState s)
    {
        var o = new JsonObject();
        foreach (var p in s.Rng.GetType().GetProperties().Where(p => p.PropertyType == typeof(GameRng)))
        {
            o[p.Name == "CombatOrbs" ? "CombatOrbGeneration" : p.Name] = ((GameRng)p.GetValue(s.Rng)!).CallCount;
        }

        var pl = new JsonObject();
        foreach (var p in s.PlayerRng.GetType().GetProperties().Where(p => p.PropertyType == typeof(GameRng)))
        {
            pl[p.Name] = ((GameRng)p.GetValue(s.PlayerRng)!).CallCount;
        }

        return new JsonObject { ["rng"] = o, ["player_rng"] = pl };
    }

    static JsonObject? GameLog(JsonObject game, int i) =>
        (game["setup_log"] as JsonArray)?.Select(x => x as JsonObject).FirstOrDefault(x => (int?)x?["i"] == i);

    /// <summary>Items before the fight item (all of them without one), in the map phase after Neow.</summary>
    internal static void MapItems(RunEngine e, JsonObject game, JsonArray log)
    {
        var items = Items(game);
        int fight = FightItem(items);
        if (items.Count > 0)
        {
            var before = new JsonObject { ["i"] = -1, ["cmd"] = "before", ["ok"] = true, ["msg"] = "" };
            foreach (var kv in Counters(e.State)) before[kv.Key] = kv.Value!.DeepClone();
            log.Add(before);
        }

        Run(e, game, items, 0, fight < 0 ? items.Count : fight, log);
    }

    /// <summary>Items after the fight item, at the fight's start (turn 1, hand drawn).</summary>
    internal static void CombatItems(RunEngine e, JsonObject game, JsonArray log)
    {
        var items = Items(game);
        int fight = FightItem(items);
        if (fight >= 0) Run(e, game, items, fight + 1, items.Count, log);
    }

    static void Run(RunEngine e, JsonObject game, JsonArray items, int from, int to, JsonArray log)
    {
        for (int i = from; i < to; i++)
        {
            var item = items[i]!.AsObject();
            string cmd = (string?)item["cmd"] ?? "";
            var entry = new JsonObject { ["i"] = i, ["cmd"] = cmd, ["id"] = item["id"]?.DeepClone() };
            var g = GameLog(game, i);
            if (g != null && (bool?)g["ok"] == false && cmd != "potion")
            {
                entry["ok"] = false;
                entry["msg"] = "skipped: the game refused it (" + (string?)g["msg"] + ")";
            }
            else
            {
                string? problem;
                bool ok;
                try
                {
                    (ok, problem) = Item(e, item, Choices(game, i));
                }
                catch (Exception ex) when (ex is KeyNotFoundException or InvalidOperationException or ArgumentException)
                {
                    (ok, problem) = (false, "unsupported: " + ex.Message);
                }

                entry["ok"] = ok;
                entry["msg"] = problem ?? "";
                if (problem != null && problem.StartsWith("unsupported")) entry["unsupported"] = true;
            }

            foreach (var kv in Counters(e.State)) entry[kv.Key] = kv.Value!.DeepClone();
            log.Add(entry);
        }
    }

    /// <summary>The card choices the game logged against item i (tagged step -(i+1)).</summary>
    static List<JsonObject> Choices(JsonObject game, int i) =>
        (game["choices"] as JsonArray ?? new JsonArray()).Select(c => c as JsonObject)
            .Where(c => c != null && (int?)c["step"] == -(i + 1) && (string?)c["kind"] == "cards").Select(c => c!).ToList();

    static (bool ok, string? msg) Item(RunEngine e, JsonObject item, List<JsonObject> choices)
    {
        var s = e.State;
        string cmd = (string?)item["cmd"] ?? "";
        string id = ((string?)item["id"] ?? "").ToUpperInvariant();
        var c = s.ActiveCombat;
        switch (cmd)
        {
            case "relic":
            {
                if (c != null) return (false, "unsupported: relic in a fight");
                int relic = RelicId(id) ?? throw new KeyNotFoundException("no emulator relic " + id);
                var followUp = RunNonCombatEffects.ApplyRelicPickup(s, relic, SelectionReturn.Map);
                if (followUp == RunFollowUp.TransformSelect)
                {
                    s.Phase = RunPhase.TransformSelect;
                }
                else if (followUp != RunFollowUp.None)
                {
                    return (false, "unsupported: relic follow-up " + followUp);
                }

                string? gap = AnswerDeckSelection(e, choices);
                return gap == null ? (true, null) : (false, "unsupported: " + gap);
            }
            case "potion":
            {
                int potion = PotionId(id) ?? throw new KeyNotFoundException("no emulator potion " + id);
                bool ok = c != null ? AddToCombatBelt(c, potion) : RunRewardGenerator.AddPotion(s, potion);
                return (ok, ok ? null : "refused");
            }
            case "card":
            {
                string pile = ((string?)item["pile"] ?? "Deck").ToLowerInvariant();
                if (pile != "deck" || c != null) return (false, "unsupported: card into " + pile + (c != null ? " in a fight" : ""));
                RunNonCombatEffects.AddCardToDeck(s, new CardInstance(Fight.CardIdBySnake(id), Upgraded: (bool?)item["upgraded"] == true));
                return (true, null);
            }
            case "power":
            {
                if (c == null) return (false, "unsupported: power outside a fight");
                var buff = Reanchor.BuffFor(id) ?? throw new KeyNotFoundException("no emulator buff for " + id);
                string target = (string?)item["target"] ?? "player";
                int at = target == "player" ? -1 : target.StartsWith("enemy:") ? int.Parse(target[6..]) : throw new ArgumentException("target " + target);
                bool ok = ConsoleSetup.ApplyPowerWithoutApplier(c, at, buff, (int?)item["amount"] ?? 1);
                return (ok, ok ? null : "no living target");
            }
            case "energy":
            {
                if (c == null) return (false, "unsupported: energy outside a fight");
                Sts2Emulator.Core.Effects.CardEffects.GainEnergy(c, (int?)item["amount"] ?? 1);
                return (true, null);
            }
            default:
                return (false, "unsupported: " + cmd);
        }
    }

    /// <summary>PotionCmd.TryToProcure in a fight: Sozu refuses, else the first free slot.</summary>
    static bool AddToCombatBelt(CombatState c, int potion)
    {
        if (c.Relics.Any(r => r.DefId == Sts2Emulator.Core.Effects.RelicEffects.Sozu)) return false;
        for (int i = 0; i < Math.Min(c.MaxPotionSlots, c.PotionSlots.Length); i++)
        {
            if (c.PotionSlots[i] != 0) continue;
            c.PotionSlots[i] = potion;
            return true;
        }

        return false;
    }

    /// <summary>A screen an item opened, answered with the first option as the referee's selector does.</summary>
    static void AnswerScreens(RunEngine e)
    {
        var s = e.State;
        int guard = 0;
        while (s.Phase != RunPhase.Map && s.Phase != RunPhase.Combat && guard++ < 10)
        {
            e.Step(0, -1, out _, out _, out _);
            if (s.Phase == RunPhase.BundleSelect)
            {
                e.Step(RunConstants.BundleConfirmAction, -1, out _, out _, out _);
            }
        }
    }

    /// <summary>
    /// A deck selection a relic's pickup opened, answered with the cards the game's selector took
    /// (matched by card id among the cards the emulator lets the player pick). Returns a reason when
    /// the two screens disagree: the emulator wants more picks than the game's screen required (it has
    /// no way to stop early), or a picked card is not selectable. The run is then walked on with the
    /// first selectable cards so the fight can still be replayed, and the item counts as a mirror gap.
    /// </summary>
    static string? AnswerDeckSelection(RunEngine e, List<JsonObject> choices)
    {
        var s = e.State;
        string? gap = null;
        var mask = new int[4096];
        // A grid of offered cards is indexed by the offer, any other selection by the deck.
        bool Grid() => s.PendingOfferCards.Length > 0;
        int IdAt(int i) => Grid() ? Math.Abs(s.PendingOfferCards[i]) : s.Deck[i].DefId;
        int[] Selectable()
        {
            e.WriteActionMask(mask);
            int n = Grid() ? s.PendingOfferCards.Length : s.Deck.Count;
            return Enumerable.Range(0, n).Where(i => mask[i] != 0).ToArray();
        }

        var want = new Queue<int>();
        foreach (var c in choices)
        {
            var opts = c["options"]!.AsArray().Select(x => (string)x!).ToList();
            foreach (var p in c["picked"]!.AsArray()) want.Enqueue(Fight.CardIdBySnake(opts[(int)p!]));
        }

        int guard = 0;
        while (s.Phase == RunPhase.TransformSelect && guard++ < 64)
        {
            var ok = Selectable();
            if (ok.Length == 0) { gap ??= "selection with nothing selectable"; break; }
            int pick;
            if (want.Count > 0)
            {
                int id = want.Dequeue();
                pick = ok.Where(i => IdAt(i) == id).DefaultIfEmpty(-1).First();
                if (pick < 0) { gap ??= $"card {id} the game picked is not selectable here"; pick = ok[0]; }
            }
            else if (Grid() && s.PendingOfferOptional)
            {
                pick = s.PendingOfferCards.Length;  // done: the game took no more
            }
            else
            {
                if (choices.Count > 0) gap ??= "the emulator's selection wants more cards than the game's screen required";
                pick = ok[0];
            }

            e.Step(pick, -1, out _, out _, out _);
        }

        if (want.Count > 0) gap ??= $"the game picked {want.Count} more cards than the emulator's selection took";
        AnswerScreens(e);
        return gap;
    }

    /// <summary>
    /// For the scenario generator: the follow-up each relic's pickup opens on the map (None and
    /// TransformSelect are mirrored; any other is a mirror gap), probed on a fresh run past Neow.
    /// </summary>
    internal static JsonObject Info()
    {
        var relics = new JsonObject();
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            var e = new RunEngine();
            e.Reset("SETUPINFO1", 10);
            AnswerScreens(e);
            string followUp;
            try
            {
                followUp = RunNonCombatEffects.ApplyRelicPickup(e.State, r.Id, SelectionReturn.Map).ToString();
            }
            catch (Exception ex)
            {
                followUp = "error: " + ex.GetType().Name;
            }

            relics[r.Entry] = followUp;
        }

        var keys = Reanchor.PowerKeys();
        return new JsonObject { ["relics"] = relics, ["buff_keys"] = keys["buff_keys"]!.DeepClone(), ["internal_keys"] = keys["internal_keys"]!.DeepClone() };
    }

    /// <summary>
    /// The fight item: the same map option the referee walked onto, meeting the encounter the game
    /// entered, found by its place in the game's act lists (act, kind, index) and taken from the
    /// emulator's own copy of that list (tools/fidelity/table_check.py checks the two agree).
    /// Returns false without a fight item; throws when the encounter cannot be named.
    /// </summary>
    internal static bool EnterFight(RunEngine e, JsonObject game, int option, JsonObject log)
    {
        if (FightItem(Items(game)) < 0) return false;
        var f = game["fight"] as JsonObject ?? throw new InvalidOperationException("setup has a fight item but the game recorded no fight");
        string actName = (string?)f["act"] ?? throw new InvalidOperationException("the game lists " + f["id"] + " in no act");
        int act = Tables.ActIds.First(a => a.Name == actName).Act;
        string kind = (string)f["kind"]!;
        var list = Tables.EncounterList(act, kind);
        int index = (int)f["index"]!;
        if (index >= list.Length) throw new InvalidOperationException($"{actName} {kind} has {list.Length} encounters, the game's {f["id"]} is at {index}");
        int encounter = list[index];
        int node = kind switch { "elite" => RunConstants.NodeElite, "boss" => RunConstants.NodeBoss, _ => RunConstants.NodeNormal };
        log["encounter"] = ((CombatFactory.ActOneEncounter)encounter).ToString();
        if (e.EnterEncounterDebug(option, encounter, node, weakVariant: kind == "weak") != 0)
            throw new InvalidOperationException("EnterEncounterDebug refused option " + option);
        return true;
    }
}
