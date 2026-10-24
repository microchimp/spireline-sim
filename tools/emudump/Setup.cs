// The setup mirror's driver (referee/driver/Setup.cs): the same items, in the same order, at the same moments, in
// terms of IEngine only. Map items run in the run's map phase after Neow (IEngine.StartFight), the fight item walks
// onto the referee's row-1 coordinate but meets the named encounter, and combat items run before the step-0 dump,
// so a wrong mirror shows at step 0. Each item is logged with the run's RNG counters (tools/fidelity/setup_compare.py
// lines them up with the game's). An item the engine cannot build is logged "unsupported": a mirror gap, not an
// emulator divergence. An item the game refused (ok false) is skipped, so both sides stay alike.
// What each item does to a run is the engine's: the old engine's is Engines/Old/OldSetup.cs.
// Three combat items are the scenarios' own (Scenario.cs, docs/rebuild/SCENARIOS.md), with no referee counterpart yet:
//   {"cmd": "block", "amount": N}                              the player's block becomes N (IEngine.SetBlock)
//   {"cmd": "enemy_hp", "slot": S, "max": M, "current": C}     the enemy in slot S: max HP, then current HP; either optional (IEngine.SetEnemyHp)
//   {"cmd": "enemy_block", "slot": S, "amount": N}             the enemy in slot S's block becomes N (IEngine.SetEnemyBlock)
// The engine refuses a slot with no living enemy, a negative value and a current HP above the max HP in force (ok false, not a gap).
using System.Text.Json.Nodes;
using EmuDump.Engine;

static class Setup
{
    /// <summary>Whether the fight being replayed records the setup extras (potions in each snapshot).</summary>
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

    static JsonObject? GameLog(JsonObject game, int i) =>
        (game["setup_log"] as JsonArray)?.Select(x => x as JsonObject).FirstOrDefault(x => (int?)x?["i"] == i);

    /// <summary>Items before the fight item (all of them without one), in the map phase after Neow.</summary>
    internal static void MapItems(IEngine e, JsonObject game, JsonArray log)
    {
        var items = Items(game);
        int fight = FightItem(items);
        if (items.Count > 0)
        {
            var before = new JsonObject { ["i"] = -1, ["cmd"] = "before", ["ok"] = true, ["msg"] = "" };
            foreach (var kv in e.RngCounters()) before[kv.Key] = kv.Value!.DeepClone();
            log.Add(before);
        }

        Run(e, game, items, 0, fight < 0 ? items.Count : fight, log);
    }

    /// <summary>Items after the fight item, at the fight's start (turn 1, hand drawn).</summary>
    internal static void CombatItems(IEngine e, JsonObject game, JsonArray log)
    {
        var items = Items(game);
        int fight = FightItem(items);
        if (fight >= 0) Run(e, game, items, fight + 1, items.Count, log);
    }

    static void Run(IEngine e, JsonObject game, JsonArray items, int from, int to, JsonArray log)
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

            foreach (var kv in e.RngCounters()) entry[kv.Key] = kv.Value!.DeepClone();
            log.Add(entry);
        }
    }

    /// <summary>The card choices the game logged against item i (tagged step -(i+1)).</summary>
    static List<JsonObject> Choices(JsonObject game, int i) =>
        (game["choices"] as JsonArray ?? new JsonArray()).Select(c => c as JsonObject)
            .Where(c => c != null && (int?)c["step"] == -(i + 1) && (string?)c["kind"] == "cards").Select(c => c!).ToList();

    static SetupResult Item(IEngine e, JsonObject item, List<JsonObject> choices)
    {
        string cmd = (string?)item["cmd"] ?? "";
        string id = ((string?)item["id"] ?? "").ToUpperInvariant();
        switch (cmd)
        {
            case "relic":
                return e.ObtainRelic(id, choices);
            case "potion":
                return e.ProcurePotion(id);
            case "card":
                return e.AddCard(id, (bool?)item["upgraded"] == true, ((string?)item["pile"] ?? "Deck").ToLowerInvariant(),
                    onTop: (string?)item["pos"] == "top");
            case "power":
            {
                string target = (string?)item["target"] ?? "player";
                int at = target == "player" ? -1 : target.StartsWith("enemy:") ? int.Parse(target[6..]) : throw new ArgumentException("target " + target);
                return e.ApplyPower(id, at, (int?)item["amount"] ?? 1);
            }
            case "energy":
                return e.GainEnergy((int?)item["amount"] ?? 1);
            case "hp":
                return e.SetHp((int?)item["max"], (int?)item["current"]);
            case "block":
                return e.SetBlock((int?)item["amount"] ?? throw new ArgumentException("block: no amount"));
            case "enemy_hp":
                return e.SetEnemyHp(Slot(item), (int?)item["max"], (int?)item["current"]);
            case "enemy_block":
                return e.SetEnemyBlock(Slot(item), (int?)item["amount"] ?? throw new ArgumentException("enemy_block: no amount"));
            default:
                return e.RefsetItem(item, choices) ?? new SetupResult(false, "unsupported: " + cmd);
        }
    }

    /// <summary>The creature slot an enemy item names (<c>slot</c>, an integer); throws when it names none.</summary>
    static int Slot(JsonObject item) =>
        item["slot"] is JsonValue v && v.TryGetValue<int>(out int slot) ? slot : throw new ArgumentException((string?)item["cmd"] + ": no slot");

    /// <summary>
    /// The fight item: the same map option the referee walked onto, meeting the encounter the game entered, found by
    /// its place in the game's act lists (act, kind, index). Fills <c>encounter</c> of <paramref name="log"/>.
    /// Returns false without a fight item; throws when the encounter cannot be named.
    /// </summary>
    internal static bool EnterFight(IEngine e, JsonObject game, int option, JsonObject log)
    {
        if (FightItem(Items(game)) < 0) return false;
        var f = game["fight"] as JsonObject ?? throw new InvalidOperationException("setup has a fight item but the game recorded no fight");
        string actName = (string?)f["act"] ?? throw new InvalidOperationException("the game lists " + f["id"] + " in no act");
        log["encounter"] = e.EnterEncounter(option, actName, (string)f["kind"]!, (int)f["index"]!, f["id"]?.ToString() ?? "");
        return true;
    }
}
