// The emulator's mirror of the reference-set setup items (referee/driver/RefsetSetup.cs): the same
// algorithm on the run state, so a real run's deck, relics (with their saved state), HP, gold and potions are
// in place before its fight, in its own act. An item it cannot build is "unsupported" (a mirror gap), as in
// Setup.cs and Engines/Old/OldSetup.cs.
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class RefsetSetup
{
    static Dictionary<int, string>? _cardNames;

    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    static string CardName(int defId)
    {
        if (_cardNames == null)
        {
            var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!["cards"]!.AsObject();
            _cardNames = map.ToDictionary(kv => (int)kv.Value!, kv => Snake(kv.Key));
        }

        return _cardNames.TryGetValue(defId, out var n) ? n : defId.ToString();
    }

    static string RelicEntry(int defId)
    {
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            if (r.Id == defId) return r.Entry;
        }

        return defId.ToString();
    }

    /// <summary>The deck in order, as the referee logs it (referee/driver/RefsetSetup.cs DeckLine).</summary>
    static string DeckLine(RunState s) => " | deck: " + string.Join(",", s.Deck.Select(c =>
        CardName(c.DefId) + (c.Upgraded ? "+" : "") + (c.Enchantment != Enchantment.None ? "*" + Snake(c.Enchantment.ToString()) + c.EnchantAmount : "")));

    readonly record struct Key(string Id, bool Upgraded, string Enchantment, int Amount);

    static Key KeyOf(JsonObject o) => new(((string?)o["id"] ?? "").ToUpperInvariant(), (bool?)o["upgraded"] == true,
        ((string?)o["enchantment"] ?? "").ToUpperInvariant(), (int?)o["amount"] ?? 0);

    static Key KeyOf(CardInstance c) => new(CardName(c.DefId), c.Upgraded,
        c.Enchantment == Enchantment.None ? "" : Snake(c.Enchantment.ToString()), c.Enchantment == Enchantment.None ? 0 : c.EnchantAmount);

    static (List<int> extra, List<Key> missing) Diff(List<Key> deck, List<Key> want)
    {
        var left = want.ToList();
        var extra = new List<int>();
        for (int i = 0; i < deck.Count; i++)
        {
            int at = left.IndexOf(deck[i]);
            if (at >= 0) left.RemoveAt(at);
            else extra.Add(i);
        }

        return (extra, left);
    }

    /// <summary>
    /// Saved properties the emulator has no state for and that do nothing in a fight: what an ancient's
    /// relic picked at its event (Archaic Tooth's cards, Dusty Tome's card, Touch of Orobas' relics, Sea
    /// Glass' character), skins, and map-only state (Golden Compass' act, Spoils Map's act). Any other
    /// saved property with a value other than the one a fresh model has, and that RunReplay.AnchorHoldings
    /// does not set, is a mirror gap.
    /// </summary>
    static readonly HashSet<string> NoFightEffect =
    [
        "ARCHAIC_TOOTH.StarterCard", "ARCHAIC_TOOTH.AncientCard", "DUSTY_TOME.AncientCard",
        "TOUCH_OF_OROBAS.StarterRelic", "TOUCH_OF_OROBAS.UpgradedRelic", "SEA_GLASS.CharacterId",
        "BYRDPIP.Skin", "PAELS_LEGION.Skin", "GOLDEN_COMPASS.GoldenPathAct", "SPOILS_MAP.SpoilsActIndex",
    ];

    /// <summary>Saved properties RunReplay.AnchorHoldings puts on the emulator's state.</summary>
    static HashSet<string> Anchored() => RunSnap.RelicProps.Select(p => p.Relic + "." + p.Name)
        .Concat(["MAD_SCIENCE.TinkerTimeType", "MAD_SCIENCE.TinkerTimeRider", "GUILTY.CombatsSeen"])
        .Concat(Sts2Emulator.GeneratedData.Relics.All.ToArray().SelectMany(r => new[] { r.Entry + ".IsWax", r.Entry + ".IsMelted" }))
        .ToHashSet();

    static string Bare(string? id) => id == null ? "" : id.Contains('.') ? id[(id.IndexOf('.') + 1)..] : id;

    /// <summary>
    /// The player_load item (referee/driver/RefsetSetup.cs): the run's holdings SET, not earned, through the
    /// run replay's own anchoring of a save's player (RunReplay.AnchorHoldings, RunEngine.DebugSetPlayer); the
    /// held relics leave the grab bags as the game's load takes them out (RelicCmd.Obtain's rule: all but
    /// the stackable Circlet).
    /// </summary>
    static (bool ok, string? msg) PlayerLoad(RunEngine e, JsonObject item)
    {
        var s = e.State;
        int max = (int?)item["max_hp"] ?? s.PlayerMaxHp;
        var potions = new JsonArray();
        int slot = 0;
        foreach (var x in item["potions"] as JsonArray ?? new JsonArray())
            potions.Add(new JsonObject { ["id"] = "POTION." + ((string?)x ?? "").ToUpperInvariant(), ["slot_index"] = slot++ });
        var p = new JsonObject
        {
            ["current_hp"] = Math.Min((int?)item["hp"] ?? s.PlayerHp, max), ["max_hp"] = max, ["gold"] = (int?)item["gold"] ?? s.Gold,
            ["deck"] = (item["deck"] ?? new JsonArray()).DeepClone(), ["relics"] = (item["relics"] ?? new JsonArray()).DeepClone(),
            ["potions"] = potions, ["max_potion_slot_count"] = (int?)item["max_potion_slots"] ?? s.MaxPotionSlots,
        };
        RunReplay.AnchorHoldings(e, p, new JsonArray());
        foreach (var r in s.Relics)
        {
            if (RelicEntry(r.DefId) == "CIRCLET") continue;
            s.RelicBag.Remove(r.DefId);
            s.SharedRelicBag.Remove(r.DefId);
        }

        // Saved properties the emulator cannot hold, with a value a fresh model would not have.
        var anchored = Anchored();
        var gaps = new List<string>();
        foreach (var model in (item["relics"] as JsonArray ?? new JsonArray()).Concat(item["deck"] as JsonArray ?? new JsonArray()))
        {
            string id = Bare((string?)model!["id"]);
            foreach (var (kind, list) in model["props"]?.AsObject() ?? new JsonObject())
            {
                foreach (var prop in list?.AsArray() ?? new JsonArray())
                {
                    string key = id + "." + (string?)prop!["name"];
                    if (anchored.Contains(key) || NoFightEffect.Contains(key) || IsFresh(kind, prop["value"])) continue;
                    gaps.Add(key);
                }
            }
        }

        string tail = DeckLine(s) + " | relics: " + string.Join(",", s.Relics.Select(r => RelicEntry(r.DefId)));
        return gaps.Count == 0 ? (true, tail) : (false, "unsupported: saved state " + string.Join(",", gaps.Distinct()) + tail);
    }

    /// <summary>A saved value a freshly made model has too (0, false, an empty list or none).</summary>
    static bool IsFresh(string kind, JsonNode? v) => v == null || kind switch
    {
        "ints" => (int)v == 0,
        "bools" => !(bool)v,
        "card_arrays" or "int_arrays" => v.AsArray().Count == 0,
        _ => false,
    };

    /// <summary>A reference-set item, or null when <paramref name="item" /> is not one.</summary>
    internal static (bool ok, string? msg)? Item(RunEngine e, JsonObject item, List<JsonObject> choices)
    {
        var s = e.State;
        string cmd = (string?)item["cmd"] ?? "";
        if (cmd is not ("relics_set" or "deck_set" or "potions_set" or "player_load" or "act")) return null;
        if (s.ActiveCombat != null) return (false, cmd + ": map only");
        switch (cmd)
        {
            case "player_load":
                return PlayerLoad(e, item);
            case "act":
            {
                // RunManager.EnterAct(N - 1): the emulator's own act transition (RunEngine.EnterNextAct, which
                // the boss path and the debug hook share), then onto the act's start point the way the game's
                // EnterMapCoordDebug walks straight onto the fight's row-1 coordinate.
                int n = (int?)item["index"] ?? 1;
                if (n < 1 || n > s.Acts.Count) return (false, $"no act {n} (the run has {s.Acts.Count})");
                if (s.CurrentActIndex == n - 1) return (true, null);
                if (s.CurrentActIndex > n - 1) return (false, "unsupported: an act behind the current one");
                while (s.CurrentActIndex < n - 1)
                {
                    if (!e.EnterNextAct()) return (false, "unsupported: no act after " + s.CurrentActIndex);
                }

                s.AwaitingActStartNode = false;
                s.CurrentMapCoord = (RunConstants.MapStartCol, 0);
                RunMapGenerator.RefreshMapOptions(s);
                return (true, null);
            }
            case "relics_set":
            {
                var want = (item["ids"] as JsonArray ?? new JsonArray()).Select(x => ((string?)x ?? "").ToUpperInvariant()).ToList();
                var held = s.Relics.Select(r => RelicEntry(r.DefId)).ToList();
                var queue = new Queue<JsonObject>(choices);
                string? gap = null;
                var notes = new List<string>();  // where the game's screen and the emulator's listed other cards (no gap)
                foreach (string id in want)
                {
                    if (held.Remove(id)) continue;
                    int relic = OldSetup.RelicId(id) ?? throw new KeyNotFoundException("no emulator relic " + id);
                    var followUp = RunNonCombatEffects.ApplyRelicPickup(s, relic, SelectionReturn.Map);
                    if (followUp == RunFollowUp.TransformSelect)
                    {
                        s.Phase = RunPhase.TransformSelect;
                        var mine = queue.Count > 0 ? new List<JsonObject> { queue.Dequeue() } : new List<JsonObject>();
                        if (gap == null)
                        {
                            gap = OldSetup.AnswerDeckSelection(e, mine, out var pickNote);
                            if (pickNote != null) notes.Add(id + ": " + pickNote);
                        }
                    }
                    else if (followUp != RunFollowUp.None)
                    {
                        return (false, "unsupported: relic follow-up " + followUp + " of " + id);
                    }
                }

                var keep = want.ToList();
                for (int i = 0; i < s.Relics.Count; i++)
                {
                    if (keep.Remove(RelicEntry(s.Relics[i].DefId))) continue;
                    s.Relics.RemoveAt(i--);  // RelicCmd.Remove: no relic overrides AfterRemoved
                }

                // The deck line stays the message's first part (refset_score.setup_deck reads up to the first " | " after it).
                return gap == null ? (true, DeckLine(s) + (notes.Count > 0 ? " | note: " + string.Join("; ", notes) : "")) : (false, "unsupported: " + gap);
            }
            case "deck_set":
            {
                var want = (item["cards"] as JsonArray ?? new JsonArray()).Select(x => KeyOf(x!.AsObject())).ToList();
                string residual = "";
                for (int pass = 0; pass < 3; pass++)
                {
                    var (extra, missing) = Diff(s.Deck.Select(KeyOf).ToList(), want);
                    if (extra.Count == 0 && missing.Count == 0) { residual = ""; break; }
                    foreach (int i in extra.OrderByDescending(i => i)) s.Deck.RemoveAt(i);  // CardPileCmd.RemoveFromDeck
                    foreach (var k in missing)
                    {
                        int at = s.Deck.Count;
                        RunNonCombatEffects.AddCardToDeck(s, new CardInstance(OldIds.CardIdBySnake(k.Id), Upgraded: k.Upgraded));
                        if (k.Enchantment == "") continue;
                        if (!Enum.TryParse<Enchantment>(k.Enchantment.Replace("_", ""), true, out var ench))
                            return (false, "unsupported: enchantment " + k.Enchantment);
                        if (at < s.Deck.Count) s.Deck[at] = s.Deck[at] with { Enchantment = ench, EnchantAmount = k.Amount };
                    }

                    var after = Diff(s.Deck.Select(KeyOf).ToList(), want);
                    residual = after.extra.Count + after.missing.Count == 0 ? ""
                        : $"residual: {after.extra.Count} extra, missing {string.Join(",", after.missing.Select(m => m.Id + (m.Upgraded ? "+" : "")))}";
                }

                // As the referee: a deck the item could not build fails it.
                return (residual == "", residual + DeckLine(s));
            }
            default: // potions_set
            {
                for (int i = 0; i < s.PotionSlots.Length; i++) s.PotionSlots[i] = 0;  // PotionCmd.Discard
                var refused = new List<string>();
                foreach (var x in item["ids"] as JsonArray ?? new JsonArray())
                {
                    string id = ((string?)x ?? "").ToUpperInvariant();
                    int potion = OldSetup.PotionId(id) ?? throw new KeyNotFoundException("no emulator potion " + id);
                    if (!RunRewardGenerator.AddPotion(s, potion)) refused.Add(id);
                }

                return (true, refused.Count == 0 ? null : "refused: " + string.Join(",", refused));
            }
        }
    }
}
