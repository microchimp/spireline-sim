// The emulator's mirror of the reference-set setup items (referee/driver/RefsetSetup.cs): the same
// algorithm on the run state, so a real run's deck, relics, HP and potions are in place before its fight.
// An item it cannot build is "unsupported" (a mirror gap), as in Setup.cs.
using System.Text.Json.Nodes;
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

    /// <summary>A reference-set item, or null when <paramref name="item" /> is not one.</summary>
    internal static (bool ok, string? msg)? Item(RunEngine e, JsonObject item, List<JsonObject> choices)
    {
        var s = e.State;
        string cmd = (string?)item["cmd"] ?? "";
        if (cmd is not ("relics_set" or "deck_set" or "hp" or "potions_set")) return null;
        if (s.ActiveCombat != null) return (false, cmd + ": map only");
        switch (cmd)
        {
            case "relics_set":
            {
                var want = (item["ids"] as JsonArray ?? new JsonArray()).Select(x => ((string?)x ?? "").ToUpperInvariant()).ToList();
                var held = s.Relics.Select(r => RelicEntry(r.DefId)).ToList();
                var queue = new Queue<JsonObject>(choices);
                string? gap = null;
                foreach (string id in want)
                {
                    if (held.Remove(id)) continue;
                    int relic = Setup.RelicId(id) ?? throw new KeyNotFoundException("no emulator relic " + id);
                    var followUp = RunNonCombatEffects.ApplyRelicPickup(s, relic, SelectionReturn.Map);
                    if (followUp == RunFollowUp.TransformSelect)
                    {
                        s.Phase = RunPhase.TransformSelect;
                        var mine = queue.Count > 0 ? new List<JsonObject> { queue.Dequeue() } : new List<JsonObject>();
                        gap ??= Setup.AnswerDeckSelection(e, mine);
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

                return gap == null ? (true, DeckLine(s)) : (false, "unsupported: " + gap);
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
                        RunNonCombatEffects.AddCardToDeck(s, new CardInstance(Fight.CardIdBySnake(k.Id), Upgraded: k.Upgraded));
                        if (k.Enchantment == "") continue;
                        if (!Enum.TryParse<Enchantment>(k.Enchantment.Replace("_", ""), true, out var ench))
                            return (false, "unsupported: enchantment " + k.Enchantment);
                        if (at < s.Deck.Count) s.Deck[at] = s.Deck[at] with { Enchantment = ench, EnchantAmount = k.Amount };
                    }

                    var after = Diff(s.Deck.Select(KeyOf).ToList(), want);
                    residual = after.extra.Count + after.missing.Count == 0 ? ""
                        : $"residual: {after.extra.Count} extra, missing {string.Join(",", after.missing.Select(m => m.Id + (m.Upgraded ? "+" : "")))}";
                }

                return (true, residual + DeckLine(s));
            }
            case "hp":
            {
                int max = (int?)item["max"] ?? s.PlayerMaxHp;
                int cur = Math.Min((int?)item["current"] ?? s.PlayerHp, max);
                if (max < 1 || cur < 1) return (false, "hp below 1");
                s.PlayerMaxHp = max;  // CreatureCmd.SetMaxHp, then SetCurrentHp
                s.PlayerHp = cur;
                return (true, null);
            }
            default: // potions_set
            {
                for (int i = 0; i < s.PotionSlots.Length; i++) s.PotionSlots[i] = 0;  // PotionCmd.Discard
                var refused = new List<string>();
                foreach (var x in item["ids"] as JsonArray ?? new JsonArray())
                {
                    string id = ((string?)x ?? "").ToUpperInvariant();
                    int potion = Setup.PotionId(id) ?? throw new KeyNotFoundException("no emulator potion " + id);
                    if (!RunRewardGenerator.AddPotion(s, potion)) refused.Add(id);
                }

                return (true, refused.Count == 0 ? null : "refused: " + string.Join(",", refused));
            }
        }
    }
}
