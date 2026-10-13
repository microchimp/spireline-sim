// Job kind "export": writes the game's own content tables (pool orders, rarities, costs, act
// encounter and event lists, Neow's option list) as JSON, for tools/fidelity/table_check.py.
// Reads only ModelDb's public API. The output is game data: keep it local, never commit it.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Unlocks;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>Runs one section; a failure is recorded under "errors", never thrown.</summary>
    private static void Section(JsonObject root, string name, Func<JsonNode?> body)
    {
        try
        {
            root[name] = body();
        }
        catch (Exception ex)
        {
            var errors = root["errors"] as JsonArray ?? new JsonArray();
            root["errors"] = errors;
            errors.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static JsonArray Ids(IEnumerable<AbstractModel> models) =>
        new(models.Select(m => (JsonNode)m.Id.Entry).ToArray());

    private static JsonNode? Safe(Func<object?> f)
    {
        try
        {
            object? v = f();
            return v is null ? null : JsonValue.Create(v.ToString());
        }
        catch (Exception ex)
        {
            return "ERR " + ex.GetType().Name;
        }
    }

    private static JsonObject CardInfo(CardModel c) => new()
    {
        ["id"] = c.Id.Entry,
        ["type"] = Safe(() => c.Type),
        ["rarity"] = Safe(() => c.Rarity),
        ["cost"] = Safe(() => c.EnergyCost.Canonical),
        ["cost_x"] = Safe(() => c.EnergyCost.CostsX),
        ["star_cost"] = Safe(() => c.CanonicalStarCost),
        ["target"] = Safe(() => c.TargetType),
        ["max_upgrade_level"] = Safe(() => c.MaxUpgradeLevel),
        ["keywords"] = Safe(() => string.Join(",", c.CanonicalKeywords.Select(k => k.ToString()))),
        ["multiplayer_constraint"] = Safe(() => c.MultiplayerConstraint),
    };

    private static JsonObject Export()
    {
        var root = new JsonObject { ["game"] = "v0.107.1" };
        Section(root, "characters", () => Ids(ModelDb.AllCharacters));
        Section(root, "card_pools", () =>
        {
            var o = new JsonObject();
            // The game's own order: character pools, then the shared ones.
            foreach (CardPoolModel p in ModelDb.AllCardPools)
            {
                string name = p.GetType().Name;
                try
                {
                    var all = p.AllCards.ToList();
                    o[name] = new JsonObject
                    {
                        ["colorless"] = Safe(() => p.IsColorless),
                        ["all"] = new JsonArray(all.Select(c => (JsonNode)CardInfo(c)).ToArray()),
                        ["unlocked"] = Ids(p.GetUnlockedCards(UnlockState.all, CardMultiplayerConstraint.SingleplayerOnly)),
                    };
                }
                catch (Exception ex)
                {
                    o[name] = new JsonObject { ["error"] = ex.ToString() };
                }
            }

            return o;
        });
        Section(root, "character_info", () =>
        {
            var o = new JsonObject();
            foreach (CharacterModel c in ModelDb.AllCharacters)
            {
                o[c.Id.Entry] = new JsonObject
                {
                    ["card_pool"] = c.CardPool.GetType().Name,
                    ["relic_pool"] = c.RelicPool.GetType().Name,
                    ["potion_pool"] = c.PotionPool.GetType().Name,
                    ["starting_deck"] = Ids(c.StartingDeck),
                    ["starting_relics"] = Ids(c.StartingRelics),
                };
            }

            return o;
        });
        Section(root, "relic_pools", () =>
        {
            var o = new JsonObject();
            foreach (RelicPoolModel p in ModelDb.AllRelicPools)
            {
                o[p.GetType().Name] = new JsonObject
                {
                    ["all"] = new JsonArray(p.AllRelics.Select(r => (JsonNode)new JsonObject
                    {
                        ["id"] = r.Id.Entry,
                        ["rarity"] = Safe(() => r.Rarity),
                    }).ToArray()),
                    ["unlocked"] = Ids(p.GetUnlockedRelics(UnlockState.all)),
                };
            }

            return o;
        });
        Section(root, "potion_pools", () =>
        {
            var o = new JsonObject();
            foreach (PotionPoolModel p in ModelDb.AllPotionPools)
            {
                o[p.GetType().Name] = new JsonObject
                {
                    ["all"] = new JsonArray(p.AllPotions.Select(r => (JsonNode)new JsonObject
                    {
                        ["id"] = r.Id.Entry,
                        ["rarity"] = Safe(() => r.Rarity),
                        ["usage"] = Safe(() => r.Usage),
                        ["target"] = Safe(() => r.TargetType),
                    }).ToArray()),
                    ["unlocked"] = Ids(p.GetUnlockedPotions(UnlockState.all)),
                };
            }

            return o;
        });
        Section(root, "acts_by_index", () => new JsonArray(ModelDb.ActsByIndex.Select(g =>
            (JsonNode)new JsonArray(g.Select(a => (JsonNode)a.Id.Entry).ToArray())).ToArray()));
        Section(root, "acts", () =>
        {
            var o = new JsonObject();
            foreach (ActModel a in ModelDb.Acts)
            {
                var act = new JsonObject
                {
                    ["index"] = Safe(() => a.Index),
                    ["is_default"] = Safe(() => a.IsDefault),
                };
                o[a.Id.Entry] = act;
                void Part(string key, Func<JsonNode?> f)
                {
                    try { act[key] = f(); }
                    catch (Exception ex) { act[key] = "ERR " + ex.GetType().Name + ": " + ex.Message; }
                }

                Part("weak", () => Ids(a.AllWeakEncounters));
                Part("normal", () => Ids(a.AllRegularEncounters));
                Part("elite", () => Ids(a.AllEliteEncounters));
                Part("boss", () => Ids(a.AllBossEncounters));
                Part("boss_discovery_order", () => Ids(a.BossDiscoveryOrder));
                Part("all_encounters", () => Ids(a.AllEncounters));
                Part("events", () => Ids(a.AllEvents));
                Part("ancients", () => Ids(a.AllAncients));
                Part("number_of_rooms", () => (int)a.GetNumberOfRooms(false));
            }

            return o;
        });
        Section(root, "shared_events", () => Ids(ModelDb.AllSharedEvents));
        Section(root, "shared_ancients", () => Ids(ModelDb.AllSharedAncients));
        Section(root, "neow_options", () =>
        {
            var neow = ModelDb.AllAncients.First(x => x.Id.Entry == "NEOW");
            return new JsonArray(neow.AllPossibleOptions.Select(o =>
                (JsonNode)new JsonObject { ["text_key"] = o.TextKey, ["relic"] = o.Relic?.Id.Entry }).ToArray());
        });
        return root;
    }
}
