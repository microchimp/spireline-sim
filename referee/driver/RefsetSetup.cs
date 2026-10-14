// Reference-set setup items (referee/refset.py): put a real run's state in place before its fight, through
// the game's own paths. They complement Setup.cs's items and keep its rule: only what SETS UP a state, never
// a command that changes or skips a rule.
//
//   {"cmd": "player_load", "deck": [<card>...], "relics": [<relic>...], "potions": [...], "max_potion_slots": S,
//    "hp": C, "max_hp": M, "gold": G}
//                                              The run's holdings at that floor, through the game's save-load
//                                              path: each card and relic is the save's own shape (SerializableCard,
//                                              SerializableRelic: id, upgrade level, enchantment, saved properties,
//                                              floor) read by the game's serializer (JsonSerializationUtility) and
//                                              built by CardModel.FromSerializable / RunState.LoadCard and
//                                              RelicModel.FromSerializable, which restore the [SavedProperty] state
//                                              (Mad Science's type and rider, Archaic Tooth's cards, a counter).
//                                              They replace the player's deck, relics, potions, HP, gold and potion
//                                              slots with the steps Player.SyncWithSerializedPlayer and
//                                              Player.LoadInventory use: no pickup effect (AfterObtained) and no
//                                              add-to-deck hook runs, because the record's holdings already include
//                                              every one, as a save does. Relics leave the grab bags as
//                                              RelicCmd.Obtain takes them out. ok=false when what the game holds
//                                              afterwards is not what was asked (the message names the difference).
//   {"cmd": "act", "index": N}                 ActConsoleCmd with an integer: RunManager.EnterAct(N - 1), so the
//                                              fight is met in its own act (act-dependent monsters and relics).
//   {"cmd": "relics_set", "ids": [...]}       RelicCmd.Obtain each listed relic not held (in list order,
//                                              answering any screen it opens), then RelicCmd.Remove each
//                                              held relic the list does not name (RelicConsoleCmd's remove).
//   {"cmd": "deck_set", "cards": [{"id", "upgraded", "enchantment", "amount"}, ...]}
//                                              CardPileCmd.RemoveFromDeck each deck card the list does not
//                                              hold (RemoveCardConsoleCmd), then for each missing card:
//                                              CreateCard, CardCmd.Upgrade, CardPileCmd.Add to the deck and
//                                              CardCmd.Enchant (EnchantConsoleCmd). Up to three passes, since
//                                              an add can trigger a relic (eggs upgrade, Bing Bong clones); a
//                                              deck that still differs fails the item (ok=false, "residual").
//   {"cmd": "hp", "max": M, "current": C}     CreatureCmd.SetMaxHp, then CreatureCmd.SetCurrentHp (events' own).
//   {"cmd": "potions_set", "ids": [...]}      PotionCmd.Discard every held potion, then PotionCmd.TryToProcure
//                                              each listed one (PotionConsoleCmd).
// player_load, relics_set and deck_set end their message with the deck in order (" | deck: A,B+,C*SHARP1"),
// which EmuDump writes the same way, so a deck the emulator orders or builds differently shows at the item,
// not at step 0. All are map items (before "fight"). EmuDump mirrors them in tools/emudump/RefsetSetup.cs.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>A reference-set item, or null when <paramref name="item" /> is not one.</summary>
    private static async Task<(bool ok, string msg)?> RefsetItem(JsonObject item, RunState state, Player player)
    {
        string cmd = (string?)item["cmd"] ?? "";
        if (cmd is not ("relics_set" or "deck_set" or "hp" or "potions_set" or "player_load" or "act")) return null;
        if (CombatManager.Instance.IsInProgress) return (false, cmd + ": map only");
        switch (cmd)
        {
            case "player_load":
                return await PlayerLoad(item, state, player);
            case "act":
                return await EnterActItem(item, state);
            case "relics_set":
            {
                var want = (item["ids"] as JsonArray ?? new JsonArray()).Select(x => ((string?)x ?? "").ToUpperInvariant()).ToList();
                var missing = new List<string>();
                var held = player.Relics.Select(r => r.Id.Entry).ToList();
                foreach (string id in want)
                {
                    if (held.Remove(id)) continue;
                    var relic = ModelDb.AllRelics.FirstOrDefault(r => r.Id.Entry == id);
                    if (relic == null) { missing.Add(id); continue; }
                    try
                    {
                        await AwaitSetup(RelicCmd.Obtain(relic.ToMutable(), player), "RelicCmd.Obtain " + id);
                    }
                    catch (Exception ex)
                    {
                        return (false, "RelicCmd.Obtain " + id + ": " + ex.GetType().Name + ": " + ex.Message);
                    }

                    await Idle();
                }

                var keep = want.ToList();
                foreach (var relic in player.Relics.ToList())
                {
                    if (keep.Remove(relic.Id.Entry)) continue;
                    await AwaitSetup(RelicCmd.Remove(relic), "RelicCmd.Remove " + relic.Id.Entry);
                }

                return (missing.Count == 0, (missing.Count == 0 ? "" : "no relic " + string.Join(",", missing)) + DeckLine(player));
            }
            case "deck_set":
            {
                var want = (item["cards"] as JsonArray ?? new JsonArray()).Select(x => DeckKey(x!.AsObject())).ToList();
                string residual = "";
                for (int pass = 0; pass < 3; pass++)
                {
                    var deck = PileType.Deck.GetPile(player).Cards.ToList();
                    var (extra, missing) = DeckDiff(deck.Select(DeckKey).ToList(), want);
                    if (extra.Count == 0 && missing.Count == 0) { residual = ""; break; }
                    foreach (int i in extra.OrderByDescending(i => i))
                    {
                        await AwaitSetup(CardPileCmd.RemoveFromDeck(deck[i], showPreview: false), "CardPileCmd.RemoveFromDeck");
                    }

                    foreach (var k in missing)
                    {
                        var canonical = ModelDb.AllCards.FirstOrDefault(c => c.Id.Entry == k.id);
                        if (canonical == null) return (false, "no card " + k.id);
                        var card = state.CreateCard(canonical, player);
                        if (k.upgraded)
                        {
                            if (!card.IsUpgradable) return (false, "not upgradable " + k.id);
                            CardCmd.Upgrade(card);
                        }

                        await AwaitSetup(CardPileCmd.Add(card, PileType.Deck, CardPilePosition.Bottom, skipVisuals: true), "CardPileCmd.Add");
                        if (k.enchantment != "")
                        {
                            var ench = ModelDb.DebugEnchantments.FirstOrDefault(e => e.Id.Entry == k.enchantment);
                            if (ench == null) return (false, "no enchantment " + k.enchantment);
                            var mutable = ench.ToMutable();
                            if (!mutable.CanEnchant(card)) return (false, k.enchantment + " cannot enchant " + k.id);
                            CardCmd.Enchant(mutable, card, k.amount);
                        }
                    }

                    await Idle();
                    var after = DeckDiff(PileType.Deck.GetPile(player).Cards.Select(DeckKey).ToList(), want);
                    residual = after.extra.Count + after.missing.Count == 0 ? ""
                        : $"residual: {after.extra.Count} extra, missing {string.Join(",", after.missing.Select(m => m.id + (m.upgraded ? "+" : "")))}";
                }

                return DeckResidualFails(residual, DeckLine(player));
            }
            case "hp":
            {
                int max = (int?)item["max"] ?? player.Creature.MaxHp;
                int cur = Math.Min((int?)item["current"] ?? player.Creature.CurrentHp, max);
                if (max < 1 || cur < 1) return (false, "hp below 1");
                await AwaitSetup(CreatureCmd.SetMaxHp(player.Creature, max), "CreatureCmd.SetMaxHp");
                await AwaitSetup(CreatureCmd.SetCurrentHp(player.Creature, cur), "CreatureCmd.SetCurrentHp");
                return (true, "");
            }
            default: // potions_set
            {
                foreach (var p in player.PotionSlots.ToList())
                {
                    if (p != null) await AwaitSetup(PotionCmd.Discard(p), "PotionCmd.Discard");
                }

                var refused = new List<string>();
                foreach (var x in item["ids"] as JsonArray ?? new JsonArray())
                {
                    string id = ((string?)x ?? "").ToUpperInvariant();
                    var potion = ModelDb.AllPotions.FirstOrDefault(p => p.Id.Entry == id);
                    if (potion == null) return (false, "no potion " + id);
                    var res = await PotionCmd.TryToProcure(potion.ToMutable(), player);
                    if (!res.success) refused.Add(id);
                }

                return (true, refused.Count == 0 ? "" : "refused: " + string.Join(",", refused));
            }
        }
    }

    /// <summary>A deck the item could not build is a failed item: the fight would not start from the asked state.</summary>
    private static (bool ok, string msg) DeckResidualFails(string residual, string deckLine) => (residual == "", residual + deckLine);

    /// <summary>ActConsoleCmd's integer form: travel enabled, then RunManager.EnterAct(N - 1).</summary>
    private static async Task<(bool ok, string msg)> EnterActItem(JsonObject item, RunState state)
    {
        int n = (int?)item["index"] ?? 1;
        if (n < 1 || n > state.Acts.Count) return (false, $"no act {n} (the run has {state.Acts.Count})");
        if (state.CurrentActIndex == n - 1) return (true, "already in act " + state.Act.Id.Entry);
        NMapScreen.Instance?.SetTravelEnabled(enabled: true);
        await AwaitSetup(RunManager.Instance.EnterAct(n - 1), "RunManager.EnterAct");
        await Idle();
        return (state.CurrentActIndex == n - 1, "act " + state.Act.Id.Entry);
    }

    private static System.Reflection.MethodInfo? _setMaxPotionCount;

    /// <summary>The player_load item (see the file comment): a run's holdings through the game's save-load path.</summary>
    private static async Task<(bool ok, string msg)> PlayerLoad(JsonObject item, RunState state, Player player)
    {
        var cardInfo = JsonSerializationUtility.GetTypeInfo<SerializableCard>();
        var relicInfo = JsonSerializationUtility.GetTypeInfo<SerializableRelic>();
        var deck = new List<SerializableCard>();
        foreach (var x in item["deck"] as JsonArray ?? new JsonArray())
        {
            var c = JsonSerializer.Deserialize(x!.ToJsonString(), cardInfo);
            if (c?.Id == null || ModelDb.AllCards.All(m => m.Id != c.Id)) return (false, "no card " + x["id"]);
            deck.Add(c);
        }

        var relics = new List<SerializableRelic>();
        foreach (var x in item["relics"] as JsonArray ?? new JsonArray())
        {
            var r = JsonSerializer.Deserialize(x!.ToJsonString(), relicInfo);
            if (r?.Id == null || ModelDb.AllRelics.All(m => m.Id != r.Id)) return (false, "no relic " + x["id"]);
            relics.Add(r);
        }

        var potions = new List<PotionModel>();
        foreach (var x in item["potions"] as JsonArray ?? new JsonArray())
        {
            string id = ((string?)x ?? "").ToUpperInvariant();
            var p = ModelDb.AllPotions.FirstOrDefault(m => m.Id.Entry == id);
            if (p == null) return (false, "no potion " + id);
            potions.Add(p);
        }

        int maxHp = (int?)item["max_hp"] ?? player.Creature.MaxHp;
        int hp = Math.Min((int?)item["hp"] ?? player.Creature.CurrentHp, maxHp);
        int slots = (int?)item["max_potion_slots"] ?? player.MaxPotionCount;
        if (maxHp < 1 || hp < 1) return (false, "hp below 1");
        if (potions.Count > slots) return (false, $"{potions.Count} potions for {slots} slots");

        // Out, as SyncWithSerializedPlayer clears them (relics without AfterRemoved, cards out of the run's scope).
        foreach (var r in player.Relics.ToList()) player.RemoveRelicInternal(r);
        foreach (var c in player.Deck.Cards.ToList())
        {
            c.RemoveFromCurrentPile();
            c.RemoveFromState();
        }

        foreach (var p in player.PotionSlots.ToList())
        {
            if (p != null) player.DiscardPotionInternal(p);
        }

        // The scalars, as SyncWithSerializedPlayer sets them.
        player.Creature.SetMaxHpInternal(maxHp);
        player.Creature.SetCurrentHpInternal(hp);
        if (item["gold"] is JsonNode g) player.Gold = (int)g;
        _setMaxPotionCount ??= AccessTools.Method(typeof(Player), "SetMaxPotionCountInternal")
            ?? throw new MissingMethodException(nameof(Player), "SetMaxPotionCountInternal");
        _setMaxPotionCount.Invoke(player, new object[] { slots });

        // In, as LoadInventory populates them (PopulateDeck, PopulateRelics, LoadPotions; not silent, so the
        // top bar shows them).
        foreach (var c in deck) player.Deck.AddInternal(state.LoadCard(c, player));
        foreach (var r in relics)
        {
            var relic = RelicModel.FromSerializable(r);
            player.AddRelicInternal(relic);
            if (!relic.IsStackable)
            {
                player.RelicGrabBag.Remove(relic);
                state.SharedRelicGrabBag.Remove(relic);
            }
        }

        for (int i = 0; i < potions.Count; i++) player.AddPotionInternal(potions[i].ToMutable(), i);
        await Idle();

        // What the game holds now must be what was asked.
        var problems = new List<string>();
        var wantDeck = deck.Select(c => (c.Id!.Entry, c.CurrentUpgradeLevel, c.Enchantment?.Id?.Entry ?? "", c.Enchantment?.Amount ?? 0)).ToList();
        var haveDeck = player.Deck.Cards.Select(c => (c.Id.Entry, c.CurrentUpgradeLevel, c.Enchantment?.Id.Entry ?? "", c.Enchantment?.Amount ?? 0)).ToList();
        if (!wantDeck.SequenceEqual(haveDeck)) problems.Add($"deck {haveDeck.Count} cards, asked {wantDeck.Count}");
        if (!relics.Select(r => r.Id!.Entry).SequenceEqual(player.Relics.Select(r => r.Id.Entry))) problems.Add("relics differ");
        if (!potions.Select(p => p.Id.Entry).SequenceEqual(player.PotionSlots.Where(p => p != null).Select(p => p!.Id.Entry)))
            problems.Add("potions differ");
        if (player.Creature.MaxHp != maxHp || player.Creature.CurrentHp != hp) problems.Add($"hp {player.Creature.CurrentHp}/{player.Creature.MaxHp}");
        if (item["gold"] is JsonNode g2 && player.Gold != (int)g2) problems.Add("gold " + player.Gold);
        if (player.MaxPotionCount != slots) problems.Add("potion slots " + player.MaxPotionCount);
        return (problems.Count == 0, string.Join("; ", problems) + DeckLine(player)
            + " | relics: " + string.Join(",", player.Relics.Select(r => r.Id.Entry)));
    }

    /// <summary>The deck in order, for the log: EmuDump writes the same line, so a different order shows at the item.</summary>
    private static string DeckLine(Player player) => " | deck: " + string.Join(",", PileType.Deck.GetPile(player).Cards.Select(c =>
        c.Id.Entry + (c.IsUpgraded ? "+" : "") + (c.Enchantment != null ? "*" + c.Enchantment.Id.Entry + c.Enchantment.Amount : "")));

    private readonly record struct CardKey(string id, bool upgraded, string enchantment, int amount);

    private static CardKey DeckKey(JsonObject o) => new(((string?)o["id"] ?? "").ToUpperInvariant(), (bool?)o["upgraded"] == true,
        ((string?)o["enchantment"] ?? "").ToUpperInvariant(), (int?)o["amount"] ?? 0);

    private static CardKey DeckKey(CardModel c) => new(c.Id.Entry, c.IsUpgraded, c.Enchantment?.Id.Entry ?? "", c.Enchantment?.Amount ?? 0);

    /// <summary>Deck positions the wanted list does not hold, and the wanted cards the deck lacks (multisets).</summary>
    private static (List<int> extra, List<CardKey> missing) DeckDiff(List<CardKey> deck, List<CardKey> want)
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
}
