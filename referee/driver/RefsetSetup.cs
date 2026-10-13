// Reference-set setup items (referee/refset.py): put a real run's state in place before its fight, through
// the game commands the developers' console and the game's own events call. They complement Setup.cs's
// items and keep its rule: only commands that SET UP a state, never one that changes or skips a rule.
//
//   {"cmd": "relics_set", "ids": [...]}       RelicCmd.Obtain each listed relic not held (in list order,
//                                              answering any screen it opens), then RelicCmd.Remove each
//                                              held relic the list does not name (RelicConsoleCmd's remove).
//   {"cmd": "deck_set", "cards": [{"id", "upgraded", "enchantment", "amount"}, ...]}
//                                              CardPileCmd.RemoveFromDeck each deck card the list does not
//                                              hold (RemoveCardConsoleCmd), then for each missing card:
//                                              CreateCard, CardCmd.Upgrade, CardPileCmd.Add to the deck and
//                                              CardCmd.Enchant (EnchantConsoleCmd). Up to three passes, since
//                                              an add can trigger a relic (eggs upgrade, Bing Bong clones);
//                                              what still differs is reported in the item's message.
//   {"cmd": "hp", "max": M, "current": C}     CreatureCmd.SetMaxHp, then CreatureCmd.SetCurrentHp (events' own).
//   {"cmd": "potions_set", "ids": [...]}      PotionCmd.Discard every held potion, then PotionCmd.TryToProcure
//                                              each listed one (PotionConsoleCmd).
// relics_set and deck_set end their message with the deck in order (" | deck: A,B+,C*SHARP1"), which EmuDump
// writes the same way, so a deck the emulator orders or builds differently shows at the item, not at step 0.
// All four are map items (before "fight"). EmuDump mirrors them in tools/emudump/RefsetSetup.cs.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>A reference-set item, or null when <paramref name="item" /> is not one.</summary>
    private static async Task<(bool ok, string msg)?> RefsetItem(JsonObject item, RunState state, Player player)
    {
        string cmd = (string?)item["cmd"] ?? "";
        if (cmd is not ("relics_set" or "deck_set" or "hp" or "potions_set")) return null;
        if (CombatManager.Instance.IsInProgress) return (false, cmd + ": map only");
        switch (cmd)
        {
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

                return (true, residual + DeckLine(player));
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
