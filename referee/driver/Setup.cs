// Setup fights: a fight spec's "setup" list, run through the game commands the developers' own
// console commands call (MegaCrit.Sts2.Core.DevConsole.ConsoleCommands), so every state it builds is
// one the game's code produces. Owner's rule: only commands that SET UP a situation (which fight,
// cards incl. upgraded, relics, potions, powers, energy); never one that changes or skips a rule
// (godmode, kill, win, damage/heal shortcuts), so those are refused here.
//
// Spec: {"kind": "combat", ..., "setup": [{"cmd": "relic", "id": "VAJRA"}, {"cmd": "potion", "id": "FIRE_POTION"},
//        {"cmd": "card", "id": "BASH", "upgraded": true}, {"cmd": "fight", "id": "<ENCOUNTER>"},
//        {"cmd": "power", "id": "VULNERABLE_POWER", "amount": 1, "target": "enemy:0"}],
//        "use_potions": true, "trace_setup": false, "script": null}
// Items before "fight" run on the map after Neow and its rewards; items after it run once the fight's
// room has settled (turn 1, hand drawn), before the first player action. Each item is logged with the
// run's RNG counters; a choice an item opens is tagged with step -(i+1).
// "script": a list of recorded actions played instead of random ones (the shrinker's reproduction).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>Whether this fight records the setup extras (potions and relics in each snapshot).</summary>
    private static bool _extended;

    /// <summary>A spec that uses any of the setup features; an old spec stays byte-identical.</summary>
    private static bool Extended(JsonObject spec) =>
        spec["setup"] is JsonArray || (bool?)spec["use_potions"] == true || spec["script"] is JsonArray;

    private static JsonArray SetupItems(JsonObject spec) => spec["setup"] as JsonArray ?? new JsonArray();

    private static int FightItem(JsonArray items)
    {
        for (int i = 0; i < items.Count; i++)
        {
            if ((string?)items[i]?["cmd"] == "fight") return i;
        }

        return -1;
    }

    /// <summary>Runs items [from, to) in order and logs each one.</summary>
    private static async Task RunSetup(JsonArray items, int from, int to, RunState state, Player player, JsonArray log, bool trace)
    {
        if (from == 0 && items.Count > 0)
        {
            // The baseline: where the run stands after Neow, before any item.
            log.Add(new JsonObject { ["i"] = -1, ["cmd"] = "before", ["ok"] = true, ["msg"] = "",
                ["rng"] = Counters(state.Rng), ["player_rng"] = Counters(player.PlayerRng) });
        }

        for (int i = from; i < to; i++)
        {
            var item = items[i]!.AsObject();
            CurrentStep = -(i + 1);  // a choice this item opens is logged against it
            string cmd = (string?)item["cmd"] ?? "";
            (bool ok, string msg) r;
            try
            {
                r = await SetupItem(item, state, player);
            }
            catch (Exception ex)
            {
                r = (false, ex.GetType().Name + ": " + ex.Message);
            }

            bool idle = await Idle();
            if (!CombatManager.Instance.IsInProgress && TopRewardsSet() != null)
            {
                await ClaimRewards(ScreenLog!);
            }

            var entry = new JsonObject
            {
                ["i"] = i, ["cmd"] = cmd, ["id"] = item["id"]?.DeepClone(), ["ok"] = r.ok, ["msg"] = r.msg,
                ["rng"] = Counters(state.Rng), ["player_rng"] = Counters(player.PlayerRng),
            };
            if (!idle) entry["idle_timeout"] = true;
            if (trace && CombatManager.Instance.IsInProgress) entry["state"] = Snapshot(state, player);
            log.Add(entry);
            GD.Print($"[referee] setup {i}: {cmd} {item["id"]} ok={r.ok} {r.msg}");
        }

        CurrentStep = 0;
    }

    /// <summary>Waits for a setup command, answering screens it opens (rewards on the map, bundles).</summary>
    private static async Task AwaitSetup(Task task, string what)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (!task.IsCompleted && clock.Elapsed.TotalSeconds < 30)
        {
            AnswerOpenScreens();
            if (!CombatManager.Instance.IsInProgress && TopRewardsSet() != null)
            {
                await ClaimRewards(ScreenLog!);
                continue;
            }

            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        if (!task.IsCompleted) throw new TimeoutException(what + " did not finish within 30 s");
        await task;
    }

    private static async Task<(bool ok, string msg)> SetupItem(JsonObject item, RunState state, Player player)
    {
        string cmd = (string?)item["cmd"] ?? "";
        string id = ((string?)item["id"] ?? "").ToUpperInvariant();
        bool inCombat = CombatManager.Instance.IsInProgress;
        var combat = CombatManager.Instance.DebugOnlyGetState();
        switch (cmd)
        {
            case "relic":
            {
                // RelicConsoleCmd: RelicCmd.Obtain(relic.ToMutable(), player), here awaited and by exact id.
                if (inCombat) return (false, "relic: map only");
                var relic = ModelDb.AllRelics.FirstOrDefault(r => r.Id.Entry == id);
                if (relic == null) return (false, "no relic " + id);
                await AwaitSetup(RelicCmd.Obtain(relic.ToMutable(), player), "RelicCmd.Obtain");
                return (true, "");
            }
            case "potion":
            {
                // PotionConsoleCmd: PotionCmd.TryToProcure (Sozu refuses, a full belt fails).
                var potion = ModelDb.AllPotions.FirstOrDefault(p => p.Id.Entry == id);
                if (potion == null) return (false, "no potion " + id);
                var res = await PotionCmd.TryToProcure(potion.ToMutable(), player);
                return (res.success, res.success ? "" : res.failureReason.ToString());
            }
            case "card":
            {
                // CardConsoleCmd: CreateCard in the pile's scope, then CardPileCmd.Add. An upgraded card
                // is upgraded (CardCmd.Upgrade) before it joins the pile, as a reward card is made.
                if (!Enum.TryParse<PileType>((string?)item["pile"] ?? "Deck", true, out var pile) || pile is PileType.None or PileType.Play)
                    return (false, "bad pile " + item["pile"]);
                if (pile.IsCombatPile() && !inCombat) return (false, "combat pile outside a fight");
                if (pile == PileType.Hand && PileType.Hand.GetPile(player).Cards.Count >= CardPile.MaxCardsInHand) return (false, "hand full");
                var canonical = ModelDb.AllCards.FirstOrDefault(c => c.Id.Entry == id);
                if (canonical == null) return (false, "no card " + id);
                ICardScope scope = pile.IsCombatPile() ? combat! : state;
                var card = scope.CreateCard(canonical, player);
                if ((bool?)item["upgraded"] == true)
                {
                    if (!card.IsUpgradable) return (false, "not upgradable");
                    CardCmd.Upgrade(card);
                }

                var pos = (string?)item["pos"] == "top" ? CardPilePosition.Top : CardPilePosition.Bottom;
                await AwaitSetup(CardPileCmd.Add(card, pile, pos, skipVisuals: true), "CardPileCmd.Add");
                return (true, "");
            }
            case "upgrade":
            {
                // UpgradeCardConsoleCmd: CardCmd.Upgrade of a hand card, here named by id.
                if (!inCombat) return (false, "upgrade: combat only");
                var card = PileType.Hand.GetPile(player).Cards.FirstOrDefault(c => c.Id.Entry == id && c.IsUpgradable);
                if (card == null) return (false, "no upgradable " + id + " in hand");
                CardCmd.Upgrade(card);
                return (true, "");
            }
            case "power":
            {
                // ApplyPowerConsoleCmd: PowerCmd.ModifyAmount on a power the creature already has (not
                // instanced), else PowerCmd.Apply; no applier and no source card either way.
                if (!inCombat || combat == null) return (false, "power: combat only");
                var power = ModelDb.AllPowers.FirstOrDefault(p => p.Id.Entry == id);
                if (power == null) return (false, "no power " + id);
                int amount = (int?)item["amount"] ?? 1;
                Creature? target = Target(combat, player, (string?)item["target"] ?? "player");
                if (target == null || !target.IsAlive) return (false, "no living target " + item["target"]);
                var existing = target.Powers.FirstOrDefault(p => p.GetType() == power.GetType());
                var ctx = new BlockingPlayerChoiceContext();
                Task task = power.InstanceType == PowerInstanceType.None && existing != null
                    ? PowerCmd.ModifyAmount(ctx, existing, amount, null, null)
                    : PowerCmd.Apply(ctx, power.ToMutable(), target, amount, null, null);
                await AwaitSetup(task, "PowerCmd");
                return (true, existing != null ? "modified" : "applied");
            }
            case "energy":
            {
                // EnergyConsoleCmd: PlayerCmd.GainEnergy.
                if (!inCombat) return (false, "energy: combat only");
                await AwaitSetup(PlayerCmd.GainEnergy((int?)item["amount"] ?? 1, player), "PlayerCmd.GainEnergy");
                return (true, "");
            }
            default:
                return (false, "not a setup command: " + cmd);
        }
    }

    /// <summary>"player", or "enemy:&lt;slot&gt;" (the slot in the combat's enemy list, dead or alive).</summary>
    private static Creature? Target(CombatState combat, Player player, string name)
    {
        if (name == "player") return player.Creature;
        if (name.StartsWith("enemy:") && int.TryParse(name[6..], out int slot))
        {
            var enemies = combat.Enemies.ToList();
            return slot >= 0 && slot < enemies.Count ? enemies[slot] : null;
        }

        return null;
    }

    /// <summary>
    /// FightConsoleCmd's room entry with two differences that make it a run's own room: the encounter's
    /// Rng is NOT randomized from the clock (DebugRandomizeRng), so it seeds from the run seed, the floor
    /// and its id as in any run; and it enters at the row-1 map coordinate, setting ActFloor and pausing
    /// the executor as EnterMapPointInternal does (EnterMapCoordDebug records the visited coordinate).
    /// Returns where the game lists the encounter (act, kind, position), for EmuDump to name the same one.
    /// </summary>
    private static async Task<JsonObject> EnterFight(JsonObject item, RunState state, MapPoint target)
    {
        string id = ((string?)item["id"] ?? "").ToUpperInvariant();
        var canonical = ModelDb.AllEncounters.FirstOrDefault(e => e.Id.Entry == id)
            ?? throw new InvalidOperationException("no encounter " + id);
        var info = new JsonObject { ["id"] = id, ["room_type"] = canonical.RoomType.ToString() };
        foreach (ActModel act in ModelDb.ActsByIndex.SelectMany(a => a))
        {
            foreach (var (kind, list) in new[] { ("weak", act.AllWeakEncounters), ("normal", act.AllRegularEncounters),
                         ("elite", act.AllEliteEncounters), ("boss", act.AllBossEncounters) })
            {
                int at = list.Select(e => e.Id.Entry).ToList().IndexOf(id);
                if (at < 0 || info["act"] != null) continue;
                info["act"] = act.Id.Entry;
                info["kind"] = kind;
                info["index"] = at;
            }
        }

        info["current_act"] = state.Act.Id.Entry;
        var enc = canonical.ToMutable();
        state.ActFloor = target.coord.row + 1;
        RunManager.Instance.ActionExecutor.Pause();
        await WithTimeout(RunManager.Instance.EnterMapCoordDebug(target.coord, enc.RoomType, MapPointType.Unassigned, enc), 60, "EnterMapCoordDebug");
        return info;
    }

    /// <summary>NPotionPopup's use button: enabled for an AnyTime or CombatOnly potion on the player's turn.</summary>
    private static bool PotionUsable(PotionModel p, Player player) =>
        !p.IsQueued && player.Creature.IsAlive && player.CanRemovePotions && p.PassesCustomUsabilityCheck
        && p.Usage is PotionUsage.AnyTime or PotionUsage.CombatOnly;

    /// <summary>Potion options for the random chooser, as (-2 - slot, target), after the card options.</summary>
    private static void AddPotionOptions(List<(int hand, int target)> options, Player player, List<Creature> enemies)
    {
        for (int s = 0; s < player.PotionSlots.Count; s++)
        {
            var p = player.PotionSlots[s];
            if (p == null || !PotionUsable(p, player)) continue;
            if (p.TargetType == TargetType.AnyEnemy)
            {
                for (int t = 0; t < enemies.Count; t++) options.Add((-2 - s, t));
            }
            else
            {
                options.Add((-2 - s, -1));
            }
        }
    }

    /// <summary>The player's own potion-use path (PotionModel.EnqueueManualUse), as the popup's use button.</summary>
    private static JsonObject UsePotion(int slot, Creature? tgt, int ti, CombatState combat, Player player)
    {
        var p = player.PotionSlots[slot]!;
        var action = new JsonObject
        {
            ["kind"] = "potion", ["slot"] = slot, ["potion"] = p.Id.Entry, ["target"] = ti,
            ["target_slot"] = tgt != null ? combat.Enemies.ToList().IndexOf(tgt) : -1,
        };
        p.EnqueueManualUse(tgt);
        return action;
    }

    /// <summary>
    /// One recorded action to replay (script mode): the card by id, at its recorded hand index if the card
    /// is still there, else the first playable copy; the potion likewise by slot then id. Null and a reason
    /// when the action is not legal now (the shrinker counts such a subset as unresolved).
    /// </summary>
    private static (int hand, int target)? Scripted(JsonObject want, IReadOnlyList<CardModel> hand, List<Creature> enemies,
        CombatState combat, Player player, out string? why)
    {
        why = null;
        string kind = (string?)want["kind"] ?? "";
        int targetSlot = (int?)want["target_slot"] ?? -1;
        int? EnemyIndex(TargetType tt)
        {
            if (tt != TargetType.AnyEnemy) return -1;
            var all = combat.Enemies.ToList();
            if (targetSlot < 0 || targetSlot >= all.Count || !all[targetSlot].IsAlive) return null;
            return enemies.IndexOf(all[targetSlot]);
        }

        if (kind == "end_turn") return (-1, -1);
        if (kind == "play")
        {
            string card = (string?)want["card"] ?? "";
            int at = (int?)want["hand_index"] ?? -1;
            int h = at >= 0 && at < hand.Count && hand[at].Id.Entry == card && hand[at].CanPlay() ? at
                : Enumerable.Range(0, hand.Count).FirstOrDefault(i => hand[i].Id.Entry == card && hand[i].CanPlay(), -1);
            if (h < 0) { why = card + " not playable from hand"; return null; }
            int? t = EnemyIndex(hand[h].TargetType);
            if (t == null) { why = "target slot " + targetSlot + " not alive"; return null; }
            return (h, t.Value);
        }

        if (kind == "potion")
        {
            string potion = (string?)want["potion"] ?? "";
            int at = (int?)want["slot"] ?? -1;
            bool Ok(int s) => s >= 0 && s < player.PotionSlots.Count && player.PotionSlots[s] is { } p && p.Id.Entry == potion && PotionUsable(p, player);
            int slot = Ok(at) ? at : Enumerable.Range(0, player.PotionSlots.Count).FirstOrDefault(Ok, -1);
            if (slot < 0) { why = potion + " not usable"; return null; }
            int? t = EnemyIndex(player.PotionSlots[slot]!.TargetType);
            if (t == null) { why = "target slot " + targetSlot + " not alive"; return null; }
            return (-2 - slot, t.Value);
        }

        why = "unknown action kind " + kind;
        return null;
    }

    /// <summary>The setup extras of a snapshot: the potion belt by slot and the relics in order.</summary>
    private static void Decorate(JsonObject snap, Player player)
    {
        snap["potions"] = new JsonArray(player.PotionSlots.Select(p => p == null ? null : (JsonNode)p.Id.Entry).ToArray());
        snap["relics"] = new JsonArray(player.Relics.Select(r => (JsonNode)r.Id.Entry).ToArray());
    }
}
