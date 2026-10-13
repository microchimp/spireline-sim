// Fight comparison, game side: choose a Neow option, walk into the first row-1 monster room, then
// play seeded random legal actions, recording each action and the state after it.
// Spec: {"kind": "combat", "seed", "character", "ascension", "neow": 0, "chooser_seed": 1, "max_actions": 80}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>The step whose action is running, for tagging choices; 0 before the fight.</summary>
    internal static int CurrentStep;

    private static async Task<JsonObject> PlayCombat(JsonObject spec)
    {
        CurrentStep = 0;
        var phase = System.Diagnostics.Stopwatch.StartNew();
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        ulong f0 = Engine.GetProcessFrames();
        double cpu0 = proc.TotalProcessorTime.TotalMilliseconds;
        int gc0 = GC.CollectionCount(2);
        JsonObject Cost() { proc.Refresh(); return new JsonObject {
            ["frames"] = (long)(Engine.GetProcessFrames() - f0),
            ["cpu_ms"] = Math.Round(proc.TotalProcessorTime.TotalMilliseconds - cpu0),
            ["gen2_gcs"] = GC.CollectionCount(2) - gc0 }; }
        var (state, dump) = await BeginRun(spec);
        var timing = new JsonObject { ["begin_run"] = phase.ElapsedMilliseconds };
        dump["phase_ms"] = timing;
        var run = RunManager.Instance;
        var steps = new JsonArray();
        var selector = new FirstChoiceSelector();
        using var selection = MegaCrit.Sts2.Core.Commands.CardSelectCmd.UseSelector(selector);
        dump["choices"] = selector.Log;
        ScreenLog = selector.Log;
        dump.Remove("save");
        dump["steps"] = steps;
        try
        {
            // Coverage: cards added to the deck before Neow, the way any card joins a deck
            // (CardPileCmd.Add to the Deck pile), so each fight exercises more than the starter.
            var added = new JsonArray();
            foreach (var id in spec["deck_add"]?.AsArray() ?? new JsonArray())
            {
                var canonical = ModelDb.AllCards.First(c => c.Id.Entry == (string)id!);
                var card = state.CreateCard(canonical, state.Players[0]);
                await MegaCrit.Sts2.Core.Commands.CardPileCmd.Add(card, MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck, skipVisuals: true);
                added.Add((string)id!);
            }

            dump["deck_add"] = added;
            await Idle();

            // Neow: record the offer, take the requested option.
            var neow = run.EventSynchronizer.GetLocalEvent();
            dump["neow_options"] = new JsonArray(neow.CurrentOptions
                .Select(o => (JsonNode)(o.Relic?.Id.Entry ?? o.TextKey)).ToArray());
            int pick = (int?)spec["neow"] ?? 0;
            GD.Print($"[referee] neow options {dump["neow_options"]!.ToJsonString()} finished={neow.IsFinished}; choosing {pick}");
            dump["neow_pick"] = pick;
            run.EventSynchronizer.ChooseLocalOption(pick);
            await Idle();
            await ClaimRewards(selector.Log);
            GD.Print($"[referee] neow chosen; finished={neow.IsFinished} options={neow.CurrentOptions.Count}");

            // First monster room on row 1, lowest column.
            var target = state.Map.GetPointsInRow(1).Where(p => p.PointType == MapPointType.Monster)
                .OrderBy(p => p.coord.col).First();
            timing["neow"] = phase.ElapsedMilliseconds;
            Heartbeat();
            dump["room"] = new JsonArray(target.coord.col, target.coord.row);
            GD.Print($"[referee] entering {target.coord.col},{target.coord.row}");
            await WithTimeout(run.EnterMapCoord(target.coord), 60, "EnterMapCoord");
            await Idle();
            GD.Print($"[referee] in room; combat={CombatManager.Instance.IsInProgress}");

            timing["enter_room"] = phase.ElapsedMilliseconds;
            dump["cost_setup"] = Cost();
            var rng = new Random((int?)spec["chooser_seed"] ?? 1);
            int maxActions = (int?)spec["max_actions"] ?? 80;
            Player player = state.Players[0];
            steps.Add(new JsonObject { ["action"] = null, ["state"] = Snapshot(state, player) });
            for (int i = 0; i < maxActions && CombatManager.Instance.IsInProgress; i++)
            {
                CurrentStep = i + 1;  // choices this action opens are logged against its step
                var combat = CombatManager.Instance.DebugOnlyGetState()!;
                var enemies = combat.Enemies.Where(e => e.IsAlive).ToList();
                var options = new List<(int hand, int target)> { (-1, -1) };  // -1: end turn
                var hand = player.PlayerCombatState!.Hand.Cards;
                for (int h = 0; h < hand.Count; h++)
                {
                    if (!hand[h].CanPlay()) continue;
                    // Only AnyEnemy asks the player for a target. RandomEnemy and AllEnemies pick
                    // (or hit) their own; passing them a target would make the play fail.
                    if (hand[h].TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                    {
                        for (int t = 0; t < enemies.Count; t++) options.Add((h, t));
                    }
                    else
                    {
                        options.Add((h, -1));
                    }
                }

                var (hi, ti) = options[rng.Next(options.Count)];
                JsonObject action;
                if (hi < 0)
                {
                    action = new JsonObject { ["kind"] = "end_turn" };
                    run.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, combat.RoundNumber));
                }
                else
                {
                    var card = hand[hi];
                    Creature? tgt = ti >= 0 ? enemies[ti] : null;
                    action = new JsonObject
                    {
                        ["kind"] = "play", ["hand_index"] = hi, ["card"] = card.Id.Entry,
                        ["target"] = ti, ["target_slot"] = tgt != null ? combat.Enemies.ToList().IndexOf(tgt) : -1,
                    };
                    // A refused play is still a step: the emulator must refuse it the same way.
                    action["played"] = card.TryManualPlay(tgt);
                }

                if (!await Idle())
                {
                    // The game never settled (a screen or loop the driver cannot answer). Keep what
                    // was recorded and end the fight here instead of letting the watchdog kill the copy.
                    dump["error"] = $"idle_timeout after step {i}: {action.ToJsonString()}";
                    break;
                }
                steps.Add(new JsonObject { ["action"] = action, ["state"] = Snapshot(state, player) });
                Heartbeat();
                GD.Print($"[referee] step {i}: {action.ToJsonString()}");
            }
        }
        catch (Exception ex)
        {
            dump["error"] = ex.ToString();
        }

        timing["fight_end"] = phase.ElapsedMilliseconds;
        dump["cost"] = Cost();
        return dump;
    }

    private static async Task WithTimeout(Task task, double seconds, string what, Action? eachFrame = null)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (!task.IsCompleted && clock.Elapsed.TotalSeconds < seconds)
        {
            eachFrame?.Invoke();
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        if (!task.IsCompleted)
        {
            throw new TimeoutException($"{what} did not finish within {seconds} s");
        }

        await task;
    }

    /// <summary>Frames until no action runs and, in combat, the player may act or the fight is over.</summary>
    /// <summary>
    /// Quiet frames needed before the game counts as idle: no action running, the queue empty,
    /// and (in a fight) the player's turn with actions enabled. In Instant fast mode the game
    /// does no timed waits, so a few frames suffice; a job may raise it ("idle_frames").
    /// </summary>
    internal static double IdleFrames = 6;

    /// <returns>false if the game did not settle within maxSeconds (below the heartbeat watchdog's stall_s).</returns>
    private static async Task<bool> Idle(double maxSeconds = 10)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        string last = "";
        int stable = 0;
        // Idle = nothing running AND the visible state unchanged for IdleFrames frames in a row.
        // Quiet frames alone are not enough: even in Instant mode cards are drawn over several
        // frames outside the action queue.
        while (clock.Elapsed.TotalSeconds < maxSeconds && stable < (int)IdleFrames)
        {
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            AnswerOpenScreens();
            var cm = CombatManager.Instance;
            var combat = cm.DebugOnlyGetState();
            bool busy = RunManager.Instance.ActionExecutor.IsRunning
                || !RunManager.Instance.ActionQueueSet.IsEmpty
                || (cm.IsInProgress && (cm.PlayerActionsDisabled || combat?.CurrentSide != CombatSide.Player));
            string sig = busy ? "busy" : Signature(combat);
            stable = !busy && sig == last ? stable + 1 : 0;
            last = sig;
        }

        return stable >= (int)IdleFrames;
    }

    private static string Signature(CombatState? combat)
    {
        if (combat == null) return "none";
        var sb = new System.Text.StringBuilder();
        foreach (var p in combat.Players)
        {
            var pcs = p.PlayerCombatState;
            sb.Append(pcs?.Hand.Cards.Count).Append('/').Append(pcs?.DrawPile.Cards.Count).Append('/')
              .Append(pcs?.DiscardPile.Cards.Count).Append('/').Append(pcs?.ExhaustPile.Cards.Count).Append('/')
              .Append(pcs?.Energy).Append('/').Append(p.Creature.CurrentHp).Append('/').Append(p.Creature.Block).Append(';');
        }

        foreach (var e in combat.Enemies) sb.Append(e.CurrentHp).Append(',').Append(e.Block).Append(',').Append(e.Powers.Count).Append(';');
        sb.Append(combat.RoundNumber);
        return sb.ToString();
    }

    private static JsonObject Snapshot(RunState run, Player player)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var pcs = player.PlayerCombatState;
        static JsonArray Cards(IEnumerable<CardModel>? cards) => new((cards ?? Enumerable.Empty<CardModel>())
            .Select(c => (JsonNode)new JsonArray(c.Id.Entry, c.IsUpgraded)).ToArray());
        static JsonArray Powers(Creature c) => new(c.Powers.Select(p => (JsonNode)new JsonArray(p.Id.Entry, p.Amount)).ToArray());
        var me = player.Creature;
        return new JsonObject
        {
            ["in_combat"] = CombatManager.Instance.IsInProgress,
            ["round"] = combat?.RoundNumber,
            ["player"] = new JsonObject
            {
                ["hp"] = me.CurrentHp, ["max_hp"] = me.MaxHp, ["block"] = me.Block,
                ["energy"] = pcs?.Energy, ["powers"] = Powers(me),
            },
            ["hand"] = Cards(pcs?.Hand.Cards),
            ["draw"] = Cards(pcs?.DrawPile.Cards),
            ["discard"] = Cards(pcs?.DiscardPile.Cards),
            ["exhaust"] = Cards(pcs?.ExhaustPile.Cards),
            ["enemies"] = new JsonArray((combat?.Enemies ?? Array.Empty<Creature>()).Select(e => (JsonNode)new JsonObject
            {
                ["name"] = e.Monster?.Id.Entry, ["hp"] = e.CurrentHp, ["max_hp"] = e.MaxHp, ["block"] = e.Block,
                ["powers"] = Powers(e),
            }).ToArray()),
            ["rng"] = Counters(run.Rng),
        };
    }
}
