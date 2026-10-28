// Whole-run referee, game side: one seeded run played from Neow onwards with random legal choices
// on every screen, through the game's own synchronizers and commands (never field writes), and the
// run state (the game's own save, RunManager.ToSave) recorded after every decision.
// Spec: {"kind": "run", "seed", "character", "ascension", "chooser_seed": 1,
//        "max_decisions": 1500, "max_actions": 400 (per fight), "buff_max_hp": 0,
//        "avoid": ["Unknown", "Shop"] (map point types not chosen while another is offered),
//        "stop_after_act": 3, "card_skip": 0.2}
// Output: "decisions": [{"i", "screen", "floor", "offer"?, "choice", "selections"?, "fight"?, "snap"}],
// "maps": [act map per act], "end": why the run stopped.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>The run being played, for the selector to log its choices into the open decision.</summary>
    private sealed class RunLog
    {
        public readonly JsonArray Decisions = new();
        public JsonObject? Open;
        public readonly Random Rng;
        public readonly RunState State;
        public RunLog(RunState state, int seed) { State = state; Rng = new Random(seed); }

        public JsonObject Begin(string screen)
        {
            Open = new JsonObject { ["i"] = Decisions.Count, ["screen"] = screen, ["floor"] = State.TotalFloor };
            Decisions.Add(Open);
            return Open;
        }

        /// <summary>Close the open decision with the run state after it.</summary>
        public void End()
        {
            if (Open == null) return;
            Open["snap"] = RunSnap(State);
            GD.Print($"[referee] run decision {Open["i"]} {Open["screen"]} {Open["choice"]?.ToJsonString()}");
            Open = null;
            Heartbeat();
        }
    }

    /// <summary>
    /// The game's save of the run, minus what is not run state (timestamps, drawings, unlocks,
    /// discovered lists) and minus the act map (recorded once per act in "maps").
    /// </summary>
    private static JsonNode RunSnap(RunState state)
    {
        var save = JsonNode.Parse(SaveManager.ToJson(RunManager.Instance.ToSave(null)))!.AsObject();
        foreach (var k in new[] { "save_time", "start_time", "run_time", "win_time", "map_drawings", "platform_type", "schema_version" })
            save.Remove(k);
        foreach (var a in save["acts"]!.AsArray()) a!.AsObject().Remove("saved_map");
        foreach (var p in save["players"]!.AsArray())
            foreach (var k in new[] { "unlock_state", "discovered_cards", "discovered_enemies", "discovered_epochs", "discovered_potions", "discovered_relics" })
                p!.AsObject().Remove(k);
        // Per floor only what was entered; the stats are the run history screen's.
        var hist = new JsonArray();
        foreach (var act in save["map_point_history"]?.AsArray() ?? new JsonArray())
            hist.Add(new JsonArray(act!.AsArray().Select(e => (JsonNode)new JsonObject
            {
                ["type"] = e!["map_point_type"]?.DeepClone(),
                ["rooms"] = new JsonArray(e["rooms"]!.AsArray().Select(r => (JsonNode)new JsonObject
                {
                    ["type"] = r!["room_type"]?.DeepClone(), ["id"] = r["model_id"]?.DeepClone(),
                }).ToArray()),
            }).ToArray()));
        save["map_point_history"] = hist;
        return save;
    }

    /// <summary>
    /// Card choices outside a fight are seeded random and logged into the open decision with each
    /// card's deck index; inside a fight they are the first options, as FirstChoiceSelector answers
    /// a fight spec, logged with the step in the fight's own choice list.
    /// </summary>
    private sealed class RunSelector : ICardSelector
    {
        private readonly RunLog _run;
        private readonly FirstChoiceSelector _fight = new();
        public JsonArray FightLog => _fight.Log;
        public RunSelector(RunLog run) { _run = run; }

        public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
        {
            if (CombatManager.Instance.IsInProgress) return _fight.GetSelectedCards(options, minSelect, maxSelect);
            var list = options.ToList();
            int n = Math.Min(list.Count, Math.Max(minSelect, Math.Min(1, maxSelect)));
            var idx = Env.DecideMany("select_cards", list.Select(c => (JsonNode)CardJson(c)).ToList(), n,
                () => Enumerable.Range(0, list.Count).OrderBy(_ => _run.Rng.Next()).Take(n).OrderBy(i => i).ToList(),
                allowNone: Env.Extras && minSelect <= 0 && n > 0);
            var deck = _run.State.Players[0].Deck.Cards.ToList();
            Log(new JsonObject
            {
                ["kind"] = "cards",
                ["options"] = new JsonArray(list.Select(c => (JsonNode)CardJson(c)).ToArray()),
                ["min"] = minSelect, ["max"] = maxSelect,
                ["picked"] = new JsonArray(idx.Select(i => (JsonNode)i).ToArray()),
                ["deck_index"] = new JsonArray(idx.Select(i => (JsonNode)deck.IndexOf(list[i])).ToArray()),
            });
            return Task.FromResult<IEnumerable<CardModel>>(idx.Select(i => list[i]).ToList());
        }

        public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        {
            if (CombatManager.Instance.IsInProgress) return _fight.GetSelectedCardReward(options, alternatives);
            // With extra_actions the screen's alternatives (Skip, Reroll, a relic's own) follow the cards as actions of their own.
            var alts = Env.Extras ? alternatives : Array.Empty<CardRewardAlternative>();
            int pick = options.Count == 0 ? -1 : Env.Decide("card_reward", options.Count + alts.Count,
                () => CardRewardActions(options.Select(o => o.Card).ToList(), alts), () => _run.Rng.Next(options.Count));
            Log(new JsonObject
            {
                ["kind"] = "card_reward",
                ["options"] = new JsonArray(options.Select(o => (JsonNode)CardJson(o.Card)).ToArray()),
                ["picked"] = pick,
            });
            if (pick >= options.Count && options.Count > 0) return new CardRewardSelection { card = null, alternative = alternatives[pick - options.Count] };
            return pick < 0 ? default : new CardRewardSelection { card = options[pick].Card, alternative = null };
        }

        private void Log(JsonObject o)
        {
            var d = _run.Open ?? _run.Begin("selection");
            (d["selections"] ??= new JsonArray()).AsArray().Add(o);
        }
    }

    private static JsonArray CardJson(CardModel c) => new(c.Id.Entry, c.IsUpgraded ? 1 : 0);

    /// <summary>A card reward screen's actions: each card, then each alternative ("skip", or "alt:" and its id in lower case).</summary>
    private static JsonArray CardRewardActions(IReadOnlyList<CardModel> cards, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        var a = new JsonArray(cards.Select(c => (JsonNode)CardJson(c)).ToArray());
        foreach (var alt in alternatives)
            a.Add(alt.OptionId.Equals("Skip", StringComparison.OrdinalIgnoreCase) ? "skip" : "alt:" + alt.OptionId.ToLowerInvariant());
        return a;
    }

    private static async Task<JsonObject> PlayRun(JsonObject spec)
    {
        var (state, dump) = await BeginRun(spec);
        var run = RunManager.Instance;
        var player = state.Players[0];
        var log = new RunLog(state, (int?)spec["chooser_seed"] ?? 1);
        var selector = new RunSelector(log);
        Env.Run = state;  // Env.cs: decision points ask the agent (or trace) only inside a run
        Env.Extras = (bool?)spec["extra_actions"] == true;
        Env.AnchorState = (bool?)spec["anchor_state"] == true;  // Env.cs: the lockstep emulator side's hidden state
        Env.AnchorPrevKind = null;
        Env.FullSelect = (bool?)spec["full_select"] == true;
        Env.AltChecked = Env.AltMismatch = 0;
        Env.Trace = Env.Tracing ? new JsonArray() : null;
        FirstChoiceSelector.PickHook = Env.Live || Env.Tracing ? FightPick : null;
        FirstChoiceSelector.CardRewardHook = Env.Live || Env.Tracing ? FightCardReward : null;
        using var selection = CardSelectCmd.UseSelector(selector);
        ScreenLog = selector.FightLog;
        int maxDecisions = (int?)spec["max_decisions"] ?? 1500;
        int stopAfterAct = (int?)spec["stop_after_act"] ?? 3;
        var avoid = (spec["avoid"]?.AsArray() ?? new JsonArray()).Select(x => (string)x!).ToHashSet();
        dump.Remove("save");
        dump["kind"] = "run";
        dump["chooser_seed"] = (int?)spec["chooser_seed"] ?? 1;
        dump["decisions"] = log.Decisions;
        var maps = new JsonArray { Map(state) };
        dump["maps"] = maps;
        string end = "max_decisions";
        try
        {
            log.Begin("start");
            log.End();
            await Idle();

            // Neow: the offer, a random option, then its rewards.
            var neow = run.EventSynchronizer.GetLocalEvent();
            await PlayEvent(log, neow, "neow", spec);

            int buff = (int?)spec["buff_max_hp"] ?? 0;
            if (buff > 0)
            {
                // Recorded and mirrored by the emulator (RunNonCombatEffects.GainMaxHp), so random
                // play reaches later floors. The game's own command, so its hooks run.
                var d = log.Begin("buff");
                d["choice"] = buff;
                await CreatureCmd.GainMaxHp(player.Creature, buff);
                await Idle();
                log.End();
            }

            while (log.Decisions.Count < maxDecisions)
            {
                if (player.Creature.IsDead) { end = "death"; break; }
                var options = TravelOptions(state);
                if (options.Count == 0) { end = "no_map_options"; break; }
                var preferred = options.Where(p => !avoid.Contains(p.PointType.ToString())).ToList();
                var pickFrom = preferred.Count > 0 ? preferred : options;
                var target = options[await DecideP("map", options.Count,
                    () => new JsonArray(options.Select(p => (JsonNode)new JsonArray(p.coord.col, p.coord.row, p.PointType.ToString())).ToArray()),
                    () => options.IndexOf(pickFrom[log.Rng.Next(pickFrom.Count)]))];
                var md = log.Begin("map");
                md["offer"] = new JsonArray(options.Select(p => (JsonNode)new JsonArray(p.coord.col, p.coord.row, p.PointType.ToString())).ToArray());
                md["choice"] = new JsonArray(target.coord.col, target.coord.row);
                int act = state.CurrentActIndex;
                // PunchOff.AfterEventStarted starts an endless punching animation whose waits are
                // Cmd.Wait, which returns at once in Instant mode: the loop then never yields and the
                // game freezes. Only an Unknown point can hold it, so those are entered in the
                // player's Fast mode, kept while that event's room is up (its loop ends when the
                // room is left or the fight begins).
                if (target.PointType == MapPointType.Unknown) SetFastMode(MegaCrit.Sts2.Core.Settings.FastModeType.Fast);
                await WithTimeout(run.EnterMapCoord(target.coord), 60, "EnterMapCoord");
                if (!IsPunchOff(state)) SetFastMode(MegaCrit.Sts2.Core.Settings.FastModeType.Instant);
                await Idle();
                var room = state.CurrentRoom;
                md["room"] = new JsonArray(room?.RoomType.ToString(), room?.ModelId?.Entry);
                Task? chest = null;
                if (room is TreasureRoom)
                {
                    // Opening the chest is part of walking in: NTreasureRoom.OpenChest, which the
                    // chest's click runs (gold, any extra rewards, then the relic collection that
                    // awards a picked relic). Extra rewards are claimed while it waits on them.
                    chest = OpenChest();
                    var wait = new Budget(20);
                    while (!chest.IsCompleted && !wait.Spent
                           && !(TreasureChestOpened() && RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics != null))
                    {
                        if (TopRewardsSet() != null) await TakeRewards(log, takeAll: false);
                        await IdleBeat();
                    }

                    md["treasure_relic"] = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics?.FirstOrDefault()?.Id.Entry;
                    if (chest.IsFaulted) md["chest_error"] = chest.Exception?.ToString();
                }

                log.End();

                string? stop = room switch
                {
                    CombatRoom c => await PlayCombatRoom(log, state, c, spec, stopAfterAct),
                    RestSiteRoom => await PlayRestSite(log, state),
                    TreasureRoom => await PlayTreasure(log, chest!),
                    MerchantRoom shop => await PlayShop(log, state, shop),
                    EventRoom => await PlayEvent(log, run.EventSynchronizer.GetLocalEvent(),
                        target.PointType == MapPointType.Ancient ? "ancient" : "event", spec),
                    _ => $"unsupported_room:{room?.GetType().Name}",
                };
                if (state.CurrentActIndex != act) maps.Add(Map(state));
                if (stop != null) { end = stop; break; }
            }
        }
        catch (Env.Closed)
        {
            throw;
        }
        catch (Exception ex)
        {
            end = "error";
            dump["error"] = ex.ToString();
        }
        finally
        {
            Env.Run = null;
            FirstChoiceSelector.PickHook = null;
            FirstChoiceSelector.CardRewardHook = null;
        }

        if (Env.Trace != null) dump["env_trace"] = Env.Trace;
        Env.Trace = null;
        dump["end"] = end;
        dump["floor"] = state.TotalFloor;
        dump["fight_selections"] = selector.FightLog;
        GD.Print($"[referee] run {dump["seed"]} ended: {end} after {log.Decisions.Count} decisions, floor {state.TotalFloor}");
        return dump;
    }

    /// <summary>NMapScreen.RecalculateTravelability: the points the map lets the player travel to.</summary>
    private static List<MapPoint> TravelOptions(RunState state)
    {
        var map = state.Map;
        if (state.VisitedMapCoords.Count == 0) return new List<MapPoint> { map.StartingMapPoint };
        var last = state.VisitedMapCoords[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new List<MapPoint> { map.SecondBossMapPoint };
        if (last == map.BossMapPoint.coord || (map.SecondBossMapPoint != null && last == map.SecondBossMapPoint.coord))
            return new List<MapPoint>();
        if (last.row == map.GetRowCount() - 1) return new List<MapPoint> { map.BossMapPoint };
        var cur = map.GetPoint(last);
        if (cur == null) return new List<MapPoint> { map.StartingMapPoint };
        return MapTravel.GetTravelablePointsFrom(state, cur).OrderBy(p => p.coord.col).ToList();
    }

    /// <summary>An event (Neow included): random enabled options until it finishes, rewards taken as offered.</summary>
    private static async Task<string?> PlayEvent(RunLog log, EventModel ev, string screen, JsonObject spec)
    {
        var run = RunManager.Instance;
        var state = log.State;
        for (int guard = 0; guard < 30; guard++)
        {
            await IdleBeat();
            if (CombatManager.Instance.IsInProgress && state.CurrentRoom is CombatRoom fightRoom)
            {
                SetFastMode(MegaCrit.Sts2.Core.Settings.FastModeType.Instant);  // PunchOff's loop ends with FIGHT
                // EventModel.EnterCombatWithoutExitingEvent: the fight is a room on top of the event's.
                string? stop = await PlayCombatRoom(log, state, fightRoom, spec, stopAfterAct: 99);
                if (stop != null) return stop;
                if (!fightRoom.ShouldResumeParentEventAfterCombat) return null;
                await WaitFor(run.ProceedFromTerminalRewardsScreen(), "resume event");  // the rewards screen's Proceed
                await IdleBeat();
                ev = run.EventSynchronizer.GetLocalEvent();
                continue;
            }

            if (ev.IsFinished) break;
            // Custom screens the driver does not answer yet: the Crystal Sphere's grid
            // (CrystalSphereMinigame) and the Fake Merchant's stall (an event with a merchant
            // inventory and no options). The run ends there, recorded, rather than hanging.
            if (ev is MegaCrit.Sts2.Core.Models.Events.FakeMerchant fake) return await PlayFakeMerchant(log, fake);
            if (ev.CurrentOptions.Count == 0)
            {
                // Some events lay out their page a few frames after the room opens.
                var wait = new Budget(5);
                while (ev.CurrentOptions.Count == 0 && !ev.IsFinished && !wait.Spent) await IdleBeat();
                if (ev.CurrentOptions.Count == 0 && !ev.IsFinished) return $"event_no_options:{ev.Id.Entry}";
                continue;
            }

            var opts = ev.CurrentOptions.ToList();
            var enabled = Enumerable.Range(0, opts.Count).Where(i => !opts[i].IsLocked).ToList();
            if (enabled.Count == 0) return $"event_all_locked:{ev.Id.Entry}";
            int pick = enabled[await DecideP(screen, enabled.Count,
                () => new JsonArray(enabled.Select(i => (JsonNode)(opts[i].Relic?.Id.Entry ?? opts[i].TextKey)).ToArray()),
                () => log.Rng.Next(enabled.Count), () => ev.Id.Entry)];
            var d = log.Begin(screen);
            d["event"] = ev.Id.Entry;
            d["offer"] = new JsonArray(opts.Select(o => (JsonNode)new JsonArray(o.Relic?.Id.Entry ?? o.TextKey, o.IsLocked)).ToArray());
            d["choice"] = opts[pick].Relic?.Id.Entry ?? opts[pick].TextKey;
            d["choice_index"] = pick;
            run.EventSynchronizer.ChooseLocalOption(pick);
            // The option's own task (EventSynchronizer's pending option tasks) runs past the action
            // queue: a card added after a preview, a rewards screen it awaits. Wait for it, answering
            // reward screens, unless it opened a fight or the Crystal Sphere's grid, which wait on us.
            var option = run.EventSynchronizer.AwaitPendingOptionTasks();
            var budget = new Budget(60);
            while (!option.IsCompleted && !budget.Spent && !CombatManager.Instance.IsInProgress
                   && !CrystalSphereOpen())
            {
                if (TopRewardsSet() != null) { log.End(); await TakeRewards(log, takeAll: screen is "neow" or "ancient"); continue; }
                // A screen the option opens outside the selector hook (Scroll Boxes' bundles off Neow): answered here,
                // or the option's task waits out the 60 s budget (2 of 54 random runs) before TakeRewards answers it.
                AnswerOpenScreens();
                Heartbeat();
                await ((SceneTree)Engine.GetMainLoop()).ToSignal(Engine.GetMainLoop(), SceneTree.SignalName.ProcessFrame);
            }

            await IdleBeat();
            log.End();
            if (await PlayCrystalSphere(log) is { } sphereStop) return sphereStop;
            await TakeRewards(log, takeAll: screen is "neow" or "ancient");
            if (screen == "neow") break;  // Neow's finished page leads to the map
        }

        return null;
    }

    private static void SetFastMode(MegaCrit.Sts2.Core.Settings.FastModeType mode) => SaveManager.Instance.PrefsSave.FastMode = mode;

    private static bool IsPunchOff(RunState state) =>
        state.CurrentRoom is EventRoom && RunManager.Instance.EventSynchronizer.GetLocalEvent()?.Id.Entry == "PUNCH_OFF";

    /// <summary>
    /// The Fake Merchant's stall (an event whose layout is a MerchantInventory of relics and no
    /// options): random purchases through each entry's OnTryPurchaseWrapper, then leave, which
    /// is the stall's Proceed opening the map. Throwing a Foul Potion is not tried yet.
    /// </summary>
    private static async Task<string?> PlayFakeMerchant(RunLog log, MegaCrit.Sts2.Core.Models.Events.FakeMerchant fake)
    {
        var inv = fake.Inventory;
        JsonArray Stock() => new(inv.RelicEntries.Select(e => e.Model is { } m ? (JsonNode)new JsonArray(m.Id.Entry, e.Cost) : null).ToArray());
        for (int guard = 0; guard < 8; guard++)
        {
            var entries = inv.RelicEntries.ToList();
            var cands = Enumerable.Range(0, entries.Count).Where(i => entries[i].IsStocked && entries[i].EnoughGold).ToList();
            int fi = Env.Decide("fake_merchant", cands.Count + 1,
                () => new JsonArray(cands.Select(i => (JsonNode)new JsonArray("buy", entries[i].Model?.Id.Entry, entries[i].Cost)).Append(new JsonArray("leave")).ToArray()),
                () => cands.Count == 0 || log.Rng.NextDouble() < 0.4 ? cands.Count : log.Rng.Next(cands.Count));
            if (fi == cands.Count) break;
            int slot = cands[fi];
            var d = log.Begin("fake_merchant");
            d["offer"] = Stock();
            d["choice"] = new JsonObject { ["buy"] = "relic", ["slot"] = slot };
            d["taken"] = await WaitAnswering(log, entries[slot].OnTryPurchaseWrapper(inv), "fake merchant purchase");
            await IdleBeat();
            log.End();
            await TakeRewards(log, takeAll: true);
        }

        var leave = log.Begin("fake_merchant");
        leave["offer"] = Stock();
        leave["choice"] = "leave";
        log.End();
        return null;
    }

    /// <summary>
    /// The Crystal Sphere's grid (NCrystalSphereScreen over a CrystalSphereMinigame): random
    /// clicks, each with a random tool, among the cells whose area still holds fog (a click on
    /// clear ground only wastes a divination), through the minigame's own SetTool and
    /// CellClicked, as the screen's buttons and cells call them. Its rewards are claimed as they
    /// come; the screen's Proceed (ProceedFromTerminalRewardsScreen) is pressed at the end.
    /// Returns null when no sphere is open or once it is done.
    /// </summary>
    private static object? CrystalSphereScreen() => MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance?.GetChildren()
        .FirstOrDefault(n => n.GetType().Name == "NCrystalSphereScreen");

    private static bool CrystalSphereOpen() => CrystalSphereScreen() != null;

    private static async Task<string?> PlayCrystalSphere(RunLog log)
    {
        if (CrystalSphereScreen() is not { } screen) return null;
        var game = screen.GetType().GetField("_entity", Private)!.GetValue(screen)!;
        var gt = game.GetType();
        var adjacent = gt.GetMethod("GetAdjacentCells", Private)!;
        var toolType = gt.GetNestedType("CrystalSphereToolType")!;
        for (int guard = 0; guard < 20 && !(bool)gt.GetProperty("IsFinished")!.GetValue(game)!; guard++)
        {
            var cells = (Array)gt.GetField("cells")!.GetValue(game)!;
            bool Hidden(int x, int y) => (bool)cells.GetValue(x, y)!.GetType().GetProperty("IsHidden")!.GetValue(cells.GetValue(x, y))!;
            var cands = new List<(string tool, int x, int y)>();
            for (int x = 0; x < cells.GetLength(0); x++)
                for (int y = 0; y < cells.GetLength(1); y++)
                {
                    var area = ((System.Collections.IEnumerable)adjacent.Invoke(game, new object[] { x, y })!).Cast<object>()
                        .Select(v => ((Godot.Vector2I)v)).ToList();
                    if (area.Any(c => Hidden(c.X, c.Y))) cands.Add(("Big", x, y));
                    if (Hidden(x, y)) cands.Add(("Small", x, y));
                }

            if (cands.Count == 0) break;
            var (tool, cx, cy) = cands[Env.Decide("crystal_sphere", cands.Count,
                () => new JsonArray(cands.Select(c => (JsonNode)new JsonArray(c.tool, c.x, c.y)).ToArray()), () => log.Rng.Next(cands.Count))];
            var d = log.Begin("crystal_sphere");
            d["choice"] = new JsonObject { ["tool"] = tool, ["x"] = cx, ["y"] = cy };
            gt.GetMethod("SetTool")!.Invoke(game, new[] { Enum.Parse(toolType, tool) });
            await WaitAnswering(log, (Task)gt.GetMethod("CellClicked")!.Invoke(game, new[] { cells.GetValue(cx, cy) })!, "crystal sphere click");
            await IdleBeat();
            log.End();
        }

        // CompleteMinigame: its rewards screen, then the screen's Proceed.
        var wait = new Budget(20);
        while (!wait.Spent && !RunManager.Instance.EventSynchronizer.GetLocalEvent().IsFinished)
        {
            if (TopRewardsSet() != null) await TakeRewards(log, takeAll: false);
            await IdleBeat();
        }

        if (!RunManager.Instance.EventSynchronizer.GetLocalEvent().IsFinished) return "crystal_sphere_unfinished";
        await WaitFor(RunManager.Instance.ProceedFromTerminalRewardsScreen(), "crystal sphere proceed");
        await IdleBeat();
        return null;
    }

    /// <summary>Idle, and a heartbeat for the watchdog: out of a fight the driver waits on screens, not actions.</summary>
    private static async Task<bool> IdleBeat()
    {
        bool idle = await Idle(5);
        Heartbeat();
        return idle;
    }

    /// <summary>A fight with the existing combat loop, then its rewards; null to go on.</summary>
    private static async Task<string?> PlayCombatRoom(RunLog log, RunState state, CombatRoom room, JsonObject spec, int stopAfterAct)
    {
        var d = log.Begin("combat");
        d["encounter"] = room.Encounter?.Id.Entry;
        var steps = new JsonArray();
        var fight = new JsonObject { ["steps"] = steps };
        d["fight"] = fight;
        int choicesBefore = ScreenLog?.Count ?? 0;
        string? error = await FightLoop(state, new Random(log.Rng.Next()), (int?)spec["max_actions"] ?? 400, steps,
            potions: (bool?)spec["potions"] == true);
        fight["choices"] = new JsonArray((ScreenLog ?? new JsonArray()).Skip(choicesBefore).Select(c => c!.DeepClone()).ToArray());
        if (error != null) fight["error"] = error;
        var player = state.Players[0];
        d["result"] = player.Creature.IsDead ? "death" : CombatManager.Instance.IsInProgress ? "unfinished" : "win";
        await Idle();
        log.End();
        if (player.Creature.IsDead) return "death";
        if (CombatManager.Instance.IsInProgress) return "fight_unfinished";
        bool boss = room.RoomType == RoomType.Boss && room.ParentEventId == null;
        await TakeRewards(log, takeAll: false, leave: !boss);
        if (!boss) return null;

        // NRewardsScreen.OnProceedButtonPressed on a boss's (terminal) screen: with a second boss
        // still ahead it goes back to the map; otherwise the act changes (ActChangeSynchronizer),
        // and after the last act's boss that leads into The Architect's room.
        var map = state.Map;
        if (map.SecondBossMapPoint != null && state.CurrentMapCoord == map.BossMapPoint.coord)
        {
            var bp = log.Begin("boss_proceed");
            bp["choice"] = "proceed";
            await WaitFor(RunManager.Instance.ProceedFromTerminalRewardsScreen(), "boss proceed");
            await IdleBeat();
            log.End();
            return null;
        }

        if (state.CurrentActIndex + 1 >= stopAfterAct && stopAfterAct < state.Acts.Count)
            return $"stop_after_act_{state.CurrentActIndex + 1}";
        int act = state.CurrentActIndex;
        var ac = log.Begin("act_change");
        ac["choice"] = "proceed";
        RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
        var wait = new Budget(60);
        while (state.CurrentActIndex == act && !(state.CurrentRoom is EventRoom) && !wait.Spent)
        {
            await IdleBeat();
        }

        await IdleBeat();
        ac["room"] = new JsonArray(state.CurrentRoom?.RoomType.ToString(), state.CurrentRoom?.ModelId?.Entry);
        log.End();
        if (state.CurrentActIndex == act && state.CurrentRoom is not EventRoom) return "act_change_timeout";
        if (state.CurrentRoom is EventRoom)
        {
            // The Architect (RunManager.EnterNextAct after the last act's boss): its PROCEED wins the run.
            string? stop = await PlayEvent(log, RunManager.Instance.EventSynchronizer.GetLocalEvent(), "event", spec);
            return stop ?? "victory";
        }

        return null;
    }

    /// <summary>
    /// The rewards screen(s) on top of the stack: each reward taken or left (seeded), card rewards
    /// answered with a random card, then the screen left. Neow's rewards cannot be skipped.
    /// </summary>
    /// <param name="leave">false for a boss's screen, which is left by the act change instead.</param>
    /// <param name="depth">How deep this screen sits: a reward that opens a screen of its own
    /// (Lost Coffer off Neow's Bones) is answered at depth 1.</param>
    /// <param name="only">At depth 1 and below: the nested screen to answer, and nothing under it (Rewards.cs
    /// NestedAbove: once a nested screen completes, what is on top is its parent, mid-reward).</param>
    private static async Task TakeRewards(RunLog log, bool takeAll, bool leave = true, int depth = 0, object? only = null)
    {
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var tree = (SceneTree)Engine.GetMainLoop();
        for (int guard = 0; guard < 20 && (only == null ? TopRewardsSet() : RewardsStack().Contains(only) ? only : null) is { } top; guard++)
        {
            var set = (RewardsSet)top.GetType().GetField("set")!.GetValue(top)!;
            var player = log.State.Players[0];
            foreach (var reward in set.Rewards.ToList())
            {
                if (reward.SuccessfullySelected) continue;
                if (!RewardsStack().Contains(top)) break;  // the set is complete: nothing left to select into it
                // Gold and special cards are always taken; a card, potion or relic reward is the agent's take or skip
                // (the seeded chooser: a card skipped one time in five, a potion taken while a slot is open, a relic
                // always), unless the screen takes all or cannot be skipped.
                bool canTake = !(reward is PotionReward && player.PotionSlots.All(p => p != null) && !set.DisallowSkipping);
                bool take;
                if (!canTake) take = false;
                else if (takeAll || set.DisallowSkipping || reward is GoldReward or SpecialCardReward) take = true;
                else if (reward is CardReward or PotionReward or RelicReward)
                    take = await DecideP("reward", 2, () => new JsonArray("take", "skip"),
                        () => reward is CardReward && log.Rng.NextDouble() < 0.2 ? 1 : 0, () => RewardJson(reward)) == 0;
                else take = false;
                if (!take) continue;
                var d = log.Begin("rewards");
                if (depth > 0) d["depth"] = depth;
                d["offer"] = new JsonArray(set.Rewards.Where(r => !r.SuccessfullySelected).Select(r => (JsonNode)RewardJson(r)).ToArray());
                d["choice"] = RewardJson(reward);
                Task<bool> task = sync.SelectLocalReward(reward);
                var budget = new Budget(60);
                while (!task.IsCompleted)
                {
                    if (budget.Spent) throw new TimeoutException("SelectLocalReward " + reward.GetType().Name);
                    AnswerCardRewardRandom(reward, log, d);
                    AnswerOpenScreens();  // a screen the reward opens outside the selector hook (Scroll Boxes' bundles), as in ClaimSet
                    if (NestedAbove(top) is { } nested && depth < 5)
                    {
                        log.End();
                        await TakeRewards(log, takeAll: true, depth: depth + 1, only: nested);
                        d = log.Open ?? log.Begin("rewards_continued");
                        continue;
                    }

                    await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
                }

                d["taken"] = await task;
                await Idle();
                log.End();
            }

            if (!leave) return;
            if (ReferenceEquals(TopRewardsSet(), top))
            {
                var d = log.Begin("rewards");
                d["offer"] = new JsonArray(set.Rewards.Where(r => !r.SuccessfullySelected).Select(r => (JsonNode)RewardJson(r)).ToArray());
                d["choice"] = "proceed";
                sync.SkipLocalRewardsSet();  // leave the screen, as Proceed does
                await Idle();
                log.End();
            }
        }
    }

    private static JsonObject RewardJson(Reward r) => r switch
    {
        GoldReward g => new JsonObject { ["type"] = "gold", ["amount"] = g.Amount },
        PotionReward p => new JsonObject { ["type"] = "potion", ["id"] = p.Potion?.Id.Entry },
        RelicReward rr => new JsonObject { ["type"] = "relic", ["id"] = rr.Relic?.Id.Entry },
        CardReward c => new JsonObject { ["type"] = "card", ["cards"] = new JsonArray(c.Cards.Select(x => (JsonNode)CardJson(x)).ToArray()) },
        SpecialCardReward sc => new JsonObject
        {
            ["type"] = "special_card",
            ["card"] = sc.GetType().GetField("_card", Private)?.GetValue(sc) is CardModel card ? CardJson(card) : null,
        },
        _ => new JsonObject { ["type"] = r.GetType().Name },
    };

    /// <summary>The card reward screen, answered with a random card (AnswerCardRewardScreen's way in).</summary>
    private static void AnswerCardRewardRandom(Reward reward, RunLog log, JsonObject d)
    {
        if (reward is not CardReward cr) return;
        var screen = reward.GetType().GetField("_currentlyShownScreen", Private)?.GetValue(reward);
        if (screen == null) return;
        var source = screen.GetType().GetField("_completionSource", Private)?.GetValue(screen) as TaskCompletionSource<int?>;
        if (source == null || source.Task.IsCompleted) return;
        int n = cr.Cards.Count();
        IReadOnlyList<CardRewardAlternative> alts = Array.Empty<CardRewardAlternative>();
        if (Env.Extras && (Env.Live || Env.Tracing))
        {
            // The screen's own list gives the answer's index after the cards; Generate is the cross-check (order and ids), counted.
            var own = screen.GetType().GetField("_extraOptions", Private)?.GetValue(screen) as IReadOnlyList<CardRewardAlternative>;
            var generated = CardRewardAlternative.Generate(cr);
            Env.AltChecked++;
            if (own == null || !own.Select(x => x.OptionId).SequenceEqual(generated.Select(x => x.OptionId))) Env.AltMismatch++;
            alts = own ?? generated;
        }

        int pick = Env.Decide("card_reward", n + alts.Count, () => CardRewardActions(cr.Cards.Select(x => x).ToList(), alts),
            () => log.Rng.Next(n));
        if (pick < n) d["card"] = CardJson(cr.Cards.ElementAt(pick));
        else d["alternative"] = alts[pick - n].OptionId;
        source.SetResult(pick);
    }

    private static async Task<string?> PlayRestSite(RunLog log, RunState state)
    {
        var sync = RunManager.Instance.RestSiteSynchronizer;
        var opts = sync.GetLocalOptions().ToList();
        var enabled = Enumerable.Range(0, opts.Count).Where(i => opts[i].IsEnabled).ToList();
        if (enabled.Count > 0)
        {
            int pick = enabled[await DecideP("rest", enabled.Count,
                () => new JsonArray(enabled.Select(i => (JsonNode)opts[i].OptionId).ToArray()), () => log.Rng.Next(enabled.Count))];
            var d = log.Begin("rest");
            d["offer"] = new JsonArray(opts.Select(o => (JsonNode)new JsonArray(o.OptionId, o.IsEnabled)).ToArray());
            d["choice"] = opts[pick].OptionId;
            d["taken"] = await WaitAnswering(log, sync.ChooseLocalOption(pick), "rest option");
            await Idle();
            log.End();
        }

        var p = log.Begin("rest");
        p["choice"] = "proceed";
        log.End();
        return null;
    }

    private static object? TreasureRoomNode() =>
        ((SceneTree)Engine.GetMainLoop()).Root.GetNodeOrNull("/root/Game/RootSceneContainer/Run/RoomContainer/TreasureRoom");

    /// <summary>NTreasureRoom.OpenChest, as the chest button's release calls it.</summary>
    private static Task OpenChest()
    {
        var node = TreasureRoomNode() ?? throw new InvalidOperationException("treasure room node not found");
        var open = node.GetType().GetMethod("OpenChest", Private) ?? throw new MissingMethodException("NTreasureRoom.OpenChest");
        return (Task)open.Invoke(node, null)!;
    }

    private static bool TreasureChestOpened() =>
        TreasureRoomNode() is { } node && node.GetType().GetField("_hasChestBeenOpened", Private)?.GetValue(node) is true;

    private static async Task<string?> PlayTreasure(RunLog log, Task chest)
    {
        var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        var relics = sync.CurrentRelics;
        if (relics == null)
        {
            // The empty chest. TreasureRoomRelicSynchronizer.BeginRelicPicking (run at room entry by
            // TreasureRoom.EnterInternal) generates a relic only while Hook.ShouldGenerateTreasure holds;
            // SilverCrucible.ShouldGenerateTreasure is its one override (the relic's first treasure room),
            // and with nothing generated the synchronizer ends the voting at once, so CurrentRelics is
            // null before the chest is clicked. NTreasureRoomRelicCollection.InitializeRelics then shows
            // the empty chest and CompleteWithNoRelics closes the picking, which the chest task waits on.
            // A singleplayer chest with a relic cannot reach here with null (the list is set at room
            // entry and cleared by the award or the room's exit), so this is the relic's rule, not a
            // stop: until 2026-10-28 the run ended here as "treasure_without_relic_picking" (6% of the
            // generation-7 night's runs, every one a Silver Crucible taken at Neow, at the Act-1 chest).
            if (chest.IsFaulted) return "treasure_chest_error";
            var e = log.Begin("treasure");
            e["offer"] = new JsonArray();
            e["choice"] = "empty";
            try { await WaitFor(chest, "empty chest"); }
            catch (TimeoutException) { log.End(); return "treasure_chest_timeout"; }
            await IdleBeat();
            log.End();
            return null;
        }
        var r = log.Begin("treasure");
        r["offer"] = new JsonArray((relics ?? Array.Empty<RelicModel>()).Select(x => (JsonNode)x.Id.Entry).ToArray());
        if (relics is { Count: > 0 } && Env.Decide("treasure", 2, () => new JsonArray(new JsonArray("take", relics[0].Id.Entry), "skip"),
                () => log.Rng.NextDouble() >= 0.1 ? 0 : 1) == 0)
        {
            // A relic holder's click (NTreasureRoomRelicCollection); the collection then awards it.
            r["choice"] = "take";
            sync.PickRelicLocally(0);
            await WaitAnswering(log, chest, "treasure relic award");
        }
        else
        {
            // The Skip button (NTreasureRoom.OnProceedButtonPressed); the chest's task stays parked.
            r["choice"] = "skip";
            sync.SkipRelicLocally();
        }

        await IdleBeat();
        log.End();
        return null;
    }

    /// <summary>
    /// A merchant: random purchases (each through the entry's own OnTryPurchaseWrapper, which
    /// charges the gold and hands the item over), the card removal service among them, then
    /// leave. Every decision records the stock with its prices.
    /// </summary>
    private static async Task<string?> PlayShop(RunLog log, RunState state, MerchantRoom room)
    {
        var inv = room.GetLocalInventory();
        var player = state.Players[0];
        for (int guard = 0; guard < 12; guard++)
        {
            var cands = new List<(string kind, int slot, MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry)>();
            var chars = inv.CharacterCardEntries.ToList();
            var colorless = inv.ColorlessCardEntries.ToList();
            for (int i = 0; i < chars.Count; i++) cands.Add(("card", i, chars[i]));
            for (int i = 0; i < colorless.Count; i++) cands.Add(("card", chars.Count + i, colorless[i]));
            var relics = inv.RelicEntries.ToList();
            for (int i = 0; i < relics.Count; i++) cands.Add(("relic", i, relics[i]));
            var potions = inv.PotionEntries.ToList();
            if (player.HasOpenPotionSlots)
                for (int i = 0; i < potions.Count; i++) cands.Add(("potion", i, potions[i]));
            if (inv.CardRemovalEntry is { } removal) cands.Add(("remove", 0, removal));
            cands = cands.Where(c => c.entry.IsStocked && c.entry.EnoughGold).ToList();
            int si = await DecideP("shop", cands.Count + 1,
                () => new JsonArray(cands.Select(c => (JsonNode)new JsonArray(c.kind, c.slot, c.entry.Cost)).Append(new JsonArray("leave")).ToArray()),
                () => cands.Count == 0 || log.Rng.NextDouble() < 0.35 ? cands.Count : log.Rng.Next(cands.Count), () => ShopJson(inv));
            if (si == cands.Count) break;
            var (kind, slot, entry) = cands[si];
            var d = log.Begin("shop");
            d["offer"] = ShopJson(inv);
            d["choice"] = new JsonObject { ["buy"] = kind, ["slot"] = slot };
            Task<bool> buy = entry is MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry r
                ? r.OnTryPurchaseWrapper(inv)
                : entry.OnTryPurchaseWrapper(inv);
            d["taken"] = await WaitAnswering(log, buy, "shop purchase");
            await IdleBeat();
            log.End();
            await TakeRewards(log, takeAll: true);  // a relic bought can open a screen of its own
        }

        var leave = log.Begin("shop");
        leave["offer"] = ShopJson(inv);
        leave["choice"] = "leave";
        log.End();
        return null;
    }

    private static JsonObject ShopJson(MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv)
    {
        JsonNode? Card(MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardEntry e) => e.CreationResult?.Card is { } c
            ? new JsonArray(c.Id.Entry, c.IsUpgraded ? 1 : 0, e.Cost) : null;
        return new JsonObject
        {
            ["cards"] = new JsonArray(inv.CharacterCardEntries.Concat(inv.ColorlessCardEntries).Select(Card).ToArray()),
            ["relics"] = new JsonArray(inv.RelicEntries.Select(e => e.Model is { } m ? (JsonNode)new JsonArray(m.Id.Entry, e.Cost) : null).ToArray()),
            ["potions"] = new JsonArray(inv.PotionEntries.Select(e => e.Model is { } m ? (JsonNode)new JsonArray(m.Id.Entry, e.Cost) : null).ToArray()),
            ["removal"] = inv.CardRemovalEntry is { IsStocked: true } r ? r.Cost : null,
        };
    }

    private static async Task<T> WaitFor<T>(Task<T> task, string what)
    {
        await WithTimeout(task, 60, what);
        return await task;
    }

    private static async Task WaitFor(Task task, string what) => await WithTimeout(task, 60, what);

    /// <summary>
    /// Waits on a choice whose effect can open a rewards screen and await it (Dream Catcher's card
    /// on a rest, a relic bought that offers rewards): such screens are answered while it waits,
    /// as decisions after the one that opened them.
    /// </summary>
    private static async Task<bool> WaitAnswering(RunLog log, Task task, string what)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        var budget = new Budget(60);
        while (!task.IsCompleted)
        {
            if (budget.Spent) throw new TimeoutException($"{what} did not finish within 60 s");
            if (TopRewardsSet() != null)
            {
                log.End();
                await TakeRewards(log, takeAll: false);
                continue;
            }

            Heartbeat();
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        await task;
        return task is Task<bool> b ? b.Result : true;
    }

    /// <summary>
    /// Combat.cs's fight loop (seeded random legal plays, a snapshot after each), for a fight the
    /// run has already entered. Kept identical to that loop so run fights join the fight corpus.
    /// </summary>
    /// <param name="potions">
    /// Also drink potions (spec "potions"): with a one-in-eight chance per action, a random usable
    /// potion (CombatOnly or AnyTime, as NPotionPopup's Use button allows) through
    /// PotionModel.EnqueueManualUse, at a random living enemy when it needs one.
    /// </param>
    private static async Task<string?> FightLoop(RunState state, Random rng, int maxActions, JsonArray steps, bool potions = false)
    {
        var run = RunManager.Instance;
        Player player = state.Players[0];
        await Idle();
        steps.Add(new JsonObject { ["action"] = null, ["state"] = Snapshot(state, player) });
        for (int i = 0; i < maxActions && CombatManager.Instance.IsInProgress; i++)
        {
            CurrentStep = i + 1;
            var combat = CombatManager.Instance.DebugOnlyGetState()!;
            var enemies = combat.Enemies.Where(e => e.IsAlive).ToList();
            var options = new List<(int hand, int target)> { (-1, -1) };
            var hand = player.PlayerCombatState!.Hand.Cards;
            for (int h = 0; h < hand.Count; h++)
            {
                if (!hand[h].CanPlay()) continue;
                if (hand[h].TargetType == TargetType.AnyEnemy)
                {
                    for (int t = 0; t < enemies.Count; t++) options.Add((h, t));
                }
                else
                {
                    options.Add((h, -1));
                }
            }

            var usable = potions ? player.PotionSlots.Select((p, slot) => (p, slot))
                .Where(x => x.p is { IsQueued: false, PassesCustomUsabilityCheck: true }
                            && x.p.Usage is MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly or MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime
                            && player.CanRemovePotions)
                .ToList() : new();
            // One list of legal actions: end turn and plays (options), then each usable potion at each target.
            var potionActs = new List<(PotionModel potion, int slot, int target)>();
            foreach (var (pm, ps) in usable)
            {
                if (pm!.TargetType == TargetType.AnyEnemy && enemies.Count > 0)
                    for (int t = 0; t < enemies.Count; t++) potionActs.Add((pm, ps, t));
                else potionActs.Add((pm, ps, -1));
            }

            int choice = Env.Decide("combat", options.Count + potionActs.Count, () => new JsonArray(options
                    .Select(o => (JsonNode)(o.hand < 0 ? new JsonArray("end") : new JsonArray("play", o.hand, hand[o.hand].Id.Entry, o.target)))
                    .Concat(potionActs.Select(x => (JsonNode)new JsonArray("potion", x.slot, x.potion.Id.Entry, x.target))).ToArray()),
                () =>
                {
                    if (usable.Count > 0 && rng.Next(8) == 0)
                    {
                        var (p0, s0) = usable[rng.Next(usable.Count)];
                        int t0 = p0!.TargetType == TargetType.AnyEnemy && enemies.Count > 0 ? rng.Next(enemies.Count) : -1;
                        return options.Count + potionActs.FindIndex(x => x.slot == s0 && x.target == t0);
                    }

                    return rng.Next(options.Count);
                });
            if (choice >= options.Count)
            {
                var (potion, slot, pt) = potionActs[choice - options.Count];
                Creature? at = pt >= 0 ? enemies[pt] : null;
                var pa = new JsonObject
                {
                    ["kind"] = "potion", ["slot"] = slot, ["potion"] = potion.Id.Entry,
                    ["target_slot"] = at != null ? combat.Enemies.ToList().IndexOf(at) : -1,
                };
                potion.EnqueueManualUse(at);
                if (!await Idle())
                {
                    CurrentStep = 0;
                    return $"idle_timeout after step {i}: {pa.ToJsonString()}";
                }

                steps.Add(new JsonObject { ["action"] = pa, ["state"] = Snapshot(state, player) });
                Heartbeat();
                continue;
            }

            var (hi, ti) = options[choice];
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
                action["played"] = card.TryManualPlay(tgt);
            }

            if (!await Idle())
            {
                CurrentStep = 0;
                return $"idle_timeout after step {i}: {action.ToJsonString()}";
            }

            steps.Add(new JsonObject { ["action"] = action, ["state"] = Snapshot(state, player) });
            Heartbeat();
        }

        CurrentStep = 0;
        return CombatManager.Instance.IsInProgress ? "max_actions" : null;
    }
}
