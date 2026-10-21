// Run-level modes of EmuDump.
//   EmuDump runstart <ascension> <seed> [<seed> ...]   the run snapshot right after Reset, per seed
//   EmuDump run <referee out.json> [--no-fights]        replays each recorded run (referee/driver/RunPlay.cs)
//                                                        decision by decision, by id, not by position
// A run decision is applied as the emulator's own actions (RunEngine.Step): map coords, Neow relics,
// reward types, card ids, rest option ids and deck indices are looked up in the emulator's state.
// A choice the emulator does not offer is a divergence ("error"), and the replay stops there. Steps
// the game takes without a decision (Neow's last Proceed, leaving an emptied rewards screen or a chest)
// are taken before the next map move ("auto").
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class RunReplay
{
    /// <returns>true when args named a run-level mode (and it ran).</returns>
    public static bool Main(string[] args)
    {
        if (args.Length == 0) return false;
        switch (args[0])
        {
            case "runstart":
            {
                int asc = int.Parse(args[1]);
                var results = new JsonArray();
                foreach (string seed in args.Skip(2))
                {
                    var e = new RunEngine();
                    e.Reset(seed, asc);
                    results.Add(new JsonObject { ["seed"] = e.State.StringSeed, ["snap"] = RunSnap.Export(e) });
                }

                Console.WriteLine(new JsonObject { ["results"] = results }.ToJsonString());
                return true;
            }

            case "run":
            {
                bool fights = !args.Contains("--no-fights");
                Anchoring = !args.Contains("--no-anchor");
                var refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
                var results = new JsonArray(refs.Select(r => (JsonNode)(r?["decisions"] == null
                    ? new JsonObject { ["seed"] = r?["seed"]?.DeepClone(), ["error"] = "no run recorded" }
                    : Replay(r.AsObject(), fights))).ToArray());
                Console.WriteLine(new JsonObject { ["results"] = results }.ToJsonString());
                return true;
            }
        }

        return false;
    }

    sealed class Divergence(string message) : Exception(message);

    static readonly string[] NodeTypeNames = ["Unassigned", "Monster", "Elite", "RestSite", "Shop", "Treasure", "Boss", "Unknown", "Ancient"];

    /// <summary>RestSiteOption.OptionId to the emulator's rest action.</summary>
    static readonly Dictionary<string, int> RestActions = new()
    {
        ["HEAL"] = RunConstants.RestHealAction, ["SMITH"] = RunConstants.RestUpgradeAction,
        ["CLONE"] = RunConstants.RestCloneAction, ["LIFT"] = RunConstants.RestLiftAction,
        ["DIG"] = RunConstants.RestDigAction, ["HATCH"] = RunConstants.RestHatchAction,
        ["COOK"] = RunConstants.RestCookAction, ["KINDLE"] = RunConstants.RestKindleAction,
    };

    static JsonObject Replay(JsonObject game, bool fights)
    {
        var e = new RunEngine();
        e.Reset((string)game["seed"]!, (int)game["ascension"]!);
        var outp = new JsonObject { ["seed"] = e.State.StringSeed };
        var decisions = new JsonArray();
        outp["decisions"] = decisions;
        var all = game["decisions"]!.AsArray();
        for (int k = 0; k < all.Count; k++)
        {
            var d = all[k]!.AsObject();
            NextDecision = k + 1 < all.Count ? all[k + 1]!.AsObject() : null;
            var od = new JsonObject { ["i"] = d["i"]?.DeepClone(), ["screen"] = d["screen"]?.DeepClone() };
            decisions.Add(od);
            try
            {
                Apply(ref e, d, od, fights);
            }
            catch (Exception ex)
            {
                od["error"] = ex is Divergence ? ex.Message : ex.ToString();
            }

            od["phase"] = e.State.Phase.ToString();
            od["snap"] = RunSnap.Export(e);
            if (od["error"] != null) break;
        }

        return outp;
    }

    static void Step(RunEngine e, int action, JsonObject od, string what)
    {
        var mask = new int[RunConstants.MaxActions];
        e.WriteActionMask(mask);
        if ((uint)action >= (uint)mask.Length || mask[action] == 0)
            throw new Divergence($"{what}: action {action} not offered in phase {e.State.Phase}");
        if (e.Step(action, -1, out _, out _, out _) != 0)
            throw new Divergence($"{what}: action {action} refused in phase {e.State.Phase}");
        (od["actions"] ??= new JsonArray()).AsArray().Add(action);
    }

    static void Apply(ref RunEngine e, JsonObject d, JsonObject od, bool fights)
    {
        var s = e.State;
        switch ((string)d["screen"]!)
        {
            case "start":
                return;
            case "buff":
                RunNonCombatEffects.GainMaxHp(s, (int)d["choice"]!);
                return;
            case "neow":
            case "ancient":
            {
                if (s.Phase == RunPhase.Ancient && s.NeowAwaitingProceed) throw new Divergence("ancient: emulator already answered");
                Expect(e, RunPhase.Ancient, (string)d["screen"]!);
                od["offer"] = new JsonArray(s.NeowOptions.Select(r => (JsonNode)new JsonArray(RunSnap.RelicEntry(r), false)).ToArray());
                string want = (string)d["choice"]!;
                int idx = Array.FindIndex(s.NeowOptions, r => RunSnap.RelicEntry(r) == want);
                if (idx < 0) throw new Divergence($"ancient option {want} not offered");
                Step(e, idx, od, "ancient");
                Selections(e, d, od);
                return;
            }

            case "map":
            {
                AutoAdvance(e, od);
                Expect(e, RunPhase.Map, "map");
                var offer = new JsonArray();
                for (int i = 0; i < s.MapOptionCoords.Length; i++)
                {
                    if (s.MapOptionCoords[i] is { } c && s.MapNodeTypes[i] != RunConstants.NodeNone)
                        offer.Add(new JsonArray(c.Col, c.Row, NodeTypeNames[s.MapNodeTypes[i]]));
                }

                od["offer"] = offer;
                var want = d["choice"]!.AsArray();
                (int Col, int Row) coord = ((int)want[0]!, (int)want[1]!);
                int idx = Array.FindIndex(s.MapOptionCoords, c => c == coord);
                if (idx < 0) throw new Divergence($"map point {coord} not offered");
                Step(e, idx, od, "map");
                od["room"] = new JsonArray(s.Phase.ToString(),
                    s.Phase == RunPhase.Combat ? RunSnap.EncounterEntry(s.ActiveCombat?.EncounterId ?? 0)
                    : s.Phase is RunPhase.Event or RunPhase.CrystalSphere or RunPhase.FakeMerchant ? RunSnap.EventEntry(s.EventId)
                    : s.Phase == RunPhase.Ancient ? s.Ancient
                    : null,
                    RoomType(s));
                if (s.Phase == RunPhase.Treasure)
                    od["treasure_relic"] = RunSnap.RelicEntry(s.RelicReward);
                return;
            }

            case "combat":
                Combat(ref e, d, od, fights);
                return;
            case "rewards":
                Rewards(e, d, od);
                return;
            case "rest":
            {
                string choice = (string)d["choice"]!;
                if (choice == "proceed")
                {
                    if (s.Phase == RunPhase.Rest) Step(e, RunConstants.RewardSkipAction, od, "rest proceed");
                    return;
                }

                Expect(e, RunPhase.Rest, "rest");
                var mask = new int[RunConstants.MaxActions];
                e.WriteActionMask(mask);
                od["offer"] = new JsonArray(RestActions.Where(kv => mask[kv.Value] != 0)
                    .Select(kv => (JsonNode)new JsonArray(kv.Key, true)).ToArray());
                if (!RestActions.TryGetValue(choice, out int action)) throw new Divergence($"rest option {choice} unknown to the emulator");
                Step(e, action, od, $"rest {choice}");
                Selections(e, d, od);
                return;
            }

            case "treasure":
            {
                Expect(e, RunPhase.Treasure, "treasure");
                od["offer"] = new JsonArray(s.RelicReward == 0 ? Array.Empty<JsonNode>() : new JsonNode[] { RunSnap.RelicEntry(s.RelicReward) });
                if ((string)d["choice"]! == "take") Step(e, RunConstants.TreasureTakeRelicAction, od, "treasure take");
                else Step(e, RunConstants.RewardSkipAction, od, "treasure skip");
                return;
            }

            case "event":
            {
                // An event's options are its option list in the game's order, locked ones included
                // (EventOptions.g.cs), so the game's index is the emulator's action.
                Expect(e, RunPhase.Event, "event");
                if (RunSnap.EventEntry(s.EventId) != RunSnapshotName((string?)d["event"]))
                    throw new Divergence($"event {(string?)d["event"]} in the game, {RunSnap.EventEntry(s.EventId)} in the emulator");
                var mask = new int[RunConstants.MaxActions];
                e.WriteActionMask(mask);
                od["enabled"] = new JsonArray(Enumerable.Range(0, 8).Where(i => mask[i] != 0).Select(i => (JsonNode)i).ToArray());
                od["skip_action"] = RunConstants.EventSkipAction;
                od["option_count"] = OptionCount(s.EventId);
                Step(e, (int)d["choice_index"]!, od, $"event option {(string?)d["choice"]}");
                Selections(e, d, od);
                return;
            }

            case "fake_merchant":
            {
                // The stall's six relics are its actions 0..5; leaving is FakeMerchantLeaveAction.
                Expect(e, RunPhase.FakeMerchant, "fake merchant");
                od["offer"] = new JsonArray(RunNonCombatEffects.FakeMerchantStock(s)
                    .Select((r, i) => r == 0 ? null : (JsonNode)new JsonArray(RunSnap.RelicEntry(r), RunNonCombatEffects.FakeMerchantPrice(s, i))).ToArray());
                if (d["choice"] is JsonValue) { Step(e, RunConstants.FakeMerchantLeaveAction, od, "fake merchant leave"); return; }
                if ((bool?)d["taken"] == false) return;
                Step(e, (int)d["choice"]!["slot"]!, od, "fake merchant buy");
                Selections(e, d, od);
                return;
            }

            case "crystal_sphere":
            {
                // A cell with a tool: the Big tool's action is the cell, the Small tool's is past
                // the grid (RunConstants.CrystalSphereSmallToolAction).
                Expect(e, RunPhase.CrystalSphere, "crystal sphere");
                var c = d["choice"]!.AsObject();
                int cell = (int)c["x"]! * RunConstants.CrystalSphereSize + (int)c["y"]!;
                Step(e, (string)c["tool"]! == "Small" ? RunConstants.CrystalSphereSmallToolAction + cell : cell, od, "crystal sphere");
                return;
            }

            case "boss_proceed":
            case "act_change":
            {
                // The boss's rewards screen left by its Proceed: back to the map for a second boss,
                // else RunManager.EnterNextAct (the next act's map, or The Architect after the last).
                int act = s.CurrentActIndex;
                if (s.Phase == RunPhase.CardReward) Step(e, RunConstants.RewardSkipAction, od, "boss card reward skip");
                if (s.Phase == RunPhase.RelicReward) Step(e, RunConstants.RewardSkipAction, od, "boss proceed");
                if ((string)d["screen"]! == "act_change" && s.CurrentActIndex == act && s.Phase != RunPhase.Event)
                    throw new Divergence($"act change: emulator stayed in act {act + 1} ({s.Phase})");
                od["room"] = new JsonArray(s.Phase.ToString(), s.Phase == RunPhase.Event ? RunSnap.EventEntry(s.EventId) : null);
                if (s.CurrentActIndex != act) od["map"] = RunSnap.Map(s);
                return;
            }

            case "shop":
            {
                Expect(e, RunPhase.Shop, "shop");
                od["offer"] = ShopOffer(s);
                if (d["choice"] is JsonValue) { Step(e, RunConstants.ShopSkipAction, od, "shop leave"); return; }
                string kind = (string)d["choice"]!["buy"]!;
                int slot = (int)d["choice"]!["slot"]!;
                int action = kind switch
                {
                    "card" => slot, "relic" => 7 + slot, "potion" => 10 + slot,
                    "remove" => RunConstants.ShopRemoveAction,
                    _ => throw new Divergence($"shop purchase {kind} unknown"),
                };
                if ((bool?)d["taken"] == false) return;  // the game refused it; nothing changed
                Step(e, action, od, $"shop buy {kind} {slot}");
                Selections(e, d, od);
                return;
            }
            default:
                throw new Divergence($"screen {(string?)d["screen"]} not replayed yet");
        }
    }

    /// <summary>
    /// The room the emulator entered, as the game's RoomType names it (the driver records the
    /// game's): a fight by the node it resolved to, an ancient as the Event room it is in the game.
    /// </summary>
    static string? RoomType(RunState s) => s.Phase switch
    {
        RunPhase.Combat => s.LastResolvedRoomType switch
        {
            RunConstants.NodeNormal => "Monster", RunConstants.NodeElite => "Elite", RunConstants.NodeBoss => "Boss",
            _ => $"Combat({s.LastResolvedRoomType})",
        },
        RunPhase.Event or RunPhase.CrystalSphere or RunPhase.FakeMerchant or RunPhase.Ancient => "Event",
        RunPhase.Rest => "RestSite",
        RunPhase.Shop => "Shop",
        RunPhase.Treasure => "Treasure",
        _ => s.Phase.ToString(),
    };

    static MethodInfo? _optionCount;

    /// <summary>EventOptions.CountFor: the event's first-page option count, 0 where it is run state.</summary>
    static int OptionCount(int eventId)
    {
        _optionCount ??= typeof(RunEngine).Assembly.GetType("Sts2Emulator.GeneratedData.EventOptions")!
            .GetMethod("CountFor", BindingFlags.Public | BindingFlags.Static)!;
        return (int)_optionCount.Invoke(null, [eventId])!;
    }

    /// <summary>A game model id without its type prefix (EVENT.TRIAL -> TRIAL).</summary>
    static string? RunSnapshotName(string? id) => id == null ? null : id.Contains('.') ? id[(id.IndexOf('.') + 1)..] : id;

    static void Expect(RunEngine e, RunPhase phase, string what)
    {
        if (e.State.Phase != phase) throw new Divergence($"{what}: emulator is in {e.State.Phase}, not {phase}");
    }

    /// <summary>The steps the game takes without a decision, before the next map move.</summary>
    static void AutoAdvance(RunEngine e, JsonObject od)
    {
        var s = e.State;
        for (int guard = 0; guard < 6 && s.Phase != RunPhase.Map; guard++)
        {
            var auto = (od["auto"] ??= new JsonArray()).AsArray();
            if (s.Phase == RunPhase.Ancient && s.NeowAwaitingProceed) { auto.Add("ancient proceed"); e.Step(0, -1, out _, out _, out _); }
            else if (s.Phase == RunPhase.Treasure && s.RelicReward == 0) { auto.Add("leave chest"); e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _); }
            else if (s.Phase == RunPhase.Rest && s.RestResultPending) { auto.Add("rest proceed"); e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _); }
            else if (s.Phase == RunPhase.Event && s.EventId == RunConstants.EventResultPending) { auto.Add("event proceed"); e.Step(0, -1, out _, out _, out _); }
            else if (s.Phase == RunPhase.RelicReward && !RunRewardGenerator.HasPendingRewards(s)) { auto.Add("leave rewards"); e.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _); }
            else break;
        }
    }

    /// <summary>The merchant's stock as the driver's ShopJson records it.</summary>
    static JsonObject ShopOffer(RunState s) => new()
    {
        ["cards"] = new JsonArray(Enumerable.Range(0, s.ShopCards.Length).Select(i => s.ShopCards[i] == 0 ? null
            : (JsonNode)new JsonArray(RunSnap.CardEntry(s.ShopCards[i]), s.ShopCardUpgraded[i] ? 1 : 0, s.ShopCosts[i])).ToArray()),
        ["relics"] = new JsonArray(Enumerable.Range(0, s.ShopRelics.Length).Select(i => s.ShopRelics[i] == 0 ? null
            : (JsonNode)new JsonArray(RunSnap.RelicEntry(s.ShopRelics[i]), s.ShopCosts[7 + i])).ToArray()),
        ["potions"] = new JsonArray(Enumerable.Range(0, s.ShopPotions.Length).Select(i => s.ShopPotions[i] == 0 ? null
            : (JsonNode)new JsonArray(RunSnap.PotionEntry(s.ShopPotions[i]), s.ShopCosts[10 + i])).ToArray()),
        ["removal"] = s.ShopRemovalUsedThisVisit ? null : s.ShopCosts[RunConstants.ShopRemoveAction],
    };

    /// <summary>The rewards screen as the game's RewardJson shows it: what is still to take.</summary>
    static JsonArray RewardOffer(RunState s)
    {
        var o = new JsonArray();
        if (s.RewardGold != 0) o.Add(new JsonObject { ["type"] = "gold", ["amount"] = s.RewardGold });
        foreach (int g in s.PendingGoldRewards) o.Add(new JsonObject { ["type"] = "gold", ["amount"] = g });
        if (s.RewardPotion != 0) o.Add(new JsonObject { ["type"] = "potion", ["id"] = RunSnap.PotionEntry(s.RewardPotion) });
        foreach (int p in s.PendingPotionRewards.Where(p => p != 0))
            o.Add(new JsonObject { ["type"] = "potion", ["id"] = RunSnap.PotionEntry(p) });
        if (s.RelicReward != 0) o.Add(new JsonObject { ["type"] = "relic", ["id"] = RunSnap.RelicEntry(s.RelicReward) });
        foreach (int r in s.PendingBonusRelicRewards) o.Add(new JsonObject { ["type"] = "relic", ["id"] = RunSnap.RelicEntry(r) });
        if (s.RewardCardPending || s.Phase == RunPhase.CardReward)
            o.Add(CardOffer(s.RewardCards, s.RewardUpgraded));
        foreach (var off in s.PendingCardOffers) o.Add(CardOffer(off.Cards, off.Upgraded));
        foreach (var sc in s.RewardSpecialCards)
            o.Add(new JsonObject { ["type"] = "special_card", ["card"] = new JsonArray(RunSnap.CardEntry(sc.DefId), sc.Upgraded ? 1 : 0) });
        return o;
    }

    static JsonObject CardOffer(int[] cards, bool[] upg) => new()
    {
        ["type"] = "card",
        ["cards"] = new JsonArray(cards.Select((c, i) => (c, i)).Where(x => x.c != 0)
            .Select(x => (JsonNode)new JsonArray(RunSnap.CardEntry(x.c), i_upg(upg, x.i))).ToArray()),
    };

    static int i_upg(bool[] upg, int i) => i < upg.Length && upg[i] ? 1 : 0;

    static void Rewards(RunEngine e, JsonObject d, JsonObject od)
    {
        var s = e.State;
        od["offer"] = RewardOffer(s);
        if (d["choice"] is JsonValue v && (string?)v == "proceed")
        {
            if (s.Phase == RunPhase.CardReward) Step(e, RunConstants.RewardSkipAction, od, "card reward skip");
            if (s.Phase == RunPhase.RelicReward) Step(e, RunConstants.RewardSkipAction, od, "rewards proceed");
            return;
        }

        string type = (string)d["choice"]!["type"]!;

        // A lone card reward (Dream Catcher's on a rest, an event's) opens the card screen
        // straight away in the emulator; the game shows it on a rewards screen first.
        if (type == "card" && s.Phase == RunPhase.CardReward) goto card;
        Expect(e, RunPhase.RelicReward, "rewards");
        // The emulator's screen lists gold, potion, relic, card in that order (ClaimRewardAtIndex).
        int idx = 0;
        if (s.RewardGold != 0 && type == "gold" && (int?)d["choice"]!["amount"] is int amount && amount != s.RewardGold)
        {
            // Two gold piles (the room's and a hook's): the game's screen takes either; the
            // emulator's shows one at a time, the rest queued. Which pile is up front is the
            // screen's layout, not a rule, so the game's pile is brought forward.
            int at = s.PendingGoldRewards.IndexOf(amount);
            if (at < 0) throw new Divergence($"gold reward {amount} not on the emulator's screen");
            s.PendingGoldRewards[at] = s.RewardGold;
            s.RewardGold = amount;
        }

        if (s.RewardGold != 0) { if (type == "gold") goto claim; idx++; }
        if (s.RewardPotion != 0) { if (type == "potion") goto claim; idx++; }
        if (s.RelicReward != 0) { if (type == "relic") goto claim; idx++; }
        if (s.RewardCardPending || s.PendingCardOffers.Count > 0) { if (type == "card") goto claim; idx++; }
        idx += s.PendingOtherCharacterCardRewards;
        if (type == "special_card")
        {
            // A SpecialCardReward row (a Thieving Hopper's stolen card, the Lantern Key): the
            // rows sit after the card items, one each.
            var sc = d["choice"]!["card"]!.AsArray();
            int want = Fight.CardIdBySnake((string)sc[0]!);
            bool up = (int)sc[1]! != 0;
            int at = s.RewardSpecialCards.FindIndex(c => c.DefId == want && c.Upgraded == up);
            if (at < 0)
            {
                // Not on the emulator's screen (its fight went otherwise): the offer comparison
                // shows the difference, and the claim is made outside the screen so the walk goes on.
                RunNonCombatEffects.AddCardToDeck(s, new Sts2Emulator.Core.CardInstance(want, up));
                od["special_card"] = "not on the emulator's screen; claimed outside it";
                return;
            }

            idx += at;
            goto claim;
        }
        throw new Divergence($"reward {type} not on the emulator's screen");
    claim:
        Step(e, idx, od, $"claim {type}");
    card:
        if (type == "card")
        {
            Expect(e, RunPhase.CardReward, "card reward");
            var card = d["card"]?.AsArray();
            if (card == null) throw new Divergence("card reward taken without a recorded card");
            string id = (string)card[0]!;
            bool up = (int)card[1]! != 0;
            int i = Enumerable.Range(0, s.RewardCards.Length)
                .FirstOrDefault(k => s.RewardCards[k] != 0 && RunSnap.CardEntry(s.RewardCards[k]) == id && s.RewardUpgraded[k] == up, -1);
            if (i < 0) throw new Divergence($"card {id}{(up ? "+" : "")} not among the emulator's offer");
            Step(e, i, od, $"card {id}");
        }

        Selections(e, d, od);
    }

    /// <summary>
    /// The card choices a decision opened (CardSelectCmd through the driver's selector): each picked
    /// card by deck index, or by id for a grid of generated cards, in the order the game took them.
    /// </summary>
    static void Selections(RunEngine e, JsonObject d, JsonObject od)
    {
        var s = e.State;
        // NChooseABundleSelectionScreen (Scroll Boxes): the driver's AnswerOpenScreens takes the
        // first bundle, which the screen confirms in two steps here.
        if (s.Phase == RunPhase.BundleSelect)
        {
            od["bundle"] = new JsonArray(s.BundleOffer.Take(3).Select(c => (JsonNode)RunSnap.CardEntry(c)).ToArray());
            Step(e, 0, od, "bundle 0");
            Step(e, RunConstants.BundleConfirmAction, od, "bundle confirm");
        }

        foreach (var selNode in d["selections"]?.AsArray() ?? new JsonArray())
        {
            var sel = selNode!.AsObject();
            if ((string?)sel["kind"] != "cards") throw new Divergence($"selection kind {(string?)sel["kind"]} not replayed yet");
            if (s.Phase != RunPhase.TransformSelect) throw new Divergence($"game chose cards; emulator is in {s.Phase}");
            var options = sel["options"]!.AsArray();
            var picked = sel["picked"]!.AsArray().Select(x => (int)x!).ToList();
            var deckIdx = sel["deck_index"]!.AsArray().Select(x => (int)x!).ToList();
            var pending = new List<int>(deckIdx);
            for (int k = 0; k < picked.Count; k++)
            {
                if (s.Phase != RunPhase.TransformSelect) throw new Divergence($"selection ended after {k} of {picked.Count} picks");
                int action;
                if (s.PendingOfferCards.Length > 0)
                {
                    string id = (string)options[picked[k]]![0]!;
                    action = Array.FindIndex(s.PendingOfferCards, c => RunSnap.CardEntry(c) == id);
                    if (action < 0) throw new Divergence($"offered card {id} not in the emulator's grid");
                }
                else
                {
                    action = pending[k];
                    if (action < 0) throw new Divergence("picked card not in the deck");
                }

                var before = s.Deck.ToList();
                Step(e, action, od, "select card");
                // A card the pick removed (or moved to the end) shifts the later picks down by one.
                bool shifted = s.Deck.Count < before.Count
                    || (s.Deck.Count == before.Count && action < before.Count - 1
                        && s.Deck.Skip(action).Take(before.Count - 1 - action).SequenceEqual(before.Skip(action + 1)));
                if (shifted)
                    for (int j = k + 1; j < pending.Count; j++) if (pending[j] > action) pending[j]--;
            }

            // A grid that allows fewer picks than it holds is closed with DONE.
            if (s.Phase == RunPhase.TransformSelect && s.PendingOfferCards.Length > 0 && s.PendingOfferOptional)
                Step(e, s.PendingOfferCards.Length, od, "selection done");
            // So is a deck selection whose minimum is zero (Kifuda, Gnarled Hammer, Claws).
            else if (s.Phase == RunPhase.TransformSelect && s.PendingSelectionKind != DeckSelection.None && s.PendingSelectionOptional)
                Step(e, s.Deck.Count, od, "selection done");
        }
    }

    static MethodInfo? _fightDump;

    /// <summary>Fight.Dump: the per-step state in the shape fight_compare reads.</summary>
    static JsonObject FightDump(RunEngine e)
    {
        _fightDump ??= typeof(Fight).GetMethod("Dump", BindingFlags.NonPublic | BindingFlags.Static);
        return (JsonObject)_fightDump!.Invoke(null, new object[] { e })!;
    }

    /// <summary>The fight's recorded actions, replayed as Fight.Replay replays a fight spec.</summary>
    static void Combat(ref RunEngine e, JsonObject d, JsonObject od, bool fights)
    {
        // A faithful copy from before the fight, in case the emulator's fight goes worse than the
        // game's and has to be won without playing it (anchoring).
        var beforeFight = Anchoring ? e.Clone() : null;
        var s = e.State;
        Expect(e, RunPhase.Combat, "combat");
        od["encounter"] = RunSnap.EncounterEntry(s.ActiveCombat?.EncounterId ?? 0);
        var fight = d["fight"]!.AsObject();
        var steps = new JsonArray();
        if (fights) od["fight"] = new JsonObject { ["steps"] = steps };
        if (fights) steps.Add(new JsonObject { ["action"] = null, ["state"] = FightDump(e) });
        var gameChoices = (fight["choices"]?.AsArray() ?? new JsonArray()).ToList();
        int stepNo = 0;
        foreach (var step in fight["steps"]!.AsArray().Skip(1))
        {
            stepNo++;
            var a = step!["action"]!.AsObject();
            if (s.ActiveCombat == null || s.Phase != RunPhase.Combat)
            {
                if (!Anchoring) throw new Divergence($"fight over in the emulator before game step {stepNo}");
                od["emulator_fight_ended"] = stepNo;
                break;
            }
            if ((string)a["kind"]! == "end_turn")
                e.Step(s.ActiveCombat.Hand.Count, -1, out _, out _, out _);
            else if ((string)a["kind"]! == "potion")
            {
                // CombatEngine's potion actions follow end turn: hand.Count + 1 + slot.
                int slot = (int)a["slot"]!;
                string want = Bare((string?)a["potion"]);
                if (RunSnap.PotionEntry(s.PotionSlots[slot]) != want)
                {
                    // The fight has already gone its own way (a potion used or made differently):
                    // a fight divergence; with anchoring the run goes on from the game's outcome.
                    string msg = $"potion slot {slot}: game {want}, emulator {RunSnap.PotionEntry(s.PotionSlots[slot])}";
                    if (!Anchoring) throw new Divergence(msg);
                    od["emulator_fight_ended"] = $"{stepNo}: {msg}";
                    break;
                }
                e.Step(s.ActiveCombat.Hand.Count + 1 + slot, (int)a["target_slot"]!, out _, out _, out _);
            }
            else if ((bool?)a["played"] != false)
                e.Step((int)a["hand_index"]!, (int)a["target_slot"]!, out _, out _, out _);
            var chosen = new JsonArray();
            var engine = e;
            foreach (var choice in gameChoices.Where(c => (int?)c!["step"] == stepNo && (string?)c["kind"] == "cards"))
                chosen.Add(Fight.ApplyChoice(() => s.ActiveCombat, i => engine.Step(i, -1, out _, out _, out _), choice!.AsObject()));
            if (fights) steps.Add(new JsonObject { ["action"] = a.DeepClone(), ["choices"] = chosen, ["state"] = FightDump(e) });
        }

        string result = (string?)d["result"] ?? "";
        // The emulator's own outcome, before anything is anchored: the comparer checks it
        // against the game's post-fight save (and the reward screen against the game's next
        // offer), so a fight identical at every step that still ends elsewhere is a divergence.
        od["fight_end"] = s.Phase.ToString();
        od["outcome_snap"] = RunSnap.Export(e);
        if (s.Phase is RunPhase.RelicReward or RunPhase.CardReward) od["outcome_offer"] = RewardOffer(s);
        if (!Anchoring)
        {
            if (result == "win" && s.Phase == RunPhase.Combat) throw new Divergence("game won the fight; emulator fight still running");
            if (result == "death" && s.Phase != RunPhase.Complete) throw new Divergence($"game died; emulator in {s.Phase}");
            return;
        }

        // Anchoring: the fight's own fidelity is the fight corpus's business; the run goes on from
        // where the GAME came out of it, so every later screen is still compared.
        if (result == "death") return;
        var anchored = new JsonArray();
        if (s.Phase == RunPhase.Combat)
        {
            e.DebugResolveCombat();  // the game won: win it here too, through FinishCombat
            anchored.Add("outcome");
        }
        else if (s.Phase == RunPhase.Complete)
        {
            // The emulator's fight killed the player where the game's did not: go back to the
            // fight's start and win it there without playing it.
            e = beforeFight!;
            s = e.State;
            e.DebugResolveCombat();
            anchored.Add("outcome (emulator died)");
        }

        Anchor(e, d["snap"]!.AsObject(), anchored);
        if (anchored.Count > 0 && NextDecision is { } next && (string?)next["screen"] == "rewards" && next["offer"] is JsonArray offer
            && s.Phase == RunPhase.RelicReward && RewardOffer(s).ToJsonString() != offer.ToJsonString())
        {
            ForceRewardScreen(s, offer);
            anchored.Add("reward_screen");
        }

        if (anchored.Count > 0) od["anchored"] = anchored;
    }

    /// <summary>The game decision after the one being applied, for anchoring.</summary>
    static JsonObject? NextDecision;

    /// <summary>
    /// A diverged fight's rewards are the emulator's own rolls off its own fight; the game's
    /// screen is put up instead (its first rewards decision's offer), so the run stays comparable.
    /// </summary>
    static void ForceRewardScreen(RunState s, JsonArray offer)
    {
        RunRewardGenerator.ClearRewardScreen(s);
        s.PendingPotionRewards.Clear();
        s.PendingBonusRelicRewards.Clear();
        s.PendingRelicReward = false;
        s.PendingCardOffers.Clear();
        foreach (var r in offer)
        {
            switch ((string?)r!["type"])
            {
                case "gold":
                    if (s.RewardGold == 0) s.RewardGold = (int)r["amount"]!; else s.PendingGoldRewards.Add((int)r["amount"]!);
                    break;
                case "potion":
                    int pot = PotionId(Bare((string?)r["id"]));
                    if (s.RewardPotion == 0) s.RewardPotion = pot; else s.PendingPotionRewards.Add(pot);
                    break;
                case "relic":
                    int rel = RelicId(Bare((string?)r["id"]));
                    if (s.RelicReward == 0) s.RelicReward = rel; else s.PendingBonusRelicRewards.Add(rel);
                    break;
                case "card":
                    var cards = r["cards"]!.AsArray().Select(c => Fight.CardIdBySnake(Bare((string?)c![0]))).ToArray();
                    var upg = r["cards"]!.AsArray().Select(c => (int)c![1]! != 0).ToArray();
                    if (!s.RewardCardPending) { new CardOffer(cards, upg).ShowOn(s); s.RewardCardPending = true; }
                    else s.PendingCardOffers.Add(new CardOffer(cards, upg));
                    break;
            }
        }
    }

    /// <summary>Whether fights are anchored to the game's outcome (on unless --no-anchor).</summary>
    internal static bool Anchoring = true;

    static Dictionary<string, int>? _relicIds, _potionIds;

    static int RelicId(string entry) => (_relicIds ??= Sts2Emulator.GeneratedData.Relics.All.ToArray().ToDictionary(r => r.Entry, r => r.Id))[entry];

    static int PotionId(string entry)
    {
        _potionIds ??= Enumerable.Range(1, 200).Where(i => { try { return Sts2Emulator.GeneratedData.Potions.Get(i).Name != null; } catch { return false; } })
            .GroupBy(i => RunSnap.PotionEntry(i)).ToDictionary(g => g.Key, g => g.First());
        return _potionIds[entry];
    }

    static string Bare(string? id) => id == null ? "" : id.Contains('.') ? id[(id.IndexOf('.') + 1)..] : id;

    /// <summary>
    /// Puts the emulator's holdings and every RNG counter where the game's save has them
    /// (RunEngine.DebugSetPlayer for the holdings), naming each field that had to move.
    /// </summary>
    static void Anchor(RunEngine e, JsonObject save, JsonArray anchored)
    {
        var s = e.State;
        var p = save["players"]![0]!.AsObject();
        AnchorHoldings(e, p, anchored);
        AnchorStreams(e, save, p, anchored);
    }

    /// <summary>
    /// The holdings half of <see cref="Anchor" />: HP, gold, deck, relics, potions and slots from a save's
    /// player (the game's save shape), then Toy Box's wax, the relics' saved properties RunSnap.RelicProps
    /// compares, Mad Science's type and rider and Guilty's combats. The reference set's player_load item
    /// (RefsetSetup.cs) puts a real run's holdings in place with it.
    /// </summary>
    internal static void AnchorHoldings(RunEngine e, JsonObject p, JsonArray anchored)
    {
        var s = e.State;
        int hp = (int)p["current_hp"]!, maxHp = (int)p["max_hp"]!, gold = (int)p["gold"]!;
        if (hp != s.PlayerHp) anchored.Add("hp");
        if (maxHp != s.PlayerMaxHp) anchored.Add("max_hp");
        if (gold != s.Gold) anchored.Add("gold");
        int[]? cards = null;
        var deck = p["deck"]!.AsArray().Select(c => (
            Id: Fight.CardIdBySnake(Bare((string?)c!["id"])),
            Up: ((int?)c["current_upgrade_level"] ?? 0) > 0,
            Ench: c["enchantment"] is { } en ? Enum.Parse<Enchantment>(Pascal(Bare((string?)en["id"]))) : Enchantment.None,
            Amt: c["enchantment"] is { } en2 ? (int?)en2["amount"] ?? 0 : 0)).ToList();
        if (!deck.SequenceEqual(s.Deck.Select(c => (c.DefId, c.Upgraded, c.Enchantment, c.Enchantment == Enchantment.None ? 0 : c.EnchantAmount))))
        {
            anchored.Add("deck");
            cards = deck.SelectMany(c => new[] { c.Id, c.Up ? 1 : 0, (int)c.Ench, c.Amt }).ToArray();
        }

        int[]? relics = null;
        // A melted wax relic stays in the game's list, inert (the hooks skip IsMelted); the
        // emulator drops it as it melts, so only the unmelted ones are the emulator's relics.
        var held = p["relics"]!.AsArray().Where(r => !GameFlag(r!, "IsMelted")).ToList();
        var rel = held.Select(r => RelicId(Bare((string?)r!["id"]))).ToArray();
        if (!rel.SequenceEqual(s.Relics.Select(r => r.DefId))) { anchored.Add("relics"); relics = rel; }
        int[]? potions = null;
        int slots = (int)p["max_potion_slot_count"]!;
        var pots = new int[slots];
        foreach (var pot in p["potions"]?.AsArray() ?? new JsonArray())
            if ((int)pot!["slot_index"]! < slots) pots[(int)pot["slot_index"]!] = PotionId(Bare((string?)pot["id"]));
        if (!pots.SequenceEqual(s.PotionSlots.Take(slots)) || slots != s.MaxPotionSlots) { anchored.Add("potions"); potions = pots; }
        e.DebugSetPlayer(hp, maxHp, gold, cards, relics, potions, slots);

        // Toy Box's wax relics, in the game's relic order (the first held one melts next).
        var wax = held.Where(r => GameFlag(r!, "IsWax")).Select(r => RelicId(Bare((string?)r!["id"]))).ToList();
        if (!wax.SequenceEqual(s.Relics.Where(r => s.WaxRelics.Contains(r.DefId)).Select(r => r.DefId)))
        {
            anchored.Add("wax");
            s.WaxRelics.RemoveAll(id => s.Relics.Any(r => r.DefId == id));
            s.WaxRelics.InsertRange(0, wax);
        }

        // The relics' saved properties the snapshot compares (RunSnap.RelicProps).
        foreach (var prop in RunSnap.RelicProps)
        {
            int i = s.Relics.FindIndex(r => RunSnap.RelicEntry(r.DefId) == prop.Relic);
            var gr = held.FirstOrDefault(r => Bare((string?)r!["id"]) == prop.Relic);
            if (i < 0 || gr == null || (GameProp(gr, prop.Kind, prop.Name) ?? prop.Absent) is not { } want
                || prop.Get(s, s.Relics[i]) == want) continue;
            anchored.Add($"relic_props.{prop.Relic}.{prop.Name}");
            prop.Set(s, i, want);
        }

        // Card properties: Mad Science's type and rider by deck position (the deck is the
        // game's by now), Guilty's combats served as the game's counts.
        var gdeck = p["deck"]!.AsArray();
        var guilty = new List<int>();
        for (int k = 0; k < gdeck.Count && k < s.Deck.Count; k++)
        {
            string id = Bare((string?)gdeck[k]!["id"]);
            if (id == "GUILTY") guilty.Add(GameProp(gdeck[k]!, "ints", "CombatsSeen") ?? 0);
            if (id != "MAD_SCIENCE") continue;
            var type = (Sts2Emulator.Core.CardType)Enum.Parse(typeof(Sts2Emulator.Core.CardType), GameCardTypes[GameProp(gdeck[k]!, "ints", "TinkerTimeType") ?? 0]);
            var rider = (Sts2Emulator.Core.TinkerRider)(GameProp(gdeck[k]!, "ints", "TinkerTimeRider") ?? 0);
            if (s.Deck[k].TinkerType == type && s.Deck[k].TinkerRider == rider) continue;
            anchored.Add("card_props.MAD_SCIENCE");
            s.Deck[k] = s.Deck[k] with { TinkerType = type, TinkerRider = rider };
        }

        if (!guilty.OrderBy(x => x).SequenceEqual(s.GuiltyCombatsServed.Take(guilty.Count).Concat(Enumerable.Repeat(0, Math.Max(0, guilty.Count - s.GuiltyCombatsServed.Count))).OrderBy(x => x)))
        {
            anchored.Add("card_props.GUILTY");
            s.GuiltyCombatsServed.Clear();
            s.GuiltyCombatsServed.AddRange(guilty);
        }
    }

    /// <summary>The RNG counters, odds and relic queues half of <see cref="Anchor" />.</summary>
    static void AnchorStreams(RunEngine e, JsonObject save, JsonObject p, JsonArray anchored)
    {
        var s = e.State;

        void Counters(object set, JsonObject counters, string prefix)
        {
            foreach (var prop in set.GetType().GetProperties().Where(x => x.PropertyType == typeof(Sts2Emulator.Core.Rng.GameRng)))
            {
                string key = RunSnap.Snake(prop.Name).ToLowerInvariant();
                if (counters[key] is not JsonValue v) continue;
                var rng = (Sts2Emulator.Core.Rng.GameRng)prop.GetValue(set)!;
                int want = (int)v;
                if (rng.CallCount == want) continue;
                anchored.Add($"{prefix}.{key}");
                if (rng.CallCount < want) { rng.AdvanceToCallCount(want); continue; }
                // Behind the emulator: a fresh copy of the stream, moved to the game's position.
                var fresh = new Sts2Emulator.Core.Rng.GameRng((uint)RunSeedOf(set), key);
                fresh.AdvanceToCallCount(want);
                RewindTo(rng, fresh);
            }
        }

        // The odds a diverged fight's own rewards moved (card rarity, potion reward), and the map's.
        void Odds(string name, double want, Func<double> get, Action<double> set)
        {
            if (Math.Round(get(), 6) == Math.Round(want, 6)) return;
            anchored.Add($"odds.{name}");
            set(want);
        }

        var po = p["odds"]!;
        var ro = save["odds"]!;
        Odds("card_rarity", (double)po["card_rarity_odds_value"]!, () => s.CardRarityOffset, v => s.CardRarityOffset = v);
        Odds("potion_reward", (double)po["potion_reward_odds_value"]!, () => s.PotionRewardOdds, v => s.PotionRewardOdds = v);
        Odds("unknown_monster", (double)ro["unknown_map_point_monster_odds_value"]!, () => s.UnknownMapPointMonsterOdds, v => s.UnknownMapPointMonsterOdds = v);
        Odds("unknown_elite", (double)ro["unknown_map_point_elite_odds_value"]!, () => s.UnknownMapPointEliteOdds, v => s.UnknownMapPointEliteOdds = v);
        Odds("unknown_treasure", (double)ro["unknown_map_point_treasure_odds_value"]!, () => s.UnknownMapPointTreasureOdds, v => s.UnknownMapPointTreasureOdds = v);
        Odds("unknown_shop", (double)ro["unknown_map_point_shop_odds_value"]!, () => s.UnknownMapPointShopOdds, v => s.UnknownMapPointShopOdds = v);

        Counters(s.Rng, save["rng"]!["counters"]!.AsObject(), "rng");
        Counters(s.PlayerRng, p["rng"]!["counters"]!.AsObject(), "player_rng");

        // The relic queues: a diverged fight's own rewards pulled the emulator's relic, not the game's.
        void Bag(RelicGrabBag bag, JsonObject? lists, string name)
        {
            if (lists == null) return;
            var want = lists.Select(kv => (Rarity: Enum.Parse<Sts2Emulator.Core.RelicRarity>(kv.Key, ignoreCase: true),
                Relics: kv.Value!.AsArray().Select(x => RelicId(Bare((string?)x))).ToList())).ToList();
            var have = bag.Lists.ToDictionary(l => l.Rarity, l => l.Relics.ToList());
            if (want.All(w => have.TryGetValue(w.Rarity, out var h) ? h.SequenceEqual(w.Relics) : w.Relics.Count == 0)) return;
            anchored.Add(name);
            bag.ReplaceListsForReplay(want);
        }

        Bag(s.RelicBag, p["relic_grab_bag"]?["relic_id_lists"]?.AsObject(), "grab_bag");
        Bag(s.SharedRelicBag, save["shared_relic_grab_bag"]?["relic_id_lists"]?.AsObject(), "shared_grab_bag");
    }

    /// <summary>The game's CardType names in enum order (its save writes the number).</summary>
    internal static readonly string[] GameCardTypes = ["None", "Attack", "Skill", "Power", "Status", "Curse", "Quest"];

    /// <summary>A saved property off a relic or card in the game's save (props.ints / props.bools), bools as 0/1.</summary>
    static int? GameProp(JsonNode model, string kind, string name)
    {
        foreach (var it in model["props"]?[kind]?.AsArray() ?? new JsonArray())
        {
            if ((string?)it!["name"] != name) continue;
            return it["value"] is JsonValue v && v.TryGetValue<bool>(out bool b) ? (b ? 1 : 0) : (int?)it["value"];
        }

        return null;
    }

    static bool GameFlag(JsonNode model, string name) => GameProp(model, "bools", name) == 1;

    static uint RunSeedOf(object set) => (uint)set.GetType().GetProperty("Seed")!.GetValue(set)!;

    /// <summary>A stream set back to another copy's position: the tool's own reach into GameRng, never the engine's.</summary>
    static void RewindTo(Sts2Emulator.Core.Rng.GameRng rng, Sts2Emulator.Core.Rng.GameRng fresh)
    {
        const BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;
        var gen = typeof(Sts2Emulator.Core.Rng.GameRng).GetField("_rng", F)!;
        gen.SetValue(rng, gen.GetValue(fresh));
        typeof(Sts2Emulator.Core.Rng.GameRng).GetField("<CallCount>k__BackingField", F)!.SetValue(rng, fresh.CallCount);
    }

    static string Pascal(string snake) => string.Concat(snake.Split('_').Select(w => w.Length == 0 ? w : w[0] + w[1..].ToLowerInvariant()));
}
