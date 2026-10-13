// The codex replay: walk one real run's history (a Spire Codex / game `.run` record: seed,
// character, ascension and, floor by floor, what was offered and what was picked) through the
// emulator's run engine, and compare at every floor what the emulator OFFERS with what the game
// offered. Fights are not played: each is resolved, and HP, max HP, gold, potions and the deck are
// set to what the history shows after it. The walk stops at the first difference.
//
// Every difference is classed by where the emulator's offer came from:
//   EMULATOR        the offer draws only on streams and state no fight touches;
//   FIGHT-DEPENDENT the offer drew on a stream a resolved fight left behind the game's (the fight's
//                   own turns, a creature that joined mid-fight, a reward the fight added);
//   WALKER          the history cannot be followed here (an unmodelled screen, an unreadable pick).
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Codex;

public sealed class Walker
{
    public const string Emulator = "EMULATOR";
    public const string FightDependent = "FIGHT-DEPENDENT";
    public const string WalkerLimit = "WALKER";

    /// <summary>
    /// The player's profile decided it, and the history does not carry the profile: which acts
    /// were discovered (`ActModel.GetRandomList` forces an undiscovered alternate act, and only
    /// an unlocked one can be rolled), and which characters are unlocked (Kaleidoscope is only
    /// offered at Neow with every character's pool unlocked).
    /// </summary>
    public const string Profile = "PROFILE-DEPENDENT";

    /// <summary>How the class of a difference is decided.</summary>
    enum Source
    {
        /// <summary>Generation no fight reaches: act and map rolls, encounters, Neow, unknown rooms.</summary>
        Static,

        /// <summary>The Rewards stream: fight-dependent once a fight has added rewards of its own.</summary>
        Rewards,

        /// <summary>Whatever the emulator drew since the last mark, against the tainted streams.</summary>
        Drawn,

        /// <summary>Whatever the emulator drew on the floor so far (after any fight was set up).</summary>
        FloorDrawn,
    }

    sealed class Stop(JsonObject mismatch) : Exception
    {
        public JsonObject Mismatch { get; } = mismatch;
    }

    RunEngine _e = new();
    readonly EventOptionIndex _events;
    RunState S => _e.State;

    // --- per run ---
    readonly Dictionary<string, int[]> _checks = []; // component -> [floors checked, floors matched]
    readonly HashSet<string> _floorComponents = [];
    readonly HashSet<string> _floorFailed = [];
    readonly Dictionary<string, string> _taint = []; // stream -> why
    readonly int[] _mark = new int[Streams.Names.Length];
    readonly int[] _floorMark = new int[Streams.Names.Length];
    readonly int[] _now = new int[Streams.Names.Length];
    int _act, _floor, _ambiguousFloors, _fights, _checkableFloors, _uncheckableFloors, _fightAddedRewards;
    bool _floorUsedTaint;
    List<string[]> _actTypes = [];
    int _actPos;

    // --- per floor: the history's lists and how far into each the walk is ---
    JsonElement _stats;
    bool _hasStats;
    int _upgradeCursor, _removeCursor, _transformCursor, _enchantCursor, _eventCursor,
        _restCursor, _gainCursor, _shopBuyCursor;
    readonly HashSet<int> _relicChoiceUsed = [];
    readonly HashSet<int> _potionChoiceUsed = [];
    readonly List<string> _discardsLeft = [];
    bool _rewardsCompared, _shopRemovalDone;

    public Walker(EventOptionIndex events) => _events = events;

    // =====================================================================================
    // entry
    // =====================================================================================

    public JsonObject Walk(JsonElement rec)
    {
        _checks.Clear();
        _taint.Clear();
        _ambiguousFloors = _fights = _checkableFloors = _uncheckableFloors = _fightAddedRewards = 0;
        string seed = rec.GetProperty("seed").GetString() ?? "";
        int asc = rec.GetProperty("ascension").GetInt32();
        var character = Names.Character(rec.GetProperty("players")[0].GetProperty("character").GetString()) ?? PlayerCharacter.Ironclad;
        bool win = rec.TryGetProperty("win", out var w) && w.ValueKind == JsonValueKind.True;
        var history = rec.GetProperty("map_point_history");
        int floorsRecorded = history.EnumerateArray().Sum(a => a.GetArrayLength());
        var result = new JsonObject
        {
            ["seed"] = seed,
            ["ascension"] = asc,
            ["win"] = win,
            ["abandoned"] = rec.TryGetProperty("was_abandoned", out var ab) && ab.ValueKind == JsonValueKind.True,
            ["floors_recorded"] = floorsRecorded,
        };
        _floor = 0;
        try
        {
            // A fresh engine a run: Reset reuses its state object, and a field it forgets to
            // clear would carry one run's leftovers into the next.
            long r0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _e = new RunEngine();
            _e.Reset(seed, asc, character);
            _belt.Clear();
            _eventsSeen = 0;
            ResetSeconds += System.Diagnostics.Stopwatch.GetElapsedTime(r0).TotalSeconds;
            Mark();
            BeginFloorBookkeeping();
            string recActs = string.Join(",", rec.GetProperty("acts").EnumerateArray().Select(a => Names.Bare(a.GetString())));
            string emuActs = string.Join(",", S.Acts.Select(a => Names.Act(a.Act)));
            if (recActs != emuActs)
            {
                // The act roll is the profile's: an alternate act not yet discovered is taken
                // without a draw, and one not unlocked cannot be drawn.
                Fail("acts", "acts", recActs, emuActs, Profile, "act 1 is forced (undiscovered) or barred (locked) by the profile");
            }

            Count("acts", true);
            EndFloorBookkeeping();

            _actTypes = history.EnumerateArray()
                .Select(a => a.EnumerateArray().Select(p => p.GetProperty("map_point_type").GetString() ?? "").ToArray())
                .ToList();
            int act = 0;
            foreach (var points in history.EnumerateArray())
            {
                _act = act;
                int pos = 0;
                foreach (var point in points.EnumerateArray())
                {
                    _floor++;
                    _actPos = pos++;
                    bool last = _floor == floorsRecorded && !win;
                    WalkFloor(point, last);
                }

                act++;
            }

            result["status"] = "complete";
        }
        catch (Stop stop)
        {
            result["status"] = "mismatch";
            result["stop"] = stop.Mismatch;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            result["status"] = "error";
            result["error"] = $"{ex.GetType().Name}: {ex.Message}";
            result["error_floor"] = _floor;
        }

        result["floors_walked"] = _floor;
        result["fights"] = _fights;
        result["ambiguous_path_floors"] = _ambiguousFloors;
        result["fight_added_rewards"] = _fightAddedRewards;
        result["checkable_floors"] = _checkableFloors;
        result["uncheckable_floors"] = _uncheckableFloors;
        var checks = new JsonObject();
        foreach (var (k, v) in _checks.OrderBy(kv => kv.Key))
        {
            checks[k] = new JsonArray(v[0], v[1]);
        }

        result["checks"] = checks;
        return result;
    }

    // =====================================================================================
    // one floor
    // =====================================================================================

    void WalkFloor(JsonElement point, bool lastOfLostRun)
    {
        string kind = point.GetProperty("map_point_type").GetString() ?? "";
        var rooms = point.TryGetProperty("rooms", out var r) ? r.EnumerateArray().ToList() : [];
        _hasStats = point.TryGetProperty("player_stats", out var ps) && ps.GetArrayLength() > 0;
        _stats = _hasStats ? ps[0] : default;
        ResetFloorCursors();
        BeginFloorBookkeeping();
        var deckBefore = DeckKeys();
        var relicsBefore = S.Relics.Select(x => Names.Relic(x.DefId)).ToList();
        int goldBefore = S.Gold;
        Mark();
        MarkFloor();

        var room = rooms.Count > 0 ? rooms[0] : default;
        string roomType = rooms.Count > 0 && room.TryGetProperty("room_type", out var rt) ? rt.GetString() ?? "" : "";
        // A floor with a fight in it (an event's included) has the fight's HP, gold and deck.
        bool combat = rooms.Any(x => x.TryGetProperty("room_type", out var t) && t.GetString() is "monster" or "elite" or "boss");
        _combatFloor = combat;

        if (kind == "ancient")
        {
            AtAncient(room);
        }
        else
        {
            if (S.Phase != RunPhase.Map)
            {
                Fail("flow", "screen", "map", S.Phase.ToString(), WalkerLimit, "the previous floor did not end on the map");
            }

            TakeNode(kind);
            EnterRoom(kind, roomType, rooms, lastOfLostRun);
            if (lastOfLostRun)
            {
                EndFloorBookkeeping();
                return;
            }
        }

        if (lastOfLostRun)
        {
            EndFloorBookkeeping();
            return;
        }

        Settle(kind, rooms);
        AfterFloor(kind, combat, deckBefore, relicsBefore, goldBefore);
        EndFloorBookkeeping();
    }

    void ResetFloorCursors()
    {
        _upgradeCursor = _removeCursor = _transformCursor = _enchantCursor = 0;
        _eventCursor = _restCursor = _gainCursor = _shopBuyCursor = 0;
        _relicChoiceUsed.Clear();
        _potionChoiceUsed.Clear();
        _discardsLeft.Clear();
        _discardsLeft.AddRange(StrList("potion_discarded"));
        _discardsLeft.AddRange(StrList("potion_used"));
        _rewardsCompared = false;
        _shopRemovalDone = false;
        _floorUsedTaint = false;
        _cardCursor = 0;
        _goldUnknown = false;
        _skippedPotions.Clear();
        _skippedRelics.Clear();
        _shopSeen = false;
        _shopGoldLent = 0;
        _shopSpent = 0;
        _fakeBuyCursor = 0;
        _fightJustResolved = false;
    }

    // --- the ancient -----------------------------------------------------------------------

    void AtAncient(JsonElement room)
    {
        if (_act > 0)
        {
            // A later act's ancient is the first point of its map, walked onto.
            if (S.Phase != RunPhase.Map || S.CurrentActIndex != _act)
            {
                Fail("flow", "act start", $"act {_act + 1} map", $"{S.Phase} act {S.CurrentActIndex + 1}", WalkerLimit);
            }

            int at = Array.FindIndex(S.MapNodeTypes, t => t == RunConstants.NodeAncient);
            if (at < 0)
            {
                Fail("map", "ancient node", "ancient", string.Join(",", S.MapNodeTypes.Select(Names.MapType)), Source.Static);
            }

            Do(at);
        }

        if (S.Phase != RunPhase.Ancient)
        {
            Fail("flow", "screen", "ancient", S.Phase.ToString(), WalkerLimit);
        }

        string model = room.ValueKind == JsonValueKind.Object && room.TryGetProperty("model_id", out var m) ? Names.Bare(m.GetString()) : "";
        Check("ancient", "ancient", model, S.Ancient, Source.Static);
        var real = Arr("ancient_choice").Select(o => o.GetProperty("TextKey").GetString() ?? "").ToList();
        var emu = S.NeowOptions.Where(x => x != 0).Select(Names.Relic).ToList();
        if (real.Count == 0)
        {
            // An abandoned run can end on the ancient before anything was chosen or recorded.
            return;
        }

        if (_act == 0 && string.Join(",", real) != string.Join(",", emu) && NeowWithout(real) is string barred)
        {
            Fail("ancient", "options", string.Join(",", real), string.Join(",", emu), Profile,
                $"Neow's options match with {barred} barred, which the profile decides");
        }

        Check("ancient", "options", string.Join(",", real), string.Join(",", emu), _act == 0 ? Source.Static : Source.Drawn);
        string? chosen = Arr("ancient_choice").Where(o => o.TryGetProperty("was_chosen", out var c) && c.ValueKind == JsonValueKind.True)
            .Select(o => o.GetProperty("TextKey").GetString()).FirstOrDefault();
        int index = chosen is null ? -1 : emu.IndexOf(chosen);
        if (index < 0)
        {
            Fail("ancient", "choice", chosen ?? "(none)", string.Join(",", emu), WalkerLimit);
        }

        // The blessing is the floor's first relic choice.
        ConsumeRelicChoice(Names.Relic(S.NeowOptions.Where(x => x != 0).ElementAt(index)));
        Mark();
        Do(index);
    }

    /// <summary>
    /// Neow's options as `Neow.GenerateInitialOptions` rolls them when `IsAllowedAtNeow` bars
    /// Kaleidoscope (not every character unlocked) or Scroll Boxes (no bundles to make), the same
    /// draws the emulator's own `GenerateNeowOptions` makes: which barring, if any, gives the
    /// history's options.
    /// </summary>
    string? NeowWithout(List<string> real)
    {
        int[][] barrings = [[RunConstants.RelicKaleidoscope], [RunConstants.RelicScrollBoxes],
            [RunConstants.RelicKaleidoscope, RunConstants.RelicScrollBoxes]];
        foreach (var barred in barrings)
        {
            var rng = S.Rng.NeowRng();
            int[] curses = RunConstants.NeowCurseOptions.ToArray().Where(r => !barred.Contains(r)).ToArray();
            int cursed = curses[rng.NextInt(curses.Length)];
            var positive = RunConstants.NeowPositiveOptions.ToArray().ToList();
            if (cursed == RunConstants.RelicCursedPearl) positive.Remove(RunConstants.RelicGoldenPearl);
            else if (cursed == RunConstants.RelicHeftyTablet) positive.Remove(RunConstants.RelicArcaneScroll);
            else if (cursed == RunConstants.RelicLeafyPoultice) positive.Remove(RunConstants.RelicNewLeaf);
            else if (cursed == RunConstants.RelicPrecariousShears) positive.Remove(RunConstants.RelicPreciseScissors);
            if (cursed != RunConstants.RelicLargeCapsule)
            {
                positive.Add(rng.NextBool() ? RunConstants.RelicLavaRock : RunConstants.RelicSmallCapsule);
            }

            positive.Add(rng.NextBool() ? RunConstants.RelicNutritiousOyster : RunConstants.RelicStoneHumidifier);
            positive.Add(rng.NextBool() ? RunConstants.RelicNeowsTalisman : RunConstants.RelicPomander);
            positive.RemoveAll(barred.Contains);
            rng.Shuffle(positive);
            var options = new[] { positive[0], positive[1], cursed }.Select(Names.Relic);
            if (options.SequenceEqual(real))
            {
                return string.Join(" and ", barred.Select(Names.Relic));
            }
        }

        return null;
    }

    /// <summary>
    /// `Overgrowth.ApplyActDiscoveryOrderModifications`: on a profile's first run ever the act's
    /// first rooms are swapped into a set order. Whether the history's room is the one that
    /// order puts at this position.
    /// </summary>
    bool FirstRunOrder(string model)
    {
        if (_act != 0 || S.Act != RunConstants.ActOvergrowth)
        {
            return false;
        }

        string[] normal = ["NIBBITS_WEAK", "SLIMES_WEAK", "SHRINKER_BEETLE_WEAK", "INKLETS_NORMAL", "MAWLER_NORMAL", "RUBY_RAIDERS_NORMAL", "NIBBITS_NORMAL"];
        string[] elite = ["BYRDONIS_ELITE", "PHROG_PARASITE_ELITE"];
        string[] events = ["BYRDONIS_NEST", "SAPPHIRE_SEED"];
        if (S.Phase == RunPhase.Combat && S.ActiveCombat is not null)
        {
            var order = S.CurrentNodeType == RunConstants.NodeElite ? elite : normal;
            int at = (S.CurrentNodeType == RunConstants.NodeElite ? S.EliteEncountersVisited : S.NormalEncountersVisited) - 1;
            return at >= 0 && at < order.Length && order[at] == model;
        }

        return _eventsSeen < events.Length && events[_eventsSeen] == model;
    }

    int _eventsSeen;

    // --- the map ---------------------------------------------------------------------------

    void TakeNode(string kind)
    {
        int wanted = kind switch
        {
            "monster" => RunConstants.NodeNormal,
            "elite" => RunConstants.NodeElite,
            "rest_site" => RunConstants.NodeRest,
            "shop" => RunConstants.NodeShop,
            "treasure" => RunConstants.NodeRelic,
            "boss" => RunConstants.NodeBoss,
            "unknown" => RunConstants.NodeEvent,
            _ => -1,
        };
        if (wanted < 0)
        {
            Fail("map", "point type", kind, "(unmodelled)", WalkerLimit);
        }

        var ahead = _actTypes[_act].Skip(_actPos + 1).ToArray();
        var candidates = new List<int>();
        for (int i = 0; i < S.MapNodeTypes.Length; i++)
        {
            if (S.MapNodeTypes[i] == wanted && S.MapOptionCoords[i] is { } c && Walkable(c, ahead))
            {
                candidates.Add(i);
            }
        }

        if (candidates.Count == 0)
        {
            if (Trace)
            {
                foreach (var row in S.MapNodes.Values.GroupBy(n => n.Row).OrderBy(g => g.Key))
                {
                    Console.Error.WriteLine($"    row {row.Key}: " + string.Join("  ", row.OrderBy(n => n.Col).Select(n =>
                        $"{n.Col}:{Names.MapType(n.NodeType)}->{string.Join("/", n.Children.Select(c => c.Col))}")));
                }
            }

            string offered = string.Join(",", Enumerable.Range(0, S.MapNodeTypes.Length)
                .Where(i => S.MapNodeTypes[i] != RunConstants.NodeNone).Select(i => Names.MapType(S.MapNodeTypes[i])));
            Fail("map", "path", string.Join(">", new[] { kind }.Concat(ahead)), offered, Source.Static,
                "no node of the history's type from which the rest of its act can be walked");
        }

        Count("map", true);
        if (candidates.Count > 1)
        {
            _ambiguousFloors++;
        }

        Do(candidates[0]);
    }

    /// <summary>Whether the history's remaining point types can be walked from a node.</summary>
    bool Walkable((int Col, int Row) from, string[] ahead)
    {
        var memo = new Dictionary<((int, int), int), bool>();
        bool Go((int Col, int Row) node, int step)
        {
            if (step >= ahead.Length)
            {
                return true;
            }

            if (memo.TryGetValue((node, step), out bool known))
            {
                return known;
            }

            bool ok = false;
            if (S.MapNodes.TryGetValue(node, out var n))
            {
                foreach (var child in n.Children)
                {
                    if (S.MapNodes.TryGetValue(child, out var cn) && Names.MapType(cn.NodeType) == ahead[step] && Go(child, step + 1))
                    {
                        ok = true;
                        break;
                    }
                }
            }

            memo[(node, step)] = ok;
            return ok;
        }

        return Go(from, 0);
    }

    // --- entering the room -----------------------------------------------------------------

    void EnterRoom(string kind, string roomType, List<JsonElement> rooms, bool lastOfLostRun)
    {
        string emuRoom = S.Phase switch
        {
            RunPhase.Combat => S.CurrentNodeType switch
            {
                RunConstants.NodeElite => "elite",
                RunConstants.NodeBoss => "boss",
                _ => "monster",
            },
            RunPhase.Event or RunPhase.CrystalSphere or RunPhase.FakeMerchant or RunPhase.TransformSelect
                or RunPhase.BundleSelect or RunPhase.RelicReward or RunPhase.CardReward => "event",
            RunPhase.Shop => "shop",
            RunPhase.Treasure => "treasure",
            RunPhase.Rest => "rest_site",
            _ => S.Phase.ToString(),
        };
        // An unknown room's monster fight is a monster room; elites never come out of one.
        Check(kind == "unknown" ? "unknown_room" : "room", "room type", roomType, emuRoom, Source.Static);
        var room = rooms[0];
        switch (roomType)
        {
            case "monster":
            case "elite":
            case "boss":
                CompareEncounter(room);
                if (lastOfLostRun)
                {
                    return;
                }

                ResolveFight(room);
                break;
            case "event":
                CompareEvent(room);
                break;
            case "shop":
                break;
            case "treasure":
                CompareTreasure();
                break;
        }
    }

    void CompareEncounter(JsonElement room)
    {
        if (S.ActiveCombat is not { } fight)
        {
            Fail("flow", "fight", "combat", S.Phase.ToString(), WalkerLimit);
            return;
        }

        string model = room.TryGetProperty("model_id", out var m) ? m.GetString() ?? "" : "";
        string wanted = Names.EncounterModelToEmulator(model), got = Names.EncounterName(fight.EncounterId);
        if (wanted != got && FirstRunOrder(Names.Bare(model)))
        {
            Fail("encounter", "encounter", wanted, got, Profile, "the profile's first run: Overgrowth presents its rooms in a set order");
        }

        Check("encounter", "encounter", wanted, got, Source.Static);
        var real = room.TryGetProperty("monster_ids", out var ids)
            ? ids.EnumerateArray().Select(x => Names.MonsterBase(Names.Bare(x.GetString()))).Where(x => !Names.Allies.Contains(x)).ToList()
            : [];
        var emu = fight.Enemies.Select(x => Names.MonsterBase(Names.Enemy(x.DefId))).Where(x => !Names.Allies.Contains(x)).ToList();
        // The history lists every creature the fight had, the ones that joined it as well; the
        // opening roster is the first of them, in order.
        bool prefix = real.Count >= emu.Count && real.Take(emu.Count).SequenceEqual(emu);
        Check("monsters", "opening roster", string.Join(",", real.Take(Math.Max(emu.Count, 1))), string.Join(",", emu),
            Source.Static, prefix ? null : "the history's monsters, as many as the emulator opens with");
        if (real.Count > emu.Count)
        {
            // A creature joined mid-fight and its HP was rolled on Niche.
            _taint.TryAdd("Niche", $"floor {_floor}: {string.Join(",", real.Skip(emu.Count))} joined the fight");
        }
    }

    void ResolveFight(JsonElement room)
    {
        _fights++;
        foreach (string st in Streams.FightTurns)
        {
            _taint.TryAdd(st, $"floor {_floor}: the fight's turns were not played");
        }

        // Potions drunk in the fight are gone before its rewards are claimed.
        foreach (string p in StrList("potion_used"))
        {
            if (RemovePotion(p))
            {
                _discardsLeft.Remove(p);
            }
        }
        Mark();
        MarkFloor();
        // What the fight did that its rewards read, as far as the history shows it. The Gremlin
        // Merc's gold is rolled at half, or not at all, when the Fat Gremlin escapes (with the
        // Merc's loot): no gold taken is the second, a short purse the first.
        bool merc = S.ActiveCombat?.EncounterId == RunConstants.GremlinMercEncounterId;
        int gained = Int("gold_gained");
        int Poverty(int g) => Ascension.Has(S.AscensionLevel, Ascension.Poverty) ? (int)(g * Ascension.PovertyGoldMultiplier) : g;
        int fullMin = Poverty(10), halfMax = (int)Math.Round(Poverty(20) * 0.5, MidpointRounding.AwayFromZero);
        bool fatEscaped = merc && gained < fullMin;
        bool mercStole = merc && gained == 0;
        // A purse both a full roll and a half one can give: either way it is one Rewards draw,
        // so only its amount is unknown.
        _goldUnknown = merc && gained >= fullMin && gained <= halfMax;

        long f0 = System.Diagnostics.Stopwatch.GetTimestamp();
        _e.DebugResolveCombat(fatGremlinEscaped: fatEscaped, mercGoldWasStolen: mercStole);
        FightSeconds += System.Diagnostics.Stopwatch.GetElapsedTime(f0).TotalSeconds;
        _rewardsCompared = false;
        _fightJustResolved = true;
    }

    void CompareEvent(JsonElement room)
    {
        string model = room.TryGetProperty("model_id", out var m) ? Names.Bare(m.GetString()) : "";
        string emu = S.Phase switch
        {
            RunPhase.CrystalSphere => "CRYSTAL_SPHERE",
            RunPhase.FakeMerchant => "FAKE_MERCHANT",
            _ => Names.Event(S.EventId),
        };
        if (model != emu && FirstRunOrder(model))
        {
            Fail("event", "event", model, emu, Profile, "the profile's first run: Overgrowth presents its rooms in a set order");
        }

        Check("event", "event", model, emu, Source.Static);
        _eventsSeen++;
        Mark();
    }

    void CompareTreasure()
    {
        var real = Arr("relic_choices").Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
        string emu = S.RelicReward != 0 ? Names.Relic(S.RelicReward) : "";
        if (real.Count == 0)
        {
            // A chest left with its relic in it is not in the history.
            return;
        }

        Check("treasure", "chest relic", real.FirstOrDefault() ?? "", emu, Source.Drawn);
        if (emu.Length > 0)
        {
            ConsumeRelicChoice(emu);
        }
    }

    // =====================================================================================
    // answering the floor's screens the way the history says
    // =====================================================================================

    void Settle(string kind, List<JsonElement> rooms)
    {
        for (int guard = 0; guard < 200; guard++)
        {
            switch (S.Phase)
            {
                case RunPhase.Map:
                case RunPhase.Complete:
                    return;
                case RunPhase.Ancient when S.NeowAwaitingProceed:
                    Do(0);
                    break;
                case RunPhase.Ancient:
                    // The next act's ancient comes off the map; this one is the floor's own, answered.
                    return;
                case RunPhase.Combat:
                    // A fight an event started: its encounter is the floor's second room.
                    if (rooms.Count > 1)
                    {
                        CompareEncounter(rooms[1]);
                    }

                    ResolveFight(rooms.Count > 1 ? rooms[1] : rooms[0]);
                    break;
                case RunPhase.RelicReward:
                    RewardScreen();
                    break;
                case RunPhase.CardReward:
                    CardReward();
                    break;
                case RunPhase.TransformSelect:
                    DeckSelect();
                    break;
                case RunPhase.BundleSelect:
                    Bundles();
                    break;
                case RunPhase.Event:
                    EventOption();
                    break;
                case RunPhase.Shop:
                    Shop();
                    break;
                case RunPhase.Rest:
                    Rest();
                    break;
                case RunPhase.FakeMerchant:
                    FakeMerchant(rooms);
                    break;
                case RunPhase.Treasure:
                    Treasure();
                    break;
                default:
                    Fail("flow", "screen", "(any)", S.Phase.ToString(), WalkerLimit, "screen not followed by the walker");
                    break;
            }
        }

        Fail("flow", "settle", "map", S.Phase.ToString(), WalkerLimit, "floor never settled");
    }

    // --- rewards ---------------------------------------------------------------------------

    void RewardScreen()
    {
        if (!_rewardsCompared)
        {
            _rewardsCompared = true;
            // A potion the history was offered after the fight that the emulator's screen lacks.
            // A fight's own potions (Petrified Toad's rock, Alchemize) are listed too, taken; one
            // left on the screen can only have been the reward.
            if (S.RewardPotion == 0 && S.PendingPotionRewards.Count == 0 && _fightJustResolved)
            {
                var declined = Arr("potion_choices").Select((c, i) => (c, i))
                    .Where(t => !_potionChoiceUsed.Contains(t.i) && !Picked("potion_choices", t.i))
                    .Select(t => Names.Bare(t.c.GetProperty("choice").GetString())).FirstOrDefault();
                if (declined is not null)
                {
                    Check("potion_reward", "potion", declined, "", Source.Rewards);
                }
            }
        }

        _fightJustResolved = false;

        // Claim what the history took, in the screen's order; then leave.
        int index = 0;
        if (S.RewardGold != 0)
        {
            if (Int("gold_gained") > 0 || !_hasStats)
            {
                Do(index);
                return;
            }

            index++;
        }

        if (S.RewardPotion != 0)
        {
            if (!_skippedPotions.Contains(S.RewardPotion))
            {
                string potion = Names.Potion(S.RewardPotion);
                int at = FindChoice("potion_choices", potion, _potionChoiceUsed);
                Check("potion_reward", "potion", at >= 0 ? potion : string.Join("|", OpenChoices("potion_choices", _potionChoiceUsed)), potion, Source.Rewards);
                _potionChoiceUsed.Add(at);
                if (Picked("potion_choices", at))
                {
                    FreeSlot();

                    Do(index);
                    return;
                }

                _skippedPotions.Add(S.RewardPotion);
            }

            index++;
        }

        if (S.RelicReward != 0)
        {
            if (!_skippedRelics.Contains(S.RelicReward))
            {
                string relic = Names.Relic(S.RelicReward);
                int at = FindChoice("relic_choices", relic, _relicChoiceUsed);
                Check("relic_reward", "relic", at >= 0 ? relic : FirstOpen("relic_choices", _relicChoiceUsed), relic, Source.Rewards);
                _relicChoiceUsed.Add(at);
                if (Picked("relic_choices", at))
                {
                    Mark();
                    Do(index);
                    return;
                }

                _skippedRelics.Add(S.RelicReward);
            }

            index++;
        }

        if (S.RewardCardPending || S.PendingCardOffers.Count > 0 || S.PendingOtherCharacterCardRewards > 0)
        {
            // Opened only if the history shows the offer.
            if (Arr("card_choices").Count() > _cardCursor)
            {
                Do(index);
                return;
            }
        }

        // More card offers in the history than the emulator made: the fight added a reward
        // (The Hunt's kill, say), rolled on the Rewards stream after the fight's own.
        if (_combatFloor && Arr("card_choices").Count() > _cardCursor)
        {
            // Only The Hunt adds a card reward from inside a fight (`CombatRoom.AddExtraReward`);
            // anything else is an offer the emulator did not make.
            if (!S.Deck.Any(c => Names.Card(c.DefId) == "THE_HUNT"))
            {
                Fail("card_reward", "offers", Arr("card_choices").Count().ToString(), _cardCursor.ToString(), Source.Rewards,
                    "card offers the history shows on the floor, against the emulator's");
            }

            _fightAddedRewards++;
            _taint.TryAdd("Rewards", $"floor {_floor}: The Hunt added a card reward");
            _cardCursor = Arr("card_choices").Count();
        }

        Do(RunConstants.RewardSkipAction);
    }

    bool _fightJustResolved, _combatFloor, _goldUnknown;
    readonly HashSet<int> _skippedPotions = [];
    readonly HashSet<int> _skippedRelics = [];
    int _cardCursor;

    List<string> OpenChoices(string list, HashSet<int> used) =>
        Arr(list).Select((c, i) => (c, i)).Where(t => !used.Contains(t.i)).Select(t => Names.Bare(t.c.GetProperty("choice").GetString())).ToList();

    string FirstOpen(string list, HashSet<int> used) =>
        Arr(list).Select((c, i) => (c, i)).Where(t => !used.Contains(t.i)).Select(t => Names.Bare(t.c.GetProperty("choice").GetString()))
            .FirstOrDefault() ?? "";

    void CardReward()
    {
        var emu = new List<string>();
        for (int i = 0; i < S.RewardCards.Length; i++)
        {
            if (S.RewardCards[i] != 0)
            {
                emu.Add(CardName(S.RewardCards[i], S.RewardUpgraded[i]));
            }
        }

        bool candy = S.LastingCandyRewardCard != 0;
        if (candy)
        {
            emu.Add(CardName(S.LastingCandyRewardCard, S.LastingCandyRewardUpgraded));
        }

        var all = Arr("card_choices").ToList();
        var offered = all.Skip(_cardCursor).Take(emu.Count).ToList();
        var real = offered.Select(c => CardName(c.GetProperty("card"))).ToList();
        _cardCursor += emu.Count;
        int pick = offered.FindIndex(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True);
        if (pick < 0 && S.CardRewardRerollAvailable && all.Count >= _cardCursor + emu.Count)
        {
            // Driftwood: nothing taken from these three, and three more follow -- a reroll.
            Check("card_reward", "cards", string.Join(",", real), string.Join(",", emu), Source.Rewards);
            Do(RunConstants.RewardRerollAction);
            return;
        }

        // The history lists the card taken first and the others in the screen's order.
        var shown = emu.ToList();
        if (pick >= 0 && shown.IndexOf(real[pick]) is int taken and >= 0)
        {
            shown.RemoveAt(taken);
            shown.Insert(0, real[pick]);
        }

        Check("card_reward", "cards", string.Join(",", real), string.Join(",", shown), Source.Rewards);
        // The pick, as the emulator's screen has it.
        pick = pick < 0 ? -1 : emu.IndexOf(real[pick]);
        if (pick < 0)
        {
            Do(RunConstants.RewardSkipAction);
            return;
        }

        if (candy && pick == emu.Count - 1)
        {
            Do(RunConstants.RewardFourthCardAction);
            return;
        }

        // The ordinary three sit in their slots, possibly with a gap.
        int slot = -1, seen = 0;
        for (int i = 0; i < S.RewardCards.Length; i++)
        {
            if (S.RewardCards[i] != 0)
            {
                if (seen == pick)
                {
                    slot = i;
                    break;
                }

                seen++;
            }
        }

        _gainCursor++;
        Do(slot);
    }

    // --- deck selections -------------------------------------------------------------------

    void DeckSelect()
    {
        if (S.PendingOfferCards.Length > 0)
        {
            // A grid of cards to take from: the floor's gained cards say which.
            var gained = Arr("cards_gained").Select(c => Names.Bare(c.GetProperty("id").GetString())).ToList();
            var offer = S.PendingOfferCards.Select(Names.Card).ToList();
            for (int g = _gainCursor; g < gained.Count; g++)
            {
                int at = offer.IndexOf(gained[g]);
                if (at >= 0)
                {
                    _gainCursor = g + 1;
                    Do(at);
                    return;
                }
            }

            if (S.PendingOfferOptional)
            {
                Do(S.PendingOfferCards.Length);
                return;
            }

            Fail("card_offer", "grid", string.Join(",", gained.Skip(_gainCursor)), string.Join(",", offer), Source.Drawn,
                "the cards the floor gained are not in the emulator's grid");
        }

        List<(string Id, bool Up)> wanted;
        ref int cursor = ref _upgradeCursor;
        string what;
        if (S.PendingRestUpgrade || S.PendingSelectionKind == DeckSelection.Upgrade)
        {
            wanted = StrList("upgraded_cards").Select(x => (x, false)).ToList();
            cursor = ref _upgradeCursor;
            what = "upgrade";
        }
        else
        {
            switch (S.PendingSelectionKind)
            {
                case DeckSelection.Remove:
                case DeckSelection.RemoveUpgradable:
                case DeckSelection.RemoveTaggedBasic:
                    wanted = Arr("cards_removed").Select(CardKey).ToList();
                    cursor = ref _removeCursor;
                    what = "remove";
                    break;
                case DeckSelection.TransformTo:
                case DeckSelection.TransformToCarrying:
                case DeckSelection.TransformToRandom:
                case DeckSelection.TransformToRandomUpgraded:
                    wanted = Arr("cards_transformed").Select(t => CardKey(t.GetProperty("original_card"))).ToList();
                    cursor = ref _transformCursor;
                    what = "transform";
                    break;
                case DeckSelection.Enchant:
                    wanted = Arr("cards_enchanted").Select(t => CardKey(t.GetProperty("card"))).Select(k => (k.Id, k.Up)).ToList();
                    cursor = ref _enchantCursor;
                    what = "enchant";
                    break;
                case DeckSelection.Duplicate:
                    wanted = Arr("cards_gained").Select(CardKey).ToList();
                    cursor = ref _gainCursor;
                    what = "duplicate";
                    break;
                default:
                    Fail("flow", "deck selection", "(known kind)", S.PendingSelectionKind.ToString(), WalkerLimit);
                    return;
            }
        }

        while (cursor < wanted.Count)
        {
            var (id, up) = wanted[cursor++];
            int best = -1;
            for (int i = 0; i < S.Deck.Count; i++)
            {
                bool legal = S.PendingRestUpgrade ? RunConstants.IsRunCardUpgradable(S.Deck[i]) : RunNonCombatEffects.CanSelectCard(S, i);
                if (!legal || Names.Card(S.Deck[i].DefId) != id)
                {
                    continue;
                }

                if (S.Deck[i].Upgraded == up)
                {
                    best = i;
                    break;
                }

                best = best < 0 ? i : best;
            }

            if (best >= 0)
            {
                Mark();
                Do(best);
                return;
            }

            // A card the floor only gains later (a relic pair on one screen, claimed in the other
            // order: the emulator's screen holds one relic at a time) cannot be chosen yet.
            bool later = Arr("cards_transformed").Any(t => CardKey(t.GetProperty("final_card")).Id == id)
                || Arr("cards_gained").Any(c => CardKey(c).Id == id);
            if (later && !S.Deck.Any(c => Names.Card(c.DefId) == id))
            {
                Fail("flow", "claim order", id, "(gained later on the floor)", WalkerLimit,
                    "the history chose a card the emulator's claim order has not given yet");
            }

            Fail("deck_select", what, id, "(not selectable)", Source.Drawn, "the history's card cannot be chosen from the emulator's deck");
        }

        // Nothing named: an optional selection the history passed on.
        if (Legal(RunConstants.RewardSkipAction))
        {
            Do(RunConstants.RewardSkipAction);
            return;
        }

        Fail("deck_select", what, "(none named)", $"{S.PendingSelectionKind} x{S.PendingSelectionCount}", WalkerLimit,
            "the history does not say which card");
    }

    void Bundles()
    {
        var gained = Arr("cards_gained").Select(c => Names.Bare(c.GetProperty("id").GetString())).OrderBy(x => x).ToList();
        var offer = S.BundleOffer;
        var bundles = new List<List<string>>();
        for (int i = 0; i + 2 < offer.Length; i += 3)
        {
            bundles.Add(offer.Skip(i).Take(3).Where(c => c != 0).Select(Names.Card).OrderBy(x => x).ToList());
        }

        int pick = bundles.FindIndex(b => b.All(gained.Contains));
        Check("card_offer", "bundles", pick >= 0 ? string.Join(",", bundles[pick]) : string.Join(",", gained),
            pick >= 0 ? string.Join(",", bundles[pick]) : string.Join(" | ", bundles.Select(b => string.Join(",", b))),
            Source.Drawn);
        if (S.SelectedBundle != pick)
        {
            Do(pick);
        }

        if (S.Phase == RunPhase.BundleSelect)
        {
            Do(RunConstants.BundleConfirmAction);
        }
    }

    // --- events ----------------------------------------------------------------------------

    void EventOption()
    {
        if (S.EventId == RunConstants.EventResultPending)
        {
            Do(0);
            return;
        }

        if (S.EventId == RunConstants.EventTheArchitect)
        {
            Do(0);
            return;
        }

        var keys = Arr("event_choices")
            .Select(c => c.TryGetProperty("title", out var t) && t.TryGetProperty("key", out var k) ? k.GetString() ?? "" : "")
            .ToList();
        while (_eventCursor < keys.Count)
        {
            string key = keys[_eventCursor++];
            if (!EventOptionIndex.IsEventChoice(key))
            {
                // A relic's title: an option that IS a relic (the Doll Room's dolls). Which slot
                // holds it is run state, so each takeable option is tried on a copy of the run.
                string relic = key.EndsWith(".title") ? key[..^".title".Length] : key;
                if (Names.RelicIds.TryGetValue(relic, out int relicId) && ProbeForRelic(relicId) is int slot)
                {
                    Mark();
                    Do(slot);
                    return;
                }

                continue; // an ancient's relic title, taken with the blessing
            }

            if (!_events.Available)
            {
                Fail("flow", "event option", key, "(no decompiled events to read option slots from)", WalkerLimit);
            }

            int? index = DynamicEventSlot(key) ?? _events.IndexOf(key);
            if (index is null)
            {
                Fail("flow", "event option", key, "(slot not found in the decompiled event)", WalkerLimit);
            }

            if (!Legal(index!.Value))
            {
                Fail("event_option", "option", key, $"slot {index} not takeable ({LegalList()})", Source.Drawn);
            }

            Mark();
            Do(index.Value);
            return;
        }

        // The Fake Merchant's page has no options: its stall is walked up to.
        if (S.EventId == RunConstants.EventFakeMerchant && Legal(0))
        {
            Do(0);
            return;
        }

        // The history names nothing more: leave the way the screen allows.
        if (Legal(RunConstants.EventSkipAction))
        {
            Do(RunConstants.EventSkipAction);
            return;
        }

        if (Legal(0))
        {
            Do(0);
            return;
        }

        Fail("flow", "event", "(no more choices)", LegalList(), WalkerLimit);
    }

    /// <summary>The takeable event option that gives this relic, tried on copies of the run.</summary>
    int? ProbeForRelic(int relicId)
    {
        Array.Clear(_maskBuf);
        _e.WriteActionMask(_maskBuf);
        var legal = Enumerable.Range(0, _maskBuf.Length).Where(i => _maskBuf[i] != 0 && i != RunConstants.EventSkipAction).ToList();
        foreach (int a in legal)
        {
            var copy = _e.Clone();
            int before = copy.State.Relics.Count(r => r.DefId == relicId);
            if (copy.Step(a, -1, out _, out _, out _) != 0)
            {
                continue;
            }

            if (copy.State.Relics.Count(r => r.DefId == relicId) > before || copy.State.RelicReward == relicId)
            {
                return a;
            }
        }

        return null;
    }

    /// <summary>
    /// Events whose option list is built at run time, so no fixed slot can be read off the
    /// source: the slot each history choice takes on the emulator's screen.
    /// </summary>
    int? DynamicEventSlot(string key)
    {
        string option = key.Split(".options.")[^1].Replace(".title", "");
        switch (Names.Event(S.EventId))
        {
            case "THE_FUTURE_OF_POTIONS" when option == "POTION":
                return FutureOfPotionsSlot();
            case "SLIPPERY_BRIDGE":
                // OVERCOME, then HOLD_ON_n with n counting the holds so far.
                return option == "OVERCOME" ? 0 : option.StartsWith("HOLD_ON") ? 1 : null;
            case "ENDLESS_CONVEYOR":
                // The dish on the belt is always the first option, its key the dish's own.
                return option is "OBSERVE_CHEF" or "LEAVE" ? 1 : 0;
            case "RANWID_THE_ELDER":
                return option switch { "POTION" => 0, "GOLD" => 1, "RELIC" => 2, _ => null };
            case "HUNGRY_FOR_MUSHROOMS":
                return option switch { "BIG_MUSHROOM" => 0, "FRAGRANT_MUSHROOM" => 1, _ => null };
            case "COLORFUL_PHILOSOPHERS":
            {
                // One option per other character's pool still in the offer, in colour order.
                var order = new[] { "NECROBINDER", "IRONCLAD", "REGENT", "SILENT", "DEFECT" }
                    .Where(c => !c.Equals(S.Character.ToString(), StringComparison.OrdinalIgnoreCase)).ToArray();
                var offer = RunNonCombatEffects.ColorfulPhilosopherPools(S);
                for (int i = 0; i < offer.Length; i++)
                {
                    if (offer[i] < order.Length && order[offer[i]] == option)
                    {
                        return i;
                    }
                }

                return -1;
            }
        }

        return null;
    }

    /// <summary>The Future of Potions offers the first three potions held; the traded one is dropped.</summary>
    int? FutureOfPotionsSlot()
    {
        var held = S.PotionSlots.Where(p => p != 0).Select(Names.Potion).Take(3).ToList();
        foreach (string p in StrList("potion_discarded").Concat(StrList("potion_used")))
        {
            int at = held.IndexOf(p);
            if (at >= 0)
            {
                return at;
            }
        }

        return null;
    }

    // --- the shop --------------------------------------------------------------------------

    readonly List<string> _shopShownCards = [];
    readonly List<string> _shopShownRelics = [];
    readonly List<string> _shopShownPotions = [];
    bool _shopSeen;
    int _shopSpent;

    void Shop()
    {
        if (!_shopSeen)
        {
            _shopSeen = true;
            _shopSpent = 0;
            _shopShownCards.Clear();
            _shopCardsBought = 0;
            Array.Clear(_shelfCards);
            Array.Clear(_shelfRelics);
            Array.Clear(_shelfPotions);
            _shopShownRelics.Clear();
            _shopShownPotions.Clear();
            NoteShelf();
        }

        // The history's purchases, in the order it lists them: cards, relics, potions.
        // Bought cards: the floor's gains, less card-reward picks and relic-made copies.
        var rewardPicks = Arr("card_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => CardName(c.GetProperty("card"))).ToList();
        var wantCards = Minus(Arr("cards_gained")
            .Where(c => !(c.TryGetProperty("floor_added_to_deck", out var f) && f.ValueKind == JsonValueKind.Number && f.GetInt32() != _floor))
            .Select(c => CardName(c)).ToList(), rewardPicks);
        var wantRelics = Arr("relic_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
        var wantPotions = Arr("potion_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
        var wants = wantCards.Select(x => (Kind: 0, Item: x)).Concat(wantRelics.Select(x => (Kind: 1, Item: x)))
            .Concat(wantPotions.Select(x => (Kind: 2, Item: x))).ToList();
        while (_shopBuyCursor < wants.Count)
        {
            var (k, item) = wants[_shopBuyCursor++];
            int slot = -1;
            if (k == 0)
            {
                for (int i = 0; i < 7; i++)
                {
                    if (S.ShopCards[i] != 0 && CardName(S.ShopCards[i], S.ShopCardUpgraded[i]) == item)
                    {
                        slot = i;
                        break;
                    }
                }

                if (slot < 0)
                {
                    for (int i = 0; i < 7; i++)
                    {
                        if (S.ShopCards[i] != 0 && Names.Card(S.ShopCards[i]) == item.TrimEnd('+'))
                        {
                            slot = i;
                            break;
                        }
                    }
                }
            }
            else if (k == 1)
            {
                slot = Array.FindIndex(S.ShopRelics, r => r != 0 && Names.Relic(r) == item);
                slot = slot < 0 ? -1 : slot + 7;
            }
            else
            {
                slot = Array.FindIndex(S.ShopPotions, p => p != 0 && Names.Potion(p) == item);
                slot = slot < 0 ? -1 : slot + 10;
            }

            if (slot < 0)
            {
                // A card bought in the shop the emulator does not stock -- or gained some other
                // way on the floor; the shelf comparison at the end of the floor decides.
                continue;
            }

            if (k == 2)
            {
                FreeSlot();
            }

            BuySlot(slot);
            return;
        }

        if (Arr("cards_removed").Any() && !_shopRemovalDone)
        {
            _shopRemovalDone = true;
            if (!Legal(RunConstants.ShopRemoveAction))
            {
                int need = S.ShopCosts[RunConstants.ShopRemoveAction];
                if (S.Gold < need)
                {
                    // The price is compared through what the floor spent.
                    _shopGoldLent += need - S.Gold;
                    S.Gold = need;
                }
            }

            if (Legal(RunConstants.ShopRemoveAction))
            {
                _shopSpent += S.ShopCosts[RunConstants.ShopRemoveAction];
                Do(RunConstants.ShopRemoveAction);
                return;
            }
        }

        Do(RunConstants.ShopSkipAction);
    }

    int _shopGoldLent;

    int _shopCardsBought;

    void BuySlot(int slot)
    {
        if (slot < 7)
        {
            _shopCardsBought++;
        }

        int cost = S.ShopCosts[slot];
        if (S.Gold < cost)
        {
            // The emulator's price is above what the player had: buy anyway, so the floor can go
            // on, and let the gold spent say which price was off.
            _shopGoldLent += cost - S.Gold;
            S.Gold = cost;
        }

        _shopSpent += cost;
        Mark();
        Do(slot);
        NoteShelf();
    }

    readonly int[] _shelfCards = new int[7], _shelfRelics = new int[3], _shelfPotions = new int[3];

    void NoteShelf()
    {
        // Everything the shop has shown, slot by slot: a slot shows a new item when its id
        // changes (The Courier restocks one), and an egg's upgrade is the same card.
        Note(S.ShopCards, _shelfCards, _shopShownCards, Names.Card);
        Note(S.ShopRelics, _shelfRelics, _shopShownRelics, Names.Relic);
        Note(S.ShopPotions, _shelfPotions, _shopShownPotions, Names.Potion);
    }

    static void Note(int[] shelf, int[] last, List<string> shown, Func<int, string> name)
    {
        for (int i = 0; i < last.Length && i < shelf.Length; i++)
        {
            if (shelf[i] != 0 && shelf[i] != last[i])
            {
                shown.Add(name(shelf[i]));
            }

            last[i] = shelf[i];
        }
    }

    void CompareShop()
    {
        // The history lists the shelf's cards it did not buy, and its buys as cards gained. Card
        // offers a bought relic opened (Orrery) are in the same list, and were answered on their
        // own screens; a copy a relic made (Dolly's Mirror) keeps its first floor.
        // The shelf's own entries are the last ones: as many as the emulator showed, less its buys.
        int left = Math.Max(0, _shopShownCards.Count - _shopCardsBought);
        var rest = Arr("card_choices").Skip(_cardCursor).ToList();
        var unbought = rest.Skip(Math.Max(0, rest.Count - left)).Select(c => CardKey(c.GetProperty("card")).Id).ToList();
        var picked = Arr("card_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => CardKey(c.GetProperty("card")).Id).ToList();
        var bought = Arr("cards_gained")
            .Where(c => !(c.TryGetProperty("floor_added_to_deck", out var f) && f.ValueKind == JsonValueKind.Number && f.GetInt32() != _floor))
            .Select(c => CardKey(c).Id).ToList();
        var realCards = unbought.Concat(Minus(bought, picked)).OrderBy(x => x).ToList();
        var realRelics = Arr("relic_choices").Select(c => Names.Bare(c.GetProperty("choice").GetString())).OrderBy(x => x).ToList();
        var realPotions = Arr("potion_choices").Select(c => Names.Bare(c.GetProperty("choice").GetString())).OrderBy(x => x).ToList();
        Check("shop", "cards", string.Join(",", realCards), string.Join(",", _shopShownCards.OrderBy(x => x)), Source.Static);
        Check("shop", "relics", string.Join(",", realRelics), string.Join(",", _shopShownRelics.OrderBy(x => x)), Source.Static);
        Check("shop", "potions", string.Join(",", realPotions), string.Join(",", _shopShownPotions.OrderBy(x => x)), Source.Static);
        Check("shop", "gold spent", Int("gold_spent").ToString(), _shopSpent.ToString(), Source.Static);
    }

    // --- the Fake Merchant -----------------------------------------------------------------

    int _fakeBuyCursor;

    void FakeMerchant(List<JsonElement> rooms)
    {
        var stall = RunNonCombatEffects.FakeMerchantStock(S);
        var bought = StrList("bought_relics");
        while (_fakeBuyCursor < bought.Count)
        {
            string relic = bought[_fakeBuyCursor++];
            int slot = -1;
            for (int i = 0; i < stall.Count; i++)
            {
                if (stall[i] != 0 && Names.Relic(stall[i]) == relic)
                {
                    slot = i;
                    break;
                }
            }

            Check("fake_merchant", "stall", relic, slot >= 0 ? relic : string.Join(",", stall.Where(r => r != 0).Select(Names.Relic)), Source.Drawn);
            if (S.Gold < RunConstants.FakeMerchantRelicCost)
            {
                _shopGoldLent += RunConstants.FakeMerchantRelicCost - S.Gold;
                S.Gold = RunConstants.FakeMerchantRelicCost;
            }

            Mark();
            Do(slot);
            return;
        }

        // A fight on the floor is the Foul Potion thrown at the merchant.
        bool fight = rooms.Skip(1).Any(x => x.TryGetProperty("room_type", out var t) && t.GetString() is "monster" or "elite");
        Do(fight && Legal(RunConstants.FakeMerchantThrowAction) ? RunConstants.FakeMerchantThrowAction : RunConstants.FakeMerchantLeaveAction);
    }

    // --- rest sites and chests -------------------------------------------------------------

    void Rest()
    {
        if (S.RestResultPending)
        {
            Do(0);
            return;
        }

        var choices = StrList("rest_site_choices");
        if (_restCursor < choices.Count)
        {
            string choice = choices[_restCursor++];
            int action = choice switch
            {
                "HEAL" => RunConstants.RestHealAction,
                "SMITH" => RunConstants.RestUpgradeAction,
                "CLONE" => RunConstants.RestCloneAction,
                "LIFT" => RunConstants.RestLiftAction,
                "DIG" => RunConstants.RestDigAction,
                "HATCH" => RunConstants.RestHatchAction,
                "COOK" => RunConstants.RestCookAction,
                "KINDLE" => RunConstants.RestKindleAction,
                _ => -1,
            };
            if (action < 0)
            {
                Fail("flow", "rest option", choice, "(unmodelled)", WalkerLimit);
            }

            if (!Legal(action))
            {
                Fail("rest", "option", choice, $"not offered ({LegalList()})", Source.Drawn);
            }

            Mark();
            Do(action);
            return;
        }

        Do(RunConstants.RewardSkipAction);
    }

    void Treasure()
    {
        if (S.RelicReward != 0)
        {
            string relic = Names.Relic(S.RelicReward);
            int at = Arr("relic_choices").Select((c, i) => (c, i))
                .Where(t => Names.Bare(t.c.GetProperty("choice").GetString()) == relic && t.c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
                .Select(t => t.i).DefaultIfEmpty(-1).First();
            if (at >= 0)
            {
                Mark();
                Do(RunConstants.TreasureTakeRelicAction);
                return;
            }
        }

        Do(RunConstants.RewardSkipAction);
    }

    // =====================================================================================
    // after the floor: what the run holds
    // =====================================================================================

    void AfterFloor(string kind, bool combat, List<string> deckBefore, List<string> relicsBefore, int goldBefore)
    {
        if (_shopSeen)
        {
            CompareShop();
            _shopSeen = false;
        }

        S.Gold -= _shopGoldLent;
        _shopGoldLent = 0;

        // The deck the history says the floor left, from the deck before it.
        var expected = deckBefore.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        void Add(string k, int n)
        {
            expected[k] = expected.GetValueOrDefault(k) + n;
            if (expected[k] == 0)
            {
                expected.Remove(k);
            }
        }

        foreach (var c in Arr("cards_gained"))
        {
            Add(CardName(c), 1);
        }

        foreach (var t in Arr("cards_transformed"))
        {
            Add(CardName(t.GetProperty("original_card")), -1);
            Add(CardName(t.GetProperty("final_card")), 1);
        }

        foreach (string u in StrList("upgraded_cards"))
        {
            string id = Names.Bare(u);
            if (expected.GetValueOrDefault(id) > 0)
            {
                Add(id, -1);
                Add(id + "+", 1);
            }
        }

        foreach (string u in StrList("downgraded_cards"))
        {
            string id = Names.Bare(u);
            if (expected.GetValueOrDefault(id + "+") > 0)
            {
                Add(id + "+", -1);
                Add(id, 1);
            }
        }

        foreach (var c in Arr("cards_removed"))
        {
            string k = CardName(c);
            if (expected.GetValueOrDefault(k) > 0)
            {
                Add(k, -1);
            }
            else if (expected.GetValueOrDefault(k.TrimEnd('+')) > 0)
            {
                Add(k.TrimEnd('+'), -1);
            }
            else if (expected.GetValueOrDefault(k + "+") > 0)
            {
                Add(k + "+", -1);
            }
        }

        var emuDeck = DeckKeys();
        string Fmt(IEnumerable<string> keys) => string.Join(",", keys.OrderBy(x => x));
        var expectedList = expected.SelectMany(kv => Enumerable.Repeat(kv.Key, Math.Max(0, kv.Value))).ToList();
        string expDiff = Fmt(Minus(expectedList, emuDeck));
        string emuDiff = Fmt(Minus(emuDeck, expectedList));

        // Relics: the floor's picks, less the ones it lost.
        var expRelics = relicsBefore.ToList();
        expRelics.AddRange(Arr("relic_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => Names.Bare(c.GetProperty("choice").GetString())));
        foreach (string r in StrList("relics_removed"))
        {
            expRelics.Remove(Names.Bare(r));
        }

        var emuRelics = S.Relics.Select(x => Names.Relic(x.DefId)).ToList();
        string relExp = Fmt(Minus(expRelics, emuRelics));
        string relEmu = Fmt(Minus(emuRelics, expRelics));

        if (combat)
        {
            // A fight's own doings (a card stolen and kept, a curse added, a relic spent) are the
            // fight's, like its HP: the deck and relics are set to what the history says, unscored.
            // What the rewards added was scored on the reward screens.
            if (expDiff.Length > 0 || emuDiff.Length > 0)
            {
                SetDeck(expectedList);
            }

            int g = Int("gold_gained");
            if (Int("gold_stolen") == 0 && Int("stolen_loot") == 0 && _hasStats)
            {
                int emuGained = Math.Max(0, S.Gold - goldBefore);
                bool fightGold = S.Deck.Any(c => Names.Card(c.DefId) == "HAND_OF_GREED");
                if (!(fightGold && g != emuGained) && !_goldUnknown)
                {
                    Check("gold_reward", "gold gained", g.ToString(), emuGained.ToString(), Source.Rewards);
                }
            }
        }
        else
        {
            if (expDiff.Length > 0 || emuDiff.Length > 0)
            {
                Check("deck", "cards", "+" + expDiff, "+" + emuDiff, Source.FloorDrawn, "deck after the floor: what each side has that the other lacks");
            }
            else
            {
                Count("deck", true);
            }

            if (_hasStats)
            {
                Check("outcome", "gold", Int("current_gold").ToString(), S.Gold.ToString(), Source.FloorDrawn);
                // A potion drunk away from a fight did what it does to HP, which the walk does not
                // replay: the floor's HP is the history's then, and is not scored.
                if (!Arr("potion_used").Any())
                {
                    Check("outcome", "max hp", Int("max_hp").ToString(), S.PlayerMaxHp.ToString(), Source.FloorDrawn);
                    Check("outcome", "hp", Int("current_hp").ToString(), S.PlayerHp.ToString(), Source.FloorDrawn);
                }
            }
        }

        if (relExp.Length > 0 || relEmu.Length > 0)
        {
            Check("relics", "relics", "+" + relExp, "+" + relEmu, combat ? Source.Rewards : Source.FloorDrawn, "relics after the floor");
        }
        else
        {
            Count("relics", true);
        }

        // The belt as the history has it: what the floor picked or bought, less what it drank
        // or dropped. Potions are not scored; they are kept right because events ask for them.
        SyncPotions();
        if (_hasStats)
        {
            // Fights are not replayed: their HP and gold are the history's.
            _e.DebugSetPlayer(Int("current_hp"), Int("max_hp"), Int("current_gold"), null, null, null);
        }
    }

    readonly List<string> _belt = [];

    void SyncPotions()
    {
        var picked = Arr("potion_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
        _belt.AddRange(picked);
        _belt.AddRange(Minus(StrList("bought_potions"), picked));
        foreach (string p in StrList("potion_used").Concat(StrList("potion_discarded")))
        {
            _belt.Remove(p);
        }

        // Drop what the history no longer holds, then add what it gained, in the first free slots.
        var want = _belt.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        for (int i = 0; i < S.PotionSlots.Length; i++)
        {
            if (S.PotionSlots[i] == 0)
            {
                continue;
            }

            string name = Names.Potion(S.PotionSlots[i]);
            if (want.GetValueOrDefault(name) > 0)
            {
                want[name]--;
            }
            else
            {
                S.PotionSlots[i] = 0;
            }
        }

        foreach (var (name, n) in want)
        {
            for (int k = 0; k < n; k++)
            {
                int free = Array.IndexOf(S.PotionSlots, 0);
                if (free < 0 || !Names.PotionIds.TryGetValue(name, out int id))
                {
                    break;
                }

                if (free >= S.MaxPotionSlots)
                {
                    S.MaxPotionSlots = free + 1;
                }

                S.PotionSlots[free] = id;
            }
        }
    }

    void SetDeck(List<string> keys)
    {
        // Keep each surviving copy (its enchantment rides along); add the rest plain.
        var left = S.Deck.ToList();
        var next = new List<CardInstance>();
        foreach (string k in keys)
        {
            bool up = k.EndsWith('+');
            string id = up ? k[..^1] : k;
            int at = left.FindIndex(c => Names.Card(c.DefId) == id && c.Upgraded == up);
            if (at >= 0)
            {
                next.Add(left[at]);
                left.RemoveAt(at);
            }
            else if (Names.CardIds.TryGetValue(id, out int cid))
            {
                next.Add(new CardInstance(cid, up));
            }
        }

        S.Deck.Clear();
        S.Deck.AddRange(next);
    }

    // =====================================================================================
    // plumbing
    // =====================================================================================

    public static double ResetSeconds, FightSeconds;

    public static bool Trace = Environment.GetEnvironmentVariable("CODEX_TRACE") == "1";

    void Do(int action)
    {
        if (Trace)
        {
            Console.Error.WriteLine($"  f{_floor} {S.Phase} event={S.EventId} awaiting={S.NeowAwaitingProceed} -> {action}  [{LegalList()}]");
        }

        int status = _e.Step(action, -1, out _, out _, out _);
        if (status != 0)
        {
            Fail("flow", "action", action.ToString(), $"refused in {S.Phase} ({LegalList()})", WalkerLimit);
        }
    }

    readonly int[] _maskBuf = new int[256];

    bool Legal(int action)
    {
        Array.Clear(_maskBuf);
        _e.WriteActionMask(_maskBuf);
        return action >= 0 && action < _maskBuf.Length && _maskBuf[action] != 0;
    }

    string LegalList()
    {
        Array.Clear(_maskBuf);
        _e.WriteActionMask(_maskBuf);
        return string.Join(",", Enumerable.Range(0, _maskBuf.Length).Where(i => _maskBuf[i] != 0));
    }

    void Mark() => Streams.Snapshot(S, _mark);

    void MarkFloor() => Streams.Snapshot(S, _floorMark);

    void Count(string component, bool ok)
    {
        _floorComponents.Add(component);
        if (!ok)
        {
            _floorFailed.Add(component);
        }
    }

    void BeginFloorBookkeeping()
    {
        _floorComponents.Clear();
        _floorFailed.Clear();
    }

    void EndFloorBookkeeping()
    {
        foreach (string c in _floorComponents)
        {
            if (!_checks.TryGetValue(c, out var v))
            {
                _checks[c] = v = new int[2];
            }

            v[0]++;
            if (!_floorFailed.Contains(c))
            {
                v[1]++;
            }
        }

        if (_floor > 0)
        {
            if (_floorUsedTaint)
            {
                _uncheckableFloors++;
            }
            else
            {
                _checkableFloors++;
            }
        }
    }

    (string Class, List<string> Streams, string Why) Classify(Source source)
    {
        switch (source)
        {
            case Source.Static:
                return (Emulator, [], "");
            case Source.Rewards:
                return _taint.TryGetValue("Rewards", out var why) ? (FightDependent, ["Rewards"], why) : (Emulator, ["Rewards"], "");
            default:
                Streams.Snapshot(S, _now);
                var drawn = Streams.Moved(source == Source.FloorDrawn ? _floorMark : _mark, _now).ToList();
                var hit = drawn.Where(_taint.ContainsKey).ToList();
                return hit.Count > 0 ? (FightDependent, drawn, string.Join("; ", hit.Select(h => $"{h}: {_taint[h]}"))) : (Emulator, drawn, "");
        }
    }

    void Check(string component, string field, string expected, string emulator, Source source, string? note = null)
    {
        var (klass, _, _) = Classify(source);
        if (klass == FightDependent)
        {
            _floorUsedTaint = true;
        }

        bool ok = expected == emulator && note is null;
        Count(component, ok);
        if (!ok)
        {
            // A fight-added reward is visible in the history: more card offers than the emulator
            // made means the next ones read a Rewards stream the fight moved.
            Fail(component, field, expected, emulator, source, note);
        }
    }

    void Fail(string component, string field, string expected, string emulator, Source source, string? note = null)
    {
        var (klass, drawn, why) = Classify(source);
        Fail(component, field, expected, emulator, klass, note, drawn, why);
    }

    void Fail(string component, string field, string expected, string emulator, string klass, string? note = null,
        List<string>? drawn = null, string? why = null)
    {
        _floorComponents.Add(component);
        _floorFailed.Add(component);
        EndFloorBookkeeping();
        var m = new JsonObject
        {
            ["act"] = _act + 1,
            ["floor"] = _floor,
            ["component"] = component,
            ["field"] = field,
            ["expected"] = expected,
            ["emulator"] = emulator,
            ["class"] = klass,
        };
        if (note is not null)
        {
            m["note"] = note;
        }

        if (drawn is { Count: > 0 })
        {
            m["streams"] = new JsonArray(drawn.Select(d => (JsonNode)d).ToArray());
        }

        if (!string.IsNullOrEmpty(why))
        {
            m["tainted_by"] = why;
        }

        _floorComponents.Clear();
        throw new Stop(m);
    }

    // --- history access --------------------------------------------------------------------

    IEnumerable<JsonElement> Arr(string name) =>
        _hasStats && _stats.TryGetProperty(name, out var a) && a.ValueKind == JsonValueKind.Array ? a.EnumerateArray() : [];

    List<string> StrList(string name) => Arr(name).Select(x => x.ValueKind == JsonValueKind.String ? x.GetString() ?? "" : x.ToString())
        .Select(Names.Bare).ToList();

    int Int(string name) => _hasStats && _stats.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 0;

    bool Picked(string list, int index) =>
        index >= 0 && Arr(list).ElementAt(index) is var c && c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True;

    int FindChoice(string list, string item, HashSet<int> used)
    {
        int i = 0;
        foreach (var c in Arr(list))
        {
            if (!used.Contains(i) && Names.Bare(c.GetProperty("choice").GetString()) == item)
            {
                return i;
            }

            i++;
        }

        return -1;
    }

    void ConsumeRelicChoice(string relic)
    {
        int at = FindChoice("relic_choices", relic, _relicChoiceUsed);
        if (at >= 0)
        {
            _relicChoiceUsed.Add(at);
        }
    }

    static (string Id, bool Up) CardKey(JsonElement card) =>
        (Names.Bare(card.GetProperty("id").GetString()),
            card.TryGetProperty("current_upgrade_level", out var u) && u.ValueKind == JsonValueKind.Number && u.GetInt32() > 0);

    static string CardName(JsonElement card)
    {
        var (id, up) = CardKey(card);
        return up ? id + "+" : id;
    }

    static string CardName(int id, bool upgraded) => Names.Card(id) + (upgraded ? "+" : "");

    List<string> DeckKeys() => S.Deck.Select(c => CardName(c.DefId, c.Upgraded)).ToList();

    static List<string> Minus(List<string> a, List<string> b)
    {
        var left = b.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        var outList = new List<string>();
        foreach (string x in a)
        {
            if (left.GetValueOrDefault(x) > 0)
            {
                left[x]--;
            }
            else
            {
                outList.Add(x);
            }
        }

        return outList;
    }

    /// <summary>
    /// Make room on a full belt the way the history did: drink or drop a potion it says went on
    /// this floor (one bought here and drunk to make room included), else the first one.
    /// </summary>
    void FreeSlot()
    {
        if (S.UsablePotionSlots().Any(p => p == 0))
        {
            return;
        }

        foreach (string p in _discardsLeft.ToList())
        {
            if (RemovePotion(p))
            {
                _discardsLeft.Remove(p);
                return;
            }
        }

        int first = Array.FindIndex(S.PotionSlots, p => p != 0);
        if (first >= 0)
        {
            S.PotionSlots[first] = 0;
        }
    }

    bool RemovePotion(string p)
    {
        int at = Array.FindIndex(S.PotionSlots, x => x != 0 && Names.Potion(x) == p);
        if (at < 0)
        {
            return false;
        }

        S.PotionSlots[at] = 0;
        return true;
    }

    void RemovePotions(List<string> potions)
    {
        foreach (string p in potions)
        {
            int at = Array.FindIndex(S.PotionSlots, x => x != 0 && Names.Potion(x) == p);
            if (at >= 0)
            {
                S.PotionSlots[at] = 0;
            }
        }
    }
}
