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
//
// The run is generated for the profile its history fits (ProfileFit): the mature one unless the
// acts, the fights, the ancients or Neow's options show otherwise. A fitted profile is reported
// as "profile" (what it lacks); what is still unexplained afterwards is classed as below.
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;

namespace EmuDump.Codex;

public sealed class Walker
{
    public const string Emulator = "EMULATOR";
    public const string FightDependent = "FIGHT-DEPENDENT";
    public const string WalkerLimit = "WALKER";

    /// <summary>
    /// The player's profile decided it, and no profile the fit tries explains it: the acts, or
    /// Neow's options with Scroll Boxes barred (its bundles need an unlocked pool deep enough),
    /// or the profile's first run (Overgrowth's set room order), or content an epoch unlocks of
    /// which nothing appears anywhere in the run (the pools' epoch filters are not modelled).
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

    readonly RunEngine _e = new();
    readonly EventOptionIndex _events;
    RunState S => _e.State;

    /// <summary>The run where the last walk stopped (tests read its belt).</summary>
    public RunState State => _e.State;

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
        _restCursor, _gainCursor;
    readonly HashSet<int> _relicChoiceUsed = [];
    readonly HashSet<int> _potionChoiceUsed = [];
    readonly List<string> _discardsLeft = [];
    bool _rewardsCompared, _shopRemovalDone;

    readonly ProfileIndex? _profile;
    HashSet<string> _seen = [];

    public Walker(EventOptionIndex events, ProfileIndex? profile = null)
    {
        _events = events;
        _profile = profile;
    }

    // =====================================================================================
    // entry
    // =====================================================================================

    /// <summary>Tests: changes the run each walk starts from (a relic the record's run held from the start).</summary>
    internal Action<RunState>? AfterReset { get; init; }

    // --- hidden fight facts ------------------------------------------------------------------
    // A fact about a resolved fight that the history does not record: whether a Gremlin Merc
    // stole before it died, whether a Thieving Hopper's room kept its gold row, which of the deck's
    // identical copies a Hopper took, whether a hurt fight's damage was all unblockable. Each has
    // ways 0 .. arity-1 (for a yes-or-no fact, 1 is yes), one of them the default.
    readonly Dictionary<string, int> _facts = []; // fact -> the way this walk takes it
    readonly HashSet<string> _openFacts = []; // settled facts whose best ways tie
    readonly List<(string Key, int Arity, int Default)> _pending = []; // the facts this walk met, in order
    readonly List<(string Key, int Arity, int Default)> _factsMet = [];
    string? _taintAfterFloor;
    int _passed;

    sealed record Once(JsonObject Result, List<(string Key, int Arity, int Default)> Met, int Passed, string Ways);

    /// <summary>
    /// Walks a run. A fact of a fight the history leaves open is settled by consistency: when the
    /// walk stops, it is walked again each other way from the fact on, and the way that gets
    /// furthest through the history (the one the next observations fit) is kept. When the best
    /// ways stop at the same check, or all walk to the end, the history cannot tell them apart and
    /// the fact stays open: a fact that feeds the Rewards stream leaves it uncertain from there,
    /// so a later stop on it is FIGHT-DEPENDENT, neither a match nor a mismatch. A walk that fits
    /// the whole history settles nothing. The facts are settled in the order the walk meets them,
    /// each with the earlier ones settled.
    /// </summary>
    public JsonObject Walk(JsonElement rec)
    {
        _facts.Clear();
        _openFacts.Clear();
        var best = WalkOnce(rec);
        int walks = 1;
        var settled = new List<(string Key, int Arity, int Default)>();
        void Settle((string Key, int Arity, int Default) fact)
        {
            int bestWay = _facts.TryGetValue(fact.Key, out int walked) ? walked : fact.Default;
            bool tied = false;
            _openFacts.Remove(fact.Key);
            for (int way = 0; way < fact.Arity; way++)
            {
                if (way == bestWay)
                {
                    continue;
                }

                _facts[fact.Key] = way;
                var other = WalkOnce(rec);
                walks++;
                int further = Further(other, best);
                if (further > 0)
                {
                    (best, bestWay, tied) = (other, way, false);
                }
                else if (further == 0)
                {
                    tied = true;
                }
            }

            _facts[fact.Key] = bestWay;
            if (tied)
            {
                _openFacts.Add(fact.Key);
            }
        }

        while (best.Met.FirstOrDefault(m => !settled.Any(s => s.Key == m.Key)) is { Key: not null } fact)
        {
            settled.Add(fact);
            Settle(fact);
        }

        // Facts interact: one settled while an earlier-met fact was still at its default can tie
        // for that reason alone (Gremlin gang: the joiners' count against the Merc's theft). A fact
        // left open is settled again with all the later ones settled, until none changes.
        for (int round = 0; round < 3 && settled.Where(s => _openFacts.Contains(s.Key)).ToList() is { Count: > 0 } open; round++)
        {
            string before = Ways();
            best = WalkOnce(rec);
            walks++;
            foreach (var fact in open)
            {
                Settle(fact);
            }

            while (best.Met.FirstOrDefault(m => !settled.Any(s => s.Key == m.Key)) is { Key: not null } fresh)
            {
                settled.Add(fresh);
                Settle(fresh);
            }

            if (before == Ways())
            {
                break;
            }
        }

        if (best.Ways != Ways())
        {
            // The walk kept was made with a fact open that is settled now: walked again, so the
            // stop is classed with it settled.
            best = WalkOnce(rec);
            walks++;
        }

        if (settled.Count > 0)
        {
            best.Result["hidden_facts"] = new JsonArray(settled.Select(f =>
            {
                string[] parts = f.Key.Split(':');
                int way = _facts[f.Key];
                return (JsonNode)new JsonObject
                {
                    ["floor"] = int.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture),
                    ["fact"] = parts[1],
                    ["settled"] = _openFacts.Contains(f.Key) ? "open" : parts[1] == JoinedFact ? $"{way} created" : f.Arity > 2 || parts[1] == CopyFact ? $"{way + 1} of {f.Arity}" : way == 1 ? "yes" : "no",
                };
            }).ToArray());
            best.Result["walks"] = walks;
        }

        return best.Result;
    }

    /// <summary>The ways this walk takes the facts, open ones marked (what a walk's classes follow).</summary>
    string Ways() => string.Join(";", _facts.OrderBy(kv => kv.Key, StringComparer.Ordinal)
        .Select(kv => $"{kv.Key}={kv.Value}{(_openFacts.Contains(kv.Key) ? "?" : "")}"));

    /// <summary>
    /// Above 0 when walk <paramref name="a" /> got further through the history than
    /// <paramref name="b" />, below 0 when it stopped sooner, 0 when neither got further: both
    /// walked to the end, or both stopped at the same check. The two walk the same run and differ
    /// only from the fact on, so the first check where their verdicts part is where one fits.
    /// </summary>
    static int Further(Once a, Once b)
    {
        bool ca = (string?)a.Result["status"] == "complete", cb = (string?)b.Result["status"] == "complete";
        if (ca || cb)
        {
            return ca == cb ? 0 : ca ? 1 : -1;
        }

        int fa = (int?)a.Result["floors_walked"] ?? 0, fb = (int?)b.Result["floors_walked"] ?? 0;
        if (fa != fb)
        {
            return fa.CompareTo(fb);
        }

        if (a.Result["stop"] is JsonObject sa && b.Result["stop"] is JsonObject sb
            && (string?)sa["component"] == (string?)sb["component"] && (string?)sa["field"] == (string?)sb["field"])
        {
            return 0;
        }

        return a.Passed.CompareTo(b.Passed);
    }

    /// <summary>
    /// Notes that the history leaves a fact of this floor's fight open, and returns the way this
    /// walk takes it (<paramref name="defaultWay" /> unless an earlier walk settled it otherwise)
    /// and whether it is open: not settled yet, or settled with its best ways tied.
    /// </summary>
    (int Way, bool Open) Hidden(string fact, int arity = 2, int defaultWay = 0)
    {
        string key = $"{_floor}:{fact}";
        if (!_pending.Any(p => p.Key == key))
        {
            _pending.Add((key, arity, defaultWay));
        }

        return _facts.TryGetValue(key, out int way) ? (way, _openFacts.Contains(key)) : (defaultWay, true);
    }

    /// <summary>The way this walk takes a fact of this floor's fight, met or not.</summary>
    int Way(string fact) => _facts.GetValueOrDefault($"{_floor}:{fact}");

    Once WalkOnce(JsonElement rec)
    {
        _pending.Clear();
        _factsMet.Clear();
        _taintAfterFloor = null;
        _passed = 0;
        var result = WalkRecord(rec);
        if ((string?)result["status"] != "complete")
        {
            // Only a walk that stops needs its facts settled.
            _factsMet.AddRange(_pending);
        }

        return new Once(result, [.. _factsMet], _passed, Ways());
    }

    JsonObject WalkRecord(JsonElement rec)
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
            // Reset starts every run on a fresh state, so one engine walks them all. The fit resets
            // it under the profile the history shows.
            long r0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var profile = ProfileFit.Fit(rec, _e, seed, asc, character);
            if (!profile.IsAll)
            {
                result["profile"] = ProfileFit.Describe(profile);
            }

            AfterReset?.Invoke(S);
            _belt.Clear();
            _eventsSeen = 0;
            _paelsSacrifices = SavedInt(rec, "PAELS_WING", "RewardsSacrificed") ?? int.MaxValue;
            _seen = _profile is { Available: true } ? ProfileIndex.Seen(rec.GetRawText()) : [];
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
        if (_taintAfterFloor is { } why)
        {
            // An open fact of the floor's fight feeds the Rewards stream behind the floor's own
            // rewards (a Heist row is populated after them): later offers on it are uncertain,
            // this floor's are not.
            _taint.TryAdd("Rewards", why);
            _taintAfterFloor = null;
        }
    }

    void ResetFloorCursors()
    {
        _upgradeCursor = _removeCursor = _transformCursor = _enchantCursor = 0;
        _eventCursor = _restCursor = _gainCursor = 0;
        _relicChoiceUsed.Clear();
        _potionChoiceUsed.Clear();
        _discardsLeft.Clear();
        _discardsLeft.AddRange(StrList("potion_discarded"));
        _discardsLeft.AddRange(StrList("potion_used"));
        _rewardsCompared = false;
        _shopRemovalDone = false;
        _floorUsedTaint = false;
        _cardUsed.Clear();
        _deferredRelic = 0;
        _lavaLampUnsure = false;
        _floorMonsters.Clear();
        _goldUnknown = false;
        _skippedPotions.Clear();
        _skippedRelics.Clear();
        _shopSeen = false;
        _shopWants = null;
        _copyCursor = 0;
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
        var real = Arr("ancient_choice").Select(AncientOption).ToList();
        var emu = S.NeowOptions.Where(x => x != 0).Select(EmulatorAncientOption).ToList();
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
            .Select(AncientOption).FirstOrDefault();
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
    /// An ancient's option as the history names it: the relic its title key names
    /// (`KALEIDOSCOPE.title`), with the character a Sea Glass is branded with
    /// (`SEA_GLASS.DEFECT.title` is `SEA_GLASS.DEFECT`). The record's TextKey is the key's
    /// second-to-last part, which for a Sea Glass is only the character's name.
    /// </summary>
    public static string AncientOption(JsonElement option)
    {
        string text = option.TryGetProperty("TextKey", out var t) ? t.GetString() ?? "" : "";
        if (!option.TryGetProperty("title", out var title) || !title.TryGetProperty("key", out var k) || k.GetString() is not string key
            || (title.TryGetProperty("table", out var table) && table.GetString() != "relics"))
        {
            return text;
        }

        var parts = key.Split('.');
        return parts.Length switch
        {
            2 when parts[1] == "title" => parts[0],
            3 when parts[2] == "title" => $"{parts[0]}.{parts[1]}",
            _ => text,
        };
    }

    /// <summary>The emulator's option in the history's terms: a Sea Glass with its character.</summary>
    string EmulatorAncientOption(int relic) =>
        relic == RunConstants.RelicSeaGlass && S.SeaGlassCharacter >= 0
            ? "SEA_GLASS." + RunConstants.SeaGlassCharacterOf(S).ToString().ToUpperInvariant()
            : Names.Relic(relic);

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
        // Winged Boots' charges: a step off the map's edges spends one (`WingedBoots
        // .AfterRoomEntered`), and the map offers the whole next row while one is left. The
        // node taken is the one from which the rest of the act needs the fewest such steps, so
        // no charge is spent that a later act needs.
        int charges = BootsCharges();
        bool hasCurrent = S.MapNodes.TryGetValue(S.CurrentMapCoord, out var current);
        var memo = new Dictionary<((int, int), int), int>();
        var costed = new List<(int Action, int Jumps)>();
        for (int i = 0; i < S.MapNodeTypes.Length; i++)
        {
            if (S.MapNodeTypes[i] != wanted || S.MapOptionCoords[i] is not { } c)
            {
                continue;
            }

            bool edge = !hasCurrent || current!.Children.Contains(c);
            int rest = MinJumps(c, ahead, 0, memo);
            if (rest < int.MaxValue && (edge ? 0 : 1) + rest <= charges)
            {
                costed.Add((i, (edge ? 0 : 1) + rest));
            }
        }

        var candidates = costed.OrderBy(x => x.Jumps).Select(x => x.Action).ToList();
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
        if (costed.Count(x => x.Jumps == costed.Min(y => y.Jumps)) > 1)
        {
            _ambiguousFloors++;
        }

        Do(candidates[0]);
    }

    /// <summary>
    /// The fewest steps off the map's edges (Winged Boots: to any node of the next row, short of
    /// the boss, as `RunMapGenerator.TravelableCoords` allows) that walk the history's remaining
    /// point types from a node, or int.MaxValue when none does.
    /// </summary>
    int MinJumps((int Col, int Row) node, string[] ahead, int step, Dictionary<((int, int), int), int> memo)
    {
        if (step >= ahead.Length)
        {
            return 0;
        }

        if (memo.TryGetValue((node, step), out int known))
        {
            return known;
        }

        int best = int.MaxValue;
        if (S.MapNodes.TryGetValue(node, out var n))
        {
            foreach (var child in n.Children)
            {
                if (S.MapNodes.TryGetValue(child, out var cn) && Names.MapType(cn.NodeType) == ahead[step])
                {
                    best = Math.Min(best, MinJumps(child, ahead, step + 1, memo));
                }
            }

            if (best > 0 && BootsCharges() > 0 && n.Row < RunConstants.MapBossRowFor(S.Act) - 1)
            {
                foreach (var other in S.MapNodes.Values)
                {
                    if (other.Row == n.Row + 1 && !n.Children.Contains((other.Col, other.Row)) && Names.MapType(other.NodeType) == ahead[step]
                        && MinJumps((other.Col, other.Row), ahead, step + 1, memo) is int rest and < int.MaxValue)
                    {
                        best = Math.Min(best, rest + 1);
                    }
                }
            }
        }

        memo[(node, step)] = best;
        return best;
    }

    int BootsCharges()
    {
        int at = S.Relics.FindIndex(x => x.DefId == RunConstants.RelicWingedBoots);
        return at < 0 ? 0 : Math.Max(0, RunConstants.WingedBootsTravels - S.Relics[at].Counter);
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
        _floorMonsters.AddRange(real);
        _floorMonsters.AddRange(emu);
        // The history lists every creature the fight had, the ones that joined it as well; the
        // opening roster is the first of them, in order.
        bool prefix = real.Count >= emu.Count && real.Take(emu.Count).SequenceEqual(emu);
        Check("monsters", "opening roster", string.Join(",", real.Take(Math.Max(emu.Count, 1))), string.Join(",", emu),
            Source.Static, prefix ? null : "the history's monsters, as many as the emulator opens with");
        if (real.Count > emu.Count)
        {
            // A creature joined mid-fight and its HP was rolled on Niche (CreateCreature draws it
            // once per creature). The history lists the joiners it saw, at least that many.
            JoinedCreatures(real.Count - emu.Count, $"{string.Join(",", real.Skip(emu.Count))} joined the fight");
        }
        else if (emu.FirstOrDefault(Names.Spawners.Contains) is string spawner)
        {
            // The history adds a joining creature's id only if it is not listed yet, so a
            // second rat, a respawned Axebot or another Fabricator bot leaves no trace.
            _taint.TryAdd("Niche", $"floor {_floor}: {spawner} can add creatures the history does not list");
        }
    }

    const string JoinedFact = "joined";
    const int MaxJoined = 12;

    /// <summary>
    /// The number of creatures created in this floor's fight after its opening roster is a hidden
    /// fact: each one rolled its HP on Niche, and the history only names the joiners it lists (a
    /// second summon of a listed id leaves no trace). The walk draws that many Niche values (the
    /// listed count unless an earlier walk settled it otherwise); while the count is open the
    /// stream stays tainted from here, as it was before the count was a fact.
    /// </summary>
    void JoinedCreatures(int listed, string why)
    {
        var (count, open) = Hidden(JoinedFact, MaxJoined, Math.Min(listed, MaxJoined - 1));
        var niche = S.ActiveCombat?.NicheHpRng;
        for (int i = 0; i < count; i++)
        {
            if (niche is not null)
            {
                niche.Next();
            }
            else
            {
                S.Rng.Niche.NextInt();
            }
        }

        if (open)
        {
            _taint.TryAdd("Niche", $"floor {_floor}: {why}");
        }
    }

    void ResolveFight(JsonElement room)
    {
        _fights++;
        foreach (string st in Streams.FightTurns)
        {
            _taint.TryAdd(st, $"floor {_floor}: the fight's turns were not played");
        }

        // Potions drunk in the fight are gone before its rewards are made and claimed. The resolve
        // copies the fight's own belt back over the run's, so they leave the fight's belt: a slot
        // one of them freed is open when the rewards read the belt (Delicate Frond fills it) and
        // when the reward's potion is claimed, as in the game.
        int[] fightBelt = S.ActiveCombat?.PotionSlots ?? S.PotionSlots;
        foreach (string p in StrList("potion_used"))
        {
            if (RemovePotion(fightBelt, p))
            {
                _discardsLeft.Remove(p);
            }
        }
        Mark();
        MarkFloor();
        // What the fight did that its rewards read, as far as the history shows it. The Gremlin
        // Merc's gold is rolled at half, or not at all, when the Fat Gremlin escapes (with the
        // Merc's loot): no gold taken is the second, a short purse the first. Under Ectoplasm
        // every purse is empty, so only the loot tells.
        bool merc = S.ActiveCombat?.EncounterId == RunConstants.GremlinMercEncounterId;
        bool goldBlind = S.Relics.Any(x => x.DefId == RelicEffects.Ectoplasm);
        int gained = Int("gold_gained");
        int Poverty(int g) => Ascension.Has(S.AscensionLevel, Ascension.Poverty) ? (int)(g * Ascension.PovertyGoldMultiplier) : g;
        int fullMin = Poverty(10), halfMax = (int)Math.Round(Poverty(20) * 0.5, MidpointRounding.AwayFromZero);
        // Loot still missing at the end of the floor (`StolenLoot`, which a Heist's or a Swipe's
        // return takes back off) left with a creature that escaped.
        int loot = Int("stolen_loot");
        bool fatEscaped = merc && (loot > 0 || (!goldBlind && gained < fullMin));
        bool mercStole = merc && (loot > 0 || (!goldBlind && gained == 0));
        // A purse both a full roll and a half one can give: either way it is one Rewards draw,
        // so only its amount is unknown.
        _goldUnknown = merc && !goldBlind && gained >= fullMin && gained <= halfMax;
        int heistGold = merc && !fatEscaped ? HeistGold(room) : 0;
        if (heistGold > 0 && S.ActiveCombat is { } fight)
        {
            fight.PlayerGold = Math.Max(0, fight.PlayerGold - heistGold);
        }

        // A Thieving Hopper's THIEVERY took a card (`SwipePower.Steal` records it in
        // `cards_removed` and puts `stolen_loot` up); killed, it hands the card back as a row the
        // player claimed (in `cards_gained`) or left, and takes the loot back off
        // (`SwipePower.BeforeDeath`); with the loot still missing, it left with the card.
        bool hopper = S.ActiveCombat?.EncounterId == (int)CombatFactory.ActOneEncounter.ThievingHopper;
        bool kept = hopper && loot > 0;
        bool goldRow = hopper && HopperGoldRow(kept, gained, fullMin, goldBlind);
        CardInstance? stolen = null;
        if (hopper && Arr("cards_removed").FirstOrDefault() is { ValueKind: JsonValueKind.Object } taken)
        {
            string name = CardName(taken);
            int at = StolenCopy(taken, name);
            if (at >= 0)
            {
                // `SwipePower.Steal` takes the copy out of the deck (`CardPileCmd.RemoveFromDeck`).
                // Taken here, not by the resolve, which finds a stolen card by value (the first
                // equal copy) where the walk has settled which copy it was.
                stolen = S.Deck[at];
                S.Deck.RemoveAt(at);
                _removeCursor++;
                if (!kept && !Arr("cards_gained").Select(CardName).Contains(name))
                {
                    // `SpecialCardReward.OnSkipped` lists the card, unpicked, among the floor's
                    // card choices: last if the card reward was answered first, else first.
                    var choices = Arr("card_choices").ToList();
                    bool picked = choices.Any(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True);
                    foreach (int i in picked ? new[] { choices.Count - 1, 0 } : new[] { 0, choices.Count - 1 })
                    {
                        if (i >= 0 && choices.Count % 3 == 1 && CardName(choices[i].GetProperty("card")) == name)
                        {
                            _cardUsed.Add(i);
                            break;
                        }
                    }
                }
            }
        }
        // Lava Lamp's flag is the fight's: no damage taken at all is no unblocked hit, for
        // certain. Damage taken may all have been unblockable, which the flag ignores and the
        // history does not tell apart: the fact LampFact, which the card offer settles (met
        // there, when an offer upgraded is the only difference).
        if (S.Relics.Any(x => x.DefId == RelicEffects.LavaLamp))
        {
            bool hurt = Int("damage_taken") > 0;
            string key = $"{_floor}:{LampFact}";
            S.TookUnblockedDamageThisCombat = hurt && Way(LampFact) != 1;
            _lavaLampUnsure = hurt && (!_facts.ContainsKey(key) || _openFacts.Contains(key));
        }

        long f0 = System.Diagnostics.Stopwatch.GetTimestamp();
        _e.DebugResolveCombat(
            stolenBackGold: heistGold,
            fatGremlinEscaped: fatEscaped,
            mercGoldWasStolen: mercStole,
            escapedEnemies: hopper && !goldRow ? 1 : 0,
            stolenBackCards: stolen is { } back && !kept ? [back] : null
        );
        FightSeconds += System.Diagnostics.Stopwatch.GetElapsedTime(f0).TotalSeconds;
        _rewardsCompared = false;
        _fightJustResolved = true;
    }

    const string HeistFact = "heist", HopperRowFact = "hopper_gold_row", CopyFact = "stolen_copy",
        LampFact = "lava_lamp_unblockable";

    /// <summary>The most identical-copy ways a stolen card is walked (each is a walk of the run).</summary>
    const int MaxCopyWays = 6;

    /// <summary>
    /// The gold of the Heist row a Gremlin Merc fight left when its Fat Gremlin died, or 0 for
    /// none. Decompiled: on each of its moves the Merc's `ThieveryPower.Steal` takes up to 20 gold
    /// while the player has any (`PlayerCmd.LoseGold(Stolen)`: `gold_stolen` and `stolen_loot` up);
    /// `SurprisePower` hands the total to the Fat Gremlin as Heist, and `HeistPower.BeforeDeath`
    /// adds `GoldReward(Amount, wasGoldStolenBack)` (one Rewards draw, behind the room's own) and
    /// takes the loot back off `stolen_loot`. Claimed, `GainGold` takes what it pays (after gold
    /// modifiers) off `gold_stolen`. So the history shows the row only when it was left, or paid
    /// other than it took; a Heist claimed in full leaves both counters at 0, like no theft at
    /// all. The theft and the claim net out, as in the game, so the purse check still holds.
    /// </summary>
    int HeistGold(JsonElement room)
    {
        int stolen = Int("gold_stolen");
        if (stolen > 0)
        {
            // Left on the screen (or claimed under Ectoplasm, which pays nothing): its amount.
            return stolen;
        }

        int take = Math.Min(20, S.Gold);
        if (stolen < 0)
        {
            // Claimed through a gold modifier that paid more than was taken.
            return Math.Max(1, take);
        }

        int turns = room.ValueKind == JsonValueKind.Object && room.TryGetProperty("turns_taken", out var tt)
            && tt.ValueKind == JsonValueKind.Number ? tt.GetInt32() : 0;
        // `turns_taken` is the player's `TurnNumber` at the end, which starts at 1 and moves on
        // when the enemies' side hands back: a one-turn fight ended before the Merc's first move
        // was over (its gremlins, spawned at its death, outlive the enemies' turn). An empty purse
        // has nothing to take, and the Merc is the only enemy that could pay gold first.
        if (turns == 1 || take == 0)
        {
            return 0;
        }

        // Otherwise the history cannot say whether the Merc lived to move: settled by what the
        // Rewards stream gives next.
        var (way, open) = Hidden(HeistFact);
        if (open)
        {
            _taintAfterFloor ??= $"floor {_floor}: whether the Gremlin Merc stole before it died (a Heist gold row, one Rewards draw) is not in the history";
        }

        return way == 1 ? take : 0;
    }

    /// <summary>
    /// Whether a Thieving Hopper's room has its gold row (one Rewards draw).
    /// `EncounterModel.CalculateGoldProportion` is 1 - escaped / spawned kinds, so the lone
    /// Hopper's room has none once it escaped. A thief that died left one: its card was taken
    /// back off the loot (`SwipePower.BeforeDeath`). Otherwise the history leaves the row open,
    /// and the next rewards settle it. A Hopper that kept its loot escaped, and by default has no
    /// row; but the Codex has such floors that paid a purse, some with the row and some without
    /// (another purse), so the gold does not tell. One that took nothing died if a full purse was
    /// paid, by default (Ectoplasm empties every purse, so there it died by default too).
    /// </summary>
    bool HopperGoldRow(bool kept, int gained, int fullMin, bool goldBlind)
    {
        if (!kept && Arr("cards_removed").Any())
        {
            return true;
        }

        bool byDefault = !kept && (goldBlind || gained >= fullMin);
        var (way, open) = Hidden(HopperRowFact, defaultWay: byDefault ? 1 : 0);
        if (open)
        {
            _taintAfterFloor ??= $"floor {_floor}: whether the Thieving Hopper's room kept its gold row (one Rewards draw) is not in the history";
        }

        return way == 1;
    }

    /// <summary>
    /// The deck copy a Thieving Hopper took, from the history's `cards_removed` entry: its card,
    /// upgrade and enchantment say which copies it can be. Of identical copies the history does not
    /// say which (the steal is the fight's own draw), and the deck's order after the floor depends
    /// on it, as the copy leaves its place (and comes back at the bottom): a fact, with a way for
    /// each run of side-by-side copies (taking any one of a run leaves the same deck), the first
    /// by default.
    /// </summary>
    int StolenCopy(JsonElement taken, string name)
    {
        var ench = EnchantmentOf(taken);
        var same = Enumerable.Range(0, S.Deck.Count).Where(i => CardName(S.Deck[i].DefId, S.Deck[i].Upgraded) == name).ToList();
        var exact = same.Where(i => S.Deck[i].Enchantment == ench).ToList();
        var copies = (exact.Count > 0 ? exact : same).Where(i => i == 0 || S.Deck[i - 1] != S.Deck[i]).ToList();
        if (copies.Count <= 1)
        {
            return copies.Count == 1 ? copies[0] : -1;
        }

        return copies[Hidden(CopyFact, Math.Min(copies.Count, MaxCopyWays)).Way];
    }

    /// <summary>A history card's enchantment (`ENCHANTMENT.PERFECT_FIT` is PerfectFit), or None.</summary>
    static Enchantment EnchantmentOf(JsonElement card) =>
        card.TryGetProperty("enchantment", out var en) && en.ValueKind == JsonValueKind.Object && en.TryGetProperty("id", out var id)
        && Enum.TryParse<Enchantment>(string.Concat(Names.Bare(id.GetString()).Split('_', StringSplitOptions.RemoveEmptyEntries)
            .Select(w => w[..1] + w[1..].ToLowerInvariant())), out var e)
            ? e
            : Enchantment.None;

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
            if (S.RelicReward == _deferredRelic && !CardOfferWaiting())
            {
                // The card reward the history claimed first is done: now the relic.
                _deferredRelic = 0;
                Mark();
                Do(index);
                return;
            }

            if (S.RelicReward != _deferredRelic && !_skippedRelics.Contains(S.RelicReward))
            {
                string relic = Names.Relic(S.RelicReward);
                int at = FindChoice("relic_choices", relic, _relicChoiceUsed);
                Check("relic_reward", "relic", at >= 0 ? relic : FirstOpen("relic_choices", _relicChoiceUsed), relic, Source.Rewards);
                _relicChoiceUsed.Add(at);
                if (Picked("relic_choices", at))
                {
                    if (CardOfferWaiting() && CardBeforeRelic())
                    {
                        _deferredRelic = S.RelicReward;
                    }
                    else
                    {
                        Mark();
                        Do(index);
                        return;
                    }
                }
                else
                {
                    _skippedRelics.Add(S.RelicReward);
                }
            }

            index++;
        }

        // A special card row (a Thieving Hopper's stolen card, the Lantern Key off its knight),
        // which the history claimed (`cards_gained`). The rows sit after the card items. Claimed in
        // the history's order: the deck takes each card at its bottom, so which came first is the
        // deck's order, and the deck-add hooks (Book of Five Rings, Lucky Fysh) see it arrive.
        var (specialTaken, specialFirst) = SpecialCardClaim();
        if (specialTaken && specialFirst && ClaimSpecialCard(index))
        {
            return;
        }

        if (S.RewardCardPending || S.PendingCardOffers.Count > 0 || S.PendingOtherCharacterCardRewards > 0)
        {
            // Opened only if the history shows the offer.
            if (CardChoicesLeft > 0)
            {
                Do(index);
                return;
            }
        }

        if (specialTaken && ClaimSpecialCard(index))
        {
            return;
        }

        // More card offers in the history than the emulator made: the fight added a reward
        // (The Hunt's kill, say), rolled on the Rewards stream after the fight's own.
        if (_combatFloor && CardChoicesLeft > 0)
        {
            // Only The Hunt adds a card reward from inside a fight (`CombatRoom.AddExtraReward`);
            // anything else is an offer the emulator did not make.
            if (!S.Deck.Any(c => Names.Card(c.DefId) == "THE_HUNT"))
            {
                Fail("card_reward", "offers", Arr("card_choices").Count().ToString(), _cardUsed.Count.ToString(), Source.Rewards,
                    "card offers the history shows on the floor, against the emulator's");
            }

            _fightAddedRewards++;
            _taint.TryAdd("Rewards", $"floor {_floor}: The Hunt added a card reward");
            _cardUsed.UnionWith(Enumerable.Range(0, Arr("card_choices").Count()));
        }

        Do(RunConstants.RewardSkipAction);
    }

    /// <summary>Claims the screen's first special-card row, if its action is reachable.</summary>
    bool ClaimSpecialCard(int index)
    {
        if (S.RewardSpecialCards.Count == 0)
        {
            return false;
        }

        int special = index + RunRewardGenerator.CardItemCount(S);
        if (special == RunConstants.RewardSkipAction)
        {
            return false;
        }

        string offered = CardName(S.RewardSpecialCards[0].DefId, S.RewardSpecialCards[0].Upgraded);
        int at = S.Deck.Count;
        Do(special);
        // `CardPileCmd.Add` lists the card in `cards_gained` BEFORE
        // `Hook.ModifyCardBeingAddedToDeck` (an egg's upgrade), so the history names the card as
        // offered and the deck holds what landed.
        if (S.Deck.Count > at && CardName(S.Deck[at].DefId, S.Deck[at].Upgraded) is var landed && landed != offered)
        {
            _specialCardLanded = (offered, landed);
        }

        return true;
    }

    bool _fightJustResolved, _combatFloor, _goldUnknown;

    /// <summary>
    /// Whether the history claimed the screen's first special-card row (`SpecialCardReward`: the
    /// card is in `cards_gained`), and whether before the card reward's pick, which `cards_gained`
    /// lists in claim order.
    /// </summary>
    (bool Taken, bool First) SpecialCardClaim()
    {
        if (S.RewardSpecialCards.Count == 0)
        {
            return (false, false);
        }

        string name = CardName(S.RewardSpecialCards[0].DefId, S.RewardSpecialCards[0].Upgraded);
        var gainedNames = Arr("cards_gained").Select(CardName).ToList();
        if (!gainedNames.Contains(name))
        {
            return (false, false);
        }

        var picks = Arr("card_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => CardName(c.GetProperty("card"))).ToList();
        int pickAt = gainedNames.FindIndex(g => picks.Contains(g) && g != name);
        return (true, pickAt < 0 || gainedNames.IndexOf(name) < pickAt);
    }

    (string Offered, string Landed)? _specialCardLanded;
    readonly HashSet<int> _skippedPotions = [];
    readonly HashSet<int> _skippedRelics = [];
    readonly HashSet<int> _cardUsed = [];

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
        // The history lists a floor's card rewards in the order the player CLAIMED them
        // (`CardReward.OnSelect`), the emulator's screen in the order it made them: the offer
        // is the history's group with the same cards, wherever it sits among those not yet
        // walked; none such, and the next one in the list is compared.
        var names = all.Select(c => CardName(c.GetProperty("card"))).ToList();
        var window = CardGroup(names, _cardUsed, emu);
        var offered = window.Select(i => all[i]).ToList();
        var real = window.Select(i => names[i]).ToList();
        _cardUsed.UnionWith(window);
        int pick = offered.FindIndex(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True);
        if (pick < 0 && S.CardRewardRerollAvailable && CardChoicesLeft >= emu.Count)
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

        if (_lavaLampUnsure && !real.SequenceEqual(shown)
            && real.Select(x => x.TrimEnd('+')).SequenceEqual(shown.Select(x => x.TrimEnd('+'))))
        {
            // Lava Lamp upgrades the offer when the fight dealt no unblocked, blockable damage.
            // The history counts every unblocked point, unblockable ones too, so a fight that
            // only hurt through them is not told apart: the walk is tried with the lamp lit
            // (Walk), and this stop stands only if that fits no better.
            Hidden(LampFact);
            Fail("card_reward", "cards", string.Join(",", real), string.Join(",", shown), FightDependent,
                "Lava Lamp's upgrades: whether the fight's damage was all unblockable is not recorded");
        }

        Check("card_reward", "cards", string.Join(",", real), string.Join(",", shown), Source.Rewards);
        // The pick, as the emulator's screen has it.
        pick = pick < 0 ? -1 : emu.IndexOf(real[pick]);
        if (pick < 0)
        {
            if (Sacrifices())
            {
                var before = S.Relics.Select(x => Names.Relic(x.DefId)).ToList();
                Mark();
                Do(RunConstants.RewardSacrificeAction);
                foreach (string relic in Minus(S.Relics.Select(x => Names.Relic(x.DefId)).ToList(), before))
                {
                    int at = FindChoice("relic_choices", relic, _relicChoiceUsed);
                    int open = UnexplainedRelic();
                    Check("relic_reward", "Pael's Wing relic", at >= 0 ? relic : open >= 0 ? RelicChoice(open) : "(none)", relic, Source.Rewards);
                    _relicChoiceUsed.Add(at);
                }

                return;
            }

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

    /// <summary>
    /// The history's card group the emulator's offer is: among the entries not yet walked,
    /// taken a whole offer at a time from the first, the first run of side-by-side entries
    /// with the offer's cards (any order); failing that, the first entries not yet walked, as
    /// many as the offer has. Whole offers at a time, so a larger group (a fourth card the
    /// emulator did not add) is not matched by its tail.
    /// </summary>
    public static List<int> CardGroup(List<string> history, HashSet<int> used, List<string> offer)
    {
        var open = Enumerable.Range(0, history.Count).Where(i => !used.Contains(i)).ToList();
        var want = offer.OrderBy(x => x, StringComparer.Ordinal).ToList();
        for (int s = 0; offer.Count > 0 && s + offer.Count <= open.Count; s += offer.Count)
        {
            var run = open.Skip(s).Take(offer.Count).ToList();
            // A group's entries sit side by side in the list.
            if (run[^1] - run[0] != offer.Count - 1)
            {
                continue;
            }

            if (run.Select(i => history[i]).OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(want))
            {
                return run;
            }
        }

        return open.Take(offer.Count).ToList();
    }

    int CardChoicesLeft => Arr("card_choices").Count() - _cardUsed.Count;

    bool CardOfferWaiting() =>
        (S.RewardCardPending || S.PendingCardOffers.Count > 0 || S.PendingOtherCharacterCardRewards > 0) && CardChoicesLeft > 0;

    /// <summary>
    /// The screen's claim order is the player's, and the history keeps only its results: a
    /// card the floor's card reward gave that the floor also upgraded was in the deck when the
    /// relic's pickup (War Paint, Whetstone) upgraded it, so the card was claimed first.
    /// </summary>
    bool CardBeforeRelic()
    {
        var upgraded = StrList("upgraded_cards");
        return Arr("card_choices").Any(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True
            && upgraded.Contains(CardKey(c.GetProperty("card")).Id));
    }

    int _deferredRelic;

    /// <summary>
    /// Pael's Wing: an offer the history took nothing from was given up for the relic's count
    /// rather than skipped, as long as the run's final count (the relic's saved
    /// RewardsSacrificed) has sacrifices to spare. A sacrifice that completes a pair pays a
    /// relic at once, so it is made only where the floor shows a relic nothing else gave.
    /// </summary>
    bool Sacrifices()
    {
        int at = S.Relics.FindIndex(x => x.DefId == RunConstants.RelicPaelsWing);
        if (at < 0 || !Legal(RunConstants.RewardSacrificeAction))
        {
            return false;
        }

        int done = S.Relics[at].Counter;
        if (done >= _paelsSacrifices)
        {
            return false;
        }

        return (done + 1) % RunConstants.PaelsWingSacrifices != 0 || UnexplainedRelic() >= 0;
    }

    /// <summary>A relic the floor took that no screen walked so far gave, or -1.</summary>
    int UnexplainedRelic()
    {
        var bought = StrList("bought_relics");
        int i = 0;
        foreach (var c in Arr("relic_choices"))
        {
            string relic = Names.Bare(c.GetProperty("choice").GetString());
            if (!_relicChoiceUsed.Contains(i) && c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True
                && !bought.Remove(relic))
            {
                return i;
            }

            i++;
        }

        return -1;
    }

    string RelicChoice(int index) => Names.Bare(Arr("relic_choices").ElementAt(index).GetProperty("choice").GetString());

    int _paelsSacrifices = int.MaxValue;
    bool _lavaLampUnsure;

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
                // The grid takes any number, so it is done once nothing the floor gained is in
                // it; a gained card it does not hold is a grid the emulator rolled otherwise.
                var missing = gained.Skip(_gainCursor).Where(g => !offer.Contains(g)).ToList();
                if (missing.Count > 0)
                {
                    Fail("card_offer", "grid", string.Join(",", missing), string.Join(",", offer), Source.Drawn,
                        "cards the floor gained that the emulator's grid does not hold");
                }

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
                    // The copy is the gained card that carries its original's floor.
                    wanted = Arr("cards_gained").Where(IsCopy).Select(CardKey).ToList();
                    cursor = ref _copyCursor;
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

        // A deck selection whose minimum is zero (Kifuda, Gnarled Hammer, Claws) the history
        // closed with fewer picks: DONE, one past the last deck card.
        if (S.PendingSelectionKind != DeckSelection.None && S.PendingSelectionOptional && Legal(S.Deck.Count))
        {
            Do(S.Deck.Count);
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
                Fail("event_option", "option", key, index < 0 ? "(not among the options the emulator drew)" : $"slot {index} not takeable ({LegalList()})",
                    Source.Drawn);
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
            case "TINKER_TIME" when key.Contains(".pages.CHOOSE_CARD_TYPE."):
                // Two of the three types, drawn (`TakeRandom(2)`): the slot is where the
                // history's type sits in the emulator's draw.
                return Array.FindIndex(S.EventRandomOffer, t => ((CardType)t).ToString().ToUpperInvariant() == option) is int type and >= 0 ? type : -1;
            case "TINKER_TIME" when key.Contains(".pages.CHOOSE_RIDER."):
                return Array.FindIndex(S.EventRandomOffer, r => ((TinkerRider)r).ToString().ToUpperInvariant() == option) is int rider and >= 0 ? rider : -1;
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
            // The potions the history drank here, before its buys: a Foul Potion's gold pays for
            // them. What an Entropic Brew made is in the floor's potion picks that were not bought.
            Mark();
            var (usedHere, brewed) = UseAtTheMerchant(S, StrList("potion_used"), Do);
            _brewExpected = [];
            if (usedHere.Contains(EntropicBrew))
            {
                var picks = Arr("potion_choices").Where(c => c.TryGetProperty("was_picked", out var pk) && pk.ValueKind == JsonValueKind.True)
                    .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
                _brewExpected = Minus(picks, StrList("bought_potions"));
                Check("brew", "potions", string.Join(",", _brewExpected), string.Join(",", brewed), Source.Drawn);
            }
        }

        // The history's purchases, unordered: the floor's picked relics and potions, and as
        // cards the floor's gains less card-reward picks and copies (see IsCopy).
        if (_shopWants is null)
        {
            var rewardPicks = Arr("card_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
                .Select(c => CardName(c.GetProperty("card"))).ToList();
            var wantCards = Minus(Arr("cards_gained").Where(c => !IsCopy(c)).Select(c => CardName(c)).ToList(), rewardPicks);
            var wantRelics = Arr("relic_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
                .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
            var wantPotions = Arr("potion_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
                .Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList();
            _shopWants = wantCards.Select(x => (Kind: 0, Item: x)).Concat(wantRelics.Select(x => (Kind: 1, Item: x)))
                .Concat(wantPotions.Select(x => (Kind: 2, Item: x))).ToList();
        }

        // The history does not say in which order they were bought, and the order matters:
        // Membership Card and The Courier discount what is bought after them, an egg upgrades
        // the cards still on the shelf, and Dolly's Mirror copies a card already bought. So:
        // the discounts first, then the cards the shelf shows as the history has them, then the
        // other relics (eggs included), then cards only an egg's upgrade sets apart, then
        // Dolly's Mirror, then potions.
        int CardSlot(string item, bool exact)
        {
            for (int i = 0; i < 7; i++)
            {
                if (S.ShopCards[i] != 0 && (exact ? CardName(S.ShopCards[i], S.ShopCardUpgraded[i]) == item : Names.Card(S.ShopCards[i]) == item.TrimEnd('+')))
                {
                    return i;
                }
            }

            return -1;
        }

        int RelicSlot(string item) => Array.FindIndex(S.ShopRelics, r => r != 0 && Names.Relic(r) == item) is int i and >= 0 ? i + 7 : -1;
        int PotionSlot(string item) => Array.FindIndex(S.ShopPotions, p => p != 0 && Names.Potion(p) == item) is int i and >= 0 ? i + 10 : -1;
        Func<(int Kind, string Item), int>[] passes =
        [
            w => w.Kind == 1 && ShopDiscountRelics.Contains(w.Item) ? RelicSlot(w.Item) : -1,
            w => w.Kind == 0 ? CardSlot(w.Item, exact: true) : -1,
            w => w.Kind == 1 && w.Item != "DOLLYS_MIRROR" ? RelicSlot(w.Item) : -1,
            w => w.Kind == 0 ? CardSlot(w.Item, exact: false) : -1,
            w => w.Kind == 1 ? RelicSlot(w.Item) : -1,
            w => w.Kind == 2 ? PotionSlot(w.Item) : -1,
        ];
        foreach (var pass in passes)
        {
            for (int w = 0; w < _shopWants.Count; w++)
            {
                int slot = pass(_shopWants[w]);
                if (slot < 0)
                {
                    continue;
                }

                if (_shopWants[w].Kind == 2)
                {
                    FreeSlot();
                }

                _shopWants.RemoveAt(w);
                BuySlot(slot);
                return;
            }
        }

        // What is left is not on the shelf: bought in the shop where the emulator does not
        // stock it, or gained some other way on the floor; the shelf comparison decides.

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

    /// <summary>The purchases still to make on this floor, or null before the shop is seen.</summary>
    List<(int Kind, string Item)>? _shopWants;

    /// <summary>Relics that change the price of what is bought after them in the same visit.</summary>
    static readonly HashSet<string> ShopDiscountRelics = ["MEMBERSHIP_CARD", "THE_COURIER"];

    /// <summary>
    /// A gained card that carries `floor_added_to_deck` is a copy of one already in the deck
    /// (Dolly's Mirror, Bing Bong, a rest site's Clone, Reflections): a new card's entry has no
    /// floor, a copy keeps its original's.
    /// </summary>
    static bool IsCopy(JsonElement gained) => gained.TryGetProperty("floor_added_to_deck", out var f) && f.ValueKind == JsonValueKind.Number;

    int _shopCardsBought;

    int _copyCursor;

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
        var rest = Arr("card_choices").Where((_, i) => !_cardUsed.Contains(i)).ToList();
        var unbought = rest.Skip(Math.Max(0, rest.Count - left)).Select(c => CardKey(c.GetProperty("card")).Id).ToList();
        var picked = Arr("card_choices").Where(c => c.TryGetProperty("was_picked", out var p) && p.ValueKind == JsonValueKind.True)
            .Select(c => CardKey(c.GetProperty("card")).Id).ToList();
        var bought = Arr("cards_gained").Where(c => !IsCopy(c)).Select(c => CardKey(c).Id).ToList();
        var realCards = unbought.Concat(Minus(bought, picked)).OrderBy(x => x).ToList();
        var realRelics = Arr("relic_choices").Select(c => Names.Bare(c.GetProperty("choice").GetString())).OrderBy(x => x).ToList();
        // An Entropic Brew's potions are among the floor's potion choices, not the shelf's.
        var realPotions = Minus(Arr("potion_choices").Select(c => Names.Bare(c.GetProperty("choice").GetString())).ToList(), _brewExpected)
            .OrderBy(x => x).ToList();
        Check("shop", "cards", string.Join(",", realCards), string.Join(",", _shopShownCards.OrderBy(x => x)), Source.Static);
        Check("shop", "relics", string.Join(",", realRelics), string.Join(",", _shopShownRelics.OrderBy(x => x)), Source.Static);
        Check("shop", "potions", string.Join(",", realPotions), string.Join(",", _shopShownPotions.OrderBy(x => x)), Source.Static);
        Check("shop", "gold spent", Int("gold_spent").ToString(), _shopSpent.ToString(), Source.Static);
    }

    List<string> _brewExpected = [];

    const string FoulPotion = "FOUL_POTION", EntropicBrew = "ENTROPIC_BREW";

    /// <summary>
    /// The potions a history drank at the merchant, used through the engine's own off-fight use
    /// action: every Foul Potion is thrown at him first (its 100 gold may pay for the floor's
    /// buys), then every Entropic Brew is drunk. Returns the potions used, and what the brews put
    /// on the belt, brew by brew in slot order (a brew fills every open slot, its own included).
    /// </summary>
    public static (List<string> Used, List<string> Brewed) UseAtTheMerchant(RunState s, List<string> drunk, Action<int> use)
    {
        var used = new List<string>();
        var brewed = new List<string>();
        foreach (string p in drunk.Where(x => x == FoulPotion).Concat(drunk.Where(x => x == EntropicBrew)).ToList())
        {
            int at = Array.FindIndex(s.PotionSlots, x => x != 0 && Names.Potion(x) == p);
            if (at < 0)
            {
                continue;
            }

            int[] before = (int[])s.PotionSlots.Clone();
            use(RunConstants.UsePotionAction + at);
            used.Add(p);
            if (p == EntropicBrew)
            {
                for (int i = 0; i < s.PotionSlots.Length; i++)
                {
                    if (s.PotionSlots[i] != 0 && (before[i] == 0 || i == at))
                    {
                        brewed.Add(Names.Potion(s.PotionSlots[i]));
                    }
                }
            }
        }

        return (used, brewed);
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
        var choices = StrList("rest_site_choices");
        if (S.RestResultPending)
        {
            if (_restCursor < choices.Count)
            {
                // Miniature Tent leaves the other options open after the first.
                Fail("rest", "option", choices[_restCursor], "(the rest site closed after one option)", Source.Static,
                    "the history took another option at this rest site");
            }

            Do(0);
            return;
        }

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

        if (_specialCardLanded is var (offered, landed) && expected.GetValueOrDefault(offered) > 0)
        {
            Add(offered, -1);
            Add(landed, 1);
        }

        _specialCardLanded = null;

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
            // A fight's own doings (a card stolen and kept, a curse added, a card upgraded for
            // good) are the fight's, like its HP, and the history shows them mixed with what the
            // floor's screens did. What the emulator did on them (a card taken, a relic's pickup:
            // War Paint's or Whetstone's upgrades, a curse) must be among the history's changes;
            // the rest of the history's are taken as the fight's, and the deck is set to it.
            var extraAdded = Minus(Minus(emuDeck, deckBefore), Minus(expectedList, deckBefore));
            var extraRemoved = Minus(Minus(deckBefore, emuDeck), Minus(deckBefore, expectedList));
            if (extraAdded.Count > 0 || extraRemoved.Count > 0)
            {
                Check("deck", "cards (fight floor)", "+" + Fmt(Minus(expectedList, emuDeck)), "+" + Fmt(Minus(emuDeck, expectedList)), Source.FloorDrawn,
                    "a change the emulator made on the floor's screens that the history does not show");
            }
            else
            {
                Count("deck", true);
            }

            if (expDiff.Length > 0 || emuDiff.Length > 0)
            {
                SetDeck(expectedList);
            }

            // Max HP, where nothing in the fight itself can have moved it.
            if (_hasStats && !FightMovesMaxHp())
            {
                Check("outcome", "max hp (fight floor)", Int("max_hp").ToString(), S.PlayerMaxHp.ToString(), Source.FloorDrawn);
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
        // or dropped, against the emulator's less what the history drank or dropped here. A
        // fight's own potions (Alchemize, a potion a relic procures) are the history's alone.
        // Only what the walk has not taken off already (_discardsLeft): taking the floor's whole
        // list off again dropped a second copy of a potion an event had traded (two Strength
        // Potions into The Future of Potions left none).
        var emuBelt = Minus(Held(), _discardsLeft);
        SyncPotions();
        var histBelt = _belt.ToList();
        var potionsExtra = Minus(emuBelt, histBelt);
        var potionsMissing = combat ? [] : Minus(histBelt, emuBelt);
        // A potion drunk away from a fight did what it does (Entropic Brew's potions, say),
        // which the walk does not replay: that floor's belt is not scored.
        bool drankAwayFromFight = !combat && Arr("potion_used").Any();
        if (drankAwayFromFight)
        {
            Count("potions", true);
        }
        else if (potionsExtra.Count > 0 || potionsMissing.Count > 0)
        {
            Check("potions", "belt", "+" + Fmt(potionsMissing), "+" + Fmt(potionsExtra), Source.FloorDrawn,
                "potions after the floor: what each side holds that the other does not");
        }
        else if (histBelt.Count > S.MaxPotionSlots)
        {
            Check("potions", "slots", histBelt.Count.ToString(), S.MaxPotionSlots.ToString(), Source.FloorDrawn,
                "the history holds more potions than the emulator's belt has slots");
        }
        else
        {
            Count("potions", true);
        }
        if (_hasStats)
        {
            // Fights are not replayed: their HP and gold are the history's.
            _e.DebugSetPlayer(Int("current_hp"), Int("max_hp"), Int("current_gold"), null, null, null);
        }
    }

    readonly List<string> _belt = [];

    /// <summary>
    /// Whether the floor's fight may have changed max HP itself (decompiled: Feed and Brightest
    /// Flame, Fruit Juice drunk, Darkstone Periapt on a curse the fight added, Dragon Fruit on
    /// gold the fight paid, Scroll of Biting's Paper Cuts).
    /// </summary>
    bool FightMovesMaxHp() =>
        S.Deck.Any(c => Names.Card(c.DefId) is "FEED" or "BRIGHTEST_FLAME")
        || StrList("potion_used").Contains("FRUIT_JUICE")
        || S.Relics.Any(r => Names.Relic(r.DefId) is "DARKSTONE_PERIAPT")
        || (S.Relics.Any(r => Names.Relic(r.DefId) is "DRAGON_FRUIT") && S.Deck.Any(c => Names.Card(c.DefId) == "HAND_OF_GREED"))
        || _floorMonsters.Contains("SCROLL_OF_BITING");

    readonly List<string> _floorMonsters = [];

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
            Console.Error.WriteLine($"  f{_floor} {S.Phase} event={S.EventId} awaiting={S.NeowAwaitingProceed} -> {action}  [{LegalList()}]"
                + $" belt=[{string.Join("/", S.PotionSlots.Select(x => x == 0 ? "-" : Names.Potion(x)))}] max={S.MaxPotionSlots}");
        }

        var held = Held();
        int status = _e.Step(action, -1, out _, out _, out _);
        if (status != 0)
        {
            Fail("flow", "action", action.ToString(), $"refused in {S.Phase} ({LegalList()})", WalkerLimit);
        }

        TakeOff(_discardsLeft, held, Held());
    }

    List<string> Held() => S.PotionSlots.Where(x => x != 0).Select(Names.Potion).ToList();

    /// <summary>
    /// A potion a step took off the belt (an event's trade or drink, a use) that the history
    /// lists as drunk or dropped on the floor is gone already: it leaves the floor's list of
    /// potions still to take off, once per copy taken, so the floor's belt check does not take
    /// it off a second time (and with it a second copy).
    /// </summary>
    public static void TakeOff(List<string> left, List<string> before, List<string> after)
    {
        foreach (string p in Minus(before, after))
        {
            left.Remove(p);
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

        _passed++;
    }

    void Fail(string component, string field, string expected, string emulator, Source source, string? note = null)
    {
        var (klass, drawn, why) = Classify(source);
        if (klass == Emulator && _profile is { Available: true } && LockedLooking(expected, emulator) is var (entry, epoch))
        {
            // The pools leave a locked epoch's content out, so a profile without it never sees
            // what the emulator offered here.
            klass = Profile;
            note = (note is null ? "" : note + "; ") + $"the emulator offered {entry}, which the profile unlocks with {epoch}, and nothing {epoch} unlocks appears in the run";
        }

        Fail(component, field, expected, emulator, klass, note, drawn, why);
    }

    /// <summary>An entry only the emulator's side names whose epoch the run never shows, or null.</summary>
    (string, string)? LockedLooking(string expected, string emulator)
    {
        static IEnumerable<string> Tokens(string s) =>
            s.Split([',', '|', ' ', '>'], StringSplitOptions.RemoveEmptyEntries).Select(t => t.Trim('+'));
        var theirs = Tokens(expected).ToHashSet();
        return _profile!.Unseen(Tokens(emulator).Where(t => !theirs.Contains(t)), _seen);
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

    /// <summary>A relic's saved int at the run's end (`props.ints`), or null.</summary>
    static int? SavedInt(JsonElement rec, string relic, string name)
    {
        if (!rec.GetProperty("players")[0].TryGetProperty("relics", out var relics))
        {
            return null;
        }

        foreach (var r in relics.EnumerateArray())
        {
            if (Names.Bare(r.GetProperty("id").GetString()) != relic || !r.TryGetProperty("props", out var props)
                || !props.TryGetProperty("ints", out var ints))
            {
                continue;
            }

            foreach (var v in ints.EnumerateArray())
            {
                if (v.GetProperty("name").GetString() == name)
                {
                    return v.GetProperty("value").GetInt32();
                }
            }
        }

        return null;
    }

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

    bool RemovePotion(string p) => RemovePotion(S.PotionSlots, p);

    static bool RemovePotion(int[] belt, string p)
    {
        int at = Array.FindIndex(belt, x => x != 0 && Names.Potion(x) == p);
        if (at < 0)
        {
            return false;
        }

        belt[at] = 0;
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
