// The game's ModelId entries (CARD.X, RELIC.X, ENCOUNTER.X, ...) in the emulator's terms, for
// the codex replay. Built once from the emulator's own tables; nothing here is restated by hand
// except the encounter aliases, which tools/../scripts/generate_encounter_tags.py also keeps.
using System.Reflection;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;

namespace EmuDump.Codex;

public static class Names
{
    /// <summary>`StringHelper.Slugify`: an underscore before every capital but the first.</summary>
    public static string GameEntry(string className) =>
        Regex.Replace(className, "(?<!^)(?=[A-Z])", "_").ToUpperInvariant();

    /// <summary>`CARD.X` / `RELIC.X` / ... -> `X`.</summary>
    public static string Bare(string? id)
    {
        if (string.IsNullOrEmpty(id))
        {
            return "";
        }

        int dot = id.IndexOf('.');
        return dot >= 0 ? id[(dot + 1)..] : id;
    }

    static Dictionary<string, int>? _cards, _relics, _potions;
    static Dictionary<int, string>? _cardNames, _relicNames, _potionNames, _enemyNames, _eventNames, _encounterNames;

    public static Dictionary<string, int> CardIds => _cards ??= BuildCards();
    public static Dictionary<string, int> RelicIds => _relics ??= BuildRelics();
    public static Dictionary<string, int> PotionIds => _potions ??= BuildPotions();

    static Dictionary<string, int> BuildCards()
    {
        var d = new Dictionary<string, int>();
        foreach (var c in Cards.All)
        {
            d.TryAdd(c.Entry, c.Id);
        }

        return d;
    }

    static Dictionary<string, int> BuildRelics()
    {
        var d = new Dictionary<string, int>();
        foreach (var r in Relics.All)
        {
            d.TryAdd(r.Entry, r.Id);
        }

        return d;
    }

    static Dictionary<string, int> BuildPotions()
    {
        var d = new Dictionary<string, int>();
        for (int id = 1; id < 400; id++)
        {
            try
            {
                var p = Potions.Get(id);
                d.TryAdd(GameEntry(p.Name), id);
            }
            catch (ArgumentException)
            {
            }
        }

        return d;
    }

    public static string Card(int id)
    {
        _cardNames ??= CardIds.ToDictionary(kv => kv.Value, kv => kv.Key);
        return _cardNames.TryGetValue(Math.Abs(id), out var s) ? s : $"#{id}";
    }

    public static string Relic(int id)
    {
        _relicNames ??= RelicIds.ToDictionary(kv => kv.Value, kv => kv.Key);
        return _relicNames.TryGetValue(id, out var s) ? s : $"#{id}";
    }

    public static string Potion(int id)
    {
        _potionNames ??= PotionIds.ToDictionary(kv => kv.Value, kv => kv.Key);
        return _potionNames.TryGetValue(id, out var s) ? s : $"#{id}";
    }

    public static string Enemy(int defId)
    {
        _enemyNames ??= [];
        if (!_enemyNames.TryGetValue(defId, out var s))
        {
            try
            {
                s = GameEntry(Enemies.Get(defId).Name);
            }
            catch (ArgumentException)
            {
                s = $"#{defId}";
            }

            _enemyNames[defId] = s;
        }

        return s;
    }

    /// <summary>The emulator's `RunConstants.Event*` ids, by game entry.</summary>
    public static string Event(int eventId)
    {
        if (_eventNames is null)
        {
            _eventNames = [];
            foreach (var f in typeof(RunConstants).GetFields(BindingFlags.Public | BindingFlags.Static))
            {
                if (!f.IsLiteral || f.FieldType != typeof(int) || !f.Name.StartsWith("Event") || f.Name.EndsWith("Action"))
                {
                    continue;
                }

                int v = (int)f.GetRawConstantValue()!;
                if (v > 0)
                {
                    _eventNames.TryAdd(v, GameEntry(f.Name["Event".Length..]));
                }
            }
        }

        return _eventNames.TryGetValue(eventId, out var s) ? s : $"#{eventId}";
    }

    /// <summary>`CHARACTER.SILENT` -> the emulator's character, or null for one it does not have.</summary>
    public static PlayerCharacter? Character(string? id) => Bare(id) switch
    {
        "IRONCLAD" => PlayerCharacter.Ironclad,
        "SILENT" => PlayerCharacter.Silent,
        "DEFECT" => PlayerCharacter.Defect,
        "REGENT" => PlayerCharacter.Regent,
        "NECROBINDER" => PlayerCharacter.Necrobinder,
        _ => null,
    };

    public static string Act(int act) => act switch
    {
        RunConstants.ActOvergrowth => "OVERGROWTH",
        RunConstants.ActUnderdocks => "UNDERDOCKS",
        RunConstants.ActHive => "HIVE",
        RunConstants.ActGlory => "GLORY",
        _ => $"#{act}",
    };

    // --- encounters ----------------------------------------------------------------------

    /// <summary>The emulator's encounter enum names, by ordinal.</summary>
    public static string EncounterName(int encounterId)
    {
        if (_encounterNames is null)
        {
            _encounterNames = [];
            var t = typeof(CombatFactory).GetNestedType("ActOneEncounter", BindingFlags.NonPublic | BindingFlags.Public)!;
            foreach (var v in Enum.GetValues(t))
            {
                _encounterNames[(int)v] = v.ToString()!;
            }
        }

        return _encounterNames.TryGetValue(encounterId, out var s) ? s : $"#{encounterId}";
    }

    static HashSet<string>? _encounterEnum;

    // Model class name -> the emulator's ActOneEncounter name, where the two differ: the same
    // table as scripts/generate_encounter_tags.py ALIASES (verified there roster by roster).
    static readonly Dictionary<string, string> EncounterAliases = new()
    {
        ["BowlbugsNormal"] = "Bowlbugs",
        ["ChompersNormal"] = "Chompers",
        ["CorpseSlugsNormal"] = "CorpseSlugs",
        ["CorpseSlugsWeak"] = "CorpseSlugs",
        ["ExoskeletonsNormal"] = "ExoskeletonsNormal",
        ["ExoskeletonsWeak"] = "Exoskeletons",
        ["FuzzyWurmCrawlerWeak"] = "FuzzyWurmCrawler",
        ["AxebotsNormal"] = "Axebot",
        ["DecimillipedeElite"] = "Decimillipede",
        ["PunchOffEventEncounter"] = "PunchOff",
        ["KnightsElite"] = "Knights",
        ["ScrollsOfBitingNormal"] = "Scrolls",
        ["ScrollsOfBitingWeak"] = "ScrollsWeak",
        ["SeapunkNormal"] = "Seapunk",
        ["SeapunkWeak"] = "Seapunk",
        ["ShrinkerBeetleWeak"] = "ShrinkerBeetle",
        ["SlumberingBeetleNormal"] = "SlumberingBeetle",
        ["ThievingHopperWeak"] = "ThievingHopper",
        ["TunnelerNormal"] = "TunnelerAndChomper",
        ["TunnelerWeak"] = "Tunneler",
    };

    /// <summary>`ENCOUNTER.CORPSE_SLUGS_WEAK` -> the emulator's enum name, or the Pascal model.</summary>
    public static string EncounterModelToEmulator(string model)
    {
        if (_encounterEnum is null)
        {
            EncounterName(0);
            _encounterEnum = [.. _encounterNames!.Values];
        }

        string pascal = string.Concat(Bare(model).Split('_').Select(p => p.Length == 0 ? "" : p[..1] + p[1..].ToLowerInvariant()));
        if (EncounterAliases.TryGetValue(pascal, out var alias))
        {
            return alias;
        }

        if (_encounterEnum.Contains(pascal))
        {
            return pascal;
        }

        foreach (string suffix in new[] { "Weak", "Normal", "Elite", "Boss", "EventEncounter" })
        {
            if (!pascal.EndsWith(suffix))
            {
                continue;
            }

            string b = pascal[..^suffix.Length];
            foreach (string cand in new[] { b, b + "s", "The" + b, b.StartsWith("The") ? b[3..] : b })
            {
                if (_encounterEnum.Contains(cand))
                {
                    return cand;
                }
            }
        }

        return pascal;
    }

    /// <summary>Player-side creatures the history lists among the monsters.</summary>
    public static readonly HashSet<string> Allies = ["BYRDPIP", "PAELS_LEGION", "OSTY"];

    /// <summary>Subclasses the emulator models as the class they extend.</summary>
    public static string MonsterBase(string entry) => entry switch
    {
        "DECIMILLIPEDE_SEGMENT_FRONT" or "DECIMILLIPEDE_SEGMENT_MIDDLE" or "DECIMILLIPEDE_SEGMENT_BACK" => "DECIMILLIPEDE_SEGMENT",
        "MYSTERIOUS_KNIGHT" => "FLAIL_KNIGHT",
        _ => entry,
    };

    public static string MapType(int nodeType) => nodeType switch
    {
        RunConstants.NodeNormal => "monster",
        RunConstants.NodeElite => "elite",
        RunConstants.NodeRest => "rest_site",
        RunConstants.NodeShop => "shop",
        RunConstants.NodeRelic => "treasure",
        RunConstants.NodeBoss => "boss",
        RunConstants.NodeEvent => "unknown",
        RunConstants.NodeAncient => "ancient",
        _ => $"#{nodeType}",
    };
}
