// Which content the player's profile unlocks with which epoch (the game's timeline), read off the
// user's own decompiled epoch classes: each names its cards, relics and potions as `Card<X>`,
// `Relic<X>`, `Potion<X>`, and the pools leave an epoch's content out until it is revealed
// (`CardPoolModel.FilterThroughEpochs` and the like). A run history does not carry the profile,
// but it shows what the run met: an epoch nothing of which appears anywhere in the run is taken as
// locked when the emulator offers its content. The decompile is read, never copied.
using System.Text.RegularExpressions;

namespace EmuDump.Codex;

public sealed class ProfileIndex
{
    static readonly Regex Item = new(@"\b(Card|Relic|Potion)<([A-Za-z0-9]+)>", RegexOptions.Compiled);
    static readonly Regex RecordId = new("\"(?:CARD|RELIC|POTION)\\.([A-Z0-9_]+)\"", RegexOptions.Compiled);

    readonly Dictionary<string, string> _epochOf = [];
    readonly Dictionary<string, List<string>> _items = [];

    public ProfileIndex(string? decompiledEpochsDir)
    {
        if (decompiledEpochsDir is null || !Directory.Exists(decompiledEpochsDir))
        {
            return;
        }

        foreach (string file in Directory.EnumerateFiles(decompiledEpochsDir, "*Epoch.cs"))
        {
            string epoch = Path.GetFileNameWithoutExtension(file)[..^"Epoch".Length];
            foreach (Match m in Item.Matches(File.ReadAllText(file)))
            {
                string entry = Names.GameEntry(m.Groups[2].Value);
                if (_epochOf.TryAdd(entry, epoch))
                {
                    (_items.TryGetValue(epoch, out var list) ? list : _items[epoch] = []).Add(entry);
                }
            }
        }
    }

    /// <summary>Built from a decompile with epoch classes in it.</summary>
    public bool Available => _epochOf.Count > 0;

    /// <summary>The card, relic and potion entries a run's record names anywhere.</summary>
    public static HashSet<string> Seen(string recordJson) =>
        [.. RecordId.Matches(recordJson).Select(m => m.Groups[1].Value)];

    /// <summary>
    /// The first of these entries that an epoch unlocks of which nothing appears in the run,
    /// with that epoch, or null.
    /// </summary>
    public (string Entry, string Epoch)? Unseen(IEnumerable<string> entries, HashSet<string> seen)
    {
        foreach (string entry in entries)
        {
            if (_epochOf.TryGetValue(entry, out var epoch) && !_items[epoch].Any(seen.Contains))
            {
                return (entry, epoch);
            }
        }

        return null;
    }
}
