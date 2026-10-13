// Which option slot a history event choice (`EVENT.pages.PAGE.options.OPTION.title`) is, read off
// the user's own decompiled event class. The emulator's event action is the option's index in the
// list the game shows. Two shapes are read:
//   - `new EventOption[n] { ... }`: the first n option keys after it, a `_LOCKED` variant being the
//     other branch of its option's own slot (the reading of scripts/replay_real_run.py);
//   - options built one by one (`list.Add(new EventOption(...))`, a single-option page): the
//     option's position among its page's keys in the order the source names them.
// The decompile is read, never copied.
using System.Text.RegularExpressions;

namespace EmuDump.Codex;

public sealed class EventOptionIndex
{
    readonly string? _dir;
    readonly Dictionary<string, (List<List<Regex>> Arrays, List<List<Regex>> Pages)> _cache = [];

    static readonly Regex Choice = new(@"^([A-Z0-9_]+)\.pages\.([A-Z0-9_]+)\.options\.([A-Z0-9_]+)\.", RegexOptions.Compiled);

    // A literal or interpolated option key, or InitialOptionKey("X").
    static readonly Regex Option = new(
        "new EventOption\\(this, [^,]+?, (?:\\$?\"([A-Z0-9_]+\\.pages\\.[^\"]+?\\.options\\.[^\"]+)\"|InitialOptionKey\\(\"([A-Z0-9_]+)\"\\))",
        RegexOptions.Compiled);

    static readonly Regex ArrayStart = new(@"new EventOption\[(\d+)\]", RegexOptions.Compiled);

    public EventOptionIndex(string? decompiledEventsDir) => _dir = decompiledEventsDir;

    public bool Available => _dir is not null && Directory.Exists(_dir);

    /// <summary>Whether a choice key is an event option at all (an ancient's is a relic title).</summary>
    public static bool IsEventChoice(string key) => Choice.IsMatch(key);

    /// <summary>The option's index in the list that offers it, or null.</summary>
    public int? IndexOf(string key)
    {
        var m = Choice.Match(key);
        if (!m.Success)
        {
            return null;
        }

        string wanted = key.EndsWith(".title") ? key[..^".title".Length] : key;
        var (arrays, pages) = Blocks(m.Groups[1].Value);
        foreach (var group in arrays.Concat(pages))
        {
            for (int i = 0; i < group.Count; i++)
            {
                if (group[i].IsMatch(wanted))
                {
                    return i;
                }
            }
        }

        return null;
    }

    static Regex Pattern(string key) =>
        new("^" + Regex.Replace(Regex.Escape(key), @"\\\{[^}]*\}", "[A-Za-z0-9_]+") + "$", RegexOptions.Compiled);

    /// <summary>
    /// The option key of each entry of an array initializer `{ a, b, c }` starting after `from`:
    /// an inline `new EventOption(...)` names its own; a local (`eventOption`) is looked up where
    /// it was assigned. Null when an entry's key cannot be read, so the array is not trusted.
    /// </summary>
    static List<string>? ArrayEntries(string text, int from, string entry)
    {
        int open = text.IndexOf('{', from);
        if (open < 0)
        {
            return null;
        }

        var entries = new List<string>();
        int depth = 0, start = open + 1;
        bool inString = false;
        for (int i = open; i < text.Length; i++)
        {
            char c = text[i];
            if (inString)
            {
                if (c == (char)92)
                {
                    i++;
                }
                else if (c == '"')
                {
                    inString = false;
                }

                continue;
            }

            switch (c)
            {
                case '"':
                    inString = true;
                    break;
                case '(' or '[' or '{':
                    depth++;
                    break;
                case ')' or ']' or '}':
                    depth--;
                    if (depth == 0)
                    {
                        entries.Add(text[start..i]);
                        goto done;
                    }

                    break;
                case ',' when depth == 1:
                    entries.Add(text[start..i]);
                    start = i + 1;
                    break;
            }
        }

        return null;
    done:
        var keys = new List<string>();
        foreach (string raw in entries)
        {
            string e = raw.Trim();
            if (e.Length == 0)
            {
                continue;
            }

            string? key = FirstKey(e, entry);
            if (key is null && System.Text.RegularExpressions.Regex.Match(e, @"^([A-Za-z_][A-Za-z0-9_]*)") is { Success: true } id)
            {
                // A local: the statement that assigns it, before the array.
                int at = text.LastIndexOf(id.Groups[1].Value + " = ", from, StringComparison.Ordinal);
                if (at >= 0)
                {
                    int end = text.IndexOf(';', at);
                    key = FirstKey(text[at..(end < 0 ? text.Length : end)], entry);
                }
            }

            if (key is null)
            {
                return null;
            }

            keys.Add(key);
        }

        return keys;
    }

    static string? FirstKey(string code, string entry)
    {
        foreach (Match o in Option.Matches(code))
        {
            string k = o.Groups[1].Success && o.Groups[1].Value.Length > 0
                ? o.Groups[1].Value
                : $"{entry}.pages.INITIAL.options.{o.Groups[2].Value}";
            if (!k.EndsWith("_LOCKED"))
            {
                return k;
            }
        }

        return null;
    }

    (List<List<Regex>> Arrays, List<List<Regex>> Pages) Blocks(string entry)
    {
        if (_cache.TryGetValue(entry, out var cached))
        {
            return cached;
        }

        var arrays = new List<List<Regex>>();
        var pages = new List<List<Regex>>();
        if (_dir is not null)
        {
            string name = string.Concat(entry.Split('_').Select(p => p.Length == 0 ? "" : p[..1] + p[1..].ToLowerInvariant()));
            string path = Path.Combine(_dir, name + ".cs");
            if (File.Exists(path))
            {
                string text = File.ReadAllText(path);
                // Every option key in source order, with where it sits.
                var keys = new List<(int At, string Key)>();
                foreach (Match o in Option.Matches(text))
                {
                    string k = o.Groups[1].Success && o.Groups[1].Value.Length > 0
                        ? o.Groups[1].Value
                        : $"{entry}.pages.INITIAL.options.{o.Groups[2].Value}";
                    if (!k.EndsWith("_LOCKED"))
                    {
                        keys.Add((o.Index, k));
                    }
                }

                foreach (Match a in ArrayStart.Matches(text))
                {
                    var block = ArrayEntries(text, a.Index + a.Length, entry)?.Select(Pattern).ToList();
                    if (block is { Count: > 0 })
                    {
                        arrays.Add(block);
                    }
                }

                // By page, in the order the source first names each key.
                var byPage = new Dictionary<string, List<string>>();
                foreach (var (_, k) in keys)
                {
                    int p = k.IndexOf(".options.", StringComparison.Ordinal);
                    string page = k[..p];
                    if (!byPage.TryGetValue(page, out var list))
                    {
                        byPage[page] = list = [];
                    }

                    if (!list.Contains(k))
                    {
                        list.Add(k);
                    }
                }

                pages.AddRange(byPage.Values.Select(l => l.Select(Pattern).ToList()));
            }
        }

        _cache[entry] = (arrays, pages);
        return (arrays, pages);
    }
}
