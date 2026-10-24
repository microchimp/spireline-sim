// The source scan over the migrated models (src/Sts2Emulator/Core2/Models/**, docs/rebuild/TIER1_KERNEL.md section 4): until
// analyzers exist, this keeps a model handler to what a Tier 2 runtime can answer.
//   Forbidden (the test fails): a use of OldState. or OldCreatures. (the old state's own adapters: a handler reaches the
//   combat only through its HookCtx), and mutable static state in a model class (a static non-readonly field, a static
//   property with a setter, a static readonly field of a mutable collection type): a handler keeps nothing between calls,
//   because a combat is copied and replayed and a static would be shared by every copy and every thread.
//   Listed (not a failure): every use of a Tier 1 member of the context (K, World, OldBuffs, OldCombat, Enemy). A Tier 2
//   runtime has none of them, so each use is a line of the rewrite when its model moves: docs/rebuild/TIER2_REWRITE_LIST.md.
// To regenerate the list after a batch lands: SPIRELINE_WRITE_TIER2_LIST=1 dotnet test src/Sts2Emulator.Tests --filter
// FullyQualifiedName~ModelSourceScanTests (the test also prints it).
using System.Text;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Core2;

/// <summary>The scan itself: strings in, findings out, so a test can hand it a broken model.</summary>
public static class ModelScan
{
    /// <summary>The HookCtx members that exist only over the old state (Core2/HookCtx.cs, "Tier 1 members").</summary>
    public static readonly string[] Tier1Members = ["K", "World", "OldBuffs", "OldCombat", "Enemy"];

    public sealed record Finding(string File, int Line, string Rule, string Text)
    {
        public override string ToString() => $"{File}:{Line}: {Rule}: {Text}";
    }

    public sealed record Use(string File, int Line, string Member, string Text);

    private static readonly string[] MutableCollections =
    [
        "List", "Dictionary", "HashSet", "Queue", "Stack", "SortedDictionary", "SortedSet", "SortedList", "LinkedList",
        "ConcurrentDictionary", "ConcurrentQueue", "ConcurrentStack", "ConcurrentBag", "StringBuilder", "Random",
    ];

    // A type: a tuple, or a (qualified) name with generic arguments, array ranks and a nullable mark.
    private const string Type = @"(?:\([^()]*\)|[\w\.]+(?:<[^;=(){}]*>)?)(?:\[[,\s]*\])*\??";

    private static readonly Regex StaticField = new(
        @"(?<mods>(?:\b(?:public|private|internal|protected|unsafe|volatile|new|readonly)\s+)*)\bstatic\s+" +
        @"(?<mods2>(?:(?:volatile|unsafe|readonly)\s+)*)(?<type>" + Type + @")\s+(?<name>@?\w+)\s*(?:=(?!>)|;)",
        RegexOptions.Compiled);

    private static readonly Regex StaticProperty = new(
        @"\bstatic\s+(?:(?:unsafe|readonly)\s+)*" + Type + @"\s+(?<name>@?\w+)\s*\{(?<body>[^{}]*)\}", RegexOptions.Compiled);

    /// <summary>The text with comments and the contents of string and character literals blanked, newlines kept, so a
    /// match's offset still names its line and a mention in a comment or a message is no use.</summary>
    public static string Strip(string src)
    {
        var o = new StringBuilder(src.Length);
        int i = 0;
        while (i < src.Length)
        {
            char c = src[i];
            char n = i + 1 < src.Length ? src[i + 1] : '\0';
            if (c == '/' && n == '/')
            {
                while (i < src.Length && src[i] != '\n')
                {
                    o.Append(' ');
                    i++;
                }
            }
            else if (c == '/' && n == '*')
            {
                o.Append("  ");
                i += 2;
                while (i < src.Length && !(src[i] == '*' && i + 1 < src.Length && src[i + 1] == '/'))
                    o.Append(src[i++] == '\n' ? '\n' : ' ');
                if (i < src.Length)
                {
                    o.Append("  ");
                    i += 2;
                }
            }
            else if (c == '"' && src.AsSpan(i).StartsWith("\"\"\""))
            {
                int quotes = 0;
                while (i + quotes < src.Length && src[i + quotes] == '"')
                    quotes++;
                o.Append(' ', quotes);
                i += quotes;
                while (i < src.Length && !src.AsSpan(i).StartsWith(new string('"', quotes)))
                    o.Append(src[i++] == '\n' ? '\n' : ' ');
                int end = Math.Min(quotes, src.Length - i);
                o.Append(' ', end);
                i += end;
            }
            else if (c == '"' || c == '\'')
            {
                // A verbatim string (@"...", $@"...") ends at a quote not doubled; the others at a quote not escaped.
                bool verbatim = c == '"' && i > 0 && (src[i - 1] == '@' || (i > 1 && src[i - 1] == '$' && src[i - 2] == '@')
                                                       || (i > 1 && src[i - 1] == '@' && src[i - 2] == '$'));
                o.Append(c);
                i++;
                while (i < src.Length)
                {
                    char d = src[i];
                    if (verbatim && d == '"' && i + 1 < src.Length && src[i + 1] == '"')
                    {
                        o.Append("  ");
                        i += 2;
                        continue;
                    }

                    if (!verbatim && d == '\\' && i + 1 < src.Length)
                    {
                        o.Append(' ').Append(src[i + 1] == '\n' ? '\n' : ' ');
                        i += 2;
                        continue;
                    }

                    if (d == c || (!verbatim && d == '\n'))
                        break;
                    o.Append(d == '\n' ? '\n' : ' ');
                    i++;
                }

                if (i < src.Length && src[i] == c)
                {
                    o.Append(c);
                    i++;
                }
            }
            else
            {
                o.Append(c);
                i++;
            }
        }

        return o.ToString();
    }

    private static int LineOf(string text, int offset)
    {
        int line = 1;
        for (int i = 0; i < offset && i < text.Length; i++)
            if (text[i] == '\n')
                line++;
        return line;
    }

    private static string LineText(string src, int line) => src.Split('\n')[line - 1].Trim();

    /// <summary>What a model must not do: the old state's adapters, and static state a copy of the combat would share.</summary>
    public static List<Finding> Forbidden(string file, string src)
    {
        var s = Strip(src);
        var found = new List<Finding>();
        void Add(int offset, string rule)
        {
            int line = LineOf(s, offset);
            found.Add(new Finding(file, line, rule, LineText(src, line)));
        }

        foreach (Match m in Regex.Matches(s, @"\bOldState\s*\."))
            Add(m.Index, "OldState is the kernel's own adapter; ask the HookCtx");
        foreach (Match m in Regex.Matches(s, @"\bOldCreatures\s*\."))
            Add(m.Index, "OldCreatures is the kernel's own adapter; ask the HookCtx");
        foreach (Match m in StaticField.Matches(s))
        {
            bool readOnly = (m.Groups["mods"].Value + m.Groups["mods2"].Value).Contains("readonly");
            if (!readOnly)
            {
                Add(m.Index, $"static mutable field '{m.Groups["name"].Value}': a model keeps no state of its own");
                continue;
            }

            string t = m.Groups["type"].Value;
            string head = Regex.Match(t, @"^[\w\.]+").Value;
            head = head[(head.LastIndexOf('.') + 1)..];
            if (MutableCollections.Contains(head))
                Add(m.Index, $"static readonly field '{m.Groups["name"].Value}' holds a mutable {head}: shared by every copy");
        }

        foreach (Match m in StaticProperty.Matches(s))
            if (Regex.IsMatch(m.Groups["body"].Value, @"\bset\b"))
                Add(m.Index, $"static property '{m.Groups["name"].Value}' has a setter: a model keeps no state of its own");
        return found;
    }

    /// <summary>Every use of a Tier 1 member of a HookCtx in the source: through a parameter or local of type HookCtx, or
    /// through <c>rt.Ctx</c> / <c>k.Ctx</c> directly.</summary>
    public static List<Use> Tier1Uses(string file, string src)
    {
        var s = Strip(src);
        var names = new HashSet<string>();
        foreach (Match m in Regex.Matches(s, @"\bHookCtx\s+(\w+)"))
            names.Add(m.Groups[1].Value);
        foreach (Match m in Regex.Matches(s, @"\b(?:var|HookCtx)\s+(\w+)\s*=\s*[\w\.]+\.Ctx\b"))
            names.Add(m.Groups[1].Value);
        string members = string.Join("|", Tier1Members);
        var receivers = names.Count > 0 ? "(?:" + string.Join("|", names.Select(Regex.Escape)) + @")|[\w\.]+\.Ctx" : @"[\w\.]+\.Ctx";
        var uses = new List<Use>();
        foreach (Match m in Regex.Matches(s, $@"\b(?:{receivers})\s*\.\s*(?<member>{members})\b"))
        {
            int line = LineOf(s, m.Index);
            uses.Add(new Use(file, line, m.Groups["member"].Value, LineText(src, line)));
        }

        return uses;
    }

    /// <summary>One markdown document: per model file, its uses of each Tier 1 member.</summary>
    public static string Markdown(IEnumerable<(string File, List<Use> Uses)> models)
    {
        var rows = models.Where(m => m.Uses.Count > 0).OrderBy(m => m.File, StringComparer.Ordinal).ToList();
        var md = new StringBuilder();
        md.Append("# Tier 2 rewrite list\n\n");
        md.Append("Every use, in a migrated model under `src/Sts2Emulator/Core2/Models/`, of a Tier 1 member of the handler\n");
        md.Append("context (`K`, `World`, `OldBuffs`, `OldCombat`, `Enemy`: `Core2/HookCtx.cs`). A Tier 2 runtime has none of\n");
        md.Append("them, so each line is rewritten when its model moves to the Tier 2 state. Generated by\n");
        md.Append("`ModelSourceScanTests` (`SPIRELINE_WRITE_TIER2_LIST=1 dotnet test src/Sts2Emulator.Tests --filter\n");
        md.Append("FullyQualifiedName~ModelSourceScanTests`); regenerate after a batch lands. The same test fails on `OldState.`,\n");
        md.Append("`OldCreatures.` and static mutable state in a model, so those never appear here.\n\n");
        if (rows.Count == 0)
        {
            md.Append("No migrated model uses a Tier 1 member yet.\n");
            return md.ToString();
        }

        md.Append($"{rows.Sum(r => r.Uses.Count)} uses in {rows.Count} models.\n\n");
        md.Append("| model | " + string.Join(" | ", Tier1Members) + " |\n");
        md.Append("|---|" + string.Concat(Tier1Members.Select(_ => "---|")) + "\n");
        foreach (var (file, uses) in rows)
            md.Append($"| `{file}` | " + string.Join(" | ", Tier1Members.Select(m => uses.Count(u => u.Member == m).ToString())) + " |\n");
        md.Append('\n');
        foreach (var (file, uses) in rows)
        {
            md.Append($"## `{file}`\n\n");
            foreach (var u in uses)
                md.Append($"- line {u.Line}, `{u.Member}`: `{(u.Text.Length > 110 ? u.Text[..110] + "..." : u.Text)}`\n");
            md.Append('\n');
        }

        return md.ToString();
    }

    /// <summary>Scans every .cs file under a directory (Registry.cs included: it is a model file too).</summary>
    public static (List<Finding> Forbidden, List<(string File, List<Use> Uses)> Models) ScanDirectory(string dir)
    {
        var forbidden = new List<Finding>();
        var models = new List<(string, List<Use>)>();
        foreach (var path in Directory.EnumerateFiles(dir, "*.cs", SearchOption.AllDirectories).OrderBy(p => p, StringComparer.Ordinal))
        {
            string rel = Path.GetRelativePath(dir, path).Replace('\\', '/');
            string src = File.ReadAllText(path);
            forbidden.AddRange(Forbidden(rel, src));
            models.Add((rel, Tier1Uses(rel, src)));
        }

        return (forbidden, models);
    }
}

public class ModelSourceScanTests(ITestOutputHelper output)
{
    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "Sts2Emulator", "Core2", "Models")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!.FullName;
    }

    [Fact]
    public void TheMigratedModelsUseNeitherTheOldStateAdaptersNorStaticMutableState()
    {
        string root = RepoRoot();
        var (forbidden, models) = ModelScan.ScanDirectory(Path.Combine(root, "src", "Sts2Emulator", "Core2", "Models"));

        string md = ModelScan.Markdown(models);
        output.WriteLine(md);
        if (Environment.GetEnvironmentVariable("SPIRELINE_WRITE_TIER2_LIST") == "1")
            File.WriteAllText(Path.Combine(root, "docs", "rebuild", "TIER2_REWRITE_LIST.md"), md);

        Assert.True(forbidden.Count == 0, "model sources break the handler rules:\n" + string.Join("\n", forbidden));
    }

    // ---------------------------------------------------------------------------------------- the scan catches what it names

    private static List<ModelScan.Finding> Forbidden(string src) => ModelScan.Forbidden("Power/X.cs", src);

    [Theory]
    [InlineData("var w = OldState.Locate(x, y);", "OldState")]
    [InlineData("var c = OldCreatures.At(x, y);", "OldCreatures")]
    [InlineData("OldState\n    .Combat(w, l, []);", "OldState")]
    [InlineData("static int _count;", "static mutable field")]
    [InlineData("private static int _count = 0;", "static mutable field")]
    [InlineData("internal static Dictionary<int, string> Cache = new();", "static mutable field")]
    [InlineData("[ThreadStatic] private static List<int>? _scratch;", "static mutable field")]
    [InlineData("private static (int, int) _last;", "static mutable field")]
    [InlineData("static readonly List<int> Seen = new();", "mutable List")]
    [InlineData("internal static readonly System.Collections.Generic.HashSet<string> Names = [];", "mutable HashSet")]
    [InlineData("public static int Counter { get; set; }", "has a setter")]
    [InlineData("public static int Counter { get; private set; } = 3;", "has a setter")]
    public void ABrokenModelIsCaught(string member, string rule)
    {
        var f = Forbidden("namespace N;\ninternal static class X\n{\n    " + member + "\n}\n");
        Assert.Single(f);
        Assert.Contains(rule, f[0].Rule);
        Assert.True(f[0].Line is 4 or 5, $"line {f[0].Line}");
    }

    [Fact]
    public void WhatAModelMayHaveIsLeftAlone()
    {
        const string ok = """
            // OldState.Locate and static int x; are named in a comment only
            /* static List<int> y = new(); OldCreatures.At( */
            namespace N;
            internal static partial class X
            {
                private const string Name = "OldState.Locate and static int z;";
                private static readonly string Doc = @"OldCreatures.At("" static int q;";
                private static readonly decimal Half = 0.5m;
                private static readonly int[] Table = [1, 2, 3];
                private static readonly IReadOnlyList<int> Seen = [1];
                public static int Answer { get; } = 42;
                public static int Twice => 2;
                private static int Twice2 { get { return 2; } }
                static int M(int a = 3) => a;
                internal static decimal Method(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
                {
                    static int Local(int x) => x;
                    var f = static (int x) => x + 1;
                    int total = 0;
                    return total + Local(f(1));
                }
            }
            """;
        Assert.Empty(Forbidden(ok));
    }

    [Fact]
    public void ABrokenFileInADirectoryFailsTheDirectoryScanAndACleanOneDoesNot()
    {
        string dir = Path.Combine(Path.GetTempPath(), "spireline-modelscan-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(dir, "Power"));
        try
        {
            File.WriteAllText(Path.Combine(dir, "Power", "Good.cs"),
                "internal static class Good { internal static decimal M(HookCtx c, ListenerRef s) => c.Power(s)!.Value.Amount; }");
            Assert.Empty(ModelScan.ScanDirectory(dir).Forbidden);
            File.WriteAllText(Path.Combine(dir, "Power", "Bad.cs"),
                "internal static class Bad { static int _n; internal static int M(HookCtx c) => OldState.PlayerActive(c.World) ? 1 : 0; }");
            var bad = ModelScan.ScanDirectory(dir).Forbidden;
            Assert.Equal(2, bad.Count);
            Assert.All(bad, f => Assert.Equal("Power/Bad.cs", f.File));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    // ---------------------------------------------------------------------------------------- the Tier 2 list

    [Fact]
    public void TheListNamesEveryTier1UseThroughAnyNameTheContextGoesBy()
    {
        const string src = """
            internal static class X
            {
                internal static decimal A(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
                {
                    var me = c.Power(self);                       // neutral: not listed
                    var buffs = c.OldBuffs(a.Target);             // line 6
                    return CardEffects.F(c.OldCombat) + c.K.World.Combat!.PlayerHp;   // line 7 twice (OldCombat, K)
                }

                internal static HookStep B(HookCtx k, ListenerRef self, in DiedArgs a, ChoiceCtx? ctx)
                {
                    return k.Enemy(a.Who) is null || k.World is null ? HookStep.Done : HookStep.Done;  // line 12 twice
                }

                internal static HookStep Continue(HookRuntime rt, int answer)
                {
                    var ctx = rt.Ctx;
                    return ctx.K is null ? HookStep.Done : rt.Ctx.OldCombat is null ? HookStep.Done : HookStep.Done; // line 18 twice
                }

                // c.K in a comment, and "c.World" in a string, are no use
                private const string Msg = "c.World";
            }
            """;
        var uses = ModelScan.Tier1Uses("Power/X.cs", src);
        var got = uses.Select(u => (u.Line, u.Member)).OrderBy(u => u.Line).ThenBy(u => u.Member, StringComparer.Ordinal).ToList();
        var want = new[] { (6, "OldBuffs"), (7, "OldCombat"), (7, "K"), (12, "Enemy"), (12, "World"), (18, "K"), (18, "OldCombat") }
            .OrderBy(u => u.Item1).ThenBy(u => u.Item2, StringComparer.Ordinal).ToList();
        Assert.Equal(want, got);
    }

    [Fact]
    public void TheMarkdownCountsPerModelAndMember()
    {
        var uses = new List<ModelScan.Use>
        {
            new("Power/A.cs", 5, "K", "x(c.K)"), new("Power/A.cs", 9, "K", "y(c.K)"), new("Power/A.cs", 9, "World", "y(c.World)"),
        };
        string md = ModelScan.Markdown([("Power/A.cs", uses), ("Relic/B.cs", [])]);
        Assert.Contains("3 uses in 1 models", md);
        Assert.Contains("| `Power/A.cs` | 2 | 1 | 0 | 0 | 0 |", md);
        Assert.DoesNotContain("Relic/B.cs", md);
        Assert.Contains("No migrated model uses a Tier 1 member yet.", ModelScan.Markdown([("Relic/B.cs", [])]));
    }
}
