// The engine boundary of the oracle tools: only the engine folders, Engines/Old/ and Engines/Core2/ (the old engine with
// the Tier 1 kernel attached), name the old engine (Sts2Emulator, and the types declared under Engines/Old/). Every other file in tools/emudump reaches an engine through IEngine
// (Engine/IEngine.cs), except the consumers listed in Engine/not_moved.txt, which have not moved yet, and
// Engines/EngineFactory.cs, which makes the engine by name. Five rules:
//   1. A file outside Engines/Old/ that names Sts2Emulator and is not listed fails.
//   2. A file off the list and outside Engines/ (Fight.cs, Reanchor.cs, Engine/**, every adapter) also fails when it names
//      Engines.Old or a top-level type declared under Engines/Old/ (found by scanning that folder, so a new one is covered; a nested type is named only through its outer one): reaching
//      the old engine through `var` or OldEngine.Of(e).Run is a name of it too.
//   3. A `global using` of Sts2Emulator or Engines.Old in any file (Engines/Old/ too), or a <Using Include> of either in
//      EmuDump.csproj, leaks the types to every file without naming them, and fails.
//   4. A listed file that no longer names Sts2Emulator fails until it is taken off, so the list shrinks as consumers move.
//   5. The list can only shrink: NotMovedCeiling below is the set it began with (20 consumers); an entry outside it fails.
//      Growing the ceiling is a change to this file, in review, not an edit of the list.
using System.Text.RegularExpressions;
using Xunit;

public class EngineBoundaryTests
{
    const string OldFolder = "Engines/Old/";
    const string Core2Folder = "Engines/Core2/";
    const string EnginesFolder = "Engines/";

    /// <summary>The most the allowlist may hold: the consumers listed when the reanchor move finished.</summary>
    internal static readonly IReadOnlySet<string> NotMovedCeiling = new HashSet<string>(StringComparer.Ordinal)
    {
        "Codex/Driver.cs", "Codex/Names.cs", "Codex/ProfileFit.cs", "Codex/Streams.cs", "Codex/Walker.cs",
        "Digest.cs", "HookProbe.cs", "Live.cs", "Meta.cs", "MutHost.cs", "NetState.cs", "Program.cs", "RefsetSetup.cs",
        "RngLog.cs", "RunReplay.cs", "RunSnap.cs", "Setup.cs", "StateWalk.cs", "Tables.cs", "Trace.cs",
    };

    // A string literal (kept: a name inside one is still a name), a char literal ('"' and '{' are some) or a comment (dropped).
    // A char literal left alone would have its quote open a string that runs to the next quote.
    static readonly Regex StringOrComment = new(
        "@\"(?:[^\"]|\"\")*\"|\"(?:[^\"\\\\\\n]|\\\\.)*\"|'(?:[^'\\\\\\n]|\\\\(?:u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{1,4}|[^\\n]))'|//[^\\n]*|/\\*.*?\\*/",
        RegexOptions.Singleline | RegexOptions.Compiled);

    static readonly Regex Name = new(@"\bSts2Emulator\b", RegexOptions.Compiled);

    static readonly Regex OldNamespace = new(@"\bEngines\s*\.\s*Old\b", RegexOptions.Compiled);

    static readonly Regex GlobalUsing = new(@"\bglobal\s+using\b[^;]*;", RegexOptions.Compiled);

    // The ways a source declares a type: class, struct, interface, enum, record (class or struct), delegate.
    static readonly Regex TypeDeclaration = new(
        @"\b(?:class|struct|interface|enum)\s+(@?[A-Za-z_]\w*)|\brecord\s+(?:class\s+|struct\s+)?(@?[A-Za-z_]\w*)|\bdelegate\s+[^;(=]*?(@?[A-Za-z_]\w*)\s*(?:<[^>(]*>)?\s*\(",
        RegexOptions.Compiled);

    // Words after class/struct that are a constraint or a modifier, not a type's name.
    static readonly HashSet<string> NotATypeName = new(StringComparer.Ordinal) { "where", "new", "class", "struct", "notnull", "unmanaged", "record" };

    /// <summary>Source with its comments blanked; string literals kept, or blanked to "" when asked.</summary>
    internal static string Code(string source, bool keepStrings = true) =>
        StringOrComment.Replace(source, m => m.Value[0] == '/' ? " " : keepStrings ? m.Value : "\"\"");

    /// <summary>Whether C# source names the old engine's namespace (a using or a qualified name), comments aside.</summary>
    internal static bool NamesEngine(string source) => Name.IsMatch(Code(source));

    static readonly Regex NamespaceWord = new(@"\bnamespace\b", RegexOptions.Compiled);

    /// <summary>
    /// The types a source declares, in order, each with whether it is top-level (inside no type; a block namespace's braces do
    /// not count), comments and strings aside; and whether the source's braces balanced, which they do when the scan read it right.
    /// </summary>
    internal static (List<(string Name, bool TopLevel)> Types, bool Balanced) Scan(string source)
    {
        var code = Code(source, keepStrings: false);
        var types = new List<(string, bool)>();
        var braces = new Stack<bool>();  // one per open brace: true for a namespace's
        int inTypes = 0, segment = 0, at = 0, underflow = 0;

        void Advance(int to)
        {
            for (; at < to; at++)
            {
                switch (code[at])
                {
                    case '{':
                        bool ns = NamespaceWord.IsMatch(code.AsSpan(segment, at - segment));
                        braces.Push(ns);
                        if (!ns) inTypes++;
                        segment = at + 1;
                        break;
                    case '}':
                        if (braces.Count == 0) underflow++;
                        else if (!braces.Pop()) inTypes--;
                        segment = at + 1;
                        break;
                    case ';':
                        segment = at + 1;
                        break;
                }
            }
        }

        foreach (Match m in TypeDeclaration.Matches(code))
        {
            var name = m.Groups.Cast<Group>().Skip(1).First(g => g.Success).Value.TrimStart('@');
            if (NotATypeName.Contains(name)) continue;
            Advance(m.Index);
            types.Add((name, inTypes == 0));
        }

        Advance(code.Length);
        return (types, braces.Count == 0 && underflow == 0);
    }

    /// <summary>
    /// The names of the top-level types a source declares, comments and strings aside. A nested type (a `Key` inside a class) is
    /// not one: it can be named only through its outer type, and its name is also an ordinary word of other files.
    /// </summary>
    internal static IEnumerable<string> DeclaredTypes(string source) => Scan(source).Types.Where(t => t.TopLevel).Select(t => t.Name);

    /// <summary>The names of every type a source declares, nested ones too.</summary>
    internal static IEnumerable<string> AllDeclaredTypes(string source) => Scan(source).Types.Select(t => t.Name);

    /// <summary>
    /// The old engine's type names: every top-level type declared under Engines/Old/, found by scanning the files given, except
    /// a name some other file declares at top level too. A nested type (OldAnchor.Tables) is reachable only through its outer
    /// type, which is a name.
    /// </summary>
    internal static IReadOnlySet<string> OldTypes(IReadOnlyDictionary<string, string> files)
    {
        var old = files.Where(f => f.Key.StartsWith(OldFolder, StringComparison.Ordinal)).SelectMany(f => DeclaredTypes(f.Value)).ToHashSet(StringComparer.Ordinal);
        old.ExceptWith(files.Where(f => !f.Key.StartsWith(OldFolder, StringComparison.Ordinal)).SelectMany(f => DeclaredTypes(f.Value)));
        return old;
    }

    /// <summary>The old engine's names a source uses (Engines.Old, or one of <paramref name="oldTypes"/>), comments aside.</summary>
    internal static List<string> OldNames(string source, IReadOnlySet<string> oldTypes)
    {
        var code = Code(source);
        var found = OldNamespace.Matches(code).Select(m => "Engines.Old").ToList();
        if (oldTypes.Count > 0)
        {
            var types = new Regex(@"\b(?:" + string.Join("|", oldTypes.Order(StringComparer.Ordinal).Select(Regex.Escape)) + @")\b");
            found.AddRange(types.Matches(code).Select(m => m.Value));
        }

        return found.Distinct().Order(StringComparer.Ordinal).ToList();
    }

    /// <summary>What a global using in a source leaks: those that name Sts2Emulator or Engines.Old.</summary>
    internal static List<string> GlobalUsingLeaks(string source) =>
        GlobalUsing.Matches(Code(source)).Select(m => m.Value)
            .Where(u => Name.IsMatch(u) || OldNamespace.IsMatch(u)).ToList();

    /// <summary>What EmuDump.csproj leaks to every file with a &lt;Using Include&gt; item of the engine's namespaces.</summary>
    internal static List<string> ProjectProblems(string csproj)
    {
        var xml = Regex.Replace(csproj, "<!--.*?-->", " ", RegexOptions.Singleline);
        var problems = new List<string>();
        foreach (Match m in Regex.Matches(xml, "<Using\\b[^>]*?\\bInclude\\s*=\\s*\"([^\"]*)\"", RegexOptions.Singleline | RegexOptions.IgnoreCase))
        {
            foreach (var ns in m.Groups[1].Value.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                if (Name.IsMatch(ns) || OldNamespace.IsMatch(ns)) problems.Add($"EmuDump.csproj has <Using Include=\"{ns}\">: it leaks the engine's types to every file");
            }
        }

        return problems;
    }

    /// <summary>
    /// What breaks the boundary. <paramref name="files"/>: relative path (forward slashes, from tools/emudump) to source;
    /// <paramref name="notMoved"/>: the allowlist; <paramref name="ceiling"/>: the most the allowlist may hold.
    /// </summary>
    internal static List<string> Problems(IReadOnlyDictionary<string, string> files, IReadOnlySet<string> notMoved, IReadOnlySet<string> ceiling)
    {
        var problems = new List<string>();
        var oldTypes = OldTypes(files);
        foreach (var (path, source) in files.OrderBy(f => f.Key, StringComparer.Ordinal))
        {
            foreach (var leak in GlobalUsingLeaks(source)) problems.Add($"{path} has `{leak}`: a global using leaks the engine's types to every file without naming them");
            if (path.StartsWith(OldFolder, StringComparison.Ordinal) || path.StartsWith(Core2Folder, StringComparison.Ordinal)) continue;
            bool names = NamesEngine(source);
            bool listed = notMoved.Contains(path);
            if (names && !listed) problems.Add($"{path} names Sts2Emulator outside {OldFolder}: reach the engine through IEngine");
            if (!names && listed) problems.Add($"{path} is on Engine/not_moved.txt but no longer names Sts2Emulator: take it off the list");
            if (!listed && !path.StartsWith(EnginesFolder, StringComparison.Ordinal))
            {
                var old = OldNames(source, oldTypes);
                if (old.Count > 0) problems.Add($"{path} names the old engine ({string.Join(", ", old)}) outside {OldFolder}: reach the engine through IEngine");
            }
        }

        foreach (var path in notMoved.Order(StringComparer.Ordinal))
        {
            if (!files.ContainsKey(path)) problems.Add($"{path} is on Engine/not_moved.txt but is not a source file of tools/emudump");
            if (path.StartsWith(OldFolder, StringComparison.Ordinal) || path.StartsWith("Engine/", StringComparison.Ordinal) || path == "Fight.cs" || path == "Reanchor.cs")
            {
                problems.Add($"{path} must not be on Engine/not_moved.txt");
            }

            if (!ceiling.Contains(path))
            {
                problems.Add($"{path} is on Engine/not_moved.txt but is not among the {ceiling.Count} files the list began with: the list only shrinks");
            }
        }

        return problems;
    }

    internal static string EmuDumpDir()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string p = Path.Combine(d.FullName, "tools", "emudump", "Engine", "not_moved.txt");
            if (File.Exists(p)) return Path.GetDirectoryName(Path.GetDirectoryName(p))!;
        }

        throw new DirectoryNotFoundException("tools/emudump/Engine/not_moved.txt not found above " + AppContext.BaseDirectory);
    }

    internal static Dictionary<string, string> Sources(string dir) =>
        Directory.EnumerateFiles(dir, "*.cs", SearchOption.AllDirectories)
            .Select(f => (Rel: Path.GetRelativePath(dir, f).Replace('\\', '/'), Path: f))
            .Where(f => !f.Rel.StartsWith("bin/") && !f.Rel.StartsWith("obj/"))
            .ToDictionary(f => f.Rel, f => File.ReadAllText(f.Path));

    static HashSet<string> NotMoved(string dir) =>
        File.ReadAllLines(Path.Combine(dir, "Engine", "not_moved.txt"))
            .Select(l => l.Trim()).Where(l => l.Length > 0 && !l.StartsWith('#')).ToHashSet(StringComparer.Ordinal);

    [Fact]
    public void OnlyTheOldEngineAndTheListedConsumersNameTheEngine()
    {
        string dir = EmuDumpDir();
        Assert.Empty(Problems(Sources(dir), NotMoved(dir), NotMovedCeiling));
    }

    [Fact]
    public void TheProjectFileLeaksNoEngineNamespaceThroughUsingItems()
    {
        string dir = EmuDumpDir();
        Assert.Empty(ProjectProblems(File.ReadAllText(Path.Combine(dir, "EmuDump.csproj"))));
    }

    [Fact]
    public void FightReanchorAndTheInterfaceNameNoEngine()
    {
        string dir = EmuDumpDir();
        var files = Sources(dir);
        var oldTypes = OldTypes(files);
        foreach (var path in new[] { "Fight.cs", "Reanchor.cs", "Engine/IEngine.cs", "Engine/ReanchorTypes.cs" })
        {
            Assert.Contains(path, files.Keys);
            Assert.False(NamesEngine(files[path]), $"{path} names Sts2Emulator");
            Assert.Empty(OldNames(files[path], oldTypes));
        }
    }

    [Fact]
    public void TheOldEnginesTypesAreFoundByScanningItsFolder()
    {
        string dir = EmuDumpDir();
        var found = OldTypes(Sources(dir));
        foreach (var type in new[] { "OldEngine", "OldChoices", "OldIds", "OldDump", "OldAnchor", "OldReanchor", "OldCarry" })
        {
            Assert.Contains(type, found);
        }

        // Nested: named only through their outer type (OldChoices.CombatScreens is private).
        Assert.DoesNotContain("Tables", found);
        Assert.DoesNotContain("CombatScreens", found);
    }

    [Fact]
    public void TheCeilingIsTheTwentyConsumersTheListBeganWith() => Assert.Equal(20, NotMovedCeiling.Count);

    // The broken cases: the same checks, on files made up here, must fail.

    static readonly Dictionary<string, string> OldFiles = new()
    {
        ["Engines/Old/OldEngine.cs"] = "using Sts2Emulator.Core.Run;\npublic sealed class OldEngine : IEngine { public RunEngine Run; public static OldEngine Of(IEngine e) => null; }",
        ["Engines/Old/OldReanchor.cs"] = "sealed class OldCarry : EngineBlock { } static class OldReanchor { }",
        ["Engines/Old/OldChoices.cs"] = "static class OldChoices { sealed class CombatScreens : IScreenSource { } }",
        ["Engines/Old/OldIds.cs"] = "static class OldIds { }",
        ["Engines/Old/OldDump.cs"] = "static class OldDump { }",
        ["Engines/Old/OldAnchor.cs"] = "static class OldAnchor { internal sealed class Tables { } }",
    };

    static Dictionary<string, string> With(params (string Path, string Source)[] more)
    {
        var files = new Dictionary<string, string>(OldFiles);
        foreach (var (path, source) in more) files[path] = source;
        return files;
    }

    static readonly IReadOnlySet<string> DigestOnly = new HashSet<string> { "Digest.cs" };

    [Fact]
    public void ABrokenCaseIsCaught_AnAdapterNamesTheEngine()
    {
        var files = With(
            ("Fight.cs", "using Sts2Emulator.Core.Run;\nstatic class Fight { RunEngine e; }"),
            ("Engine/IEngine.cs", "namespace EmuDump.Engine; interface IEngine {}"),
            ("Digest.cs", "var c = new Sts2Emulator.Core.CombatState();"));
        var problems = Problems(files, DigestOnly, DigestOnly);
        Assert.Single(problems);
        Assert.Contains("Fight.cs names Sts2Emulator", problems[0]);
    }

    [Fact]
    public void ABrokenCaseIsCaught_AListedFileThatNoLongerNamesTheEngine()
    {
        var problems = Problems(With(("Digest.cs", "static class Digest { }")), DigestOnly, DigestOnly);
        Assert.Single(problems);
        Assert.Contains("take it off the list", problems[0]);
    }

    [Fact]
    public void ABrokenCaseIsCaught_TheListNamesAFileThatIsGoneOrMustNotBeListed()
    {
        var files = With(("Fight.cs", "// IEngine only"), ("Reanchor.cs", "// IEngine only"));
        var listed = new HashSet<string> { "Gone.cs", "Fight.cs", "Reanchor.cs" };
        var problems = Problems(files, listed, listed);
        Assert.Equal(1, problems.Count(p => p.Contains("not a source file")));
        Assert.Equal(2, problems.Count(p => p.Contains("must not be on")));
    }

    [Fact]
    public void ABrokenCaseIsCaught_TheListGrewPastTheCeiling()
    {
        // A consumer added to the list so that it may name the engine: the list is checked against the ceiling, not itself.
        var files = With(("Digest.cs", "var c = new Sts2Emulator.Core.CombatState();"), ("NewConsumer.cs", "var c = new Sts2Emulator.Core.CombatState();"));
        var listed = new HashSet<string> { "Digest.cs", "NewConsumer.cs" };
        var problems = Problems(files, listed, DigestOnly);
        Assert.Single(problems);
        Assert.Contains("NewConsumer.cs is on Engine/not_moved.txt", problems[0]);
        Assert.Contains("the list only shrinks", problems[0]);
    }

    [Theory]
    [InlineData("Fight.cs", "static class Fight { void F(IEngine e) { var run = OldEngine.Of(e).Run; run.Step(0, -1, out _, out _, out _); } }", "OldEngine")]
    [InlineData("Fight.cs", "static class Fight { int F(string s) => OldIds.CardIdBySnake(s); }", "OldIds")]
    [InlineData("Reanchor.cs", "using EmuDump.Engines.Old;\nstatic class Reanchor { }", "Engines.Old")]
    [InlineData("Engine/IEngine.cs", "interface IEngine { OldCarry Carry(); }", "OldCarry")]
    [InlineData("Engine/ReanchorTypes.cs", "class X { EmuDump.Engines.Old.OldAnchor.Tables T; }", "OldAnchor")]
    [InlineData("Codex/Items.cs", "static class Items { object Run(IEngine e) => OldEngine.Of(e).Run; }", "OldEngine")]
    [InlineData("Prof.cs", "static class Prof { object B() => OldChoices.Screens(null, null); }", "OldChoices")]
    [InlineData("Prof.cs", "static class Prof { object B() => System.Type.GetType(\"EmuDump.Engines.Old.OldDump\"); }", "OldDump")]
    public void ABrokenCaseIsCaught_ANeutralFileNamesTheOldEngineWithoutSts2Emulator(string path, string source, string name)
    {
        var problems = Problems(With((path, source)), new HashSet<string>(), NotMovedCeiling);
        Assert.Single(problems);
        Assert.Contains(path + " names the old engine", problems[0]);
        Assert.Contains(name, problems[0]);
        Assert.False(NamesEngine(source));
    }

    [Fact]
    public void TheCore2EngineFolderMayNameTheEngineAndNoOtherEnginesFolderMay()
    {
        var files = With(
            ("Engines/Core2/Core2Engine.cs", "using Sts2Emulator.Core2;\nusing EmuDump.Engines.Old;\nclass Core2Engine : OldEngine {}"),
            ("Engines/Core3/X.cs", "using Sts2Emulator.Core2;"));
        var problems = Problems(files, new HashSet<string>(), NotMovedCeiling);
        Assert.Equal(["Engines/Core3/X.cs names Sts2Emulator outside Engines/Old/: reach the engine through IEngine"], problems);
    }

    [Fact]
    public void ABrokenCaseIsCaught_ATypeAddedToTheOldFolderIsForbiddenAtOnce()
    {
        var files = With(
            ("Engines/Old/OldWidget.cs", "namespace EmuDump.Engines.Old; record struct OldWidget(int A); delegate void OldHook(int a); enum OldMode { A }"),
            ("Fight.cs", "static class Fight { OldMode m; }"),
            ("Reanchor.cs", "static class Reanchor { OldHook h; }"));
        var problems = Problems(files, new HashSet<string>(), NotMovedCeiling);
        Assert.Equal(2, problems.Count);
        Assert.Contains("OldMode", problems[0]);
        Assert.Contains("OldHook", problems[1]);
    }

    [Fact]
    public void TheOldNamesAreAllowedWhereTheyBelong()
    {
        var files = With(
            ("Engines/EngineFactory.cs", "using EmuDump.Engines.Old;\nstatic class EngineFactory { IEngine Create() => new OldEngine(); }"),
            ("Digest.cs", "static class Digest { object R(IEngine e) => OldEngine.Of(e).Run; var d = Sts2Emulator.GeneratedData.Cards.Get(1); }"),
            ("Fight.cs", "// OldEngine and Engines.Old in a comment only\nstatic class Fight { string Old; }"));
        Assert.Empty(Problems(files, DigestOnly, DigestOnly));
    }

    [Theory]
    [InlineData("class A {} interface IB {} struct C {} enum D { X }", "A,IB,C,D")]
    [InlineData("public sealed record R(int X); record struct S(int Y); record class T(int Z);", "R,S,T")]
    [InlineData("internal delegate int Pick<T>(T x); delegate void Hook(int a);", "Pick,Hook")]
    [InlineData("class G<T> where T : class, new() { }", "G")]
    [InlineData("// class Hidden {}\nclass V { string s = \"class InAString\"; }", "V")]
    [InlineData("class Outer { internal sealed class Key { } enum Mode { A } record struct Pair(int A); delegate void Hook(); }\nstruct After {}", "Outer,After")]
    [InlineData("namespace N { class A { class Key {} } interface IB {} }\nclass C {}", "A,IB,C")]
    [InlineData("namespace N;\nclass A { class Key {} }", "A")]
    [InlineData("class A { char Open = '{'; char Quote = '\"'; char Esc = '\\''; class Key {} }\nclass B { char Close = '}'; }", "A,B")]
    [InlineData("class A { void F() { class Local {} } }\nclass B {}", "A,B")]
    public void TheScannerFindsTheTypesASourceDeclares(string source, string names) =>
        Assert.Equal(names.Split(','), DeclaredTypes(source).ToArray());

    [Fact]
    public void TheScannerCanTellANestedTypeFromATopLevelOne()
    {
        var (types, balanced) = Scan("namespace N { class A { class Key {} } }");
        Assert.True(balanced);
        Assert.Equal([("A", true), ("Key", false)], types);
        Assert.Equal(["A", "Key"], AllDeclaredTypes("namespace N { class A { class Key {} } }"));
        Assert.False(Scan("class A {").Balanced);
        Assert.False(Scan("class A { } }").Balanced);
    }

    [Fact]
    public void TheScannerReadsEveryFileOfTheToolsToTheEnd()
    {
        var unbalanced = Sources(EmuDumpDir()).Where(f => !Scan(f.Value).Balanced).Select(f => f.Key).ToList();
        Assert.Empty(unbalanced);
    }

    [Fact]
    public void ANestedTypesNameIsNotAnOldNameWhereAFileUsesTheWord()
    {
        // OldWidget.Key is nested: a file that says Key (a KeyValuePair's, a tuple element's) does not name the old engine.
        var files = With(
            ("Engines/Old/OldWidget.cs", "namespace EmuDump.Engines.Old; static class OldWidget { internal sealed class Key { } internal sealed class Tables { } }"),
            ("Fight.cs", "static class Fight { object F(Dictionary<string, int> d) { foreach (var kv in d) return kv.Key; var keys = new List<(int At, string Key)>(); return keys; } }"),
            ("Codex/Items.cs", "static class Items { string Key; Dictionary<string, string> Tables; }"));
        Assert.DoesNotContain("Key", OldTypes(files));
        Assert.DoesNotContain("Tables", OldTypes(files));
        Assert.Contains("OldWidget", OldTypes(files));
        Assert.Empty(Problems(files, new HashSet<string>(), NotMovedCeiling));
        files["Fight.cs"] = "static class Fight { object F() => new OldWidget(); }";
        Assert.Contains(Problems(files, new HashSet<string>(), NotMovedCeiling), p => p.Contains("OldWidget"));
    }

    [Theory]
    [InlineData("global using Sts2Emulator.Core;", true)]
    [InlineData("global using static Sts2Emulator.GeneratedData.CardPools;", true)]
    [InlineData("global using Pools = Sts2Emulator.GeneratedData.CardPools;", true)]
    [InlineData("global using EmuDump.Engines.Old;", true)]
    [InlineData("global  using\n  Sts2Emulator.Core\n  ;", true)]
    [InlineData("global using System.Text.Json.Nodes;", false)]
    [InlineData("// global using Sts2Emulator.Core;\nclass A {}", false)]
    [InlineData("using Sts2Emulator.Core;\nclass A {}", false)]
    public void TheScannerFindsAGlobalUsingOfTheEngine(string source, bool leaks) => Assert.Equal(leaks, GlobalUsingLeaks(source).Count > 0);

    [Theory]
    [InlineData("Engines/Old/OldEngine.cs")]
    [InlineData("Fight.cs")]
    [InlineData("Digest.cs")]
    public void ABrokenCaseIsCaught_AGlobalUsingAnywhere(string path)
    {
        var problems = Problems(With((path, "global using Sts2Emulator.Core;")), DigestOnly, DigestOnly);
        Assert.Contains(problems, p => p.StartsWith(path + " has `global using Sts2Emulator.Core;`"));
    }

    [Theory]
    [InlineData("<ItemGroup><Using Include=\"Sts2Emulator.Core\" /></ItemGroup>", 1)]
    [InlineData("<ItemGroup>\n  <Using\n    Static=\"true\"\n    Include=\"Sts2Emulator.GeneratedData.CardPools\" />\n</ItemGroup>", 1)]
    [InlineData("<ItemGroup><Using Include=\"System.Text;Sts2Emulator.Core\" /></ItemGroup>", 1)]
    [InlineData("<ItemGroup><USING INCLUDE=\"EmuDump.Engines.Old\" /></ItemGroup>", 1)]
    [InlineData("<ItemGroup><Using Include=\"System.Text\" /><ProjectReference Include=\"..\\..\\src\\Sts2Emulator\\Sts2Emulator.csproj\" /></ItemGroup>", 0)]
    [InlineData("<!-- <Using Include=\"Sts2Emulator.Core\" /> --><ItemGroup />", 0)]
    public void TheScannerFindsAUsingItemOfTheEngineInTheProjectFile(string csproj, int leaks) => Assert.Equal(leaks, ProjectProblems(csproj).Count);

    [Theory]
    [InlineData("using Sts2Emulator.Core;", true)]
    [InlineData("var x = Sts2Emulator.GeneratedData.Cards.Get(1);", true)]
    [InlineData("var t = Type.GetType(\"Sts2Emulator.Core.RunState\");", true)]
    [InlineData("// Sts2Emulator in a comment only\nclass A {}", false)]
    [InlineData("/* Sts2Emulator\n in a block comment */ class A {}", false)]
    [InlineData("/// <see cref=\"x\">Sts2Emulator</see>\nclass A {}", false)]
    [InlineData("class Sts2EmulatorLike {}", false)]
    [InlineData("using EmuDump.Engine;\nclass A { IEngine e; }", false)]
    public void TheScannerFindsTheNameOutsideCommentsOnly(string source, bool names) => Assert.Equal(names, NamesEngine(source));
}
