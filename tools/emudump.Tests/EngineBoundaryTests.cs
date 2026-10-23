// The engine boundary of the oracle tools: only Engines/Old/ names the old engine (Sts2Emulator). Every
// other file in tools/emudump reaches an engine through IEngine (Engine/IEngine.cs), except the consumers
// listed in Engine/not_moved.txt, which have not moved yet. The list only shrinks: a listed file that no
// longer names Sts2Emulator fails the test until it is taken off, and a new file that names it fails until
// it is moved (it cannot be added to the list: it is checked against the list as it stands).
using System.Text.RegularExpressions;
using Xunit;

public class EngineBoundaryTests
{
    const string OldFolder = "Engines/Old/";

    // A string literal (kept: a name inside one is still a name) or a comment (dropped).
    static readonly Regex StringOrComment = new(
        "@\"(?:[^\"]|\"\")*\"|\"(?:[^\"\\\\\\n]|\\\\.)*\"|//[^\\n]*|/\\*.*?\\*/", RegexOptions.Singleline | RegexOptions.Compiled);

    static readonly Regex Name = new(@"\bSts2Emulator\b", RegexOptions.Compiled);

    /// <summary>Whether C# source names the old engine's namespace (a using or a qualified name), comments aside.</summary>
    internal static bool NamesEngine(string source) =>
        Name.IsMatch(StringOrComment.Replace(source, m => m.Value.StartsWith('"') || m.Value.StartsWith("@\"") ? m.Value : " "));

    /// <summary>
    /// What breaks the boundary. <paramref name="files"/>: relative path (forward slashes, from tools/emudump) to source;
    /// <paramref name="notMoved"/>: the allowlist.
    /// </summary>
    internal static List<string> Problems(IReadOnlyDictionary<string, string> files, IReadOnlySet<string> notMoved)
    {
        var problems = new List<string>();
        foreach (var (path, source) in files.OrderBy(f => f.Key, StringComparer.Ordinal))
        {
            if (path.StartsWith(OldFolder, StringComparison.Ordinal)) continue;
            bool names = NamesEngine(source);
            bool listed = notMoved.Contains(path);
            if (names && !listed) problems.Add($"{path} names Sts2Emulator outside {OldFolder}: reach the engine through IEngine");
            if (!names && listed) problems.Add($"{path} is on Engine/not_moved.txt but no longer names Sts2Emulator: take it off the list");
        }

        foreach (var path in notMoved.Order(StringComparer.Ordinal))
        {
            if (!files.ContainsKey(path)) problems.Add($"{path} is on Engine/not_moved.txt but is not a source file of tools/emudump");
            if (path.StartsWith(OldFolder, StringComparison.Ordinal) || path.StartsWith("Engine/", StringComparison.Ordinal) || path == "Fight.cs")
            {
                problems.Add($"{path} must not be on Engine/not_moved.txt");
            }
        }

        return problems;
    }

    static string EmuDumpDir()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string p = Path.Combine(d.FullName, "tools", "emudump", "Engine", "not_moved.txt");
            if (File.Exists(p)) return Path.GetDirectoryName(Path.GetDirectoryName(p))!;
        }

        throw new DirectoryNotFoundException("tools/emudump/Engine/not_moved.txt not found above " + AppContext.BaseDirectory);
    }

    static Dictionary<string, string> Sources(string dir) =>
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
        Assert.Empty(Problems(Sources(dir), NotMoved(dir)));
    }

    [Fact]
    public void FightAndTheInterfaceNameNoEngine()
    {
        string dir = EmuDumpDir();
        var files = Sources(dir);
        Assert.Contains("Fight.cs", files.Keys);
        Assert.Contains("Engine/IEngine.cs", files.Keys);
        foreach (var (path, source) in files.Where(f => f.Key == "Fight.cs" || f.Key.StartsWith("Engine/")))
        {
            Assert.False(NamesEngine(source), $"{path} names Sts2Emulator");
        }
    }

    // The broken cases: the same check, on files made up here, must fail.

    [Fact]
    public void ABrokenCaseIsCaught_AnAdapterNamesTheEngine()
    {
        var files = new Dictionary<string, string>
        {
            ["Fight.cs"] = "using Sts2Emulator.Core.Run;\nstatic class Fight { RunEngine e; }",
            ["Engine/IEngine.cs"] = "namespace EmuDump.Engine; interface IEngine {}",
            ["Engines/Old/OldEngine.cs"] = "using Sts2Emulator.Core.Run;",
            ["Digest.cs"] = "var c = new Sts2Emulator.Core.CombatState();",
        };
        var problems = Problems(files, new HashSet<string> { "Digest.cs" });
        Assert.Single(problems);
        Assert.Contains("Fight.cs names Sts2Emulator", problems[0]);
    }

    [Fact]
    public void ABrokenCaseIsCaught_AListedFileThatNoLongerNamesTheEngine()
    {
        var files = new Dictionary<string, string> { ["Digest.cs"] = "static class Digest { }" };
        var problems = Problems(files, new HashSet<string> { "Digest.cs" });
        Assert.Single(problems);
        Assert.Contains("take it off the list", problems[0]);
    }

    [Fact]
    public void ABrokenCaseIsCaught_TheListNamesAFileThatIsGoneOrMustNotBeListed()
    {
        var files = new Dictionary<string, string> { ["Fight.cs"] = "// IEngine only" };
        Assert.Equal(2, Problems(files, new HashSet<string> { "Gone.cs", "Fight.cs" }).Count(p => p.Contains("not_moved") && (p.Contains("not a source file") || p.Contains("must not"))));
    }

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
