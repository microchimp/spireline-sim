// Engines/Old/OldAnchor.cs is the old engine's loader of a game snapshot into its combat, and round A1b moves it into the
// engine DLL behind a native export. It must therefore depend on nothing of EmuDump: no file read, no other EmuDump class,
// and only the engine's own namespace besides System. This reads its source (comments and string literals aside) and fails
// on any name it must not use. The other EmuDump classes are not listed by hand: they are every top-level type declared in
// tools/emudump outside OldAnchor.cs (found by scanning, as EngineBoundaryTests does, so a new class is covered), and any
// name qualified with `EmuDump.`. A nested type is not one (a `Key` inside some class is only a word). The adapter side
// (Reanchor.cs names no engine) is EngineBoundaryTests' job.
using System.Text.RegularExpressions;
using Xunit;

public class ReanchorBoundaryTests
{
    const string LoaderPath = "Engines/Old/OldAnchor.cs";

    static readonly string[] AllowedUsings = ["System.Text.Json.Nodes", "System.Text.RegularExpressions", "Sts2Emulator.Core"];

    // The ways to reach a file or the process's surroundings, the table dump of the tools, and any EmuDump-qualified name
    // (EmuDump.Engine.ReanchorMode, EmuDump.Engines.Old.OldIds).
    static readonly Regex Surroundings = new(
        @"\b(?:File|Directory|Path|AppContext|Environment|Process|Program)\b|\bTables\s*\.\s*Dump\b|\bEmuDump\s*\.", RegexOptions.Compiled);

    /// <summary>
    /// The names of the other EmuDump classes: the top-level types declared in <paramref name="files"/> outside the loader,
    /// less the names the loader declares itself (its nested Tables is the loader's own, the top-level Tables is the tools').
    /// </summary>
    internal static IReadOnlySet<string> ToolTypes(IReadOnlyDictionary<string, string> files)
    {
        var tools = files.Where(f => f.Key != LoaderPath).SelectMany(f => EngineBoundaryTests.DeclaredTypes(f.Value)).ToHashSet(StringComparer.Ordinal);
        if (files.TryGetValue(LoaderPath, out var loader)) tools.ExceptWith(EngineBoundaryTests.AllDeclaredTypes(loader));
        return tools;
    }

    /// <summary>What in a loader's source ties it to EmuDump: a using outside the allowed ones, a forbidden name.</summary>
    internal static List<string> Problems(string source, IReadOnlySet<string> toolTypes)
    {
        // The loader's own namespace declaration (`namespace EmuDump.Engines.Old;`) is not a use of another EmuDump name.
        var code = Regex.Replace(EngineBoundaryTests.Code(source, keepStrings: false), @"\bnamespace\s+[\w.]+", "namespace");
        var problems = new List<string>();
        foreach (Match u in Regex.Matches(code, @"^\s*using\s+([\w.]+)\s*;", RegexOptions.Multiline))
        {
            if (!AllowedUsings.Contains(u.Groups[1].Value)) problems.Add("using " + u.Groups[1].Value);
        }

        foreach (Match m in Surroundings.Matches(code)) problems.Add("names " + Regex.Replace(m.Value, @"\s+", ""));
        if (toolTypes.Count > 0)
        {
            var types = new Regex(@"\b(?:" + string.Join("|", toolTypes.Order(StringComparer.Ordinal).Select(Regex.Escape)) + @")\b");
            foreach (Match m in types.Matches(code)) problems.Add("names " + m.Value);
        }

        return problems;
    }

    static Dictionary<string, string> RealFiles() => EngineBoundaryTests.Sources(EngineBoundaryTests.EmuDumpDir());

    static readonly Lazy<IReadOnlySet<string>> RealToolTypes = new(() => ToolTypes(RealFiles()));

    [Fact]
    public void OldAnchorDependsOnTheEngineAndTheJsonItIsGivenOnly()
    {
        var files = RealFiles();
        Assert.Empty(Problems(files[LoaderPath], ToolTypes(files)));
    }

    [Fact]
    public void TheOtherClassesAreFoundByScanningTheToolsNotListedByHand()
    {
        foreach (var type in new[] { "Digest", "Live", "Meta", "MutHost", "RngLog", "RunReplay", "RunSnap", "StateWalk", "Names", "Streams",
                     "Walker", "Driver", "ProfileFit", "Setup", "Fight", "Prof", "Reanchor", "IEngine", "EngineBlock",
                     "OldIds", "OldDump", "OldChoices", "OldReanchor", "OldEngine", "OldCarry", "EngineFactory" })
        {
            Assert.Contains(type, RealToolTypes.Value);
        }

        // Not the loader's own names (its nested Tables is its own; the tools' Tables.Dump is checked by itself), nor a nested
        // name that is only a word in other files.
        Assert.DoesNotContain("OldAnchor", RealToolTypes.Value);
        Assert.DoesNotContain("Tables", RealToolTypes.Value);
        Assert.DoesNotContain("Key", RealToolTypes.Value);
    }

    // The broken cases: the same check, on sources made up here, must fail.

    [Theory]
    [InlineData("using EmuDump.Engine;\nstatic class A {}", "using EmuDump.Engine")]
    [InlineData("static class A { string S() => File.ReadAllText(\"id_map.json\"); }", "names File")]
    [InlineData("static class A { int P(string s) => Setup.PotionId(s) ?? 0; }", "names Setup")]
    [InlineData("static class A { int C(string s) => OldIds.CardIdBySnake(s); }", "names OldIds")]
    [InlineData("static class A { string B() => AppContext.BaseDirectory; }", "names AppContext")]
    [InlineData("static class A { int T(int x) => RngLog.Take(x); }", "names RngLog")]
    [InlineData("static class A { string D(CombatState q) => Digest.Of(q); }", "names Digest")]
    [InlineData("static class A { void W(Walker w, ProfileFit f, MutHost h) { } }", "names Walker")]
    [InlineData("static class A { void W(Walker w, ProfileFit f, MutHost h) { } }", "names ProfileFit")]
    [InlineData("static class A { void W(Walker w, ProfileFit f, MutHost h) { } }", "names MutHost")]
    [InlineData("static class A { object M() => Meta.Load(); Live l; RunSnap s; StateWalk w; }", "names Live")]
    [InlineData("static class A { object M() => EmuDump.Engine.ReanchorMode.Scramble; }", "names EmuDump.")]
    [InlineData("static class A { void E(EmuDump.Engine.IEngine e) { } }", "names EmuDump.")]
    [InlineData("static class A { void E(Engine.IEngine e) { } }", "names IEngine")]
    [InlineData("static class A { string T() => Tables . Dump(1); }", "names Tables.Dump")]
    public void ABrokenCaseIsCaught(string source, string problem) => Assert.Contains(problem, Problems(source, RealToolTypes.Value));

    [Fact]
    public void TheLoadersOwnNamespaceDeclarationIsNotADependencyButAQualifiedUseIs()
    {
        Assert.Empty(Problems("namespace EmuDump.Engines.Old;\nusing Sts2Emulator.Core;\nstatic class A { }", RealToolTypes.Value));
        Assert.Empty(Problems("namespace EmuDump.Engines.Old { static class A { } }", RealToolTypes.Value));
        Assert.Contains("names EmuDump.", Problems("namespace EmuDump.Engines.Old;\nstatic class A { object O() => new EmuDump.Engines.Old.OldIds(); }", RealToolTypes.Value));
    }

    [Fact]
    public void ANameInACommentOrAStringIsNotADependency() =>
        Assert.Empty(Problems("using Sts2Emulator.Core;\n// reads File and Setup\nstatic class A { string S = \"File.ReadAllText RngLog\"; char C = '\"'; }", RealToolTypes.Value));

    [Fact]
    public void TheWordKeyIsNotADependency()
    {
        // A nested type named Key lives in some tools class; the loader and anyone may say the word.
        Assert.Empty(Problems("using Sts2Emulator.Core;\nstatic class A { int K(Dictionary<string, int> d) { foreach (var kv in d) return kv.Key; var k = new List<(int At, string Key)>(); return k.Count; } }",
            RealToolTypes.Value));
        var files = new Dictionary<string, string>
        {
            ["Setup.cs"] = "static class Setup { sealed class Key { } }",
            ["Digest.cs"] = "static class Digest { }",
            [LoaderPath] = "static class OldAnchor { internal sealed class Tables { } static int K(Tables t, string Key) => Key.Length; }",
        };
        var tools = ToolTypes(files);
        Assert.Equal(["Digest", "Setup"], tools.Order(StringComparer.Ordinal));
        Assert.Empty(Problems(files[LoaderPath], tools));
    }

    [Fact]
    public void TheLoadersOwnNestedTablesIsNotTheToolsTables()
    {
        var files = new Dictionary<string, string>
        {
            ["Tables.cs"] = "static class Tables { internal static string Dump(int x) => \"\"; }",
            [LoaderPath] = "static class OldAnchor { internal sealed class Tables { } static void P(Tables t) { } }",
        };
        var tools = ToolTypes(files);
        Assert.Empty(Problems(files[LoaderPath], tools));
        Assert.Contains("names Tables.Dump", Problems("static class OldAnchor { string D() => Tables.Dump(1); }", tools));
    }
}
