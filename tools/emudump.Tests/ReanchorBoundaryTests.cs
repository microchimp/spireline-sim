// Engines/Old/OldAnchor.cs is the old engine's loader of a game snapshot into its combat, and round A1b moves it into the
// engine DLL behind a native export. It must therefore depend on nothing of EmuDump: no file read, no other EmuDump class,
// and only the engine's own namespace besides System. This reads its source (comments and string literals aside) and fails
// on any name it must not use. The adapter side (Reanchor.cs names no engine) is EngineBoundaryTests' job.
using System.Text.RegularExpressions;
using Xunit;

public class ReanchorBoundaryTests
{
    static readonly Regex StringOrComment = new(
        "@\"(?:[^\"]|\"\")*\"|\"(?:[^\"\\\\\\n]|\\\\.)*\"|//[^\\n]*|/\\*.*?\\*/", RegexOptions.Singleline | RegexOptions.Compiled);

    static readonly string[] AllowedUsings = ["System.Text.Json.Nodes", "System.Text.RegularExpressions", "Sts2Emulator.Core"];

    // The other EmuDump classes, and the ways to reach a file or the process's surroundings.
    static readonly Regex Forbidden = new(
        @"\b(File|Directory|Path|AppContext|Environment|Process|Setup|Fight|Prof|StdOut|EmuTrace|Reanchor|Program|Tables\.Dump|"
        + @"OldIds|OldDump|OldChoices|OldReanchor|OldEngine|OldCarry|EngineFactory|IEngine|EngineBlock)\b", RegexOptions.Compiled);

    /// <summary>What in a loader's source ties it to EmuDump: a using outside the allowed ones, a forbidden name.</summary>
    internal static List<string> Problems(string source)
    {
        var code = StringOrComment.Replace(source, m => m.Value.StartsWith('"') || m.Value.StartsWith("@\"") ? "\"\"" : " ");
        var problems = new List<string>();
        foreach (Match u in Regex.Matches(code, @"^\s*using\s+([\w.]+)\s*;", RegexOptions.Multiline))
        {
            if (!AllowedUsings.Contains(u.Groups[1].Value)) problems.Add("using " + u.Groups[1].Value);
        }

        foreach (Match m in Forbidden.Matches(code)) problems.Add("names " + m.Value);
        return problems;
    }

    static string OldAnchorSource()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string p = Path.Combine(d.FullName, "tools", "emudump", "Engines", "Old", "OldAnchor.cs");
            if (File.Exists(p)) return File.ReadAllText(p);
        }

        throw new FileNotFoundException("tools/emudump/Engines/Old/OldAnchor.cs not found above " + AppContext.BaseDirectory);
    }

    [Fact]
    public void OldAnchorDependsOnTheEngineAndTheJsonItIsGivenOnly() => Assert.Empty(Problems(OldAnchorSource()));

    // The broken cases: the same check, on sources made up here, must fail.

    [Theory]
    [InlineData("using EmuDump.Engine;\nstatic class A {}", "using EmuDump.Engine")]
    [InlineData("static class A { string S() => File.ReadAllText(\"id_map.json\"); }", "names File")]
    [InlineData("static class A { int P(string s) => Setup.PotionId(s) ?? 0; }", "names Setup")]
    [InlineData("static class A { int C(string s) => OldIds.CardIdBySnake(s); }", "names OldIds")]
    [InlineData("static class A { string B() => AppContext.BaseDirectory; }", "names AppContext")]
    public void ABrokenCaseIsCaught(string source, string problem) => Assert.Contains(problem, Problems(source));

    [Fact]
    public void ANameInACommentOrAStringIsNotADependency() =>
        Assert.Empty(Problems("using Sts2Emulator.Core;\n// reads File and Setup\nstatic class A { string S = \"File.ReadAllText\"; }"));
}
