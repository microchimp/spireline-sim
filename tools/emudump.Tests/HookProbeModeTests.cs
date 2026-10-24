// EmuDump hookprobe's bad calls (HookProbe.cs): too few arguments, a scenarios file that is missing, unreadable or not a JSON
// object, and an engine that is not instrumented each write the reason to stderr and return 2, and write no output file.
using System.Text.Json.Nodes;
using Xunit;

public class HookProbeModeTests
{
    static (int Code, string Err) Run(params string[] args)
    {
        var err = new StringWriter();
        var before = Console.Error;
        Console.SetError(err);
        try
        {
            return (HookProbeMode.Run(args), err.ToString());
        }
        finally
        {
            Console.SetError(before);
        }
    }

    static string Fresh(string name) => Path.Combine(Directory.CreateTempSubdirectory("hookprobe").FullName, name);

    [Fact]
    public void TooFewArgumentsWriteTheUsageAndReturn2()
    {
        var (code, err) = Run("scenarios.json");
        Assert.Equal(2, code);
        Assert.Contains("usage: EmuDump hookprobe", err);
    }

    [Fact]
    public void AMissingScenariosFileIsNamedOnStderrAndReturns2()
    {
        string missing = Fresh("missing.json"), output = Fresh("out.json");
        var (code, err) = Run(missing, output);
        Assert.Equal(2, code);
        Assert.StartsWith("hookprobe: cannot read the scenarios file " + missing, err);
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void AnUnreadableScenariosFileIsNamedOnStderrAndReturns2()
    {
        // A directory where the file should be: File.ReadAllText refuses it on every OS.
        string dir = Directory.CreateTempSubdirectory("hookprobe-dir").FullName, output = Fresh("out.json");
        var (code, err) = Run(dir, output);
        Assert.Equal(2, code);
        Assert.StartsWith("hookprobe: cannot read the scenarios file " + dir, err);
        Assert.False(File.Exists(output));
    }

    [Theory]
    [InlineData("{ not json")]
    [InlineData("[]")]
    [InlineData("null")]
    public void AScenariosFileThatIsNotAJsonObjectReturns2(string text)
    {
        string bad = Fresh("bad.json"), output = Fresh("out.json");
        File.WriteAllText(bad, text);
        var (code, err) = Run(bad, output);
        Assert.Equal(2, code);
        Assert.StartsWith("hookprobe: cannot read the scenarios file " + bad, err);
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void AnUninstrumentedEngineSaysSoOnStderrAndReturns2()
    {
        // The test build has no Sts2Emulator.HookProbe: only the scratch copy hook_order.py instruments does (ProbeSurfaceTests).
        string spec = Fresh("scenarios.json"), output = Fresh("out.json");
        File.WriteAllText(spec, new JsonObject { ["scenarios"] = new JsonArray() }.ToJsonString());
        var (code, err) = Run(spec, output);
        Assert.Equal(2, code);
        Assert.StartsWith("hookprobe: ", err);
        Assert.Contains("not instrumented", err);
        Assert.False(File.Exists(output));
    }
}
