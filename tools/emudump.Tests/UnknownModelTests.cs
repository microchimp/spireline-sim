// A choice that names a card the engine's id map lacks ends the fight with the engine's exception, as the replay
// always did (the corpus verdict for such a fight is E, an error, not D, a divergence): Fight.ApplyChoice asks the
// screen source to RequireModel before it matches a pick against the candidates, and the old engine's adapters answer
// with the id map's KeyNotFoundException. Before it was restored the replay logged "not among emulator candidates"
// and went on.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using Sts2Emulator.Core.Run;
using Xunit;

public class UnknownModelTests
{
    // What `EmuDump fight` (main 97b03d1e) wrote as the first line of the fight's error for an id the map lacks.
    const string MainError = "System.Collections.Generic.KeyNotFoundException: The given key 'NOT_A_CARD' was not present in the dictionary.";

    /// <summary>A recorded fight whose start opens Toolbox's screen, answered with a card no map has.</summary>
    static JsonObject ToolboxFight(string pick)
    {
        var probe = new RunEngine();
        probe.Reset("84G7XUWQD6", 10);
        for (int guard = 0; guard < 10 && probe.State.Phase != RunPhase.Map; guard++) probe.Step(0, -1, out _, out _, out _);
        var (col, row) = probe.State.MapOptionCoords[0]!.Value;
        return new JsonObject
        {
            ["seed"] = "84G7XUWQD6",
            ["ascension"] = 10,
            ["neow_pick"] = 0,
            ["room"] = new JsonArray(col, row),
            ["setup"] = new JsonArray(new JsonObject { ["cmd"] = "relic", ["id"] = "TOOLBOX" }),
            ["room_choices_from"] = 0,
            ["choices"] = new JsonArray(new JsonObject
            {
                ["step"] = 0,
                ["kind"] = "cards",
                ["options"] = new JsonArray(pick),
                ["picked"] = new JsonArray(0),
            }),
            ["steps"] = new JsonArray(new JsonObject { ["action"] = null }),
        };
    }

    static string FirstLine(string text) => text.Split('\n')[0].TrimEnd('\r');

    [Fact]
    public void AnUnknownCardInTheStartChoicesEndsTheFightWithTheMapsKeyNotFound()
    {
        var outp = Fight.Replay(ToolboxFight("NOT_A_CARD"));

        Assert.Equal(MainError, FirstLine((string)outp["error"]!));
        Assert.Empty(outp["steps"]!.AsArray());
    }

    [Fact]
    public void TheSameScreenAnsweredWithAKnownCardRunsOn()
    {
        // The screen is open (the unknown id above is read against its candidates, not skipped for want of a screen);
        // a known id that is not among them is the ordinary log, not an exception.
        var outp = Fight.Replay(ToolboxFight("STRIKE_IRONCLAD"));

        Assert.Null(outp["error"]);
        Assert.Contains("not among emulator candidates", (string)outp["start_choices"]![0]!["error"]!);
    }

    sealed class StubScreens(IReadOnlyList<Candidate> candidates, Action<string>? require = null) : IScreenSource
    {
        public int Required;

        public OpenScreen? Screen => new("Test", candidates, null, false, null);

        public void Pick(int candidate) { }

        public void Decline() { }

        public void RequireModel(string modelId)
        {
            Required++;
            require?.Invoke(modelId);
        }
    }

    static JsonObject Choice(string option) => new()
    {
        ["options"] = new JsonArray(option),
        ["picked"] = new JsonArray(0),
    };

    [Fact]
    public void ApplyChoiceAsksTheEngineAboutThePickedModelAndLetsItsExceptionThrough()
    {
        var src = new StubScreens([new Candidate("STRIKE_IRONCLAD", false)], id => throw new KeyNotFoundException("the engine's own: " + id));

        var ex = Assert.Throws<KeyNotFoundException>(() => Fight.ApplyChoice(src, Choice("NOT_A_CARD")));

        Assert.Equal("the engine's own: NOT_A_CARD", ex.Message);
    }

    [Fact]
    public void ApplyChoiceDoesNotAskWhenTheScreenHasNoCandidates()
    {
        // As before: the id map was read while matching candidates, so a screen with none never read it.
        var src = new StubScreens([]);

        var log = Fight.ApplyChoice(src, Choice("NOT_A_CARD"));

        Assert.Equal(0, src.Required);
        Assert.Equal("NOT_A_CARD not among emulator candidates", (string)log["error"]!);
    }

    [Fact]
    public void ApplyChoiceKnownIdsAreAskedAboutOncePerPick()
    {
        var src = new StubScreens([new Candidate("STRIKE_IRONCLAD", false)]);

        var log = Fight.ApplyChoice(src, Choice("STRIKE_IRONCLAD"));

        Assert.Equal(1, src.Required);
        Assert.Null(log["error"]);
    }
}
