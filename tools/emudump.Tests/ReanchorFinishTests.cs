// EmuDump's handling of a screen raised in the middle of a step (the emulator stops the step ON
// the screen, CombatEngine.AwaitSelection): the re-anchor finishes a stop nobody answered rather
// than clearing the screen and leaving the step half done, and a format-1 recording's untagged
// choices answer the screens they name.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

public class ReanchorFinishTests
{
    /// <summary>Headbutt in hand, two cards in the discard: its play stops on the pick.</summary>
    static (CombatState State, CountingRandom Rng) StoppedOnHeadbutt()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.Headbutt, false)];
        state.DiscardPile = [new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.Bash, false)];
        state.Energy = 3;
        var rng = new CountingRandom(1);
        CombatEngine.Step(state, 0, rng);
        Assert.NotNull(state.PendingSelection?.Replay);
        return (state, rng);
    }

    [Fact]
    public void ATraceHoldsAStoppedStepOnceAsTheAnswerReplayedIt()
    {
        // The stopped run marks the play, stops, and the answer replays the play and marks it again;
        // the trace keeps the replay alone (it held play, play, played).
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.Headbutt, false)];
        state.DiscardPile = [new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.Bash, false)];
        state.Energy = 3;
        var rng = new CountingRandom(1);
        EmuTrace.Begin(state);
        CombatEngine.Step(state, 0, rng);
        CombatEngine.Step(state, 0, rng);
        var marks = EmuTrace.End().Select(e => e!.AsArray()).Where(e => (string?)e[0] == "M").Select(e => (string?)e[1]).ToList();

        Assert.Equal("start", marks[0]);
        Assert.Equal(1, marks.Count(m => m == "play"));
        Assert.Equal(1, marks.Count(m => m == "played"));
        Assert.DoesNotContain("replay", marks);
    }

    [Fact]
    public void AStopNobodyAnsweredIsFinishedNotLeftHalfDone()
    {
        // Cleared instead, the play would never count and Headbutt would sit in no pile: hidden
        // state the next anchor does not patch (a Gold Axe a card short, an auto-play still owed).
        var (state, rng) = StoppedOnHeadbutt();
        Assert.Equal(0, state.CardPlaysThisTurn);

        Assert.False(Reanchor.Finish(state, rng));

        Assert.Null(state.PendingSelection);
        Assert.Equal(1, state.CardPlaysThisTurn);
        Assert.Contains(state.DiscardPile, c => c.DefId == IC.Headbutt);
        Assert.Equal(IC.StrikeIronclad, state.DrawPile[0].DefId);  // the first candidate
    }

    static JsonObject Game(params JsonObject[] choices) => new() { ["choices"] = new JsonArray(choices) };

    static JsonObject Choice(string[] options, int picked, int? step = null)
    {
        var o = new JsonObject
        {
            ["kind"] = "cards",
            ["options"] = new JsonArray(options.Select(x => (JsonNode)x).ToArray()),
            ["picked"] = new JsonArray(picked),
            ["min"] = 1,
            ["max"] = 1,
        };
        if (step is int s) o["step"] = s;
        return o;
    }

    [Fact]
    public void AFormatOneChoiceAnswersTheScreenWhoseCandidatesItNames()
    {
        // The referee's first recordings logged each screen's options and pick in order, with
        // no step: the Neow screen's deck (no combat screen offers it) is passed over, and
        // Headbutt's [Strike, Bash] takes the game's Bash.
        var (state, rng) = StoppedOnHeadbutt();
        var untagged = Fight.UntaggedChoices.Of(Game(
            Choice(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"], 0),
            Choice(["BASH", "STRIKE_IRONCLAD"], 0)))!;

        var log = new JsonArray();
        var used = untagged.AnswerOpen(() => state, i => CombatEngine.Step(state, i, rng), log);

        Assert.Single(used);
        Assert.Null(state.PendingSelection);
        Assert.Equal(IC.Bash, state.DrawPile[0].DefId);
        // Used once: the same screen again finds nothing left to take.
        var (again, rng2) = StoppedOnHeadbutt();
        Assert.Empty(untagged.AnswerOpen(() => again, i => CombatEngine.Step(again, i, rng2), log));
        Assert.NotNull(again.PendingSelection);
    }

    [Fact]
    public void ARecordingWhoseChoicesCarryTheirStepHasNoUntaggedOnes()
    {
        Assert.Null(Fight.UntaggedChoices.Of(Game(Choice(["BASH", "STRIKE_IRONCLAD"], 0, step: 3))));
        Assert.Null(Fight.UntaggedChoices.Of(Game()));
    }
}
