using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.StoppedStepDocumentTests;

namespace Sts2Emulator.Tests;

/// <summary>
/// A replay of a stopped step that does not go the way the step went is REFUSED, never guessed at:
/// at load (<c>CombatSnapshot.Read</c>), at export (<c>CombatSnapshot.Write</c>) and at the answer
/// (<c>CombatEngine.Step</c>, which puts the combat back on its screen). Round 5 put the answering
/// copy back at whatever screen the replay reached, dropped answers left over without a word, and
/// answered a pick by a card no candidate offered with candidate 0 (review, ZReviewR5Doc).
/// </summary>
public class StoppedStepRefusalTests
{
    /// <summary>The agent's document of <paramref name="screen" />'s stop, as a tree to change.</summary>
    private static JsonObject Document(string screen) =>
        JsonNode.Parse(CombatSnapshot.Write(Stopped(screen).Fight.State))!.AsObject();

    private static SnapshotException Refused(JsonObject document) =>
        Assert.Throws<SnapshotException>(() => CombatSnapshot.Read(Encoding.UTF8.GetBytes(document.ToJsonString()), seed: 5, out _));

    /// <summary><paramref name="stopped" /> with some of its record replaced: a record the step did not leave.</summary>
    private static HookChoiceReplay Altered(
        HookChoiceReplay stopped,
        IReadOnlyList<StepAnswer>? answers = null,
        IReadOnlyList<CardInstance>? taken = null,
        IReadOnlyList<StepScreen>? screens = null
    ) =>
        new()
        {
            Before = stopped.Before,
            RngBefore = stopped.RngBefore,
            Action = stopped.Action,
            TargetEnemyIndex = stopped.TargetEnemyIndex,
            Answers = answers ?? stopped.Answers,
            Hidden = answers is null ? stopped.Hidden : [.. stopped.Hidden, .. Enumerable.Repeat(ScreenHiddenState.None, answers.Count - stopped.Answers.Count)],
            Taken = taken ?? stopped.Taken,
            Screens = screens ?? stopped.Screens,
        };

    private static void Put(CombatState state, HookChoiceReplay replay) =>
        state.PendingSelection = state.PendingSelection!.Clone(replay);

    /// <summary>
    /// The answer to <paramref name="state" />'s screen is refused, and the combat is on its screen
    /// as it was: the same position, the same stopped step, the step stream where it stood.
    /// </summary>
    private static StoppedStepMismatch AnswerRefused(CombatState state, CountingRandom rng)
    {
        string before = Encoding.UTF8.GetString(CombatSnapshot.WriteSave(state));
        var replay = state.PendingSelection!.Replay;
        var streamBefore = rng.Copy();

        var refused = Assert.Throws<StoppedStepMismatch>(() => CombatEngine.Step(state, 0, rng));

        Assert.Equal(before, Encoding.UTF8.GetString(CombatSnapshot.WriteSave(state)));
        Assert.Same(replay, state.PendingSelection!.Replay);
        Assert.Equal((streamBefore.Seed, streamBefore.CallCount, streamBefore.Lanes), (rng.Seed, rng.CallCount, rng.Lanes));
        return refused;
    }

    /// <summary>Havoc's Burning Pact, with the card Havoc took off the pile named as one the pile never held.</summary>
    [Fact]
    public void ADocumentWhoseStepTookACardThePileDoesNotHoldIsRefusedAtLoad()
    {
        var document = Document("HavocBurningPact");
        var step = document["stopped_step"]!;
        Assert.Equal(IC.BurningPact, (int)step["taken"]![0]!["def_id"]!);
        step["taken"]![0]!["def_id"] = IC.Bash;

        var refused = Refused(document);

        Assert.Equal("stopped_step", refused.Path);
        Assert.Contains("take 1 off the draw pile", refused.Message);
    }

    /// <summary>A document whose last screen is not the one it shows, or that reaches it after other takes.</summary>
    [Theory]
    [InlineData("count")]
    [InlineData("source")]
    [InlineData("taken")]
    public void ADocumentWhoseLastScreenIsNotTheOneItShowsIsRefusedAtLoad(string field)
    {
        var document = Document("MayhemBurningPact");
        var last = document["stopped_step"]!["screens"]![0]!;
        last[field] = (int)last[field]! + 1;

        var refused = Refused(document);

        Assert.StartsWith("stopped_step.screens", refused.Path);
    }

    /// <summary>
    /// Entropy's screen after Hellraiser's Seeker Strike, with the Strike's screen named as another
    /// card's: the replay reaches the Seeker Strike's, which is not the recorded one.
    /// </summary>
    [Fact]
    public void AReplayThatReachesAnotherScreenOnItsWayIsRefusedAtLoad()
    {
        var document = Document("HellraiserEntropyStrikeDrawn");
        document["stopped_step"]!["screens"]![0]!["source"] = IC.Bash;

        var refused = Refused(document);

        Assert.Equal("stopped_step", refused.Path);
        Assert.Contains("screen 1 of the replay", refused.Message);
    }

    /// <summary>
    /// The Seeker Strike's pick, answered before the document was written, named as a card the pile
    /// holds but the Strike did not offer: refused, not answered with candidate 0.
    /// </summary>
    [Fact]
    public void AnEarlierPickByACardNoCandidateOffersIsRefusedAtLoad()
    {
        var (fight, _, _) = Stopped("HellraiserEntropyStrikeDrawn");
        var replay = fight.Pending!.Replay!;
        var offered = replay.Screens[0].Cards;
        var other = replay.Before.DrawPile.First(card => card.DefId != CL.SeekerStrike && !offered.Contains(card));
        var document = JsonNode.Parse(CombatSnapshot.Write(fight.State))!.AsObject();
        document["stopped_step"]!["answers"]![0]!["card"]!["def_id"] = other.DefId;

        var refused = Refused(document);

        Assert.Contains("no candidate", refused.Message);
    }

    /// <summary>Answers more than the step's screens asked for: the replay ends with one left over.</summary>
    [Fact]
    public void AnswersLeftOverAreRefusedAndTheScreenStays()
    {
        var (fight, rng, _) = Stopped("Armaments");
        var stopped = fight.Pending!.Replay!;
        Put(fight.State, Altered(stopped, answers: [new StepAnswer(0, null)], screens: [stopped.Screens[0], stopped.Screens[0]]));

        var refused = AnswerRefused(fight.State, rng);

        Assert.Contains("left over", refused.Message);
    }

    /// <summary>
    /// A record whose screen is not the one the replay reaches, in memory and from a document: the
    /// answer is refused, never resolved on another screen.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnAnswerWhoseReplayReachesAnotherScreenIsRefusedAndTheScreenStays(bool fromDocument)
    {
        var (fight, rng, _) = Stopped("MayhemBurningPact");
        var state = fromDocument ? CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _) : fight.State;
        var stopped = state.PendingSelection!.Replay!;
        var screen = stopped.Screens[^1];
        Put(state, Altered(stopped, screens: [screen with { Source = IC.Bash }]));

        var refused = AnswerRefused(state, fromDocument ? new CountingRandom(5) : rng);

        Assert.Contains("screen 1 of the replay", refused.Message);
    }

    /// <summary>A record whose takes are not the step's: refused at the take, in memory and from a document.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnAnswerWhoseReplayTakesAnotherCardIsRefusedAndTheScreenStays(bool fromDocument)
    {
        var (fight, rng, _) = Stopped("HellraiserSeekerStrike");
        var state = fromDocument ? CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _) : fight.State;
        var stopped = state.PendingSelection!.Replay!;
        Put(state, Altered(stopped, taken: [new CardInstance(IC.Havoc, false)]));

        var refused = AnswerRefused(state, fromDocument ? new CountingRandom(5) : rng);

        Assert.Contains("take 1 off the draw pile", refused.Message);
    }

    /// <summary>
    /// A stop whose replay would not reach its screen is not written as an agent's document: the
    /// export refuses (native Sts2_GetStateJson returns -3). The dump's writer and the save document,
    /// which no agent loads, still write it.
    /// </summary>
    [Fact]
    public void AStopThatDoesNotReplayToItsScreenIsNotExported()
    {
        var (fight, _, _) = Stopped("HavocBurningPact");
        var stopped = fight.Pending!.Replay!;
        Put(fight.State, Altered(stopped, taken: [new CardInstance(IC.Bash, false)]));

        var refused = Assert.Throws<SnapshotException>(() => CombatSnapshot.Write(fight.State));

        Assert.Equal("stopped_step", refused.Path);
        Assert.Contains("cannot be written", refused.Message);
        Assert.Contains("stopped_step", Encoding.UTF8.GetString(CombatSnapshot.WriteUnchecked(fight.State)));
        Assert.Contains("before_streams", Encoding.UTF8.GetString(CombatSnapshot.WriteSave(fight.State)));
    }

    /// <summary>
    /// The record comes back from the agent's document as the step left it: the action, the target,
    /// the answers (a draw-pile pick by card), the cards taken and every screen.
    /// </summary>
    [Theory]
    [InlineData("HavocBurningPact")]
    [InlineData("MayhemEntropyBurningPact")]
    [InlineData("HellraiserEntropyStrikeDrawn")]
    [InlineData("WellLaidPlans")]
    public void TheRecordComesBackFromTheDocument(string screen)
    {
        var (fight, _, _) = Stopped(screen);
        var stopped = fight.Pending!.Replay!;

        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _).PendingSelection!.Replay!;

        Assert.Equal((stopped.Action, stopped.TargetEnemyIndex), (loaded.Action, loaded.TargetEnemyIndex));
        Assert.Equal(stopped.Answers.Select(answer => answer.Pick is { } card ? $"card {card}" : $"index {answer.Index}"),
            loaded.Answers.Select(answer => answer.Pick is { } card ? $"card {card}" : $"index {answer.Index}"));
        Assert.Equal(stopped.Taken, loaded.Taken);
        Assert.Equal(stopped.Screens.Select(Text), loaded.Screens.Select(Text));
    }

    private static string Text(StepScreen screen) =>
        $"{screen.Kind} {screen.Source} {screen.Count} {screen.Taken} [{string.Join(",", screen.Cards)}] [{string.Join(",", screen.Generated)}]";

    /// <summary>
    /// A search fork of a stop on Seeker Strike's screen offers the cards the Strike offered: the
    /// resample moves them in the pile, and the screen's candidates follow them. They used to stay
    /// as indices, so a fork offered whatever its shuffle put there.
    /// </summary>
    [Fact]
    public void AForkOffersTheCardsADrawPileScreenOffered()
    {
        var (fight, _, _) = Stopped("HellraiserSeekerStrike");
        var offered = Offered(fight.State);
        Assert.Equal(3, offered.Count);

        foreach (int seed in Enumerable.Range(1, 30))
        {
            var fork = fight.State.Clone(seed);
            fork.ReshuffleUnknownDrawOrder(new CountingRandom(seed));
            Assert.Equal(offered, Offered(fork));
            Assert.True(fork.PendingSelection!.Candidates.SequenceEqual(fork.PendingSelection.Candidates.Order()));
        }
    }

    private static List<string> Offered(CombatState state) =>
        [.. state.PendingSelection!.Candidates.Select(index => state.DrawPile[index].ToString()).Order()];
}
