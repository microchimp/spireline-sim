using System.Text.Json.Nodes;
using Spireline.Referee;
using Xunit;

namespace Spireline.Referee.Tests;

// What the agent's line asks of an open decision (Reply.Of): an answer, quit, the hidden state, or a bad answer.
public class ReplyTests
{
    static JsonObject Line(string json) => JsonNode.Parse(json)!.AsObject();

    [Fact]
    public void AnAnswerInRangeIsTheAnswer()
    {
        Assert.Equal((Reply.What.Answer, 2), Reply.Of(Line("{\"a\":2}"), 3, anchorState: true));
        Assert.Equal((Reply.What.Answer, 0), Reply.Of(Line("{\"a\":0}"), 2, anchorState: false));
    }

    [Fact]
    public void AnAnswerOutOfRangeOrNotANumberIsBad()
    {
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{\"a\":3}"), 3, true).what);
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{\"a\":-1}"), 3, true).what);
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{\"a\":\"x\"}"), 3, true).what);
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{}"), 3, true).what);
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{\"cmd\":7}"), 3, true).what);  // a command that is not a name
    }

    [Fact]
    public void QuitEndsTheSessionWhateverElseTheLineHolds()
    {
        Assert.Equal(Reply.What.Quit, Reply.Of(Line("{\"cmd\":\"quit\"}"), 3, false).what);
        Assert.Equal(Reply.What.Quit, Reply.Of(Line("{\"cmd\":\"quit\",\"a\":1}"), 3, true).what);
    }

    [Fact]
    public void TheHiddenStateIsAnsweredOnlyInARunThatAskedForIt()
    {
        // Lane 22, fix d: the lockstep asks at the decision where its copy differs. A run started without spec "anchor_state" (no
        // lockstep emulator side) answers the command as a bad answer, as every driver before it did.
        Assert.Equal(Reply.What.AnchorState, Reply.Of(Line("{\"cmd\":\"anchor_state\"}"), 3, anchorState: true).what);
        Assert.Equal(Reply.What.Bad, Reply.Of(Line("{\"cmd\":\"anchor_state\"}"), 3, anchorState: false).what);
    }
}
