using Spireline.Referee;
using Xunit;

namespace Spireline.Referee.Tests;

// Env.DecideMany's pick loop (PickLoop.Run) with a fake agent. The driver itself needs the game; the loop does not.
public class PickLoopTests
{
    /// <summary>An agent that answers each ask with the next scripted index into what is left.</summary>
    private static Func<IReadOnlyList<int>, bool, int, int> Script(params int[] answers) => (left, none, k) => answers[k];

    [Fact]
    public void TwoPicksInDescendingGridOrderComeBackDescending()
    {
        // A Room Full of Cheese grid: eight cards, two picks. The agent takes card 5, then (of the seven left) index 2 = card 2.
        var picks = PickLoop.Run(offerCount: 8, cap: 2, full: false, minSelect: 2, allowNone: false, Script(5, 2));
        Assert.Equal(new[] { 5, 2 }, picks);
    }

    [Fact]
    public void AnswersIndexTheCardsLeftNotTheOffer()
    {
        // Card 6 first; then index 0 of [0,1,2,3,4,5,7] is card 0.
        var picks = PickLoop.Run(8, 2, false, 2, false, Script(6, 0));
        Assert.Equal(new[] { 6, 0 }, picks);
    }

    [Fact]
    public void AscendingPicksStayAscending()
    {
        var picks = PickLoop.Run(8, 2, false, 2, false, Script(2, 4));  // card 2, then index 4 of [0,1,3,4,5,6,7] = card 5
        Assert.Equal(new[] { 2, 5 }, picks);
    }

    [Fact]
    public void FullSelectKeepsPickOrder()
    {
        var picks = PickLoop.Run(5, 3, true, 1, false, Script(3, 0, 1));  // card 3; card 0; index 1 of [1,2,4] = card 2
        Assert.Equal(new[] { 3, 0, 2 }, picks);
    }

    [Fact]
    public void NoneOnTheFirstPickOfAScreenThatAllowsZero()
    {
        var seen = new List<(int left, bool none, int k)>();
        var picks = PickLoop.Run(4, 2, false, 0, true, (left, none, k) => { seen.Add((left.Count, none, k)); return left.Count; });
        Assert.Empty(picks);
        Assert.Equal(new[] { (4, true, 0) }, seen);
    }

    [Fact]
    public void DoneOffersFromTheMinOnAndStopsTheScreen()
    {
        var seen = new List<(int left, bool none, int k)>();
        var picks = PickLoop.Run(5, 4, true, 2, false, (left, none, k) =>
        {
            seen.Add((left.Count, none, k));
            return k < 2 ? 4 - k : left.Count;  // two picks, then "done"
        });
        Assert.Equal(new[] { 4, 3 }, picks);  // card 4; then index 3 of [0,1,2,3] = card 3
        Assert.Equal(new[] { (5, false, 0), (4, false, 1), (3, true, 2) }, seen);
    }
}
