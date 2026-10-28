// The engine round (optimisation review change 4, FIGHT_SOLVER.md section 18): the lazy pre-step record and the ordered
// piles' cache change no value; the equal-value harness holds the values, these hold the two mechanisms themselves.
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class SolverEngineRoundTests
{
    static int Card(string name) => Cards.FindId(name)!.Value;

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void ARunThatReachesAScreenWithoutItsRecordIsRunAgainWithIt(string engine)
    {
        var (s, rng) = SolverPositions.Fight(engine, 46,
            [Card("Armaments"), Card("StrikeIronclad"), Card("StrikeIronclad"), Card("DefendIronclad"), Card("DefendIronclad")], [], []);
        int play = s.Hand.FindIndex(c => c.DefId == Card("Armaments"));
        var move = Actions.Legal(s, Pruning.None).First(m => m.Action == play);
        var counters = new SolveCounters();
        var outcomes = Successors.Expand(s, rng, move, byCard: false, widthCap: 32, counters);
        Assert.NotEmpty(outcomes);
        // Every outcome stands on the screen with a replayable record: the lazy run was dropped and made again with it.
        Assert.All(outcomes, o => Assert.NotNull(o.State.PendingSelection?.Replay?.Before));
        Assert.Equal(outcomes.Count + 1, counters.Steps);
        Assert.False(CombatEngine.LazyRecord);
        // Outside a solver run a step keeps its record as before.
        var copy = s.Clone();
        Successors.Step(copy, rng.Copy(), move);
        Assert.NotNull(copy.PendingSelection?.Replay?.Before);
    }

    [Fact]
    public void TheOrderedPilesCacheEqualsTheComputationOnTheHarnessPositions()
    {
        var corpus = SolverPositions.Corpus();
        int n = 0;
        foreach (var r in SolverPositions.Recipes())
        {
            if (SolverPositions.Build(r, corpus) is not { } pos)
            {
                continue;
            }

            foreach (var mode in new[] { KeyMode.Expected, KeyMode.Hindsight })
            {
                var cached = StateKey.Readers.OrderedPiles(pos.State, mode);
                Assert.True(StateKey.Readers.Compute(pos.State, mode).SetEquals(cached), $"#{r.Id} {mode}");
                Assert.Same(cached, StateKey.Readers.OrderedPiles(pos.State, mode));
            }

            n++;
        }

        Assert.True(n > 300, $"{n} positions built");
    }
}
