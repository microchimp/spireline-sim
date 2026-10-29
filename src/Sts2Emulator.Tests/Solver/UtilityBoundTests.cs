// The bounds round (docs/rebuild/FIGHT_SOLVER.md section 19, optimisation review change 5): the root kept whole, every
// root move valued as with the bounds off, and exactness never lost by them.
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class UtilityBoundTests
{
    [Fact]
    public void TheRootsMovesAreValuedAsWithTheBoundsOff()
    {
        var corpus = SolverPositions.Corpus();
        int compared = 0;
        foreach (var recipe in SolverPositions.Recipes().Where(r => r.Bound.Length > 0))
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                continue;
            }

            var on = new Expectimax(new SolveOptions { Prune = Pruning.All, UseEstimate = false }).SolveAt(pos.State, pos.Rng, recipe.Horizon);
            var off = new Expectimax(new SolveOptions { Prune = Pruning.NoBounds, UseEstimate = false }).SolveAt(pos.State, pos.Rng, recipe.Horizon);
            Assert.Equal(off.Status == SolveStatus.Refused, on.Status == SolveStatus.Refused);
            Assert.Equal(off.Moves.Select(m => (m.Move, m.Value)), on.Moves.Select(m => (m.Move, m.Value)));
            Assert.Equal(off.Value, on.Value);
            Assert.True(on.Exact || !off.Exact, $"#{recipe.Id}: exact with the bounds off, not with them on");
            compared++;
        }

        Assert.True(compared >= 16, $"compared {compared}");
    }
}
