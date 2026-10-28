// S4's sanity gate (docs/rebuild/FIGHT_SOLVER.md section 13): on every emulator record tried, the hindsight bound's best
// utility is at least the outcome of the expected-best policy played on the same draws and rolls, and the mean of the bound
// over the records is at least the mean expected value. The records are the equal-value harness's positions
// (tests/fixtures/solver/positions.jsonl), each with its own seeded streams and draw order: the policy plays, at every
// decision, the expected solve's first best move, stepped with those streams; the bound searches every line on the same
// streams. A position whose expected solve is not exact within a small budget, or whose bound is cut, is skipped and
// counted; the records are taken in file order until a wall-time cap (the Release gate's hang guard is 15 minutes).
using Sts2Emulator.Core;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class HindsightBoundTests(ITestOutputHelper output)
{
    [Fact]
    public void TheBoundIsNeverBelowTheExpectedBestPolicyOnTheSameDraws()
    {
        var corpus = SolverPositions.Corpus();
        var opts = new SolveOptions { NodeBudget = 20_000, StepBudget = 200_000, MaxHorizon = 3 };
        var clock = System.Diagnostics.Stopwatch.StartNew();
        int tried = 0, compared = 0, notExact = 0, cut = 0, policyNull = 0, yes = 0, above = 0;
        Rational sumBound = Rational.Zero, sumExpected = Rational.Zero, sumOutcome = Rational.Zero;
        var below = new List<string>();
        foreach (var recipe in SolverPositions.Recipes())
        {
            if (clock.Elapsed > TimeSpan.FromMinutes(8))
            {
                break;
            }

            if (SolverPositions.Build(recipe, corpus) is not { } pos)
            {
                continue;
            }

            tried++;
            var expected = SolverService.SolveExpected(pos.State, pos.Rng, opts);
            if (expected.Status != SolveStatus.Exact)
            {
                notExact++;
                continue;
            }

            var bound = SolverService.SolveHindsight(pos.State, pos.Rng, new HindsightOptions { AllowHindsight = true, SearchNodeBudget = 60_000 }, trueStreams: true);
            if (!bound.Complete)
            {
                cut++;
                continue;
            }

            if (SolverService.PlayExpectedPolicy(pos.State, pos.Rng, opts) is not { } played)
            {
                policyNull++;
                continue;
            }

            compared++;
            yes += bound.WinLineExists == WinLine.Yes ? 1 : 0;
            above += bound.BestUtility > played.Outcome ? 1 : 0;
            sumBound += bound.BestUtility;
            sumExpected += expected.Value;
            sumOutcome += played.Outcome;
            if (bound.BestUtility < played.Outcome)
            {
                below.Add($"#{recipe.Id}: bound {bound.BestUtility} ({bound.BestUtilityDouble:0.0000}) < policy outcome {played.Outcome} ({played.Outcome.ToDouble():0.0000})");
            }
        }

        double n = Math.Max(1, compared);
        output.WriteLine($"hindsight vs expected: {tried} records tried, {compared} compared ({notExact} not exact, {cut} bound cut, {policyNull} policy refused); "
            + $"bound below the policy on {below.Count}, above on {above}; win line on {yes}; "
            + $"mean bound {sumBound.ToDouble() / n:0.0000}, mean expected {sumExpected.ToDouble() / n:0.0000}, mean policy outcome {sumOutcome.ToDouble() / n:0.0000}");
        Assert.True(compared >= 40, $"only {compared} records compared");
        Assert.Empty(below);
        Assert.True(sumBound >= sumExpected, $"mean bound {sumBound.ToDouble() / n} below mean expected {sumExpected.ToDouble() / n}");
    }
}
