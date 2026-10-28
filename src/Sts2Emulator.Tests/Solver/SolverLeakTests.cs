// The expected solve reads no hidden future (docs/rebuild/FIGHT_SOLVER.md section 10, the firewall; piece S4): the
// solver's twin of ResampleLeakTests. Two positions that differ only in what the player cannot see (every stream reseeded,
// the draw pile's unknown middle reshuffled: a determinization) get identical expected results, the root value and every
// root move's value as exact rationals, whether made in the library or through the native Sts2_Clone. The controls, so the
// test could see a leak: the worlds' draw piles are in another order on at least a quarter of the positions, and a solver
// that does read the future (the hindsight search, at a small budget) tells the worlds apart on some of the positions where
// the native forks are compared.
using Sts2Emulator.Core;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class SolverLeakTests(ITestOutputHelper output)
{
    static (CombatState, CountingRandom) Determinized(CombatState s, CountingRandom rng, int seed)
    {
        var copy = s.Clone(seed);
        copy.ReshuffleUnknownDrawOrder(copy.ShuffleRng ?? new CountingRandom(seed));
        return (copy, new CountingRandom(unchecked(seed * 7 + 1)));
    }

    static void SameResult(ExpectedResult a, ExpectedResult b, string tag, List<string> diffs)
    {
        if (a.Status != b.Status || a.Value != b.Value || a.Exact != b.Exact || a.ExactMass != b.ExactMass || a.Reason != b.Reason
            || a.Moves.Count != b.Moves.Count)
        {
            diffs.Add($"{tag}: {a.Status} {a.Value} {a.Reason} vs {b.Status} {b.Value} {b.Reason}");
            return;
        }

        for (int i = 0; i < a.Moves.Count; i++)
        {
            if (a.Moves[i] != b.Moves[i])
            {
                diffs.Add($"{tag}: move {a.Moves[i]} vs {b.Moves[i]}");
            }
        }
    }

    [Fact]
    public void TwoWorldsThatDifferOnlyInHiddenOrderAndStreamsSolveAlike()
    {
        var corpus = SolverPositions.Corpus();
        int compared = 0, apartPositions = 0, natives = 0, futureReaderApart = 0;
        var peek = new HindsightOptions { AllowHindsight = true, SearchNodeBudget = 3_000, SearchStepBudget = 30_000, MaxTurns = 2 };
        var diffs = new List<string>();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        foreach (var recipe in SolverPositions.Recipes().Where(r => r.Id % 2 == 0))
        {
            if (clock.Elapsed > TimeSpan.FromMinutes(8))
            {
                break;  // a wall-time cap under the gate's 15-minute hang guard
            }

            if (SolverPositions.Build(recipe, corpus) is not { } pos)
            {
                continue;
            }

            var opts = NativeSolverTests.HarnessOptions(recipe.Horizon);
            var truth = SolverService.SolveExpected(pos.State, pos.Rng, opts);
            bool apart = false, seen = false;
            var truthPeek = recipe.Id % 8 == 0 ? SolverService.SolveHindsight(pos.State, pos.Rng, peek, trueStreams: true) : null;
            for (int k = 0; k < 2; k++)
            {
                var (w, wr) = Determinized(pos.State, pos.Rng, 1000 + 17 * recipe.Id + k);
                SameResult(truth, SolverService.SolveExpected(w, wr, opts), $"#{recipe.Id} world {k}", diffs);
                apart |= !w.DrawPile.Select(c => c.Uid).SequenceEqual(pos.State.DrawPile.Select(c => c.Uid));
                if (truthPeek is not null && truthPeek.Outcome != HindsightOutcome.Refused)
                {
                    // The positive control: the hindsight search reads the draws and rolls, so it answers the worlds differently.
                    var p = SolverService.SolveHindsight(w, wr, peek, trueStreams: true);
                    seen |= p.Outcome != truthPeek.Outcome || p.BestFound != truthPeek.BestFound || !p.Line.SequenceEqual(truthPeek.Line);
                }
            }

            apartPositions += apart ? 1 : 0;
            futureReaderApart += seen ? 1 : 0;

            if (recipe.Id % 8 == 0)
            {
                // Through the native determinization (Sts2_Clone with a resample seed), which also redraws the intents' hidden particles.
                int handle = NativeSolverCalls.Adopt(pos.State, pos.Rng);
                int fork = NativeSolverCalls.Clone(handle, resample: true, 4242 + recipe.Id);
                int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(opts with { Prune = opts.Prune & ~Pruning.SubtreeReuse }));
                var a = NativeSolverCalls.Expected(solver, handle, null);
                var b = NativeSolverCalls.Expected(solver, fork, null);
                // double.Equals: a refused solve's NaN value equals the other's NaN.
                if (a.Status != b.Status || !a.Result.Value.Equals(b.Result.Value) || a.Result.Exact != b.Result.Exact || !a.Q.AsSpan().SequenceEqual(b.Q))
                {
                    diffs.Add($"#{recipe.Id} native: {a.Status} {a.Result.Value} vs {b.Status} {b.Result.Value}");
                }

                natives++;
                NativeSolverCalls.Destroy(solver);
                NativeSolverCalls.DestroyCombat(fork);
                NativeSolverCalls.DestroyCombat(handle);
            }

            compared++;
        }

        output.WriteLine($"leak: {compared} positions x 2 worlds, {natives} native forks, {apartPositions} positions with another draw order in a world, "
            + $"the hindsight search telling the worlds apart on {futureReaderApart} of the {natives}, {diffs.Count} differences");
        Assert.True(compared >= 100, $"only {compared} positions");
        Assert.True(apartPositions * 4 >= compared, $"another draw order on only {apartPositions} of {compared} positions: the worlds were hardly different");
        Assert.True(futureReaderApart >= 1, "the hindsight search told no worlds apart: the test could not see a solver that reads the future");
        Assert.Empty(diffs);
    }
}
