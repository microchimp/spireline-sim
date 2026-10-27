// The equal-value harness (docs/rebuild/FIGHT_SOLVER.md section 6, piece S3): on the seeded small positions of
// tests/fixtures/solver/positions.jsonl (corpus fixtures' setups and self-play's default fight, old and core2, cut to
// low enemy HP and small piles, with order readers, screens, draws and rolls mixed in), the pruned solver under every
// pruning set must give the reference solver's root value and every root move's value, as exact rationals: equal to
// the last bit, or the test fails. A position the reference cannot finish within its cap is skipped and counted; a
// refusal must be the same refusal on both sides.
using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class EqualValueHarnessTests(ITestOutputHelper output)
{
    /// <summary>Every pruning, and every pruning but one (so a failing rule names itself).</summary>
    static readonly (string Name, Pruning Prune)[] Configs =
    [
        ("all", Pruning.All),
        ("no-transposition", Pruning.All & ~Pruning.Transposition),
        ("no-multiset-hand", Pruning.All & ~Pruning.MultisetHand),
        ("no-interchangeable", Pruning.All & ~Pruning.InterchangeableEnemies),
        ("no-reuse", Pruning.All & ~Pruning.SubtreeReuse),
        ("no-draw-by-value", Pruning.All & ~Pruning.DrawByValue),
    ];

    const long ReferenceCap = 25_000;

    [Fact]
    public void EveryPruningEqualsTheReferenceToTheLastBit()
    {
        var corpus = SolverPositions.Corpus();
        int built = 0, compared = 0, skipped = 0, refusedAlike = 0, screens = 0, screenTrees = 0, horizon1 = 0, exact = 0;
        long refNodes = 0, prunedNodes = 0, prunedSteps = 0;
        double prunedSeconds = 0;
        var diffs = new List<string>();
        var refusals = new Dictionary<string, int>();
        foreach (var recipe in SolverPositions.Recipes())
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                continue;
            }

            built++;
            var before = CombatSnapshot.Write(pos.State);
            var reference = Reference.Solve(pos.State, pos.Rng, recipe.Horizon, ReferenceCap);
            if (reference.Refused is { } why && why.StartsWith("budget", StringComparison.Ordinal))
            {
                skipped++;
                continue;
            }

            foreach (var (name, prune) in Configs)
            {
                var solver = new Expectimax(new SolveOptions { Prune = prune, NodeBudget = 10_000_000, StepBudget = 100_000_000, UseEstimate = false });
                var got = solver.SolveAt(pos.State, pos.Rng, recipe.Horizon);
                if (name == "all")
                {
                    screenTrees += reference.Refused is null && got.Counters.Screens > 0 ? 1 : 0;
                    prunedNodes += got.Counters.Nodes;
                    prunedSteps += got.Counters.Steps;
                    prunedSeconds += got.Seconds;
                }

                if (reference.Refused is not null || got.Status == SolveStatus.Refused)
                {
                    if (Tag(reference.Refused) != Tag(got.Reason))
                    {
                        diffs.Add($"#{recipe.Id} {name}: refusal reference '{reference.Refused}' vs solver '{got.Reason}'");
                    }

                    continue;
                }

                if (got.Value != reference.Value || got.Exact != reference.Exact)
                {
                    diffs.Add($"#{recipe.Id} {name}: value {got.Value} ({got.ValueDouble:R}, exact {got.Exact}) vs reference {reference.Value} ({reference.Value.ToDouble():R}, exact {reference.Exact})");
                }

                foreach (var mv in got.Moves)
                {
                    var r = reference.Moves.FirstOrDefault(x => x.Move == mv.Move);
                    if (r is null || r.Value != mv.Value)
                    {
                        diffs.Add($"#{recipe.Id} {name}: move {mv.Move} {mv.Value} vs reference {r?.Value.ToString() ?? "missing"}");
                    }
                }

                foreach (var r in reference.Moves.Where(x => got.Moves.All(m => m.Move != x.Move)))
                {
                    // A move the pruned solver merged into an equal one: some kept move has its value.
                    if (!got.Moves.Any(m => m.Value == r.Value))
                    {
                        diffs.Add($"#{recipe.Id} {name}: merged move {r.Move} {r.Value} has no equal kept move");
                    }
                }

                if (!got.Best.ToHashSet().SetEquals(BestOf(reference.Moves).Where(m => got.Moves.Any(g => g.Move == m))))
                {
                    diffs.Add($"#{recipe.Id} {name}: best moves differ");
                }
            }

            if (reference.Refused is { } refused)
            {
                refusedAlike++;
                string tag = refused.Split(':')[0];
                refusals[tag] = refusals.GetValueOrDefault(tag) + 1;
            }
            else
            {
                compared++;
                refNodes += reference.Nodes;
                screens += pos.State.PendingSelection is not null ? 1 : 0;
                horizon1 += recipe.Horizon > 0 ? 1 : 0;
                exact += reference.Exact ? 1 : 0;
            }

            Assert.Equal(before, CombatSnapshot.Write(pos.State));
        }

        output.WriteLine($"positions built {built}, compared {compared} (open screen at the root {screens}, screens in the tree {screenTrees}, horizon 1 {horizon1}, exact {exact}), refused alike {refusedAlike} [{string.Join(", ", refusals.Select(kv => $"{kv.Key} {kv.Value}"))}], skipped over the reference cap {skipped}");
        output.WriteLine($"reference nodes {refNodes}; pruned (all) nodes {prunedNodes}, engine steps {prunedSteps}, {prunedSeconds:0.00} s, {prunedNodes / Math.Max(1e-9, prunedSeconds):0} nodes/s, {prunedSteps / Math.Max(1e-9, prunedSeconds):0} steps/s");
        foreach (var d in diffs.Take(30))
        {
            output.WriteLine(d);
        }

        Assert.Empty(diffs);
        Assert.True(compared >= 200, $"only {compared} positions compared");
        Assert.True(screens > 0 && screenTrees >= 8, $"open screens: {screens} at the root, {screenTrees} in the tree");
        Assert.True(horizon1 > 0, "no horizon-1 position in the set");
    }

    static string? Tag(string? reason) => reason?.Split(':')[0];

    static IEnumerable<Move> BestOf(IReadOnlyList<MoveValue> moves) =>
        moves.Count == 0 ? [] : moves.Where(m => m.Value == moves.Max(x => x.Value)).Select(m => m.Move);
}
