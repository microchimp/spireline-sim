// The S5 root check's utilities (docs/rebuild/FIGHT_SOLVER.md section 20): Lethal (win 1, death 0, horizon leaf 0) and
// Survival (win 1, death 0, leaf 1 at the first decision past the horizon's last turn with no screen pending). On the
// equal-value harness's positions, the pruned solver under every pruning set gives the reference solver's root value,
// every root move's value and exactness under each utility, as exact rationals; the deepening's fixed-horizon solve gives
// the same; every value lies in [0, 1]. L0 is held by EqualValueHarnessTests, which these utilities leave unchanged.
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class CheckUtilityTests(ITestOutputHelper output)
{
    static readonly (string Name, Pruning Prune)[] Configs =
    [
        ("all", Pruning.All),
        ("no-transposition", Pruning.All & ~Pruning.Transposition),
        ("no-multiset-hand", Pruning.All & ~Pruning.MultisetHand),
        ("no-interchangeable", Pruning.All & ~Pruning.InterchangeableEnemies),
        ("no-draw-by-value", Pruning.All & ~Pruning.DrawByValue),
        ("no-macro-draw", Pruning.All & ~Pruning.MacroDraw),
        ("bounds-off", Pruning.NoBounds),
        ("no-closure", Pruning.All & ~Pruning.LethalClosure),
        ("no-closer-first", Pruning.All & ~Pruning.CloserFirst),
        ("none", Pruning.None),
    ];

    /// <summary>The configurations the survival check runs (its trees span the enemy's turn and the next draw).</summary>
    static readonly string[] SurvivalConfigs = ["all", "no-transposition", "no-draw-by-value", "no-macro-draw", "bounds-off", "no-closure"];

    const long ReferenceCap = 25_000;

    [Fact]
    public void CheckUtilitiesEqualTheReferenceUnderEveryPruning()
    {
        var corpus = SolverPositions.Corpus();
        var compared = new Dictionary<string, int>();
        int skipped = 0, refusedAlike = 0, won = 0, notOne = 0;
        var diffs = new List<string>();
        // A fixed quarter of the harness's recipes for lethal at horizon 0, an eighth for survival at horizon 1 (the time).
        foreach (var recipe in SolverPositions.Recipes().Where(r => r.Id % 4 == 0))
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                continue;
            }

            var checks = recipe.Id % 8 == 0 ? new[] { (UtilityKind.Lethal, 0), (UtilityKind.Survival, 1) } : new[] { (UtilityKind.Lethal, 0) };
            foreach (var (u, h) in checks)
            {
                var want = Reference.Solve(pos.State, pos.Rng, h, ReferenceCap, utility: u);
                if (want.Refused is { } why && why.StartsWith("budget", StringComparison.Ordinal))
                {
                    skipped++;
                    continue;
                }

                string tag = $"{u}/h{h}";
                if (want.Refused is null)
                {
                    compared[tag] = compared.GetValueOrDefault(tag) + 1;
                    won += want.Value == Rational.One ? 1 : 0;
                    notOne += want.Value < Rational.One ? 1 : 0;
                    if (want.Value < Rational.Zero || want.Value > Rational.One || want.Moves.Any(m => m.Value < Rational.Zero || m.Value > Rational.One))
                    {
                        diffs.Add($"#{recipe.Id} {tag}: a value outside [0, 1]");
                    }
                }
                else
                {
                    refusedAlike++;
                }

                foreach (var (name, prune) in Configs.Where(c => u != UtilityKind.Survival || SurvivalConfigs.Contains(c.Name)))
                {
                    var opts = new SolveOptions { Prune = prune, NodeBudget = 10_000_000, StepBudget = 100_000_000, UseEstimate = false, Utility = u, MaxHorizon = h };
                    var got = new Expectimax(opts).SolveAt(pos.State, pos.Rng, h);
                    Compare($"#{recipe.Id} {tag} {name}", got, want, diffs);
                    if (name == "all")
                    {
                        // The deepening's path: one solve at the fixed horizon, no estimate.
                        var deep = new Expectimax(opts with { UseEstimate = true }).SolveExpected(pos.State, pos.Rng);
                        Compare($"#{recipe.Id} {tag} deepening", deep, want, diffs);
                        if (deep.Status != SolveStatus.Refused && (deep.Horizon != h || deep.Counters.Horizons.Count != 1 || (deep.Status == SolveStatus.Exact) != want.Exact))
                        {
                            diffs.Add($"#{recipe.Id} {tag} deepening: horizon {deep.Horizon}, iterations {deep.Counters.Horizons.Count}, status {deep.Status}");
                        }
                    }
                }
            }
        }

        output.WriteLine($"compared {string.Join(", ", compared.Select(kv => $"{kv.Key} {kv.Value}"))}; refused alike {refusedAlike}; over the reference cap {skipped}; value 1 {won}, below 1 {notOne}");
        foreach (var d in diffs.Take(40))
        {
            output.WriteLine(d);
        }

        Assert.Empty(diffs);
        Assert.True(compared.GetValueOrDefault("Lethal/h0") >= 40, $"lethal positions compared {compared.GetValueOrDefault("Lethal/h0")}");
        Assert.True(compared.GetValueOrDefault("Survival/h1") >= 10, $"survival positions compared {compared.GetValueOrDefault("Survival/h1")}");
        Assert.True(won > 0 && notOne > 0, $"values of 1 {won}, below 1 {notOne}: both kinds must be met");
    }

    static void Compare(string at, ExpectedResult got, Reference.Result want, List<string> diffs)
    {
        if (want.Refused is not null || got.Status == SolveStatus.Refused)
        {
            if ((want.Refused is null) != (got.Status != SolveStatus.Refused) || want.Refused?.Split(':')[0] != got.Reason?.Split(':')[0])
            {
                diffs.Add($"{at}: refusal reference '{want.Refused}' vs solver '{got.Reason}'");
            }

            return;
        }

        if (got.Value != want.Value || got.Exact != want.Exact)
        {
            diffs.Add($"{at}: value {got.Value} (exact {got.Exact}) vs reference {want.Value} (exact {want.Exact})");
        }

        foreach (var mv in got.Moves)
        {
            var r = want.Moves.FirstOrDefault(x => x.Move == mv.Move);
            if (r is null || r.Value != mv.Value)
            {
                diffs.Add($"{at}: move {mv.Move} {mv.Value} vs reference {r?.Value.ToString() ?? "missing"}");
            }
        }
    }
}
