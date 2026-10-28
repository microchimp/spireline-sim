// The equal-value harness (docs/rebuild/FIGHT_SOLVER.md section 6, piece S3): on the seeded small positions of
// tests/fixtures/solver/positions.jsonl (corpus fixtures' setups and self-play's default fight, old and core2, cut to
// low enemy HP and small piles, with order readers, screens, draws and rolls mixed in), the pruned solver under every
// pruning set must give the reference solver's root value and every root move's value, as exact rationals: equal to
// the last bit, or the test fails. A position the reference cannot finish within its cap is skipped and counted; a
// refusal must be the same refusal on both sides. Skips are counted by class (engine, source, horizon); horizon-1
// positions and root screens are held to a minimum for each engine, source and screen kind; the order-dependence
// positions (Hunter Killer's Tender, the Toadpoles' Thorns) must be solved, not refused or skipped.
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
        ("no-macro-draw", Pruning.All & ~Pruning.MacroDraw),
    ];

    /// <summary>The macro draw's recipes: the reference gets more room there (a five-card draw by card is 720 runs).</summary>
    const long MacroReferenceCap = 400_000;

    const long ReferenceCap = 25_000;

    /// <summary>The order-dependence positions must be solved: the reference gets more room there.</summary>
    const long NamedReferenceCap = 2_000_000;

    /// <summary>Horizon-1 positions compared, at least, on each engine and each source.</summary>
    const int MinHorizon1 = 8;

    /// <summary>Root screens compared, at least, by kind (the draw-pile choice runs the keep path).</summary>
    static readonly (string Kind, int Min)[] MinScreens = [("DrawPileToHand", 6), ("ExhaustFromDrawPile", 3), ("ExhaustFromHandThenDraw", 4), ("UpgradeInHand", 4)];

    [Fact]
    public void EveryPruningEqualsTheReferenceToTheLastBit()
    {
        var corpus = SolverPositions.Corpus();
        int built = 0, compared = 0, skipped = 0, refusedAlike = 0, screens = 0, screenTrees = 0, horizon1 = 0, exact = 0, wideRoll = 0;
        long refNodes = 0, prunedNodes = 0, prunedSteps = 0;
        double prunedSeconds = 0;
        var diffs = new List<string>();
        var refusals = new Dictionary<string, int>();
        var skippedBy = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var horizon1By = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var screenKinds = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var named = new List<string>();
        var namedSolved = new List<int>();
        int namedRecipes = SolverPositions.Recipes().Count(r => r.Encounter.Length > 0);
        int macroRecipes = SolverPositions.Recipes().Count(r => r.Macro.Length > 0);
        var macroSolved = new List<string>();
        long macroNodes = 0;
        foreach (var recipe in SolverPositions.Recipes())
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                if (recipe.Macro.Length > 0)
                {
                    output.WriteLine($"#{recipe.Id} {recipe.Macro} {recipe.Engine}: not built");
                }

                continue;
            }

            built++;
            var before = CombatSnapshot.Write(pos.State);
            var reference = Reference.Solve(pos.State, pos.Rng, recipe.Horizon, recipe.Encounter.Length > 0 ? NamedReferenceCap : recipe.Macro.Length > 0 ? MacroReferenceCap : ReferenceCap);
            if (reference.Refused is { } why && why.StartsWith("budget", StringComparison.Ordinal))
            {
                skipped++;
                if (recipe.Macro.Length > 0)
                {
                    output.WriteLine($"#{recipe.Id} {recipe.Macro} {recipe.Engine}: over the reference cap");
                }

                string cls = $"{recipe.Engine}/{recipe.Source}/h{recipe.Horizon}";
                skippedBy[cls] = skippedBy.GetValueOrDefault(cls) + 1;
                continue;
            }

            long namedNodes = 0;

            foreach (var (name, prune) in Configs)
            {
                var solver = new Expectimax(new SolveOptions { Prune = prune, NodeBudget = 10_000_000, StepBudget = 100_000_000, UseEstimate = false });
                var got = solver.SolveAt(pos.State, pos.Rng, recipe.Horizon);
                if (name == "all")
                {
                    screenTrees += reference.Refused is null && got.Counters.Screens > 0 ? 1 : 0;
                    // A roll wider than the width cap: a horizon leaf on both sides (it was a chance-width refusal before).
                    wideRoll += reference.Refused is null && got.Status != SolveStatus.Refused && got.Counters.WideRollLeaves > 0 ? 1 : 0;
                    prunedNodes += got.Counters.Nodes;
                    prunedSteps += got.Counters.Steps;
                    prunedSeconds += got.Seconds;
                    namedNodes = got.Counters.Nodes;
                    macroNodes += got.Counters.MacroNodes;
                    if (recipe.Macro.Length > 0 && got.Status != SolveStatus.Refused)
                    {
                        // The observers and the ordered hand switch it off at the root; the five-card draws take it.
                        bool off = recipe.Macro.StartsWith("observer:", StringComparison.Ordinal) || recipe.Macro == "hand-ordered";
                        string seen = $"{got.Counters.MacroRoot}, macro nodes {got.Counters.MacroNodes}, off {string.Join(" ", got.Counters.MacroOff.Select(kv => $"{kv.Key} {kv.Value}"))}";
                        if (off ? got.Counters.MacroRoot != recipe.Macro || got.Counters.MacroNodes != 0 : got.Counters.MacroRoot != "on" || got.Counters.MacroNodes == 0)
                        {
                            diffs.Add($"#{recipe.Id} {recipe.Macro} {recipe.Engine}: macro draw {seen}");
                        }

                        macroSolved.Add($"#{recipe.Id} {recipe.Macro} {recipe.Engine}: {seen}, nodes {got.Counters.Nodes}, steps {got.Counters.Steps}");
                    }
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
                if (recipe.Horizon > 0)
                {
                    horizon1By[recipe.Engine] = horizon1By.GetValueOrDefault(recipe.Engine) + 1;
                    horizon1By[recipe.Source] = horizon1By.GetValueOrDefault(recipe.Source) + 1;
                }

                if (pos.State.PendingSelection is { } open)
                {
                    screenKinds[open.Kind.ToString()] = screenKinds.GetValueOrDefault(open.Kind.ToString()) + 1;
                }

                if (recipe.Encounter.Length > 0)
                {
                    namedSolved.Add(recipe.Id);
                    named.Add($"#{recipe.Id} {recipe.Encounter} ({recipe.Until}) {recipe.Engine}: value {reference.Value.ToDouble():0.######}, exact {reference.Exact}, nodes pruned {namedNodes}, reference {reference.Nodes}, hand {pos.State.Hand.Count}");
                }
            }

            Assert.Equal(before, CombatSnapshot.Write(pos.State));
        }

        output.WriteLine($"positions built {built}, compared {compared} (open screen at the root {screens}, screens in the tree {screenTrees}, horizon 1 {horizon1}, exact {exact}), refused alike {refusedAlike} [{string.Join(", ", refusals.Select(kv => $"{kv.Key} {kv.Value}"))}], skipped over the reference cap {skipped}; compared with a wide-roll leaf {wideRoll}");
        output.WriteLine($"horizon 1 compared by engine and source: {string.Join(", ", horizon1By.Select(kv => $"{kv.Key} {kv.Value}"))}; skipped by class: {string.Join(", ", skippedBy.Select(kv => $"{kv.Key} {kv.Value}"))}");
        output.WriteLine($"open screens at the root by kind: {string.Join(", ", screenKinds.Select(kv => $"{kv.Key} {kv.Value}"))}");
        foreach (var n in named.Concat(macroSolved))
        {
            output.WriteLine(n);
        }

        output.WriteLine($"macro draw nodes (all) {macroNodes}; macro recipes compared {macroSolved.Count} of {macroRecipes}");

        output.WriteLine($"reference nodes {refNodes}; pruned (all) nodes {prunedNodes}, engine steps {prunedSteps}, {prunedSeconds:0.00} s, {prunedNodes / Math.Max(1e-9, prunedSeconds):0} nodes/s, {prunedSteps / Math.Max(1e-9, prunedSeconds):0} steps/s");
        foreach (var d in diffs.Take(30))
        {
            output.WriteLine(d);
        }

        Assert.Empty(diffs);
        Assert.True(compared >= 200, $"only {compared} positions compared");
        Assert.True(screens > 0 && screenTrees >= 8, $"open screens: {screens} at the root, {screenTrees} in the tree");
        Assert.True(horizon1 > 0, "no horizon-1 position in the set");
        foreach (var cls in new[] { "old", "core2", "corpus", "selfplay" })
        {
            Assert.True(horizon1By.GetValueOrDefault(cls) >= MinHorizon1, $"{horizon1By.GetValueOrDefault(cls)} horizon-1 positions compared on {cls}");
        }

        foreach (var (kind, min) in MinScreens)
        {
            Assert.True(screenKinds.GetValueOrDefault(kind) >= min, $"{screenKinds.GetValueOrDefault(kind)} root screens of {kind}");
        }

        Assert.True(namedSolved.Count == namedRecipes && namedRecipes >= 2, $"order-dependence positions solved {namedSolved.Count} of {namedRecipes}");
        Assert.True(macroSolved.Count == macroRecipes && macroRecipes >= 24, $"macro-draw positions compared {macroSolved.Count} of {macroRecipes}");
    }

    /// <summary>
    /// Subtree reuse across solves: each harness test above builds a fresh solver per configuration, so there "no-reuse"
    /// and "all" are the same solver. Here one solver (every pruning) solves a parent position and then, keeping its
    /// table, children of it (outcomes of its first root moves, at the turns left there); each child's value, exactness
    /// and root move values must equal the reference's on that child.
    /// </summary>
    [Fact]
    public void ReuseFromAParentSolveEqualsTheReferenceOnItsChildren()
    {
        var corpus = SolverPositions.Corpus();
        int parents = 0, children = 0;
        long hits = 0;
        var diffs = new List<string>();
        foreach (var recipe in SolverPositions.Recipes().Where(r => r.Id % 2 == 0 || r.Horizon > 0 || r.Play.Length > 0))
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                continue;
            }

            var solver = new Expectimax(new SolveOptions { Prune = Pruning.All, NodeBudget = 2_000_000, StepBudget = 20_000_000, UseEstimate = false });
            if (solver.SolveAt(pos.State, pos.Rng, recipe.Horizon).Status == SolveStatus.Refused)
            {
                continue;
            }

            parents++;
            foreach (var m in Actions.Legal(pos.State, Pruning.None).Take(3))
            {
                List<Outcome> outcomes;
                try
                {
                    outcomes = Successors.Expand(pos.State, pos.Rng, m, byCard: false, 32, new SolveCounters());
                }
                catch (SolveRefused)
                {
                    continue;
                }

                foreach (var o in outcomes.Where(o => !o.Result.Terminal).Take(2))
                {
                    int h = recipe.Horizon - (o.State.Turn - pos.State.Turn);
                    if (h < 0)
                    {
                        continue;
                    }

                    var reference = Reference.Solve(o.State, o.Rng, h, ReferenceCap);
                    if (reference.Refused is not null)
                    {
                        continue;
                    }

                    var got = solver.SolveAt(o.State, o.Rng, h);
                    children++;
                    hits += got.Counters.TableHits;
                    if (got.Status == SolveStatus.Refused || got.Value != reference.Value || got.Exact != reference.Exact)
                    {
                        diffs.Add($"#{recipe.Id} child of {m}: {got.Status} {got.Value} (exact {got.Exact}) vs reference {reference.Value} (exact {reference.Exact})");
                        continue;
                    }

                    foreach (var mv in got.Moves)
                    {
                        if (reference.Moves.FirstOrDefault(x => x.Move == mv.Move) is not { } r || r.Value != mv.Value)
                        {
                            diffs.Add($"#{recipe.Id} child of {m}: move {mv.Move} {mv.Value} vs reference");
                        }
                    }
                }
            }
        }

        output.WriteLine($"parents {parents}, children compared {children}, table hits in the children's solves {hits}");
        foreach (var d in diffs.Take(30))
        {
            output.WriteLine(d);
        }

        Assert.Empty(diffs);
        Assert.True(children >= 150 && hits > 0, $"children {children}, table hits {hits}");
    }

    static string? Tag(string? reason) => reason?.Split(':')[0];

    static IEnumerable<Move> BestOf(IReadOnlyList<MoveValue> moves) =>
        moves.Count == 0 ? [] : moves.Where(m => m.Value == moves.Max(x => x.Value)).Select(m => m.Move);
}
