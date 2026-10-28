// The solver finds the discard reshuffle and the sorted pick through explicit sites, not a stack walk (review 1 of the
// solver optimisation plan, section 6 item 1). Under NativeAOT a stack frame does not know its method, so the old walks
// (FromDiscardShuffle, FromStableShuffle) found nothing in the native library: a reshuffle inside a step was not steered
// (every Fisher-Yates call a chance node, n! replays) and the draw-tie refusal never fired. This host runs under the JIT and
// cannot see that; tests/python/test_solver_aot_sites.py solves the positions below through the published library. Here:
// no stack walk is left on a solver decision path, the markers are balanced on every exit, and the positions behave.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class SolverSiteTests
{
    static CardInstance C(string name) => new(Cards.FindId(name)!.Value, false);

    static void Fresh(CombatState state)
    {
        foreach (var pile in new[] { state.Hand, state.DrawPile, state.DiscardPile })
        {
            for (int i = 0; i < pile.Count; i++)
            {
                pile[i] = pile[i].Uid == 0 ? state.Fresh(pile[i]) : pile[i];
            }
        }

        state.KnownTopCount = 0;
        state.KnownBottomCount = 0;
    }

    /// <summary>A turn whose end reshuffles: one card left to draw, the discard and the hand to shuffle under it.</summary>
    internal static (CombatState State, CountingRandom Rng) Reshuffle()
    {
        var state = Fight
            .Hand(C("StrikeIronclad"), C("StrikeIronclad"), C("DefendIronclad"), C("Bash"), C("StrikeIronclad"))
            .Energy(3)
            .Draw(C("DefendIronclad"))
            .Discard(C("StrikeIronclad"), C("DefendIronclad"), C("DefendIronclad"))
            .State;
        Fresh(state);
        foreach (var e in state.Enemies)
        {
            e.Hp = Math.Min(e.Hp, 20);
        }

        SolverFights.SetStreams(state, 41);
        return (state, new CountingRandom(43));
    }

    /// <summary>Uproar over two Rampages, one grown: a sorted pick whose tie the multiset model cannot carry.</summary>
    internal static (CombatState State, CountingRandom Rng) DrawTie() =>
        ChanceModelTests.UproarOver(new CardInstance(Cards.FindId("Rampage")!.Value, false) { BonusDamage = 5 }, C("Rampage"));

    [Fact]
    public void NoStackWalkIsLeftOnASolverDecisionPath()
    {
        // The whole library, recursively: the solver steps the engine, so a walk anywhere in it is on a decision path.
        var dir = Path.Combine(KeyFieldCoverageTests.Repo(), "src", "Sts2Emulator");
        var files = Directory.GetFiles(dir, "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .ToList();
        Assert.True(files.Count > 100, $"only {files.Count} files scanned");
        foreach (var file in files)
        {
            string text = File.ReadAllText(file);
            int walks = System.Text.RegularExpressions.Regex.Matches(text, @"new StackTrace\(|\.GetMethod\(\)").Count;
            // ChanceHook.Site names a refused roll's caller in the exception message only (the reason the solve returns
            // carries the call, not the site), so it may walk; nothing else may.
            int allowed = Path.GetFileName(file) == "ChanceSource.cs" ? 3 : 0;
            Assert.True(walks <= allowed, $"{Path.GetFileName(file)}: {walks} stack walks");
        }

        string effects = File.ReadAllText(Path.Combine(KeyFieldCoverageTests.Repo(), "src", "Sts2Emulator", "Core", "Effects", "CardEffects.cs"));
        Assert.DoesNotContain("the solver's draw-tie looks for it", effects);
    }

    [Fact]
    public void TheMarkersAreBalancedOnEveryExitAndTheReshuffleIsSteered()
    {
        var (s, rng) = Reshuffle();
        var end = new Move(s.Hand.Count, -1);
        var c = new SolveCounters();
        var outcomes = Successors.Expand(s, rng, end, byCard: false, 32, c);
        Assert.Equal(0, CardEffects.DiscardShuffleDepth);
        Assert.Equal(0, CardEffects.StableShuffleDepth);
        Assert.Equal(Rational.One, outcomes.Aggregate(Rational.Zero, (a, o) => a + o.P));
        // Steered, the eight cards to shuffle branch by value at each drawn position (a few hundred runs at most); unsteered,
        // each Fisher-Yates call of the shuffle would be a chance node of its own (8! orders).
        Assert.InRange(c.Steps, 2, 5_000);

        var (t, trng) = DrawTie();
        var uproar = Actions.Legal(t, Pruning.None).First(m => m.Action < t.Hand.Count && t.Hand[m.Action].DefId == Cards.FindId("Uproar")!.Value);
        var e = Assert.Throws<SolveRefused>(() => Successors.Expand(t, trng, uproar, byCard: false, 32, new SolveCounters()));
        Assert.StartsWith("draw-tie", e.Reason);
        Assert.Equal(0, CardEffects.DiscardShuffleDepth);
        Assert.Equal(0, CardEffects.StableShuffleDepth);
    }

    [Fact]
    public void TheMarkersLeaveNothingMarkedAndTheStepIsDeterministic()
    {
        // The plain engine on the same position twice: equal bytes, and nothing left marked. The before/after check of the
        // markers is SolverSiteGoldenTests, pinned at the commit before them.
        var (a, ar) = Reshuffle();
        var (b, br) = Reshuffle();
        CombatEngine.Step(a, a.Hand.Count, ar);
        CombatEngine.Step(b, b.Hand.Count, br);
        Assert.Equal(SolverFights.Bytes(a), SolverFights.Bytes(b));
        Assert.Equal(0, CardEffects.DiscardShuffleDepth + CardEffects.StableShuffleDepth);
    }
}
