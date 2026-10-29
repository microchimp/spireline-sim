// The exact layer's own cases (piece S3): lethal is found where it exists, a forced death values at the loss value, the
// budget cut refuses rather than returning a wrong exact value, the position given is never changed (the solver reads
// and clones), the table answers the next decision (subtree reuse), refusal reasons, and the week-3 size estimator.
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class ExpectimaxTests(ITestOutputHelper output)
{
    static int Card(string name) => Cards.FindId(name)!.Value;

    static (CombatState State, CountingRandom Rng) Strikes(string engine, int seed, int count = 10) =>
        SolverPositions.Fight(engine, seed, [.. Enumerable.Repeat(Card("StrikeIronclad"), count)], [], []);

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void LethalIsFoundWhereItExists(string engine)
    {
        var (s, rng) = Strikes(engine, 11);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 1;
        }

        var r = new Expectimax().SolveExpected(s, rng);
        Assert.Equal(SolveStatus.Exact, r.Status);
        Assert.Equal(0, r.Horizon);
        Assert.Equal(Utility.L0(s), r.Value);
        Assert.Equal(Rational.One, r.ExactMass);
        // Every strike starts a lethal line, exactly valued at the win; end turn's cut is L0's placeholder, which ties.
        var strikes = r.Moves.Where(m => m.Move.Action < s.Hand.Count).ToList();
        Assert.NotEmpty(strikes);
        Assert.All(strikes, m =>
        {
            Assert.True(m.Exact, $"{m.Move} not exact");
            Assert.Equal(Utility.L0(s), m.Value);
        });

        // No lethal: the turn ends with the enemy alive, so the value is not exact at the first horizon.
        var (t, trng) = Strikes(engine, 11);
        foreach (var e in t.Enemies)
        {
            e.Hp = e.MaxHp = 500;
        }

        var cut = new Expectimax(new SolveOptions { MaxHorizon = 0 }).SolveExpected(t, trng);
        Assert.False(cut.Exact);
        Assert.Equal(SolveStatus.Budget, cut.Status);
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void AForcedDeathValuesAtTheLossValue(string engine)
    {
        for (int seed = 1; seed < 80; seed++)
        {
            var (s, rng) = Strikes(engine, seed, 6);
            if (!s.Enemies.Any(e => e.Hp > 0 && e.CurrentIntent.AnnouncedDamage(e.Buffs, s.PlayerBuffs) > 0))
            {
                continue;
            }

            s.PlayerHp = 1;
            foreach (var e in s.Enemies)
            {
                e.Hp = e.MaxHp = 999;
            }

            var r = new Expectimax(new SolveOptions { MaxHorizon = 1, UseEstimate = false, NodeBudget = 2_000_000 }).SolveExpected(s, rng);
            Assert.Equal(SolveStatus.Exact, r.Status);
            Assert.Equal(Utility.Loss, r.Value);
            Assert.All(r.Moves, m => Assert.Equal(Utility.Loss, m.Value));
            return;
        }

        Assert.Fail("no seed whose first enemy move attacks");
    }

    [Fact]
    public void TheBudgetCutRefusesRatherThanReturningAWrongExactValue()
    {
        var (s, rng) = SolverPositions.Fight("old", 21, TestDeck.StarterDeckIds.ToArray(), [], []);

        // Nothing completes: refused, no value.
        var none = new Expectimax(new SolveOptions { NodeBudget = 3 }).SolveExpected(s, rng);
        Assert.Equal(SolveStatus.Refused, none.Status);
        Assert.False(none.Exact);
        Assert.StartsWith("budget", none.Reason);

        // The first horizon completes, the next does not: the last completed horizon, not exact, its value the fixed-horizon solve's.
        var h0 = new Expectimax(new SolveOptions { UseEstimate = false }).SolveAt(s, rng, 0);
        var cut = new Expectimax(new SolveOptions { NodeBudget = h0.Counters.Nodes + 50, UseEstimate = false }).SolveExpected(s, rng);
        Assert.Equal(SolveStatus.Budget, cut.Status);
        Assert.False(cut.Exact);
        Assert.Equal(0, cut.Horizon);
        Assert.Equal(h0.Value, cut.Value);

        // The estimator stops the deepening before it spends: a full fight is far over a small budget.
        var est = new Expectimax(new SolveOptions { NodeBudget = h0.Counters.Nodes + 10, EstimateSlack = 0 }).SolveExpected(s, rng);
        Assert.NotEqual(SolveStatus.Exact, est.Status);
        Assert.False(est.Exact);
        Assert.StartsWith("estimate", est.Reason);
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void TheEngineIsUntouchedTheSolverOnlyReadsAndClones(string engine)
    {
        var (s, rng) = SolverPositions.Fight(engine, 31, TestDeck.StarterDeckIds.ToArray(), [], []);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 9;
        }

        var bytes = CombatSnapshot.Write(s);
        var key = StateKey.Encode(s, KeyMode.Hindsight);
        var streams = Enumerable.Range(0, StateCloning.StreamCount).Select(i => (StateCloning.Stream(s, i)?.Seed, StateCloning.Stream(s, i)?.CallCount)).ToList();
        var (seed, calls) = (rng.Seed, rng.CallCount);
        new Expectimax(new SolveOptions { MaxHorizon = 1, NodeBudget = 50_000 }).SolveExpected(s, rng);
        Reference.Solve(s, rng, 0, 20_000);
        Assert.Equal(bytes, CombatSnapshot.Write(s));
        Assert.Equal(key, StateKey.Encode(s, KeyMode.Hindsight));
        Assert.Equal(streams, Enumerable.Range(0, StateCloning.StreamCount).Select(i => (StateCloning.Stream(s, i)?.Seed, StateCloning.Stream(s, i)?.CallCount)).ToList());
        Assert.Equal((seed, calls), (rng.Seed, rng.CallCount));
        Assert.Null(rng.Chance);
        Assert.All(Enumerable.Range(0, StateCloning.StreamCount), i => Assert.Null(StateCloning.Stream(s, i)?.Chance));
    }

    [Fact]
    public void TheTableAnswersTheNextDecision()
    {
        var (s, rng) = SolverPositions.Fight("old", 41, TestDeck.StarterDeckIds.ToArray(), [], []);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 12;
        }

        // The table's own test: the bounds round's closure and cuts (on in All) leave too few nodes here for the commuting
        // plays to meet, so they are off; the table's kinds and the bounds have UtilityBoundTests.
        var solver = new Expectimax(new SolveOptions { UseEstimate = false, Prune = Pruning.NoBounds });
        var first = solver.SolveAt(s, rng, 0);
        Assert.True(first.Counters.TableHits > 0, "commuting plays did not meet in the table");
        var move = first.Moves.Where(m => !Actions.EndsTurn(s, m.Move)).MaxBy(m => m.Value)!.Move;
        var next = s.Clone();
        var nrng = rng.Copy();
        Successors.Step(next, nrng, move);
        var again = solver.SolveAt(next, nrng, 0);
        var fresh = new Expectimax(new SolveOptions { UseEstimate = false, Prune = Pruning.NoBounds }).SolveAt(next, nrng, 0);
        Assert.Equal(fresh.Value, again.Value);
        Assert.True(again.Counters.Nodes < fresh.Counters.Nodes, $"reuse {again.Counters.Nodes} nodes, fresh {fresh.Counters.Nodes}");
    }

    [Theory]
    [InlineData("old")]
    [InlineData("core2")]
    public void AStarterTurnAndTheNextSolveWithinTheBudget(string engine)
    {
        // A sample of the solver's speed: the starter deck's first turn and the next (enemy turn, draws, rolls).
        var (s, rng) = SolverPositions.Fight(engine, 71, TestDeck.StarterDeckIds.ToArray(), [], []);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 30;
        }

        new Expectimax(new SolveOptions { UseEstimate = false }).SolveAt(s, rng, 0);  // warm the JIT
        var r = new Expectimax(new SolveOptions { UseEstimate = false }).SolveAt(s, rng, 1);
        output.WriteLine($"{engine}: {r.Status} {r.ValueDouble:0.0000} exact {r.Exact} nodes {r.Counters.Nodes} steps {r.Counters.Steps} hits {r.Counters.TableHits} merged {r.Counters.Merged} {r.Seconds:0.00} s, {r.Counters.Nodes / Math.Max(1e-9, r.Seconds):0} nodes/s, {r.Counters.Steps / Math.Max(1e-9, r.Seconds):0} steps/s");
        Assert.NotEqual(SolveStatus.Refused, r.Status);
        Assert.True(r.Counters.TableHits > 0);
    }

    [Fact]
    public void ADrawPileOrderPeekRefusesWithItsReason()
    {
        var (s, rng) = SolverPositions.Fight("old", 51, [.. TestDeck.StarterDeckIds.ToArray(), Card("IAmInvincible")], [], []);
        var r = new Expectimax().SolveExpected(s, rng);
        Assert.Equal(SolveStatus.Refused, r.Status);
        Assert.StartsWith("draw-peek", r.Reason);
    }

    [Fact]
    public void TheSizeEstimatorIsTheWeekThreeCount()
    {
        // Three distinct one-cost cards with three energy: 1 + 3 + 6 + 6 ordered sequences.
        Assert.Equal(16d, SizeEstimate.Sequences([(1, 1), (1, 1), (1, 1)], 3));

        // Two copies of a zero-cost card: end now, one, two.
        Assert.Equal(3d, SizeEstimate.Sequences([(0, 2)], 0));

        // Five-card hands from three kinds of two copies each: the (2, 2, 1) arrangements.
        Assert.Equal(3d, SizeEstimate.Hands([2, 2, 2], 5));
        var (s, _) = SolverPositions.Fight("old", 61, TestDeck.StarterDeckIds.ToArray(), [], []);
        double a = SizeEstimate.Log10(s, 0), b = SizeEstimate.Log10(s, 1), c = SizeEstimate.Log10(s, 4);
        Assert.True(a < b && b < c, $"{a} {b} {c}");
        Assert.True(c > 7, "four turns of a starter deck should be over 10^7, as the week-3 count found");
    }

    [Fact]
    public void RationalsAreExact()
    {
        var third = new Rational(1, 3);
        Assert.Equal(Rational.One, third + third + third);
        Assert.NotEqual(new Rational(1, 10), Rational.FromDouble(0.1));
        Assert.Equal(0.1, Rational.FromDouble(0.1).ToDouble());
        Assert.Equal(new Rational(3, 4), Rational.FromDouble(0.75));
        Assert.True(new Rational(2, 3) > new Rational(3, 5));
    }

    /// <summary>
    /// An exact max node's mass is its exact move's: a cut move first whose bound equals its value (a horizon leaf at
    /// L0's maximum, mass 0) and an exact move of the same value make the node exact, with mass one, not the cut move's.
    /// </summary>
    [Fact]
    public void AnExactMaxTakesTheMassOfItsExactMove()
    {
        var cut = new NodeValue(Rational.One, false, Rational.Zero, Rational.One);
        var proved = new NodeValue(Rational.One, true, Rational.One, Rational.One);
        var v = NodeValue.Max([cut, proved], new CombatState());
        Assert.True(v.Exact);
        Assert.True(v.Mass == Rational.One);
        var open = NodeValue.Max([cut, cut], new CombatState());
        Assert.False(open.Exact);
    }
}
