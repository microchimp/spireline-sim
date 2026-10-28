// S4's sanity gate (docs/rebuild/FIGHT_SOLVER.md section 13): on every record, the hindsight bound is at least the outcome
// of the expected-best policy played on the same draws and rolls, and over the records where both are exact, the mean bound
// is at least the mean expected value. The records are small fights at low HP on both sides, each with every stream set,
// none won at full score: 32 with the starting deck (the player at 2-11 HP, the enemies capped at 4-16), 12 with Wounds
// shuffled into it (draws that matter), and the lethal fights of PotionCurveTests without the potion (one enemy at 20 HP,
// a hand of Wounds and Strikes, the player at 1 HP: lost whatever the line). The per-record claim holds for any policy, so
// two cheap ones are played, the expected solve's best move at horizon 1 (2,000 nodes per decision) and at horizon 0 (the
// turn alone, L0 at its end: it does not see the enemies' blows coming), and neither needs an exact solve. The claim reads
// the bound the search returns: a searched bound, or, for a cut search, its sound upper bound (L0's maximum: a line it did
// not finish may still win at full score). Exactness is asked only where a claim needs it: the mean claim takes the records
// whose expected solve is exact and whose search was complete; a complete loss needs a complete search.
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class HindsightBoundTests(ITestOutputHelper output)
{
    static readonly SolveOptions[] Policies =
    [
        new() { MaxHorizon = 1, NodeBudget = 2_000, StepBudget = 20_000, UseEstimate = false },
        new() { MaxHorizon = 0, NodeBudget = 5_000, StepBudget = 50_000, UseEstimate = false },
    ];

    static readonly SolveOptions Exact = new() { MaxHorizon = 3, NodeBudget = 10_000, StepBudget = 100_000 };
    static readonly HindsightOptions Search = new() { AllowHindsight = true, SearchNodeBudget = 15_000, SearchStepBudget = 150_000 };

    /// <summary>The records: small fights, every stream set (SolverPositions.Fight), at low HP on both sides.</summary>
    internal static IEnumerable<(string Tag, CombatState State, CountingRandom Rng)> Records()
    {
        int fire = Potions.FindId("FirePotion")!.Value, block = Potions.FindId("BlockPotion")!.Value;
        int[][] belts = [[], [fire], [block, fire]];
        for (int i = 0; i < 32; i++)
        {
            string engine = i % 4 == 3 ? "core2" : "old";
            int hp = 2 + 3 * (i % 4), cap = 4 + 4 * (i / 4 % 4);
            var belt = belts[i % 3];
            var (s, rng) = SolverPositions.Fight(engine, 700 + i, TestDeck.StarterDeckIds.ToArray(), [], belt, hp: hp);
            s.MaxPotionSlots = Math.Max(s.MaxPotionSlots, belt.Length);
            Array.Clear(s.PotionSlots);
            belt.CopyTo(s.PotionSlots, 0);
            foreach (var e in s.Enemies.Where(e => e.Hp > 0))
            {
                e.Hp = Math.Min(e.Hp, cap);
            }

            yield return ($"fight {700 + i} ({engine}, HP {hp}, enemies at most {cap}, {belt.Length} potions)", s, rng);
        }

        int wound = Cards.FindId("Wound")!.Value, strike = Cards.FindId("StrikeIronclad")!.Value;
        for (int i = 0; i < 12; i++)
        {
            int hp = 6 + 2 * (i % 4), cap = 10 + 3 * (i / 4);
            var (s, rng) = SolverPositions.Fight("old", 800 + i, [.. TestDeck.StarterDeckIds, .. Enumerable.Repeat(wound, 3)], [], [], hp: hp);
            foreach (var e in s.Enemies.Where(e => e.Hp > 0))
            {
                e.Hp = Math.Min(e.Hp, cap);
            }

            yield return ($"cluttered {800 + i} (HP {hp}, enemies at most {cap}, three Wounds)", s, rng);
        }

        int doomed = 0;
        foreach (var (s, rng) in PotionCurveTests.LethalFights([], [.. Enumerable.Repeat(wound, 7), .. Enumerable.Repeat(strike, 3)], 20).Take(4))
        {
            yield return ($"doomed {doomed++} (HP 1, one enemy at 20, Wounds and Strikes)", s, rng);
        }
    }

    [Fact]
    public void TheBoundIsNeverBelowTheExpectedBestPolicyOnTheSameDraws()
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var phase = new double[3];
        int records = 0, trivial = 0, policyNull = 0, searched = 0, cut = 0, completeLosses = 0, both = 0, aboveAny = 0;
        var above = new int[Policies.Length];
        Rational sumBound = Rational.Zero, sumExpected = Rational.Zero;
        var below = new List<string>();
        var losses = new List<string>();
        foreach (var (tag, s, rng) in Records())
        {
            Assert.Empty(SolverService.UnsetStreamsOf(s));
            var t = System.Diagnostics.Stopwatch.StartNew();
            var played = Policies.Select(p => SolverService.PlayExpectedPolicy(s, rng, p)).ToList();
            phase[0] += t.Elapsed.TotalSeconds;
            if (played.Any(x => x is null))
            {
                policyNull++;
                continue;
            }

            var outcomes = played.Select(x => x!.Value.Outcome).ToList();
            if (outcomes.All(o => o == Utility.Max))
            {
                trivial++;  // won at full score: nothing for the bound to exceed
                continue;
            }

            records++;
            t.Restart();
            var h = SolverService.SolveHindsight(s, rng, Search, trueStreams: true);
            phase[1] += t.Elapsed.TotalSeconds;
            Assert.NotEqual(HindsightOutcome.Refused, h.Outcome);
            var upper = h.UpperBound!.Value;
            for (int k = 0; k < Policies.Length; k++)
            {
                if (upper < outcomes[k])
                {
                    below.Add($"{tag}: bound {upper} < outcome {outcomes[k]} of policy {k}");
                }
            }

            if (h.Outcome == HindsightOutcome.Cut)
            {
                cut++;
                Assert.Null(h.Bound);
                Assert.True(double.IsNaN(h.BoundDouble));
                continue;
            }

            searched++;
            var bound = h.Bound!.Value;
            for (int k = 0; k < Policies.Length; k++)
            {
                above[k] += bound > outcomes[k] ? 1 : 0;
            }

            aboveAny += outcomes.Any(o => bound > o) ? 1 : 0;
            if (h.WinLineExists == WinLine.No)
            {
                completeLosses++;
                Assert.All(outcomes, o => Assert.Equal(Utility.Loss, o));  // no line wins on these draws, so no policy did
                losses.Add(tag);
            }

            t.Restart();
            var expected = SolverService.SolveExpected(s, rng, Exact);
            phase[2] += t.Elapsed.TotalSeconds;
            if (expected.Status == SolveStatus.Exact)
            {
                both++;
                sumBound += bound;
                sumExpected += expected.Value;
            }
        }

        double n = Math.Max(1, both);
        output.WriteLine($"hindsight vs the policies: {records} non-trivial records ({trivial} won at full score, {policyNull} policy refused), "
            + $"{searched} searched, {cut} cut; bound above the outcome on {aboveAny} searched records (horizon-1 policy {above[0]}, horizon-0 policy {above[1]}), "
            + $"below on {below.Count}; {completeLosses} complete losses; mean bound {sumBound.ToDouble() / n:0.0000} vs mean expected "
            + $"{sumExpected.ToDouble() / n:0.0000} on {both} records with both exact; {clock.Elapsed.TotalSeconds:0.0} s "
            + $"(policies {phase[0]:0.0}, hindsight {phase[1]:0.0}, expected {phase[2]:0.0})");
        foreach (var l in losses)
        {
            output.WriteLine($"  complete loss: {l}");
        }

        Assert.Empty(below);
        Assert.True(records >= 30, $"only {records} non-trivial records");
        Assert.True(searched >= 20, $"only {searched} searched records: the per-record claim would rest on the cut searches' trivial bound");
        Assert.True(aboveAny >= 1, "no searched record has a bound above a policy's outcome");
        Assert.True(completeLosses >= 1, "no complete loss among the records");
        Assert.True(both >= 5, $"only {both} records with an exact expected value and a complete search");
        Assert.True(sumBound >= sumExpected, $"mean bound {sumBound.ToDouble() / n} below mean expected {sumExpected.ToDouble() / n}");
    }
}
