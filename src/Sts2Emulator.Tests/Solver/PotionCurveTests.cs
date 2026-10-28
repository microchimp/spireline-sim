// The potion value curve (docs/rebuild/FIGHT_SOLVER.md section 8, piece S4): one solve per subset of the potions held.
// Each subset's value is the expected solve's under the same potion mask at the same horizon; allowing more potions never lowers it; the empty
// subset is the base; the chances are probabilities; and the native export writes what the library computes.
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class PotionCurveTests(ITestOutputHelper output)
{
    static readonly SolveOptions Opts = new() { NodeBudget = 200_000, StepBudget = 2_000_000, MaxHorizon = 1, UseEstimate = false };

    /// <summary>Small fights holding two or three potions, at low HP on both sides, so that potions decide some of them.</summary>
    static IEnumerable<(string Tag, CombatState State, CountingRandom Rng)> Fights()
    {
        int[][] belts = [[Potions.FindId("FirePotion")!.Value, Potions.FindId("BlockPotion")!.Value],
            [Potions.FindId("FirePotion")!.Value, Potions.FindId("FirePotion")!.Value, Potions.FindId("StrengthPotion")!.Value]];
        for (int seed = 0; seed < 4; seed++)
        {
            var belt = belts[seed % belts.Length];
            var (s, rng) = SolverPositions.Fight(seed % 3 == 2 ? "core2" : "old", 600 + seed, TestDeck.StarterDeckIds.ToArray(), [], belt, hp: 6 + seed);
            // The belt as given, whatever the deal put there: one potion per slot from the first.
            s.MaxPotionSlots = Math.Max(s.MaxPotionSlots, belt.Length);
            Array.Clear(s.PotionSlots);
            belt.CopyTo(s.PotionSlots, 0);
            foreach (var e in s.Enemies.Where(e => e.Hp > 0))
            {
                e.Hp = Math.Min(e.Hp, 6 + 2 * (seed % 3));
            }

            yield return ($"seed {600 + seed}", s, rng);
        }
    }

    [Fact]
    public void EachSubsetIsTheExpectedSolveUnderItsMaskAndMorePotionsNeverLowerIt()
    {
        int exactEntries = 0, compared = 0;
        bool printed = false;
        foreach (var (tag, s, rng) in Fights())
        {
            var held = PotionCurve.HeldSlots(s);
            Assert.True(held.Count >= 2, $"{tag}: the belt holds {held.Count}");
            var curve = PotionCurve.Solve(s, rng, Opts);
            Assert.Equal(1 << held.Count, curve.Count);
            for (int i = 0; i < curve.Count; i++)
            {
                var e = curve[i];
                Assert.Equal(i, e.Subset);
                Assert.Equal(System.Numerics.BitOperations.PopCount((uint)i), e.Held.Count);
                if (e.Status == SolveStatus.Refused)
                {
                    continue;
                }

                Assert.InRange(e.WinChance.ToDouble(), 0, 1);
                Assert.All(e.Held.Values, p => Assert.InRange(p.ToDouble(), 0, 1));
                exactEntries += e.Exact ? 1 : 0;
                var direct = new Expectimax(Opts with { AllowedPotions = PotionCurve.MaskOf(held, i) }).SolveAt(s, rng, e.Horizon);
                if (direct.Status != SolveStatus.Refused)
                {
                    compared++;
                    Assert.True(direct.Value == e.ExpectedL0, $"{tag} subset {i}: curve {e.ExpectedL0} vs solve {direct.Value} at horizon {e.Horizon}");
                }

                for (int j = 0; j < i; j++)
                {
                    if ((j & i) == j && curve[j].Status != SolveStatus.Refused && curve[j].Horizon == e.Horizon)
                    {
                        Assert.True(curve[j].ExpectedL0 <= e.ExpectedL0, $"{tag}: subset {j} {curve[j].ExpectedL0} above its superset {i} {e.ExpectedL0}");
                    }
                }
            }

            if (!printed && curve.All(e => e.Status != SolveStatus.Refused))
            {
                printed = true;
                output.WriteLine($"curve sample ({tag}, potions {string.Join(",", held.Select(h => Potions.Get(s.PotionSlots[h]).Name))}, HP {s.PlayerHp}):");
                foreach (var e in curve)
                {
                    output.WriteLine($"  subset {e.Subset} ({e.Status}, horizon {e.Horizon}, mass {e.ExactMass.ToDouble():0.000}): L0 {e.ExpectedL0.ToDouble():0.0000} win {e.WinChance.ToDouble():0.0000} win-HP {e.ExpectedWinHp:0.00} held {string.Join(" ", e.Held.Select(kv => $"{kv.Key}:{kv.Value.ToDouble():0.00}"))}");
                }
            }
        }

        output.WriteLine($"curve: {exactEntries} exact entries, {compared} compared with the masked solve at their horizon");
        Assert.True(compared >= 6, $"only {compared} entries compared");
    }

    [Fact]
    public void TheNativeCurveWritesTheLibrarysCurve()
    {
        var (tag, s, rng) = Fights().First();
        var lib = PotionCurve.Solve(s, rng, Opts);
        int handle = NativeSolverCalls.Adopt(s, rng);
        int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(Opts));
        var (n, entries) = NativeSolverCalls.Curve(solver, handle, null);
        NativeSolverCalls.Destroy(solver);
        NativeSolverCalls.DestroyCombat(handle);
        Assert.Equal(lib.Count, n);
        var held = PotionCurve.HeldSlots(s);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(lib[i].ExpectedL0.ToDouble(), entries[i].ExpectedL0);
            Assert.Equal(lib[i].WinChance.ToDouble(), entries[i].WinChance);
            Assert.Equal((int)lib[i].Status, entries[i].CurveStatus);
            for (int slot = 0; slot < 8; slot++)
            {
                double got;
                unsafe
                {
                    var e = entries[i];
                    got = e.Held[slot];
                }

                if (lib[i].Held.TryGetValue(slot, out var p))
                {
                    Assert.Equal(p.ToDouble(), got);
                }
                else
                {
                    Assert.True(double.IsNaN(got), $"{tag} subset {i} slot {slot}");
                }
            }
        }

        Assert.Equal(held.Count, lib[^1].Held.Count);
    }
}
