// The potion value curve (docs/rebuild/FIGHT_SOLVER.md section 8, piece S4): one solve per subset of the potions held, all at
// one horizon. Each subset's value is the expected solve's under the same potion mask at the curve's horizon; allowing more
// potions never lowers it (a claim between entries of one curve, which share their horizon); a curve its budget cuts keeps
// one horizon or is refused whole; the chances are probabilities; a potion that decides lethal moves the curve (so the mask
// is seen to work, not merely shared by both sides of a comparison); a zeroed native options struct masks no potion; and the
// native export writes what the library computes.
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Interop;
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

    /// <summary>
    /// A potion decides the fight: one enemy at 12 HP whose turn kills the player (1 HP), a hand of Wounds (nothing to play),
    /// and the belt given (a Fire Potion deals 20). The first fight, over the seeds and Act 1 encounters in order, with one
    /// enemy whose turn is lethal on the position's own streams.
    /// </summary>
    internal static (CombatState State, CountingRandom Rng) PotionLethal(int[] belt) =>
        LethalFights(belt, [.. Enumerable.Repeat(Cards.FindId("Wound")!.Value, 10)], 12).First();

    /// <summary>
    /// Every fight, over the Act 1 encounters and four seeds in order, with one enemy (set to <paramref name="enemyHp" />)
    /// whose turn kills the player (1 HP) on the position's own streams, the deck and the belt given.
    /// </summary>
    internal static IEnumerable<(CombatState State, CountingRandom Rng)> LethalFights(int[] belt, int[] deck, int enemyHp)
    {
        foreach (var encounter in Enum.GetValues<CombatFactory.ActOneEncounter>())
        {
            for (int seed = 0; seed < 4; seed++)
            {
                var (s, rng) = SolverPositions.Fight("old", 900 + seed, deck, [], [], hp: 1, encounter: (int)encounter);
                if (s.PendingSelection is not null || s.Enemies.Count(e => e.Hp > 0) != 1)
                {
                    continue;
                }

                s.Enemies.First(e => e.Hp > 0).Hp = enemyHp;
                var (copy, r) = (s.Clone(), rng.Copy());
                var end = Actions.Legal(copy, Pruning.None).First(m => Actions.EndsTurn(copy, m));
                var step = Successors.Step(copy, r, end);
                if (!step.Terminal || step.PlayerWon)
                {
                    continue;
                }

                s.MaxPotionSlots = Math.Max(s.MaxPotionSlots, Math.Max(1, belt.Length));
                Array.Clear(s.PotionSlots);
                belt.CopyTo(s.PotionSlots, 0);
                yield return (s, rng);
            }
        }
    }

    static void OneHorizonAndMonotone(string tag, List<CurveEntry> curve)
    {
        var horizons = curve.Where(e => e.Status != SolveStatus.Refused).Select(e => e.Horizon).Distinct().ToList();
        Assert.True(horizons.Count <= 1, $"{tag}: the curve mixes horizons {string.Join(",", horizons)}");
        if (horizons.Count == 0)
        {
            Assert.All(curve, e => Assert.StartsWith("budget:", e.Reason));
        }

        for (int i = 0; i < curve.Count; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if ((j & i) == j && curve[j].Status != SolveStatus.Refused && curve[i].Status != SolveStatus.Refused)
                {
                    Assert.True(curve[j].ExpectedL0 <= curve[i].ExpectedL0, $"{tag}: subset {j} {curve[j].ExpectedL0} above its superset {i} {curve[i].ExpectedL0}");
                }
            }
        }
    }

    [Fact]
    public void EachSubsetIsTheExpectedSolveUnderItsMaskAtTheCurvesOneHorizon()
    {
        int exactEntries = 0, compared = 0;
        bool printed = false;
        foreach (var (tag, s, rng) in Fights())
        {
            var held = PotionCurve.HeldSlots(s);
            Assert.True(held.Count >= 2, $"{tag}: the belt holds {held.Count}");
            var curve = PotionCurve.Solve(s, rng, Opts);
            Assert.Equal(1 << held.Count, curve.Count);
            OneHorizonAndMonotone(tag, curve);
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
            }

            if (!printed && curve.All(e => e.Status != SolveStatus.Refused))
            {
                printed = true;
                Print($"curve sample ({tag}, potions {string.Join(",", held.Select(h => Potions.Get(s.PotionSlots[h]).Name))}, HP {s.PlayerHp})", curve);
            }
        }

        output.WriteLine($"curve: {exactEntries} exact entries, {compared} compared with the masked solve at their horizon");
        Assert.True(compared >= 6, $"only {compared} entries compared");
    }

    void Print(string title, List<CurveEntry> curve)
    {
        output.WriteLine(title + ":");
        foreach (var e in curve)
        {
            output.WriteLine($"  subset {e.Subset} ({e.Status}, horizon {e.Horizon}, mass {e.ExactMass.ToDouble():0.000}): L0 {e.ExpectedL0.ToDouble():0.0000} win {e.WinChance.ToDouble():0.0000} win-HP {e.ExpectedWinHp:0.00} held {string.Join(" ", e.Held.Select(kv => $"{kv.Key}:{kv.Value.ToDouble():0.00}"))}");
        }
    }

    [Fact]
    public void ACurveItsBudgetCutsKeepsOneHorizonOrIsRefusedWhole()
    {
        var (tag, s, rng) = Fights().Skip(1).First();
        var horizons = new List<string>();
        foreach (long nodes in new long[] { 1, 40, 400, 4_000 })
        {
            var curve = PotionCurve.Solve(s, rng, Opts with { NodeBudget = nodes, MaxHorizon = 4 });
            OneHorizonAndMonotone($"{tag} at {nodes} nodes", curve);
            horizons.Add($"{nodes}: {string.Join(",", curve.Select(e => e.Status == SolveStatus.Refused ? "x" : $"{e.Horizon}").Distinct())}");
        }

        output.WriteLine($"{tag}, curve horizon by node budget: {string.Join("; ", horizons)}");
    }

    [Fact]
    public void APotionThatDecidesLethalMovesTheCurve()
    {
        int fire = Potions.FindId("FirePotion")!.Value;
        var (s, rng) = PotionLethal([fire]);
        var curve = PotionCurve.Solve(s, rng, Opts);
        Print($"a potion that decides lethal (HP {s.PlayerHp}, one enemy at 12, a Fire Potion in slot 0)", curve);
        Assert.Equal(2, curve.Count);
        OneHorizonAndMonotone("lethal", curve);
        var (none, all) = (curve[0], curve[1]);
        Assert.Equal(1, curve[1].Horizon);
        Assert.Equal(Rational.One, all.WinChance);  // the potion kills before the enemy's turn
        Assert.Equal(Rational.Zero, none.WinChance);  // without it nothing in the hand deals damage
        Assert.Equal(Rational.Zero, all.Held[0]);  // and the policy drinks it
        Assert.True(all.ExpectedL0 > none.ExpectedL0, $"{all.ExpectedL0} vs {none.ExpectedL0}");

        // At horizon 0 the empty subset's only move is the end turn, a leaf: L0 of the position, the unusable potion's credit
        // counted, above the same position on an empty belt.
        var flat = PotionCurve.Solve(s, rng, Opts with { MaxHorizon = 0 });
        var empty = s.Clone();
        Array.Clear(empty.PotionSlots);
        var emptyValue = SolverService.SolveExpected(empty, rng, Opts with { MaxHorizon = 0 }).Value;
        Assert.Equal(Utility.L0(s), flat[0].ExpectedL0);
        Assert.True(flat[0].ExpectedL0 > emptyValue, $"{flat[0].ExpectedL0} vs empty belt {emptyValue}");
        output.WriteLine($"horizon 0: empty subset L0 {flat[0].ExpectedL0.ToDouble():0.0000}, the same position on an empty belt {emptyValue.ToDouble():0.0000}");
    }

    [Fact]
    public void AZeroedNativeOptionsStructMasksNoPotion()
    {
        int fire = Potions.FindId("FirePotion")!.Value;
        var (s, rng) = PotionLethal([fire]);
        int handle = NativeSolverCalls.Adopt(s, rng);
        int solver = NativeSolverCalls.Create(handle, null);
        try
        {
            var zeroed = new SolveOptionsNative { MaxHorizon = 1 };  // every other field zero, as a C caller's memset leaves it
            var (status, r, _) = NativeSolverCalls.Expected(solver, handle, zeroed);
            var lib = SolverService.SolveExpected(s, rng, new SolveOptions { MaxHorizon = 1, Prune = Pruning.None, UseEstimate = false });
            Assert.Equal((int)lib.Status, status);
            Assert.Equal(-1, r.AllowedPotions);
            Assert.Equal(lib.Value.ToDouble(), r.Value);
            Assert.Equal(s.Hand.Count + 1, r.BestAction);  // the potion in slot 0

            var (_, masked, _) = NativeSolverCalls.Expected(solver, handle, zeroed with { ForbiddenPotions = 1 });
            Assert.Equal(~1, masked.AllowedPotions);
            Assert.True(masked.Value < r.Value, $"masked {masked.Value} vs {r.Value}");
            Assert.Equal(s.Hand.Count, masked.BestAction);  // the end turn: the potion is not usable
        }
        finally
        {
            NativeSolverCalls.Destroy(solver);
            NativeSolverCalls.DestroyCombat(handle);
        }
    }

    [Fact]
    public void TheNativeCurveWritesTheLibrarysCurve()
    {
        foreach (var (tag, s, rng, opts) in new[] { Fights().First() }.Select(f => (f.Tag, f.State, f.Rng, Opts))
            .Append(("cut", Fights().First().State, Fights().First().Rng, Opts with { NodeBudget = 1, MaxHorizon = 2 })))
        {
            var lib = PotionCurve.Solve(s, rng, opts);
            int handle = NativeSolverCalls.Adopt(s, rng);
            int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(opts));
            var (n, entries) = NativeSolverCalls.Curve(solver, handle, null);
            string error = NativeSolverCalls.LastError();
            NativeSolverCalls.Destroy(solver);
            NativeSolverCalls.DestroyCombat(handle);
            Assert.Equal(lib.Count, n);
            var held = PotionCurve.HeldSlots(s);
            for (int i = 0; i < n; i++)
            {
                bool refused = lib[i].Status == SolveStatus.Refused;
                Assert.Equal((int)lib[i].Status, entries[i].CurveStatus);
                Assert.Equal(PotionCurve.MaskOf(held, i), entries[i].AllowedPotions);
                Assert.Equal(refused ? double.NaN : lib[i].ExpectedL0.ToDouble(), entries[i].ExpectedL0);
                Assert.Equal(refused ? double.NaN : lib[i].WinChance.ToDouble(), entries[i].WinChance);
                for (int slot = 0; slot < 8; slot++)
                {
                    double got;
                    unsafe
                    {
                        var e = entries[i];
                        got = e.Held[slot];
                    }

                    if (!refused && lib[i].Held.TryGetValue(slot, out var p))
                    {
                        Assert.Equal(p.ToDouble(), got);
                    }
                    else
                    {
                        Assert.True(double.IsNaN(got), $"{tag} subset {i} slot {slot}");
                    }
                }
            }

            if (lib.FirstOrDefault(e => e.Status == SolveStatus.Refused) is { } first)
            {
                Assert.Equal($"subset {first.Subset}: {first.Reason}", error);
            }

            output.WriteLine($"{tag}: {string.Join(", ", lib.Select(e => $"{e.Status}@{e.Horizon}"))}");
            Assert.Equal(held.Count, lib[^1].Held.Count);
        }
    }
}
