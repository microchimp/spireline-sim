// The two outputs kept apart (docs/rebuild/FIGHT_SOLVER.md section 4a, piece S4): the expected result and the hindsight
// bound are separate types that share no field, in C# and across the native boundary, and the hindsight search refuses
// without its explicit flag, and on a handle whose streams are not the record's (a resampled fork). The owner's rule: perfect
// play is best in expectation at decision time; hindsight is an upper bound for attribution only, never a label, regret or
// grade.
using System.Reflection;
using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class HindsightApartTests
{
    static HashSet<string> Members(Type t) =>
        [.. t.GetProperties(BindingFlags.Public | BindingFlags.Instance).Select(p => p.Name)
            .Concat(t.GetFields(BindingFlags.Public | BindingFlags.Instance).Select(f => f.Name))];

    [Fact]
    public void TheResultTypesShareNoField()
    {
        foreach (var (a, b) in new[] { (typeof(ExpectedResult), typeof(HindsightResult)), (typeof(ExpectedResultNative), typeof(HindsightResultNative)) })
        {
            var shared = Members(a).Intersect(Members(b)).ToList();
            Assert.True(shared.Count == 0, $"{a.Name} and {b.Name} share {string.Join(", ", shared)}");
            Assert.False(a.IsAssignableFrom(b));
            Assert.False(b.IsAssignableFrom(a));
            Assert.Equal(a.BaseType, b.BaseType);  // object (records) or ValueType (structs): no common base of their own
        }

        // No method of the expected side returns or takes a hindsight result, and the other way round.
        foreach (var t in new[] { typeof(Expectimax), typeof(ExpectedResult) })
        {
            foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(HindsightResult), m.ReturnType);
                Assert.DoesNotContain(m.GetParameters(), p => p.ParameterType == typeof(HindsightResult));
            }
        }
    }

    static (CombatState State, CountingRandom Rng) SmallFight(int seed)
    {
        var (s, rng) = SolverPositions.Fight("old", seed, TestDeck.StarterDeckIds.ToArray(), [], []);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 6;
        }

        return (s, rng);
    }

    [Fact]
    public void HindsightRefusesWithoutItsFlag()
    {
        var (s, rng) = SmallFight(31);
        var off = new Hindsight(new HindsightOptions()).Solve(s, rng);
        Assert.Equal(HindsightOutcome.Refused, off.Outcome);
        Assert.Equal(Hindsight.NotAllowed, off.Refusal);
        Assert.Equal(WinLine.Unknown, off.WinLineExists);
        Assert.Empty(off.Line);
        Assert.Equal(HindsightOutcome.Refused, SolverService.SolveHindsight(s, rng, new HindsightOptions(), trueStreams: true).Outcome);

        var noStreams = SolverService.SolveHindsight(s, rng, new HindsightOptions { AllowHindsight = true }, trueStreams: false);
        Assert.Equal(SolverService.NoTrueStreams, noStreams.Refusal);

        var on = SolverService.SolveHindsight(s, rng, new HindsightOptions { AllowHindsight = true, SearchNodeBudget = 50_000 }, trueStreams: true);
        Assert.NotEqual(HindsightOutcome.Refused, on.Outcome);
        Assert.Equal(WinLine.Yes, on.WinLineExists);  // six HP each: a Strike line kills
        Assert.NotEmpty(on.Line);
    }

    [Fact]
    public void TheNativeHindsightRefusesWithoutTheFlagAndOnAResampledFork()
    {
        var (s, rng) = SmallFight(32);
        int handle = NativeSolverCalls.Adopt(s, rng, trueStreams: true);
        int plain = NativeSolverCalls.Create(handle, null);
        int unflagged = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(new SolveOptions()));
        int flagged = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(new SolveOptions(), allowHindsight: true));
        try
        {
            foreach (int solver in new[] { plain, unflagged })
            {
                var (status, r) = NativeSolverCalls.Hindsight(solver, handle);
                Assert.Equal(-3, status);
                Assert.Equal(-3, r.Outcome);
                Assert.Equal(Hindsight.NotAllowed, NativeSolverCalls.LastError());
            }

            var (ok, bound) = NativeSolverCalls.Hindsight(flagged, handle, 50_000);
            Assert.True(ok is 0 or 1, $"status {ok}");
            Assert.Equal(1, bound.WinLineExists);
            Assert.True(bound.LineLength > 0);

            int faithful = NativeSolverCalls.Clone(handle, resample: false, 0);
            int resampled = NativeSolverCalls.Clone(handle, resample: true, 77);
            Assert.True(NativeSolverCalls.Hindsight(flagged, faithful, 50_000).Status >= 0);
            Assert.Equal(-3, NativeSolverCalls.Hindsight(flagged, resampled, 50_000).Status);
            Assert.Equal(SolverService.NoTrueStreams, NativeSolverCalls.LastError());
            // The expected solve reads no stream: the resampled fork is solved like the faithful one.
            Assert.Equal(NativeSolverCalls.Expected(plain, faithful, null).Result.Value, NativeSolverCalls.Expected(plain, resampled, null).Result.Value);
            NativeSolverCalls.DestroyCombat(faithful);
            NativeSolverCalls.DestroyCombat(resampled);
        }
        finally
        {
            foreach (int solver in new[] { plain, unflagged, flagged })
            {
                NativeSolverCalls.Destroy(solver);
            }

            NativeSolverCalls.DestroyCombat(handle);
        }
    }

    [Fact]
    public void ADealtFightCarriesItsOwnStreamsAndAResetKeepsThem()
    {
        var (s, rng) = NativeExports.DealFight(5, Ascension.DefaultLevel, null, [], 0);
        Assert.NotNull(s);
        int handle = NativeSolverCalls.Adopt(s, rng, trueStreams: true);
        Assert.True(NativeExports.TryGetCombat(handle, out _, out _, out bool own));
        Assert.True(own);
        int resampled = NativeSolverCalls.Clone(handle, resample: true, 3);
        Assert.True(NativeExports.TryGetCombat(resampled, out _, out _, out bool forked));
        Assert.False(forked);
        NativeSolverCalls.DestroyCombat(resampled);
        NativeSolverCalls.DestroyCombat(handle);
    }
}
