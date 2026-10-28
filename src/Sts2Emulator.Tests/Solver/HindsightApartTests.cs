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

        // An unset stream: refused, whatever the flag says about the streams' origin.
        var unset = s.Clone();
        StateCloning.SetStream(unset, 0, null);
        var refused = SolverService.SolveHindsight(unset, rng, new HindsightOptions { AllowHindsight = true }, trueStreams: true);
        Assert.Equal(HindsightOutcome.Refused, refused.Outcome);
        Assert.StartsWith(SolverService.UnsetStreams, refused.Refusal);
        Assert.Null(refused.Bound);
        Assert.Null(refused.UpperBound);

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

    /// <summary>A run's first fight (the run form of EmuDump's solve): Sts2Run_Reset, then Sts2Run_StartCombat with the starting deck.</summary>
    internal static unsafe int RunInFight(string seed, int ascension, int hp, int rooms)
    {
        int run = RunNativeExports.Sts2Run_Create();
        int[] deck = [.. Sts2Emulator.Core.Run.RunConstants.StarterDeckIds];
        var obs = new int[Sts2Emulator.Core.Run.RunConstants.RunObsSize];
        byte[] s = System.Text.Encoding.UTF8.GetBytes(seed);
        fixed (byte* sp = s)
        fixed (int* d = deck)
        fixed (int* o = obs)
        {
            Assert.Equal(0, RunNativeExports.Sts2Run_Reset(run, sp, s.Length, ascension, o));
            Assert.Equal(0, RunNativeExports.Sts2Run_StartCombat(run, d, deck.Length, Sts2Emulator.Core.Run.RunConstants.SlitheringStranglerEncounterId,
                null, 0, hp, Sts2Emulator.Core.Run.RunConstants.StartingMaxHp, null, 0, 0, rooms, o));
        }

        return run;
    }

    internal static unsafe int Fork(int run, bool resample, int seed)
    {
        delegate* unmanaged<int, int, int, int*, int> fork = &NativeExports.Sts2Run_ForkCombat;
        var obs = new int[NativeExports.OBS_SIZE];
        fixed (int* o = obs)
        {
            return fork(run, resample ? 1 : 0, seed, o);
        }
    }

    [Fact]
    public unsafe void EachWayToMakeAHandleSetsItsStreamsFlagAndTheHindsightGateReadsIt()
    {
        (bool Own, int Status, string Error, double Bound) Probe(int handle)
        {
            Assert.True(NativeExports.TryGetCombat(handle, out _, out _, out bool own));
            int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(new SolveOptions(), allowHindsight: true));
            var (status, r) = NativeSolverCalls.Hindsight(solver, handle, 2_000);
            string error = status == -3 ? NativeSolverCalls.LastError() : "";
            NativeSolverCalls.Destroy(solver);
            return (own, status, error, r.BestUtility);
        }

        // Sts2_Create: the handle's own fight, with streams it leaves unset: refused as such, and no number in the bound.
        delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
        int created = create(5);
        var c = Probe(created);
        Assert.True(c.Own);
        Assert.Equal(-3, c.Status);
        Assert.StartsWith("hindsight-unset-streams", c.Error);
        Assert.True(double.IsNaN(c.Bound));

        // Sts2_LoadCombatSnapshot: the document's streams are invented.
        int loaded = NativeSolverCalls.Load(CombatSnapshot.Write(NativeExports.StateOf(created)!), 5);
        Assert.True(loaded >= 0, NativeSolverCalls.LastError());
        var l = Probe(loaded);
        Assert.False(l.Own);
        Assert.Equal(SolverService.NoTrueStreams, l.Error);

        // Sts2_Reset of the loaded handle deals a fight of the handle's own: the flag is true again (its streams unset as
        // Sts2_Create's are, so the bound is still refused, now for that).
        NativeSolverCalls.Reset(loaded);
        var r = Probe(loaded);
        Assert.True(r.Own);
        Assert.StartsWith("hindsight-unset-streams", r.Error);

        // Sts2_Clone: a faithful clone keeps the flag, a resampled one (a determinization) drops it.
        int faithful = NativeSolverCalls.Clone(created, resample: false, 0);
        int resampled = NativeSolverCalls.Clone(created, resample: true, 3);
        Assert.True(Probe(faithful).Own);
        Assert.False(Probe(resampled).Own);

        // Sts2Run_ForkCombat: as it stands, every stream set and the fight's own, so the bound is searched or cut, not
        // refused; with resampleHidden, a determinization.
        int run = RunInFight("TRUESTREAMS", 0, 64, 99);
        int asIs = Fork(run, resample: false, 0);
        int forked = Fork(run, resample: true, 9);
        Assert.True(asIs >= 0 && forked >= 0);
        Assert.Empty(SolverService.UnsetStreamsOf(NativeExports.StateOf(asIs)!));
        var f = Probe(asIs);
        Assert.True(f.Own);
        Assert.True(f.Status is 0 or 1, f.Error);
        Assert.True(f.Status == 0 || double.IsNaN(f.Bound), "a cut search exports no bound");
        var g = Probe(forked);
        Assert.False(g.Own);
        Assert.Equal(SolverService.NoTrueStreams, g.Error);

        foreach (int h in new[] { created, loaded, faithful, resampled, asIs, forked })
        {
            NativeSolverCalls.DestroyCombat(h);
        }

        RunNativeExports.Sts2Run_Destroy(run);
    }
}
