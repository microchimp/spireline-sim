using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Solver;

/// <summary>
/// The chance hook across a screen (FIGHT_SOLVER.md section 4): a copy of a stream carries no source,
/// and a step a source steered to a screen replays its answers when the screen is answered, after the
/// source is detached, on the combat's own streams.
/// </summary>
public class ChanceReplayTests
{
    [Fact]
    public void ACopyCarriesNoSourceAndAStepStartKeepsNone()
    {
        var (state, rng) = AcrobaticsOnAnEmptyPile();
        var steer = new Steer();
        ChanceHook.Attach(state, rng, steer);
        Assert.Null(rng.Copy().Chance);
        Assert.All(Streams(state.Clone()), s => Assert.Null(s.Chance));

        CombatEngine.Step(state, 0, rng);
        var stopped = state.PendingSelection?.Replay;
        Assert.NotNull(stopped);
        Assert.Null(stopped!.RngBefore!.Chance);
        Assert.All(Streams(stopped.Before), s => Assert.Null(s.Chance));
        Assert.NotEmpty(stopped.Chance);
        Assert.All(Streams(state), s => Assert.Same(steer, s.Chance));
    }

    [Fact]
    public void AScreenReachedByASteeredStepIsAnsweredOnTheSameStreamsAfterTheSourceIsDetached()
    {
        // Reference: the source stays attached through the answer.
        var (reference, refRng) = AcrobaticsOnAnEmptyPile();
        ChanceHook.Attach(reference, refRng, new Steer());
        CombatEngine.Step(reference, 0, refRng);
        var screenHand = reference.Hand.Select(c => c.DefId).ToList();
        CombatEngine.Step(reference, 0, refRng);

        // The unhooked step draws another hand, so a replay that forgot the answers would differ.
        var (plain, plainRng) = AcrobaticsOnAnEmptyPile();
        CombatEngine.Step(plain, 0, plainRng);
        Assert.NotEqual(screenHand, plain.Hand.Select(c => c.DefId).ToList());

        var (state, rng) = AcrobaticsOnAnEmptyPile();
        var steer = new Steer();
        ChanceHook.Attach(state, rng, steer);
        var shuffle = state.ShuffleRng!;
        CombatEngine.Step(state, 0, rng);
        Assert.Equal(screenHand, state.Hand.Select(c => c.DefId).ToList());
        Assert.True(steer.Asked > 0);
        _ = SolverFights.Bytes(state);  // the snapshot's replay probe follows the answers too

        ChanceHook.Attach(state, rng, null);
        steer.Dead = true;
        CombatEngine.Step(state, 0, rng);

        Assert.Null(state.PendingSelection);
        Assert.Equal(SolverFights.Bytes(reference), SolverFights.Bytes(state));
        Assert.Equal(Positions(reference, refRng), Positions(state, rng));
        Assert.Same(shuffle, state.ShuffleRng);
        Assert.All(Streams(state), s => Assert.Null(s.Chance));
        Assert.Null(rng.Chance);
    }

    [Fact]
    public void EveryOutcomeOfAStepThatStopsOnAScreenIsAnsweredAfterDetachWithoutAskingTheSourceAgain()
    {
        var (start, startRng) = AcrobaticsOnAnEmptyPile();
        var recorder = new ChanceRecorder();
        var (probe, probeRng) = (start.Clone(), startRng.Copy());
        ChanceHook.Attach(probe, probeRng, recorder);
        CombatEngine.Step(probe, 0, probeRng);
        long expected = recorder.Calls.Aggregate(1L, (n, c) => n * c.Call.Width);

        var leaves = ChanceTree.Enumerate(source =>
        {
            var copy = start.Clone();
            var r = startRng.Copy();
            ChanceHook.Attach(copy, r, source);
            CombatEngine.Step(copy, 0, r);
            Assert.NotNull(copy.PendingSelection);
            ChanceHook.Attach(copy, r, null);
            CombatEngine.Step(copy, 0, r);
            return SolverFights.Bytes(copy);
        }, maxWidth: 32, maxLeaves: 10_000);

        Assert.Equal(expected, leaves.Count);
        Assert.Equal(1d, leaves.Sum(l => l.P), 9);
        Assert.True(leaves.Select(l => l.Result).Distinct().Count() > 1);
    }

    [Fact]
    public void AnEmptyRangeThrowsHookedAsItDoesUnhookedAndNeverReachesTheSource()
    {
        var steer = new Steer();
        var hooked = new CountingRandom(3);
        hooked.SetChance(steer, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => new CountingRandom(3).Next(5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => hooked.Next(5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => hooked.Next(0));
        Assert.Equal(0, steer.Asked);

        var call = new ChanceCall(0, 0, ChanceKind.Next, 4, 4, null, null);
        Assert.Equal("empty-range", Assert.Throws<ChanceRefused>(() => call.Outcomes().ToList()).Reason);
    }

    [Fact]
    public void ACombatWithAnUnsetStreamRefusesASource()
    {
        var (state, rng) = AcrobaticsOnAnEmptyPile();
        state.NicheHpRng = null;
        Assert.Equal("null-stream", Assert.Throws<ChanceRefused>(() => ChanceHook.Attach(state, rng, new Steer())).Reason);
        Assert.Equal("null-stream", Assert.Throws<ChanceRefused>(() => ChanceHook.Attach(SolverFights.Encounter(default, 1).State, new Random(1), new Steer())).Reason);
        ChanceHook.Attach(state, rng, null);  // detaching never refuses
    }

    [Fact]
    public void AnUnhookedWeightedRollAllocatesNothing()
    {
        var rng = new CountingRandom(9);
        var branches = new List<(int Move, float Weight)> { (0, 1f), (1, 0f), (2, 2.5f) };
        double[] weights = [0.4, 0.6];
        int Rolls(int n)
        {
            int sink = 0;
            for (int i = 0; i < n; i++)
            {
                sink += Roll.Walk(rng, "test", branches, static b => b.Weight);
                sink += Roll.Branch(rng, "test", weights, static d => d <= 0.4 ? 0 : 1);
                sink += rng.Next(7);
            }

            return sink;
        }

        // The first run warms the code up (tiering and on-stack replacement allocate once); the
        // second is measured.
        Assert.True(Rolls(20_000) > 0);
        long before = GC.GetAllocatedBytesForCurrentThread();
        int sink = Rolls(20_000);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(sink > 0);
        Assert.Equal(0, allocated);
    }

    /// <summary>Acrobatics in hand, the draw pile empty and five cards in the discard: the play shuffles (Shuffle stream) and then asks for a discard.</summary>
    private static (CombatState State, CountingRandom Rng) AcrobaticsOnAnEmptyPile()
    {
        var state = Fight
            .Hand(new CardInstance(SI.Acrobatics, false), new CardInstance(IC.DefendIronclad, false))
            .Energy(3)
            .Draw()
            .Discard(
                new CardInstance(IC.StrikeIronclad, false),
                new CardInstance(IC.DefendIronclad, false),
                new CardInstance(IC.Bash, false),
                new CardInstance(IC.AscendersBane, false),
                new CardInstance(IC.StrikeIronclad, true)
            )
            .State;
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(state, i) is null)
            {
                StateCloning.SetStream(state, i, new CountingRandom(100 + i));
            }
        }

        return (state, new CountingRandom(7));
    }

    private static IEnumerable<CountingRandom> Streams(CombatState state) =>
        Enumerable.Range(0, StateCloning.StreamCount).Select(i => StateCloning.Stream(state, i)!);

    private static Dictionary<int, int> Positions(CombatState state, CountingRandom rng)
    {
        var at = new Dictionary<int, int> { [-1] = rng.CallCount };
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            at[i] = StateCloning.Stream(state, i)?.CallCount ?? 0;
        }

        return at;
    }

    /// <summary>Answers each uniform call with its top value (a Fisher-Yates that swaps nothing); asked once dead, it fails the test.</summary>
    private sealed class Steer : IChanceSource
    {
        public bool Dead;

        public int Asked;

        public int? Answer(in ChanceCall call)
        {
            Assert.False(Dead, $"a detached source was asked {call}");
            Asked++;
            return call.Kind == ChanceKind.Next ? call.Max - 1 : null;
        }
    }
}
