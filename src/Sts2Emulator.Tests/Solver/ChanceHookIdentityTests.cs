using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

/// <summary>
/// The chance hook (FIGHT_SOLVER.md section 4): every random call a step makes reaches the source,
/// and a source that lets the streams answer changes nothing.
/// </summary>
public class ChanceHookIdentityTests
{
    [Fact]
    public void AnObservingSourceSeesEveryCallOfSampleFightsAndChangesNoByte()
    {
        int fights = 0, calls = 0;
        foreach (var encounter in SolverFights.Sample(stride: 5))
        {
            for (int seed = 1; seed <= 2; seed++)
            {
                var (plain, plainRng) = SolverFights.Encounter(encounter, seed);
                var hooked = plain.Clone();
                var hookedRng = plainRng.Copy();
                var recorder = new ChanceRecorder();
                ChanceHook.Attach(hooked, hookedRng, recorder);
                var start = Positions(hooked, hookedRng);

                var policy = new Random(seed);
                for (int step = 0; step < 80; step++)
                {
                    int action = SolverFights.RandomAction(plain, policy);
                    var a = CombatEngine.Step(plain, action, plainRng);
                    var b = CombatEngine.Step(hooked, action, hookedRng);
                    Assert.Equal(a, b);
                    Assert.Equal(SolverFights.Bytes(plain), SolverFights.Bytes(hooked));
                    Assert.Equal(Positions(plain, plainRng), Positions(hooked, hookedRng));
                    if (a.Terminal)
                    {
                        break;
                    }
                }

                // Every index each stream moved through was put to the source (a replayed step
                // puts some twice), and nothing past where the stream stands.
                var end = Positions(hooked, hookedRng);
                foreach (var (stream, from) in start)
                {
                    var seen = recorder.Calls.Where(c => c.Call.Stream == stream).Select(c => c.Call.Index).ToHashSet();
                    for (int i = from; i < end[stream]; i++)
                    {
                        Assert.True(seen.Contains(i), $"{encounter} seed {seed}: stream {stream} call {i} was not seen by the hook");
                    }

                    Assert.All(seen, i => Assert.True(i < end[stream]));
                }

                calls += recorder.Calls.Count;
                fights++;
            }
        }

        Assert.True(fights >= 20, $"{fights} fights");
        Assert.True(calls > 1000, $"{calls} calls");
    }

    [Fact]
    public void AnUnhookedStreamDrawsAsItDidAndAHookedOneMovesTheSame()
    {
        var plain = new CountingRandom(42);
        var hooked = new CountingRandom(42);
        hooked.SetChance(new ChanceReplay([3, 1, 0]), 7);
        plain.Next(10);
        plain.Next(2, 9);
        Roll.Branch(plain, "test", [1, 1], static d => d < 0.5 ? 0 : 1);
        Assert.Equal(3, hooked.Next(10));
        Assert.Equal(1, hooked.Next(2, 9));
        Assert.Equal(0, Roll.Branch(hooked, "test", [1, 1], static _ => 1));
        Assert.Equal(plain.CallCount, hooked.CallCount);
        Assert.Equal(plain.Lanes, hooked.Lanes);
        Assert.Throws<ChancePending>(() => hooked.Next(4));
    }

    [Fact]
    public void ARawFloatRollRefusesAndAWeightedOneBranchesAtItsOdds()
    {
        var rng = new CountingRandom(1);
        rng.SetChance(new ChanceReplay([]), 7);
        var refused = Assert.Throws<ChanceRefused>(() => rng.NextDouble());
        Assert.Equal("unlisted-roll", refused.Reason);

        var leaves = ChanceTree.Enumerate(source =>
        {
            var r = new CountingRandom(1);
            r.SetChance(source, 7);
            return Roll.Walk(r, "test", [1f / 12f, 0f, 1f / 12f, 0.75f]);
        });
        Assert.Equal([0, 2, 3], leaves.Select(l => l.Result).Order());
        Assert.Equal(1d / 11d, leaves.Single(l => l.Result == 0).P, 6);
        Assert.Equal(9d / 11d, leaves.Single(l => l.Result == 3).P, 6);
    }

    [Fact]
    public void EveryOutcomeOfAStepIsAReplayOfItsAnswersAndTheOddsSumToOne()
    {
        int trees = 0;
        foreach (var encounter in SolverFights.Sample(stride: 9))
        {
            var (state, rng) = SolverFights.Encounter(encounter, 5);
            int endTurn = state.Hand.Count;
            List<ChanceLeaf<string>> leaves;
            try
            {
                leaves = ChanceTree.Enumerate(source => Stepped(state, rng, endTurn, source), maxWidth: 32, maxLeaves: 4000);
            }
            catch (ChanceRefused)
            {
                continue;  // a roll too wide for the cap or with no odds: refused, never guessed
            }

            Assert.Equal(1d, leaves.Sum(l => l.P), 9);
            foreach (var leaf in leaves.Take(20))
            {
                Assert.Equal(leaf.Result, Stepped(state, rng, endTurn, new ChanceReplay(leaf.Answers)));
            }

            trees++;
        }

        Assert.True(trees >= 5, $"{trees} steps enumerated");
    }

    private static string Stepped(CombatState state, CountingRandom rng, int action, IChanceSource source)
    {
        var copy = state.Clone();
        var r = rng.Copy();
        ChanceHook.Attach(copy, r, source);
        CombatEngine.Step(copy, action, r);
        return SolverFights.Bytes(copy);
    }

    private static Dictionary<int, int> Positions(CombatState state, CountingRandom rng)
    {
        var at = new Dictionary<int, int> { [-1] = rng.CallCount };
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            at[i] = StateCloning.Stream(state, i)?.CallCount ?? 0;
        }

        return at;
    }
}
