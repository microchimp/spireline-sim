using Sts2Emulator.Core;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

/// <summary>
/// The enemy move odds (FIGHT_SOLVER.md section 4): equal to the game's branch tables as its
/// state machines declare them, and to the engine's own frequencies over many seeded rolls.
/// </summary>
public class EnemyOddsTests
{
    private const int Samples = 3000;

    // The game's tables, read from each monster's move state machine (RandomBranchState branches:
    // base weight, repeat rule, cooldown) and written here as the odds they give from the named
    // history. Moves are the emulator's LastMove numbering for that monster.
    public static TheoryData<string, int, int, bool, int[], double[]> GameTables() =>
        new()
        {
            // Mawler: rip-and-tear (cannot repeat), roar (use only once), claw (cannot repeat), all weight 1.
            { "Mawler after claw", KE.Mawler, 2, false, [0, 1], [0.5, 0.5] },
            { "Mawler after roar", KE.Mawler, 1, true, [0, 2], [0.5, 0.5] },
            { "Mawler after rip, roar spent", KE.Mawler, 0, true, [2], [1] },
            // Fogmog: after swipe, swipe-random 0.4 and headbutt 0.6, both cannot repeat.
            { "Fogmog after swipe", KE.Fogmog, 1, false, [2, 3], [0.4, 0.6] },
            // Leaf Slime S: tackle and goop, cannot repeat, weight 1; the empty log allows both.
            { "LeafSlimeS opening", KE.LeafSlimeS, -1, false, [0, 1], [0.5, 0.5] },
            { "LeafSlimeS after tackle", KE.LeafSlimeS, 0, false, [1], [1] },
            // Two-Tailed Rat able to summon: scratch, bite, screech a twelfth each (cannot repeat,
            // screech with a cooldown of 3), call-for-backup 0.75 once.
            { "TwoTailedRat after scratch", KE.TwoTailedRat, 0, false, [1, 2, 3], [1d / 11, 1d / 11, 9d / 11] },
            { "TwoTailedRat after backup", KE.TwoTailedRat, 3, true, [0, 1, 2], [1d / 3, 1d / 3, 1d / 3] },
        };

    [Theory]
    [MemberData(nameof(GameTables))]
    public void TheOddsAreTheGameBranchTable(string name, int defId, int lastMove, bool onceUsed, int[] moves, double[] odds)
    {
        var roster = new List<EnemyState> { Enemy(defId, lastMove, onceUsed) };
        var got = EnemyOdds.Next(roster, 0, turn: 3, Ascension.DefaultLevel, e => e.LastMove).OrderBy(o => o.Move).ToList();
        Assert.True(moves.SequenceEqual(got.Select(o => o.Move)), $"{name}: moves {string.Join(",", got.Select(o => o.Move))}");
        for (int i = 0; i < moves.Length; i++)
        {
            Assert.Equal(odds[i], got[i].P, 6);
        }

        AssertMatchesSampling(name, roster, turn: 3);
    }

    [Fact]
    public void TheOddsMatchTheEngineRollsOverSampleFightsWithTheShownIntentFixed()
    {
        int checkedTurns = 0, branching = 0;
        foreach (var encounter in SolverFights.Sample(stride: 3))
        {
            var (state, rng) = SolverFights.Encounter(encounter, 11);
            var policy = new Random(11);
            for (int turn = 0; turn < 4 && !CombatEngine.IsOverOrEnding(state); turn++)
            {
                var shown = state.Enemies.Select(e => e.CurrentIntent).ToList();
                int endTurn = state.Hand.Count;
                List<ChanceLeaf<CombatState>> leaves;
                try
                {
                    leaves = EnemyOdds.AfterStep(state, rng, (s, r) => CombatEngine.Step(s, endTurn, r));
                }
                catch (ChanceRefused e) when (e.Reason == "unlisted-roll")
                {
                    Assert.Fail($"{encounter}: an enemy roll with no odds: {e.Message}");
                    return;
                }

                if (state.PendingSelection is null && leaves.All(l => l.Result.PendingSelection is null))
                {
                    var exact = new Dictionary<string, double>();
                    foreach (var leaf in leaves)
                    {
                        string key = Joint(leaf.Result);
                        exact[key] = exact.GetValueOrDefault(key) + leaf.P;
                    }

                    var counts = new Dictionary<string, int>();
                    for (int seed = 1; seed <= Samples; seed++)
                    {
                        var copy = state.Clone();
                        copy.AiRng = new CountingRandom(seed * 7 + 1);
                        CombatEngine.Step(copy, endTurn, rng.Copy());
                        string key = Joint(copy);
                        counts[key] = counts.GetValueOrDefault(key) + 1;
                    }

                    AssertClose($"{encounter} turn {turn}", exact, counts);
                    checkedTurns++;
                    branching += exact.Count > 1 ? 1 : 0;
                }

                // The shown intents were performed, not rolled: the state handed in is untouched.
                Assert.Equal(shown, state.Enemies.Select(e => e.CurrentIntent));

                // On to the next turn on the real streams.
                CombatEngine.Step(state, endTurn, rng);
                while (state.PendingSelection is not null && !CombatEngine.IsOverOrEnding(state))
                {
                    CombatEngine.Step(state, SolverFights.RandomAction(state, policy), rng);
                }
            }
        }

        Assert.True(checkedTurns >= 40, $"{checkedTurns} turns checked");
        Assert.True(branching >= 10, $"{branching} turns with more than one outcome");
    }

    private static EnemyState Enemy(int defId, int lastMove, bool onceUsed) =>
        new()
        {
            DefId = defId,
            Hp = 60,
            MaxHp = 60,
            Buffs = [],
            MoveIndex = 3,
            LastMove = lastMove,
            OnceOnlyMoveUsed = onceUsed,
            MoveHistory = lastMove >= 0 ? [lastMove] : [],
        };

    private static void AssertMatchesSampling(string name, List<EnemyState> roster, int turn)
    {
        var exact = new Dictionary<string, double>();
        foreach (var leaf in EnemyOdds.Pass(roster, turn, Ascension.DefaultLevel))
        {
            string key = string.Join(";", leaf.Result.Select(e => EnemyOdds.MoveOf(e).ToString()));
            exact[key] = exact.GetValueOrDefault(key) + leaf.P;
        }

        var counts = new Dictionary<string, int>();
        for (int seed = 1; seed <= Samples; seed++)
        {
            var copy = roster.Select(e => e.Clone()).ToList();
            var ai = new CountingRandom(seed * 7 + 1);
            EnemyAI.ChooseIntents(copy, turn, ai, ai, Ascension.DefaultLevel);
            string key = string.Join(";", copy.Select(e => EnemyOdds.MoveOf(e).ToString()));
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        AssertClose(name, exact, counts);
    }

    private static string Joint(CombatState state) =>
        string.Join(";", state.Enemies.Select(e => $"{e.DefId}:{EnemyOdds.MoveOf(e)}"));

    /// <summary>Every sampled outcome is in the odds' support, and each frequency is within five standard errors.</summary>
    private static void AssertClose(string what, Dictionary<string, double> exact, Dictionary<string, int> counts)
    {
        int n = counts.Values.Sum();
        Assert.Equal(1d, exact.Values.Sum(), 9);
        foreach (var key in counts.Keys)
        {
            Assert.True(exact.ContainsKey(key), $"{what}: sampled {key}, which the odds give no chance");
        }

        foreach (var (key, p) in exact)
        {
            double f = counts.GetValueOrDefault(key) / (double)n;
            double tolerance = 5 * Math.Sqrt(p * (1 - p) / n) + 1.0 / n;
            Assert.True(Math.Abs(f - p) <= tolerance, $"{what}: {key} sampled {f:F4}, odds {p:F4}");
        }
    }
}
