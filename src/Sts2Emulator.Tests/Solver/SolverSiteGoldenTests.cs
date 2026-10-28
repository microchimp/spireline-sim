// The site markers change no step (review of S4, finding 2): the engine's bytes after seeded random play through reshuffles,
// on the reshuffle position of SolverSiteTests and on harness positions, hashed and pinned at df1db6a6 (the fork's main
// before the markers replaced the stack walks). This file is self-contained so that the same code ran there. A step change
// anywhere in the engine moves the hash too: rerun this test at the commit the change is measured against, and pin anew
// only with the byte proof's verdict that the change is intended.
using System.Security.Cryptography;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class SolverSiteGoldenTests(ITestOutputHelper output)
{
    /// <summary>The hash of every position's bytes after each step, at df1db6a6.</summary>
    const string Golden = "F0BDE1604318DB91C502C0C26B634FB5BEBD8AD91A9CD51F244B86556B9C54D1";

    static CardInstance C(string name) => new(Cards.FindId(name)!.Value, false);

    /// <summary>SolverSiteTests' reshuffle position: one card left to draw, the discard and the hand to shuffle under it.</summary>
    static (CombatState State, CountingRandom Rng) Reshuffle()
    {
        var state = Fight
            .Hand(C("StrikeIronclad"), C("StrikeIronclad"), C("DefendIronclad"), C("Bash"), C("StrikeIronclad"))
            .Energy(3)
            .Draw(C("DefendIronclad"))
            .Discard(C("StrikeIronclad"), C("DefendIronclad"), C("DefendIronclad"))
            .State;
        foreach (var pile in new[] { state.Hand, state.DrawPile, state.DiscardPile })
        {
            for (int i = 0; i < pile.Count; i++)
            {
                pile[i] = pile[i].Uid == 0 ? state.Fresh(pile[i]) : pile[i];
            }
        }

        state.KnownTopCount = 0;
        state.KnownBottomCount = 0;
        foreach (var e in state.Enemies)
        {
            e.Hp = Math.Min(e.Hp, 20);
        }

        SolverFights.SetStreams(state, 41);
        return (state, new CountingRandom(43));
    }

    /// <summary>Up to <paramref name="steps" /> seeded random legal moves, each position's bytes after its step appended.</summary>
    static void Play(StringBuilder trace, CombatState s, CountingRandom rng, int seed, int steps)
    {
        var policy = new Random(seed);
        for (int i = 0; i < steps; i++)
        {
            var moves = Actions.Legal(s, Pruning.None);
            if (moves.Count == 0)
            {
                break;
            }

            var r = Successors.Step(s, rng, moves[policy.Next(moves.Count)]);
            trace.Append(SolverFights.Bytes(s)).Append('\n');
            if (r.Terminal)
            {
                break;
            }
        }
    }

    [Fact]
    public void TheEngineStepsAsAtTheCommitBeforeTheMarkers()
    {
        var trace = new StringBuilder();
        var (r, rr) = Reshuffle();
        Play(trace, r, rr, 7, 60);
        var corpus = SolverPositions.Corpus();
        int built = 0;
        foreach (var recipe in SolverPositions.Recipes().Where(x => x.Id % 23 == 0 && x.Macro.Length == 0))
        {
            if (SolverPositions.Build(recipe, corpus) is { } pos)
            {
                built++;
                Play(trace, pos.State, pos.Rng, recipe.Id, 40);
            }
        }

        string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(trace.ToString())));
        output.WriteLine($"golden: {built} harness positions and the reshuffle, {trace.Length} chars, sha256 {hash}");
        Assert.True(built >= 15, $"only {built} harness positions built");
        Assert.Equal(Golden, hash);
    }
}
