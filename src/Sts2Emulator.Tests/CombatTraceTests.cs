using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The observation points tools/emudump's trace mode pairs with the referee's trace of the game
/// (CombatTrace): they fire where the game's own trace marks its combat, see every draw of the
/// combat streams, and change nothing.
/// </summary>
public class CombatTraceTests
{
    private static CombatState Fight()
    {
        var state = CombatFactory.NewCombat(seed: 3);
        state.Energy = 3;
        state.Hand = [new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.DefendIronclad, false)];
        return state;
    }

    private static CountingRandom?[] Streams(CombatState s) =>
    [
        s.ShuffleRng, s.TargetRng, s.AiRng as CountingRandom, s.CardSelectionRng, s.CardGenerationRng,
        s.EnergyCostRng, s.PotionGenerationRng, s.OrbGenerationRng, s.NicheHpRng,
    ];

    private static void PlayAndEndTurn(CombatState state)
    {
        CombatEngine.Step(state, 0, new Random(0));
        CombatEngine.Step(state, state.Hand.Count, new Random(0));
    }

    [Fact]
    public void Marks_FollowThePlayAndTheTurnInTheGamesOrder()
    {
        var state = Fight();
        int living = state.Enemies.Count(e => e.Hp > 0);
        var marks = new List<string>();
        CombatTrace.Mark = (what, _, _) => marks.Add(what);
        try
        {
            PlayAndEndTurn(state);
        }
        finally
        {
            CombatTrace.Mark = null;
        }

        Assert.Equal(["play", "played"], marks.Take(2));
        int turnEnd = marks.IndexOf("turn_end"), enemyTurn = marks.IndexOf("enemy_turn"), turnStart = marks.IndexOf("turn_start");
        Assert.True(turnEnd == 2 && enemyTurn > turnEnd && turnStart > enemyTurn, string.Join(",", marks));
        var moves = marks.Skip(enemyTurn).Take(turnStart - enemyTurn).ToList();
        Assert.Equal(living, moves.Count(m => m == "move"));
        Assert.Equal(living, moves.Count(m => m == "moved"));
    }

    [Fact]
    public void RngCall_SeesEveryDrawOfTheCombatStreams()
    {
        var state = Fight();
        // The next hand draw has to shuffle the discard pile back in.
        state.DiscardPile = [.. state.DrawPile];
        state.DrawPile = [];
        state.ShuffleRng ??= new CountingRandom(5);
        state.AiRng ??= new CountingRandom(6);
        var streams = Streams(state).Where(r => r != null).Cast<CountingRandom>().ToList();
        int before = streams.Sum(r => r.CallCount);
        int seen = 0;
        CombatTrace.RngCall = r => seen += streams.Any(s => ReferenceEquals(s, r)) ? 1 : 0;
        try
        {
            PlayAndEndTurn(state);
        }
        finally
        {
            CombatTrace.RngCall = null;
        }

        Assert.True(seen > 0);
        Assert.Equal(streams.Sum(r => r.CallCount) - before, seen);
    }

    [Fact]
    public void Probes_ChangeNothing()
    {
        var plain = Fight();
        var traced = Fight();
        PlayAndEndTurn(plain);
        CombatTrace.Mark = (_, _, s) => CombatSnapshot.Write(s);
        CombatTrace.RngCall = r => _ = r.CallCount;
        try
        {
            PlayAndEndTurn(traced);
        }
        finally
        {
            CombatTrace.Mark = null;
            CombatTrace.RngCall = null;
        }

        Assert.Equal(CombatSnapshot.Write(plain), CombatSnapshot.Write(traced));
    }
}
