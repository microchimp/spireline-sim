// Engine fault class 2 (docs/ENGINE_FAULTS.md): OutOfMemoryException at List.AddWithResize <- StepInProgress.OnTake <-
// CombatEngine.OnDrawPileTake <- CombatState.RemoveFromDrawPileAt <- CardEffects.DrawCards <- DrawUntilNonAttack (8 traces in
// the judge corpus's worker logs, every fight one with a card-play limit -- Sloth, Ringing or Normality -- and two in the
// first judgement of generation 14, one run written as lost). Pillage draws until it draws a non-Attack, the hand fills,
// both piles are empty or the fight ends. A Strike Hellraiser plays never reaches the hand; when the limit refuses every
// auto-play, the Strike is filed in the discard unplayed and drawn again after the next shuffle, and with only Strikes left
// in the piles nothing ever changes. The game's loop has the same exit test and hangs there; the copy drew until the step's
// record of its takes ran out of memory. It now stops with GameHangs, and the native entry point turns that into a counted
// refusal. A loop that does end is drawn to its end.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Tests.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.EngineFaults;

public class EndlessDrawTests
{
    private static (CombatState State, CountingRandom Rng, Sts2Emulator.Core2.Kernel? Kernel) Bound(Fight fight, bool core2)
    {
        var kernel = core2 ? SolverPositions.KernelFor("core2") : null;
        if (kernel is not null)
        {
            CardUids.NumberNew(fight.State);
            kernel.Bind(fight.State);
        }

        return (fight.State, new CountingRandom(7), kernel);
    }

    /// <summary>Pillage with a Defend in hand, Hellraiser, and only Strikes in the draw and discard piles.</summary>
    private static Fight PillageIntoStrikes() =>
        Fight.Hand(Card(IC.Pillage), Card(IC.DefendIronclad))
            .Energy(3)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Discard(Card(IC.PommelStrike))
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.Hellraiser, 1);

    [Theory]
    [InlineData(BuffId.Ringing, false)]
    [InlineData(BuffId.Ringing, true)]
    [InlineData(BuffId.Sloth, false)]
    [InlineData(BuffId.Sloth, true)]
    public void PillageWhoseEveryStrikeIsRefusedStopsAsAHangOfTheGame(BuffId limit, bool core2)
    {
        var (s, rng, kernel) = Bound(PillageIntoStrikes().PlayerBuff(limit, 1), core2);
        int takes = 0;
        CombatEngine.TakeWatch = (_, _) => takes++;
        try
        {
            var hang = Assert.Throws<GameHangs>(() => KernelGate.Step(kernel, s, rng, 0));
            Assert.Contains("Pillage draws forever", hang.Message);
        }
        finally
        {
            CombatEngine.TakeWatch = null;
        }

        // A whole pass over the three Strikes and the window, not the twenty-thousand-draw backstop: stopped at once.
        Assert.InRange(takes, 64, 200);
    }

    /// <summary>Without the limit the same Strikes are played and kill: the loop ends with the fight, and nothing is called a hang.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PillageWhoseStrikesArePlayedDrawsToTheFightsEnd(bool core2)
    {
        var (s, rng, kernel) = Bound(PillageIntoStrikes(), core2);
        s.Enemies[0].Hp = s.Enemies[0].MaxHp = 300;

        var result = KernelGate.Step(kernel, s, rng, 0);

        Assert.True(result.Terminal && result.PlayerWon, "fifty Strikes and a Pillage kill a 300 HP enemy");
    }
}
