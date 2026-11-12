// Engine fault class 1 (docs/ENGINE_FAULTS.md): ArgumentOutOfRangeException in CardEffects.DrawCards (the teacher's round 0,
// two runs: through CombatEngine.EndTurn and through RelicEffects.ApplyAfterHandEmptied). A Strike Hellraiser plays out of a
// draw is filed on its result pile (the top of the draw pile under Nostalgia), and the draw then looked for it by position
// (the top of the draw pile, the end of another pile) to finish its draw hooks. The play's own empty-hand check could move
// it first: the copy's Unceasing Top drew after every play, nested or not, and in the turn's start, and its draw took the
// Strike back off the draw pile, leaving the pile empty where the draw looked. The game checks the empty hand only when no
// other card's or potion's effect is running, Unceasing Top draws only in the AutoPrePlay, Play and AutoPostPlay phases,
// and the game's later draw hooks act on the card itself wherever it is: so does the copy now.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Tests.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.EngineFaults;

public class DrawnCardWhereaboutsTests
{
    /// <summary>Hellraiser and Nostalgia (the first Attack or Skill a turn goes on top of the draw pile), Unceasing Top.</summary>
    private static (CombatState State, CountingRandom Rng, Sts2Emulator.Core2.Kernel? Kernel) Setup(Fight fight, bool core2)
    {
        fight.PlayerBuff(BuffId.Hellraiser, 1).PlayerBuff(BuffId.Nostalgia, 1).PlayerHp(500);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.UnceasingTop));
        var kernel = core2 ? SolverPositions.KernelFor("core2") : null;
        if (kernel is not null)
        {
            CardUids.NumberNew(fight.State);
            kernel.Bind(fight.State);
        }

        return (fight.State, new CountingRandom(5), kernel);
    }

    private static int Enemy(CombatState s) => s.Enemies.Where(e => e.Hp > 0).Sum(e => e.Hp);

    /// <summary>
    /// The last card in hand played, the hand empty: Unceasing Top draws the Strike, Hellraiser plays it onto the top of
    /// the draw pile, and that play, its effect over, finds the hand empty and draws again -- the game's chain, which ends
    /// when the Strikes kill. The first draw then looked at the top of a pile the chain had emptied.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AStrikePlayedOutOfUnceasingTopsDrawIsFoundWhereverItWent(bool core2)
    {
        var (s, rng, kernel) = Setup(Fight.Hand(Card(IC.Inflame)).Energy(3).Draw(Card(IC.StrikeIronclad)).Enemy(hp: 40), core2);

        var result = KernelGate.Step(kernel, s, rng, 0);

        Assert.True(result.Terminal && result.PlayerWon, "the Strikes' chain kills the 40 HP enemy");
    }

    /// <summary>
    /// The turn's hand draw (the game's Start phase) meets a Strike five times: Hellraiser plays each, the first onto the
    /// top of the pile (Nostalgia), and Unceasing Top draws nothing there. Five Strikes hit; the copy's relic drew inside
    /// the hand draw, and the draw looked at the top of an empty pile.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnceasingTopDoesNotDrawInTheTurnsStart(bool core2)
    {
        var (s, rng, kernel) = Setup(Fight.Hand().Energy(3).Draw(Card(IC.StrikeIronclad)).Enemy(hp: 40), core2);

        KernelGate.Step(kernel, s, rng, s.Hand.Count);

        Assert.Equal(40 - 5 * 6, Enemy(s));
        Assert.Empty(s.Hand);
    }

    /// <summary>
    /// Pillage's draw meets a Strike, which Hellraiser plays inside Pillage's effect: the hand is empty when that play ends,
    /// but another card's effect is running, so Unceasing Top does not draw; Pillage draws on to the Defend and stops. The
    /// copy's relic drew the Defend for the Strike's play, and Pillage drew on past it.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnceasingTopDoesNotDrawInsideAnotherCardsEffect(bool core2)
    {
        var (s, rng, kernel) = Setup(
            Fight.Hand(Card(IC.Pillage)).Energy(3).Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash)).Enemy(hp: 400),
            core2);
        BuffSystem.Remove(s.PlayerBuffs, BuffId.Nostalgia);

        KernelGate.Step(kernel, s, rng, 0);

        Assert.Equal([IC.DefendIronclad], s.Hand.Select(c => c.DefId));
        Assert.Equal([IC.Bash], s.DrawPile.Select(c => c.DefId));
    }
}
