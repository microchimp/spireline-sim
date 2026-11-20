using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// An action that aims at an enemy is not a legal move while no enemy can be hit. The game offers neither the card
/// (`CardModel.CanPlayTargeting` is false for one handed no target) nor the potion (`UsePotionAction` cancels it and the
/// potion stays in its slot), so the player's list at that moment is End Turn, the plays and potions that need no aim, and
/// nothing else. The Test Subject has no enemy standing between its forms: it falls to nothing and stands up on its own next
/// move. The mask used to keep every attack and every aimed potion on offer there, so the referee's lockstep (which takes a
/// lone End Turn for the copy when the game does) never saw one, and the copy stayed a turn behind the game for the rest of
/// the run: 45 of 95 Test Subject arrivals of the 2026-11-19 real block lost track of the game at this point.
/// </summary>
public class NoTargetActionsTests
{
    private static int Potion(string name) => GeneratedData.Potions.FindId(name)!.Value;

    /// <summary>A Test Subject at 1 health, a hand of two Strikes and a Defend, three energy, and a belt of an aimed potion
    /// (Vulnerable) and an unaimed one (Block).</summary>
    private static CombatState Subject()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.DrawPile.Clear();
        state.DiscardPile.Clear();
        state.Enemies =
        [
            new EnemyState
            {
                DefId = KE.TestSubject,
                Hp = 1,
                MaxHp = 111,
                CurrentIntent = new Intent(IntentType.Attack, 22),
                Buffs = [new BuffState(BuffId.Adaptable, 1)],
            },
        ];
        state.Hand =
        [
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
        ];
        state.Energy = 3;
        state.PotionSlots[0] = Potion("VulnerablePotion");
        state.PotionSlots[1] = Potion("BlockPotion");
        return state;
    }

    [Fact]
    public void WithAnEnemyStandingEveryPlayAndPotionIsLegal()
    {
        var state = Subject();

        // Hand index 0..2, End Turn (3), potion slots 0 and 1 (4 and 5).
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void WhenTheTestSubjectHasFallenNoAimedCardOrPotionIsLegal()
    {
        var state = Subject();

        CombatEngine.Step(state, 0, new Random(0)); // a Strike kills the first form
        Assert.Equal(0, state.Enemies[0].Hp);
        Assert.False(CombatEngine.HasTargetableEnemy(state));

        // The hand is now Strike (0), Defend (1); End Turn is 2, the potions 3 and 4. The Strike and the Vulnerable Potion
        // (3) aim at an enemy and are gone; the Defend, End Turn and the Block Potion (4) remain.
        Assert.Equal(new[] { 1, 2, 4 }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void WhenOnlyAimedCardsAndPotionsAreLeftEndTurnIsTheLoneAction()
    {
        var state = Subject();
        state.Hand = [new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.StrikeIronclad, false)];
        state.PotionSlots[1] = 0;
        CombatEngine.Step(state, 0, new Random(0)); // the first form falls

        // One Strike and one Vulnerable Potion are left, both aimed: what the lockstep takes for the copy without asking.
        Assert.Equal(new[] { state.Hand.Count }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void TheSolversMovesAreTheSameList()
    {
        var state = Subject();
        CombatEngine.Step(state, 0, new Random(0));

        var moves = Actions.Legal(state, Pruning.None);

        Assert.Equal(new[] { 1, 2, 4 }, moves.Select(m => m.Action).ToArray());
        Assert.All(moves, m => Assert.Equal(-1, m.Target));
    }

    [Fact]
    public void TheNextTurnTheSubjectStandsAgainAndTheAimedActionsReturn()
    {
        var state = Subject();
        CombatEngine.Step(state, 0, new Random(0));
        state.PlayerHp = 999;

        CombatEngine.Step(state, state.Hand.Count, new Random(0)); // End Turn: the respawn turn

        Assert.Equal(212, state.Enemies[0].Hp);
        Assert.True(CombatEngine.HasTargetableEnemy(state));
        int strike = state.Hand.ToList().FindIndex(c => c.DefId == IC.StrikeIronclad);
        Assert.True(strike >= 0);
        Assert.Contains(strike, CombatEngine.ValidActions(state));
    }
}
