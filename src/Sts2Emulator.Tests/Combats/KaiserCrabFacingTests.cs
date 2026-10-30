using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Kaiser Crab's SurroundedPower turns the player toward what they aim at: BeforeCardPlayed
/// (the play's target) and BeforePotionUsed (the potion's target). The half at the player's back
/// hits for half again, so the facing at the end of a turn is the player's HP after it.
/// </summary>
public class KaiserCrabFacingTests
{
    private static Fight Crab(int seed = 0) =>
        Fight.Encounter(CombatFactory.ActOneEncounter.KaiserCrab, ascension: 8, seed: seed);

    private static int Facing(Fight fight) => BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Surrounded);

    /// <summary>The enemy turn's damage to a player with no block: the half at the back for half again.</summary>
    private static int EnemyTurnLoss(Fight fight)
    {
        fight.State.PlayerHp = 500;
        fight.State.PlayerBlock = 0;
        fight.EndTurn();
        return 500 - fight.State.PlayerHp;
    }

    /// <summary>
    /// A Block Potion is aimed at the player (AnyPlayer), who has no Back Attack: the game's
    /// UpdateDirection turns no one. The copy turned the player toward the first living enemy,
    /// the Crusher, so its back went to the Rocket and the opening turn hit 16 where the game's
    /// hits 21 (A8: the Crusher's 12 from behind as 18 and the Rocket's 3, against 12 and 4).
    /// </summary>
    [Fact]
    public void APotionAimedAtThePlayerTurnsNoOne()
    {
        var fight = Crab();
        fight.State.PotionSlots[0] = Potions.FindId("BlockPotion")!.Value;

        fight.Potion(0);

        Assert.Equal(RunConstants.FacingRight, Facing(fight));
        Assert.Equal(18 + 3, EnemyTurnLoss(fight));
    }

    /// <summary>An AllEnemies potion (Explosive Ampoule) has no target either.</summary>
    [Fact]
    public void AnAllEnemiesPotionTurnsNoOne()
    {
        var fight = Crab();
        fight.State.PotionSlots[0] = Potions.FindId("ExplosiveAmpoule")!.Value;

        fight.Potion(0);

        Assert.Equal(RunConstants.FacingRight, Facing(fight));
    }

    /// <summary>
    /// Havoc plays the top card of the draw pile with `CardCmd.AutoPlay(card, null)`: a Strike
    /// rolls its target on CombatTargets, and its BeforeCardPlayed turns the player toward
    /// whichever half it hits. The copy turned the player only at a chosen play, so a Strike
    /// Havoc sent into the Crusher left the copy's player facing the Rocket. Over ten seeds
    /// the facing after Havoc is the side of the half the Strike hit, and some seeds hit the
    /// Crusher (the case the copy got wrong).
    /// </summary>
    [Fact]
    public void ACardHavocPlaysTurnsThePlayerToItsTarget()
    {
        int crusherHits = 0;
        for (int seed = 0; seed < 10; seed++)
        {
            var fight = Crab(seed);
            fight.State.Hand.Clear();
            fight.State.Hand.Add(TestDeck.Card(Core.Effects.IC.Havoc));
            fight.State.DrawPile.Clear();
            fight.State.DrawPile.Add(TestDeck.Card(Core.Effects.IC.StrikeIronclad));
            fight.State.Energy = 3;
            int[] before = fight.State.Enemies.Select(e => e.Hp).ToArray();

            fight.Play(0);

            int hit = Enumerable.Range(0, 2).Single(i => fight.State.Enemies[i].Hp < before[i]);
            crusherHits += hit == 0 ? 1 : 0;
            Assert.Equal(hit == 0 ? RunConstants.FacingLeft : RunConstants.FacingRight, Facing(fight));
        }

        Assert.InRange(crusherHits, 1, 9);
    }

    /// <summary>
    /// Control: a chosen Strike still turns the player toward its target, and a later Defend
    /// (no target) does not turn them back.
    /// </summary>
    [Fact]
    public void AChosenStrikeTurnsThePlayerAndADefendDoesNot()
    {
        var fight = Crab();
        fight.State.Hand.Clear();
        fight.State.Hand.Add(TestDeck.Card(Core.Effects.IC.StrikeIronclad));
        fight.State.Hand.Add(TestDeck.Card(Core.Effects.IC.DefendIronclad));
        fight.State.Energy = 3;

        fight.Play(0, target: 0);
        Assert.Equal(RunConstants.FacingLeft, Facing(fight));

        fight.Play(0);
        Assert.Equal(RunConstants.FacingLeft, Facing(fight));
    }

    /// <summary>
    /// Control: a potion aimed at an enemy (Weak Potion at the Crusher) does turn the player,
    /// and back again when aimed at the Rocket.
    /// </summary>
    [Fact]
    public void APotionAimedAtAHalfTurnsThePlayerToIt()
    {
        var fight = Crab();
        int weak = Potions.FindId("WeakPotion")!.Value;
        fight.State.PotionSlots[0] = weak;
        fight.State.PotionSlots[1] = weak;

        fight.Potion(0, target: 0);
        Assert.Equal(RunConstants.FacingLeft, Facing(fight));

        fight.Potion(1, target: 1);
        Assert.Equal(RunConstants.FacingRight, Facing(fight));
    }
}
