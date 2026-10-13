using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardCmd.AutoPlay`'s gate, which every auto-play passes: an Unplayable card or a
/// `Hook.ShouldPlay` refusal is filed away unplayed before anything rolls; then only a card
/// whose TargetType -- its own, read at play time -- is AnyEnemy rolls
/// `Rng.CombatTargets.NextItem(HittableEnemies)`; an AnyAlly card has nobody to aim at in a
/// solo run. `Shiv`, `SovereignBlade` and `MadScience` override TargetType.
/// </summary>
public class AutoPlayGateTests
{
    private const int DistilledChaos = 15;
    private const int BelieveInYou = 38;
    private const int MadScience = 292;
    private const int Shiv = 430;
    private const int SovereignBlade = 448;

    private static CardInstance Card(int id) => new(id, false);

    private static CountingRandom TargetStream(Fight fight)
    {
        var targets = new CountingRandom(7);
        fight.State.TargetRng = targets;
        return targets;
    }

    private static Fight TwoEnemiesWithTopCard(CardInstance top) =>
        Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 60)
            .Enemy(hp: 60)
            .Draw(top, Card(IC.DefendIronclad))
            .Energy(3);

    [Theory]
    [InlineData(Shiv, BuffId.FanOfKnives)]
    [InlineData(SovereignBlade, BuffId.SeekingEdge)]
    public void AnAutoPlayedCardThatHitsEveryoneUnderItsPowerRollsNothing(int card, BuffId power)
    {
        var fight = TwoEnemiesWithTopCard(Card(card)).PlayerBuff(power, 1);
        var targets = TargetStream(fight);

        fight.Play(0);

        Assert.Equal(0, targets.CallCount);
        Assert.All(fight.State.Enemies, enemy => Assert.True(enemy.Hp < 60));
    }

    [Theory]
    [InlineData(CardType.Attack, 1)]
    [InlineData(CardType.Skill, 0)]
    [InlineData(CardType.Power, 0)]
    public void AnAutoPlayedMadScienceRollsOnlyWhenTinkeredIntoAnAttack(CardType type, int rolls)
    {
        var fight = TwoEnemiesWithTopCard(Card(MadScience) with { TinkerType = type });
        var targets = TargetStream(fight);

        fight.Play(0);

        Assert.Equal(rolls, targets.CallCount);
    }

    /// <summary>
    /// `CardCmd.AutoPlay` rolls an AnyAlly card over the OTHER living players; a solo run
    /// has none, so nothing is drawn and the card is filed away unplayed.
    /// </summary>
    [Fact]
    public void AnAutoPlayedAllyCardDrawsNothingAndIsFiledAwayUnplayed()
    {
        Assert.Equal(CardTarget.AnyAlly, GeneratedData.Cards.Get(BelieveInYou).Target);
        var fight = Fight.Hand(Card(IC.DefendIronclad)).Enemy(hp: 60).Draw(Card(BelieveInYou));
        fight.State.PotionSlots[0] = DistilledChaos;
        var targets = TargetStream(fight);

        fight.Potion(0);

        Assert.Equal(0, targets.CallCount);
        Assert.Contains(fight.State.DiscardPile, card => card.DefId == BelieveInYou);
        Assert.Equal(0, fight.State.CardPlaysThisTurn);
    }

    /// <summary>
    /// Every `Hook.ShouldPlay` refusal gates an auto-play, before its target is rolled.
    /// </summary>
    [Theory]
    [InlineData(BuffId.Ringing)]
    [InlineData(BuffId.Sloth)]
    public void AnAutoPlayedCardThatShouldPlayRefusesIsFiledAwayWithoutARoll(BuffId refusal)
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Enemy(hp: 60)
            .Draw(Card(IC.Taunt))
            .PlayerBuff(refusal, 1);
        fight.State.PotionSlots[0] = DistilledChaos;
        var targets = TargetStream(fight);
        fight.State.CardPlaysThisTurn = 1;
        fight.State.CardsPlayedThisTurn = 1;

        fight.Potion(0);

        Assert.Equal(0, targets.CallCount);
        Assert.Contains(fight.State.DiscardPile, card => card.DefId == IC.Taunt);
        Assert.Equal(0, BuffSystem.Get(fight.State.Enemies[0].Buffs, BuffId.Vulnerable));
    }

    [Fact]
    public void AStampedeAttackThatSlothRefusesRollsNothing()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 60)
            .Draw(Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Sloth, 1)
            .PlayerBuff(BuffId.Stampede, 1);
        var targets = TargetStream(fight);
        fight.State.CardsPlayedThisTurn = 1;

        fight.EndTurn();

        Assert.Equal(0, targets.CallCount);
        Assert.Equal(60, fight.State.Enemies[0].Hp);
    }

    /// <summary>
    /// `Inky.OnPlay` reads the card's own TargetType, so an Inky Shiv under Fan of Knives
    /// Weakens every enemy it hit, not the first.
    /// </summary>
    [Fact]
    public void AnInkyShivUnderFanOfKnivesWeakensEveryEnemy()
    {
        var fight = Fight
            .Hand(new CardInstance(Shiv, false, Enchantment: Enchantment.Inky, EnchantAmount: 1))
            .Enemy(hp: 60)
            .Enemy(hp: 60)
            .PlayerBuff(BuffId.FanOfKnives, 1);

        fight.Play(0);

        Assert.All(
            fight.State.Enemies,
            enemy => Assert.Equal(1, BuffSystem.Get(enemy.Buffs, BuffId.Weak))
        );
    }
}
