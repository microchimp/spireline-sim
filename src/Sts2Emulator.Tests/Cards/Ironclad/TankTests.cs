using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Power, CardMultiplayerConstraint.MultiplayerOnly.
// MegaCrit.Sts2.Core.Models.Cards/Tank.cs applies TankPower(1) to its owner;
// MegaCrit.Sts2.Core.Models.Powers/TankPower.cs only guards OTHER players on apply, but its
// ModifyDamageMultiplicative returns 2 for a powered attack whose target is the owner. A
// player who plays it alone therefore takes double damage from every attack. OnUpgrade only
// makes it cheaper.
public class TankTests
{
    [Fact]
    public void AppliesTheTankPowerAndNothingElseOnPlay()
    {
        var fight = Fight.Hand(Card(IC.Tank)).Energy(2).PlayerHp(64).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Tank));
        Assert.Equal(64, fight.State.PlayerHp);
        Assert.Equal(0, fight.State.PlayerBlock);
        Assert.Equal(40, fight.Enemy0.Hp);
    }

    [Fact]
    public void TheOwnerTakesDoubleFromAnEnemyAttack()
    {
        var fight = Fight
            .Hand(Card(IC.Tank))
            .Energy(2)
            .PlayerHp(64)
            .Enemy(hp: 200, intent: new Intent(IntentType.Attack, 10));
        fight.Play();

        fight.EndTurn();

        Assert.Equal(44, fight.State.PlayerHp);
    }

    [Fact]
    public void WithoutItAnEnemyAttackLandsOnce()
    {
        var fight = Fight
            .Hand()
            .Energy(2)
            .PlayerHp(64)
            .Enemy(hp: 200, intent: new Intent(IntentType.Attack, 10));

        fight.EndTurn();

        Assert.Equal(54, fight.State.PlayerHp);
    }

    [Fact]
    public void CostsItsEnergyAndLeavesPlayLikeAnyPower()
    {
        var fight = Fight.Hand(Card(IC.Tank)).Energy(2).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.State.Energy);
        // A played Power goes nowhere: not back to hand, not to the discard pile.
        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.DiscardPile);
    }
}
