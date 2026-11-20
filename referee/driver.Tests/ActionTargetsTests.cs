using Spireline.Referee;
using Xunit;

namespace Spireline.Referee.Tests;

// The targets an action is offered at (ActionTargets.For). The driver needs the game; the rule does not.
public class ActionTargetsTests
{
    [Fact]
    public void AnEnemyTargetActionIsOfferedOncePerLivingEnemy()
    {
        Assert.Equal(new[] { 0, 1, 2 }, ActionTargets.For(aimsAtEnemy: true, livingEnemies: 3));
        Assert.Equal(new[] { 0 }, ActionTargets.For(true, 1));
    }

    [Fact]
    public void NoLivingCreatureMeansNoEnemyTargetAction()
    {
        // The Test Subject between its forms: nothing alive. A potion that aims at an enemy (Vulnerable, Weak, Fire, the
        // Potion-Shaped Rock) has no target, so it is not offered at all, not offered at target -1.
        Assert.Empty(ActionTargets.For(aimsAtEnemy: true, livingEnemies: 0));
    }

    [Fact]
    public void AnActionThatAimsAtNothingKeepsItsSingleTargetWithOrWithoutEnemies()
    {
        // Self and player potions (Block, Dexterity, Blood, Energy), all-enemy and random ones: target -1, always.
        Assert.Equal(new[] { -1 }, ActionTargets.For(aimsAtEnemy: false, livingEnemies: 0));
        Assert.Equal(new[] { -1 }, ActionTargets.For(false, 2));
    }

    [Fact]
    public void ThePotionListOfAFightWithNoLivingCreatureHoldsOnlyTheUntargetedPotions()
    {
        // The loop's use of the rule: (potion, aims at an enemy) for a belt of Vulnerable, Block, Weak and Energy.
        var belt = new (string potion, bool aims)[] { ("VULNERABLE_POTION", true), ("BLOCK_POTION", false), ("WEAK_POTION", true), ("ENERGY_POTION", false) };
        var offered = belt.SelectMany(b => ActionTargets.For(b.aims, 0).Select(t => (b.potion, t))).ToList();
        Assert.Equal(new[] { ("BLOCK_POTION", -1), ("ENERGY_POTION", -1) }, offered);
    }
}
