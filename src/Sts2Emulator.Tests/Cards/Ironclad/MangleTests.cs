using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

public class MangleTests
{
    [Fact]
    public void DamagesAndAppliesTempStrength()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        var rng = new Random(0);
        var enemy = state.Enemies[0];
        int hpBefore = enemy.Hp;

        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.Mangle, false));
        state.Energy = 3;

        CombatEngine.Step(state, 0, rng);
        Assert.True(enemy.Hp < hpBefore);
        Assert.Equal(-10, BuffSystem.Get(enemy.Buffs, BuffId.Strength));
        Assert.Equal(10, BuffSystem.Get(enemy.Buffs, BuffId.TemporaryStrength));
    }

    /// <summary>
    /// `Mangle.OnPlay` hits `cardPlay.Target` and then applies `ManglePower` to that same
    /// creature. A target the hit killed cannot receive it, so the Strength loss is lost; it
    /// does not move to whoever is first now.
    /// </summary>
    [Fact]
    public void AKilledTargetsStrengthLossIsLostAndNotPassedOn()
    {
        var fight = Fight.Hand(new CardInstance(IC.Mangle, false)).Energy(3).Enemy(hp: 1).Enemy(hp: 80);

        fight.Play(index: 0, target: 0);

        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(80, survivor.Hp);
        Assert.Empty(survivor.Buffs);
    }

    [Fact]
    public void TheStrengthLossLandsOnTheEnemyTheCardWasAimedAt()
    {
        var fight = Fight.Hand(new CardInstance(IC.Mangle, true)).Energy(3).Enemy(hp: 200).Enemy(hp: 200);

        fight.Play(index: 0, target: 1);

        Assert.Empty(fight.Enemy0.Buffs);
        Assert.Equal(-15, fight.EnemyBuffAmount(BuffId.Strength, 1));
        Assert.Equal(15, fight.EnemyBuffAmount(BuffId.TemporaryStrength, 1));
    }
}
