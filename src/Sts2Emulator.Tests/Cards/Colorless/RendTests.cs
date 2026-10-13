using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 2-cost Attack. MegaCrit.Sts2.Core.Models.Cards/Rend.cs: CalculationBaseVar(15m) plus
// ExtraDamageVar(5m) per power on the target that ShouldCountPower accepts; OnUpgrade
// raises both the base and the per-power damage by 3.
public class RendTests
{
    [Fact]
    public void DealsFifteenAgainstAnUnafflictedEnemy()
    {
        var fight = Fight.Hand(Card(CL.Rend)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(45, fight.Enemy0.Hp);
    }

    [Fact]
    public void GainsFivePerDebuffOnTheTarget()
    {
        var fight = Fight
            .Hand(Card(CL.Rend))
            .Energy(2)
            .Enemy(hp: 90, buffs: [new BuffState(BuffId.Weak, 2), new BuffState(BuffId.Frail, 1)]);

        // (15 + 5 x 2) with no Vulnerable in the mix to scale it.
        fight.Play();

        Assert.Equal(65, fight.Enemy0.Hp);
    }

    [Fact]
    public void UpgradedStartsFromEighteenAndGainsEight()
    {
        var fight = Fight
            .Hand(Card(CL.Rend, upgraded: true))
            .Energy(2)
            .Enemy(hp: 90, buffs: [new BuffState(BuffId.Weak, 2)]);

        // 18 + 8 x 1
        fight.Play();

        Assert.Equal(64, fight.Enemy0.Hp);
    }

    /// <summary>
    /// `target.Powers.Count(ShouldCountPower)` counts POWERS, so each instance of an Instanced
    /// debuff is one: two Knockdowns and a Tag Team are three, on top of the Weak.
    /// </summary>
    [Fact]
    public void CountsEveryInstanceOfAnInstancedDebuff()
    {
        var fight = Fight
            .Hand(Card(CL.Rend))
            .Energy(2)
            .Enemy(
                hp: 90,
                buffs:
                [
                    new BuffState(BuffId.Weak, 2),
                    new BuffState(BuffId.Knockdown, 2),
                    new BuffState(BuffId.TagTeam, 1),
                    new BuffState(BuffId.Knockdown, 3),
                ]
            );

        // 15 + 5 x 4. (A lone player's Knockdown does not scale their own damage.)
        fight.Play();

        Assert.Equal(55, fight.Enemy0.Hp);
    }

    /// <summary>
    /// The count is by the power's type at its CURRENT amount, not a list of names: a Doom is a
    /// Debuff, and an enemy pushed below zero Strength carries a Debuff StrengthPower. A
    /// temporary power (Mangle's, in the emulator TemporaryStrength) is `ITemporaryPower` and
    /// does not count, but the Strength loss it causes does.
    /// </summary>
    [Fact]
    public void CountsDoomAndANegativeStrengthButNotTheTemporaryPower()
    {
        var fight = Fight
            .Hand(Card(CL.Rend))
            .Energy(2)
            .Enemy(
                hp: 90,
                buffs:
                [
                    new BuffState(BuffId.Doom, 5),
                    new BuffState(BuffId.Strength, -10),
                    new BuffState(BuffId.TemporaryStrength, 10),
                ]
            );

        // 15 + 5 x 2: the Doom and the negative Strength. Doom 5 on 90 hp does not kill.
        fight.Play();

        Assert.Equal(65, fight.Enemy0.Hp);
    }

    /// <summary>Positive Strength is a Buff, and does not count.</summary>
    [Fact]
    public void PositiveStrengthDoesNotCount()
    {
        var fight = Fight
            .Hand(Card(CL.Rend))
            .Energy(2)
            .Enemy(hp: 90, buffs: [new BuffState(BuffId.Strength, 3)]);

        fight.Play();

        Assert.Equal(75, fight.Enemy0.Hp);
    }
}
