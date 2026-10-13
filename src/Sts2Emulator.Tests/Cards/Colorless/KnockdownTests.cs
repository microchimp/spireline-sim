using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 3-cost Attack, CardMultiplayerConstraint.MultiplayerOnly.
// MegaCrit.Sts2.Core.Models.Cards/Knockdown.cs: DamageVar(10m) then
// PowerVar<KnockdownPower>(2m) on the target; OnUpgrade raises the damage by 4 and the
// power by 1.
//
// KnockdownPower is Debuff, Counter, PowerInstanceType.Instanced, and a damage MULTIPLIER on
// its owner: ModifyDamageMultiplicative returns 1 when `dealer == base.Applier`, so it only
// ever amplifies another player's attacks and changes no number alone. It is still a visible
// Debuff -- ArtifactPower.TryModifyPowerAmountReceived eats it, the power list shows it -- and
// its AfterSideTurnEnd removes it when its OWNER's side turn ends.
//
// Stunned used to stand in for it at 2 and 3. That is a real debuff costing an enemy turns,
// handed out by a card that should do nothing beyond its damage.
public class KnockdownTests
{
    private static List<BuffState> Knockdowns(EnemyState enemy) =>
        enemy.Buffs.Where(b => b.Id == BuffId.Knockdown).ToList();

    [Fact]
    public void DealsTen()
    {
        var fight = Fight.Hand(Card(CL.Knockdown)).Energy(3).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(50, fight.Enemy0.Hp);
    }

    [Fact]
    public void UpgradedDealsFourteen()
    {
        var fight = Fight.Hand(Card(CL.Knockdown, upgraded: true)).Energy(3).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(46, fight.Enemy0.Hp);
    }

    /// <summary>The power lands on the target after the damage: 2, or 3 upgraded.</summary>
    [Fact]
    public void AppliesKnockdownTwoAndNothingElse()
    {
        var fight = Fight.Hand(Card(CL.Knockdown)).Energy(3).Enemy(hp: 60);

        fight.Play();

        Assert.Equal([new BuffState(BuffId.Knockdown, 2)], fight.Enemy0.Buffs);
    }

    [Fact]
    public void UpgradedAppliesThree()
    {
        var fight = Fight.Hand(Card(CL.Knockdown, upgraded: true)).Energy(3).Enemy(hp: 60);

        fight.Play();

        Assert.Equal([new BuffState(BuffId.Knockdown, 3)], fight.Enemy0.Buffs);
    }

    /// <summary>Instanced: a second Knockdown on the same enemy is a second power, not Knockdown 5.</summary>
    [Fact]
    public void EachPlayIsItsOwnInstance()
    {
        var fight = Fight.Hand(Card(CL.Knockdown), Card(CL.Knockdown, upgraded: true))
            .Energy(6)
            .Enemy(hp: 200);

        fight.Play();
        fight.Play();

        Assert.Equal(
            [new BuffState(BuffId.Knockdown, 2), new BuffState(BuffId.Knockdown, 3)],
            Knockdowns(fight.Enemy0)
        );
    }

    /// <summary>It goes on the enemy the card was aimed at and on no other.</summary>
    [Fact]
    public void LandsOnlyOnTheAimedEnemy()
    {
        var fight = Fight.Hand(Card(CL.Knockdown)).Energy(3).Enemy(hp: 60).Enemy(hp: 60);

        fight.Play(index: 0, target: 1);

        Assert.Empty(fight.Enemy0.Buffs);
        Assert.Equal([new BuffState(BuffId.Knockdown, 2)], fight.Enemy1.Buffs);
    }

    /// <summary>
    /// A hit that kills its target applies nothing, and the power does not pass to the next
    /// enemy: `PowerCmd.Apply` returns for a creature that cannot receive powers, and the
    /// card's target is `cardPlay.Target`, not "whoever is first now".
    /// </summary>
    [Fact]
    public void AKilledTargetGetsNothingAndNeitherDoesItsNeighbour()
    {
        var fight = Fight.Hand(Card(CL.Knockdown)).Energy(3).Enemy(hp: 10).Enemy(hp: 60);

        fight.Play(index: 0, target: 0);

        // The corpse has left the combat; the survivor is the second enemy, untouched.
        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(60, survivor.Hp);
        Assert.Empty(survivor.Buffs);
    }

    /// <summary>An Artifact eats one application whole, and every play is one application.</summary>
    [Fact]
    public void ArtifactEatsOnePlayEach()
    {
        var fight = Fight.Hand(Card(CL.Knockdown), Card(CL.Knockdown))
            .Energy(6)
            .Enemy(hp: 200, buffs: new BuffState(BuffId.Artifact, 1));

        fight.Play();
        Assert.Empty(Knockdowns(fight.Enemy0));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));

        fight.Play();
        Assert.Equal([new BuffState(BuffId.Knockdown, 2)], Knockdowns(fight.Enemy0));
    }

    /// <summary>`AfterSideTurnEnd` removes EVERY instance when the enemy side's turn ends.</summary>
    [Fact]
    public void AllInstancesAreGoneAfterTheEnemyTurn()
    {
        var fight = Fight.Hand(Card(CL.Knockdown), Card(CL.Knockdown)).Energy(6).Enemy(hp: 200);
        fight.State.PlayerHp = 999;
        fight.Play();
        fight.Play();
        Assert.Equal(2, Knockdowns(fight.Enemy0).Count);

        fight.EndTurn();

        Assert.Empty(Knockdowns(fight.Enemy0));
    }

    /// <summary>Not a stun: Stunned stays Whistle's.</summary>
    [Fact]
    public void AppliesNoStunAndNoOtherDebuff()
    {
        var fight = Fight.Hand(Card(CL.Knockdown)).Energy(3).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Stunned));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Vulnerable));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Weak));
    }

    /// <summary>
    /// Whistle shares this case body and DOES stun — it calls `CreatureCmd.Stun` outright,
    /// which is why that branch stays while Knockdown's went.
    /// </summary>
    [Fact]
    public void WhistleStillStuns()
    {
        const int whistle = 539;
        var fight = Fight.Hand(new CardInstance(whistle, false)).Energy(3).Enemy(hp: 90);

        fight.Play();

        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
    }
}
