using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 2-cost Attack, CardMultiplayerConstraint.MultiplayerOnly.
// MegaCrit.Sts2.Core.Models.Cards/TagTeam.cs: DamageVar(11m) then TagTeamPower(1) on the
// target; OnUpgrade raises the damage by 4.
//
// TagTeamPower is Debuff, Counter, PowerInstanceType.Instanced, with DisplayAmount 1.
// ModifyCardPlayCount adds a play to an Attack aimed at its owner, but returns early when the
// card's owner is the applier -- always, alone -- so AfterModifyingCardPlayCount never removes
// it and it stays on the enemy for the whole fight. It is a visible Debuff: Artifact eats it
// and the power list shows TAG_TEAM on the enemy.
public class TagTeamTests
{
    private static List<BuffState> TagTeams(EnemyState enemy) =>
        enemy.Buffs.Where(b => b.Id == BuffId.TagTeam).ToList();

    [Fact]
    public void DealsEleven()
    {
        var fight = Fight.Hand(Card(CL.TagTeam)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(49, fight.Enemy0.Hp);
    }

    [Fact]
    public void UpgradedDealsFifteen()
    {
        var fight = Fight.Hand(Card(CL.TagTeam, upgraded: true)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal(45, fight.Enemy0.Hp);
    }

    [Fact]
    public void AppliesTagTeamOneAndNothingElse()
    {
        var fight = Fight.Hand(Card(CL.TagTeam)).Energy(2).Enemy(hp: 60);

        fight.Play();

        Assert.Equal([new BuffState(BuffId.TagTeam, 1)], fight.Enemy0.Buffs);
    }

    /// <summary>Instanced: two plays are two powers of 1, never one of 2.</summary>
    [Fact]
    public void EachPlayIsItsOwnInstance()
    {
        var fight = Fight.Hand(Card(CL.TagTeam), Card(CL.TagTeam)).Energy(4).Enemy(hp: 200);

        fight.Play();
        fight.Play();

        Assert.Equal(
            [new BuffState(BuffId.TagTeam, 1), new BuffState(BuffId.TagTeam, 1)],
            TagTeams(fight.Enemy0)
        );
    }

    [Fact]
    public void LandsOnlyOnTheAimedEnemy()
    {
        var fight = Fight.Hand(Card(CL.TagTeam)).Energy(2).Enemy(hp: 60).Enemy(hp: 60);

        fight.Play(index: 0, target: 1);

        Assert.Empty(fight.Enemy0.Buffs);
        Assert.Equal([new BuffState(BuffId.TagTeam, 1)], fight.Enemy1.Buffs);
    }

    [Fact]
    public void AKilledTargetGetsNothingAndNeitherDoesItsNeighbour()
    {
        var fight = Fight.Hand(Card(CL.TagTeam)).Energy(2).Enemy(hp: 11).Enemy(hp: 60);

        fight.Play(index: 0, target: 0);

        // The corpse has left the combat; the survivor is the second enemy, untouched.
        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(60, survivor.Hp);
        Assert.Empty(survivor.Buffs);
    }

    /// <summary>Nothing removes it: it is still there after the enemy's turn.</summary>
    [Fact]
    public void ItStaysAcrossTheEnemyTurn()
    {
        var fight = Fight.Hand(Card(CL.TagTeam)).Energy(2).Enemy(hp: 200);
        fight.State.PlayerHp = 999;
        fight.Play();

        fight.EndTurn();

        Assert.Equal([new BuffState(BuffId.TagTeam, 1)], fight.Enemy0.Buffs);
    }

    [Fact]
    public void ArtifactEatsIt()
    {
        var fight = Fight.Hand(Card(CL.TagTeam))
            .Energy(2)
            .Enemy(hp: 60, buffs: new BuffState(BuffId.Artifact, 1));

        fight.Play();

        Assert.Empty(TagTeams(fight.Enemy0));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));
    }
}
