using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardSelectCmd.FromHand` / `FromCombatPile` / `FromSimpleGrid` / `FromHandForUpgrade` hand
/// back the pool without showing a screen when it is no bigger than the minimum and the
/// prefs do not `RequireManualConfirmation` (CardSelectorPrefs: min != max). Found by the
/// night03 referee: Brand, Armaments, Secret Weapon and Seeker Strike with a pool of one
/// were left waiting on a screen the game never showed.
/// </summary>
public class LoneSelectionTests
{
    [Fact]
    public void Brand_WithOneCardLeftInHand_ExhaustsItWithoutAsking()
    {
        var fight = Fight.Hand(Card(IC.Brand), Card(CL.Jackpot)).Energy(3).Enemy(hp: 40);
        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Empty(fight.State.Hand);
        Assert.Equal([CL.Jackpot], Fight.Ids(fight.State.ExhaustPile));
    }

    [Fact]
    public void Brand_WithTwoCardsLeftInHand_StillAsks()
    {
        var fight = Fight
            .Hand(Card(IC.Brand), Card(CL.Jackpot), Card(IC.Bash))
            .Energy(3)
            .Enemy(hp: 40);
        fight.Play();

        Assert.NotNull(fight.Pending);
    }

    [Fact]
    public void Armaments_WithOneUpgradableCard_UpgradesItWithoutAsking()
    {
        var fight = Fight
            .Hand(Card(IC.Armaments), Card(IC.StrikeIronclad), Card(IC.AscendersBane))
            .Energy(3)
            .Enemy(hp: 40);
        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Contains(fight.State.Hand, c => c.DefId == IC.StrikeIronclad && c.Upgraded);
    }

    [Fact]
    public void SecretWeapon_WithOneAttackInTheDrawPile_TakesItWithoutAsking()
    {
        var fight = Fight
            .Hand(Card(CL.SecretWeapon))
            .Draw(Card(IC.Bash), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .Energy(3)
            .Enemy(hp: 40);
        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
    }

    [Fact]
    public void SeekerStrike_WithOneCardInTheDrawPile_TakesItWithoutAsking()
    {
        var fight = Fight
            .Hand(Card(CL.SeekerStrike))
            .Draw(Card(IC.Bash))
            .Energy(3)
            .Enemy(hp: 40);
        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
    }

    /// <summary>The control: Purity's minimum is zero, so it asks even with a single card.</summary>
    [Fact]
    public void Purity_WithOneCardInHand_StillAsks()
    {
        var fight = Fight.Hand(Card(CL.Purity), Card(IC.Bash)).Energy(3).Enemy(hp: 40);
        fight.Play();

        Assert.NotNull(fight.Pending);
    }

}
