using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The combat upgrade predicate and the run-layer one are the same rule: <c>CardModel.IsUpgradable</c>
/// is <c>CurrentUpgradeLevel &lt; MaxUpgradeLevel</c>, and the cards that override
/// <c>MaxUpgradeLevel</c> to zero are the curses, the statuses and the three Quest cards.
/// </summary>
/// <remarks>
/// The combat copy used to test the card TYPE (not Status, not Curse), so Byrdonis Egg, Lantern
/// Key and Spoils Map took an upgrade in combat that the game's <c>CardCmd.Upgrade</c> refuses.
/// No game card overrides <c>MaxUpgradeLevel</c> to anything above one, so a card is upgradable
/// exactly when it is not yet upgraded and its definition does not cap it at zero; there is no
/// repeatable-upgrade card for the boolean to get wrong.
/// </remarks>
public class CombatUpgradableRuleTests
{
    private static int Id(string name) =>
        GeneratedData.Cards.All.ToArray().Single(card => card.Name == name).Id;

    [Theory]
    [InlineData("AscendersBane")]
    [InlineData("Clumsy")]
    [InlineData("Greed")]
    [InlineData("ByrdonisEgg")]
    [InlineData("LanternKey")]
    [InlineData("SpoilsMap")]
    public void CursesAndQuestCardsAreNotUpgradableInCombat(string name)
    {
        Assert.False(CardEffects.IsUpgradable(new CardInstance(Id(name), false)), name);
    }

    [Fact]
    public void AnOrdinaryCardIsUpgradableOnceOnly()
    {
        Assert.True(CardEffects.IsUpgradable(new CardInstance(IC.StrikeIronclad, false)));
        Assert.False(CardEffects.IsUpgradable(new CardInstance(IC.StrikeIronclad, true)));
    }

    [Fact]
    public void TheCombatRuleIsTheRunRuleForEveryCard()
    {
        foreach (var def in GeneratedData.Cards.All.ToArray())
        {
            foreach (bool upgraded in new[] { false, true })
            {
                var card = new CardInstance(def.Id, upgraded);
                Assert.True(
                    RunConstants.IsRunCardUpgradable(card) == CardEffects.IsUpgradable(card),
                    $"{def.Name} upgraded={upgraded}"
                );
            }
        }
    }

    [Fact]
    public void UpgradeAllCardsInHandLeavesQuestCardsAndCursesAlone()
    {
        var fight = Fight.Hand(
            new CardInstance(Id("ByrdonisEgg"), false),
            new CardInstance(Id("LanternKey"), false),
            new CardInstance(Id("SpoilsMap"), false),
            new CardInstance(IC.AscendersBane, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, true)
        );

        CardEffects.UpgradeAllCardsInHand(fight.State);

        Assert.All(fight.State.Hand.Take(4), card => Assert.False(card.Upgraded));
        Assert.True(fight.State.Hand[4].Upgraded);
        Assert.True(fight.State.Hand[5].Upgraded);
    }

    [Fact]
    public void UpgradedArmamentsSkipsAQuestCardInHand()
    {
        var fight = Fight
            .Hand(
                new CardInstance(IC.Armaments, true),
                new CardInstance(Id("ByrdonisEgg"), false),
                new CardInstance(IC.StrikeIronclad, false)
            )
            .Energy(3);

        fight.Play(0);

        Assert.False(fight.State.Hand.Single(c => c.DefId == Id("ByrdonisEgg")).Upgraded);
        Assert.True(fight.State.Hand.Single(c => c.DefId == IC.StrikeIronclad).Upgraded);
    }

    [Fact]
    public void UnupgradedArmamentsDoesNotOfferAQuestCard()
    {
        var fight = Fight
            .Hand(
                new CardInstance(IC.Armaments, false),
                new CardInstance(Id("SpoilsMap"), false),
                new CardInstance(IC.StrikeIronclad, false),
                new CardInstance(IC.DefendIronclad, false)
            )
            .Energy(3);

        fight.Play(0);

        Assert.NotNull(fight.Pending);
        Assert.Equal(2, fight.Pending!.Candidates.Count);
        Assert.DoesNotContain(
            fight.Pending.Candidates,
            at => fight.State.Hand[at].DefId == Id("SpoilsMap")
        );
    }

    /// <summary>
    /// Apotheosis upgrades hand, draw, discard and exhaust piles; a Quest card in any of them
    /// stays as it is.
    /// </summary>
    [Fact]
    public void ApotheosisLeavesQuestCardsInEveryPile()
    {
        var egg = new CardInstance(Id("ByrdonisEgg"), false);
        var strike = new CardInstance(IC.StrikeIronclad, false);
        var fight = Fight.Hand(new CardInstance(Id("Apotheosis"), false), egg, strike).Energy(3);
        fight.State.DrawPile = [egg, strike];
        fight.State.DiscardPile = [egg, strike];
        fight.State.ExhaustPile = [egg, strike];

        fight.Play(0);

        foreach (var pile in new[] { fight.State.Hand, fight.State.DrawPile, fight.State.DiscardPile, fight.State.ExhaustPile })
        {
            Assert.False(pile.Single(c => c.DefId == egg.DefId).Upgraded);
            Assert.True(pile.Single(c => c.DefId == strike.DefId).Upgraded);
        }
    }
}
