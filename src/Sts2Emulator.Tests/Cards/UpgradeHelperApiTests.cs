using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>The shape of the helper: <c>CardCmd.Upgrade(card)</c> and <c>CardCmd.Upgrade(cards)</c>.</summary>
public class UpgradeHelperApiTests
{
    private static int Id(string name) =>
        GeneratedData.Cards.All.ToArray().Single(card => card.Name == name).Id;

    [Fact]
    public void UpgradeCardsSkipsWhatCannotBeUpgradedAndKeepsTheGivenOrder()
    {
        var pile = new List<CardInstance>
        {
            new(IC.StrikeIronclad, false),
            new(Id("Greed"), false),
            new(IC.DefendIronclad, false),
            new(IC.Bludgeon, true),
        };

        CardEffects.UpgradeCards(pile, new[] { 2, 1, 3, 0, 9 });

        Assert.True(pile[0].Upgraded);
        Assert.False(pile[1].Upgraded);
        Assert.True(pile[2].Upgraded);
        Assert.True(pile[3].Upgraded);
        Assert.False(CardEffects.UpgradeCard(pile, 1));
        Assert.False(CardEffects.UpgradeCard(pile, 0));
        Assert.False(CardEffects.UpgradeCard(pile, -1));
    }

    [Fact]
    public void UpgradedCopyLeavesAnUnupgradableCardAlone()
    {
        var egg = new CardInstance(Id("ByrdonisEgg"), false);
        Assert.Equal(egg, CardEffects.UpgradedCopy(egg));
        Assert.True(CardEffects.UpgradedCopy(new CardInstance(IC.StrikeIronclad, false)).Upgraded);
    }
}
