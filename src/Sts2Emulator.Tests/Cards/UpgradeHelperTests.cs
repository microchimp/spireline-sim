using System;
using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Every combat upgrade goes through <c>CardCmd.Upgrade</c> in the game, which skips a card whose
/// <c>CardModel.IsUpgradable</c> is false. Bellows, Bone Tea and the rest used to write the flag
/// onto every card of the pile, curses included.
/// </summary>
public class UpgradeHelperTests
{
    private static int Id(string name) =>
        GeneratedData.Cards.All.ToArray().Single(card => card.Name == name).Id;

    [Fact]
    public void BellowsUpgradesTheOpeningHandButNotACurse()
    {
        var fight = Fight.WithRelics(RelicEffects.Bellows);
        fight.State.Hand =
        [
            new CardInstance(IC.AscendersBane, false),
            new CardInstance(IC.StrikeIronclad, false),
        ];

        RelicEffects.ApplyAfterPlayerTurnStart(fight.State, 1, new Random(0));

        Assert.False(fight.State.Hand[0].Upgraded);
        Assert.True(fight.State.Hand[1].Upgraded);
    }

    [Fact]
    public void BoneTeaUpgradesTheOpeningHandButNotACurse()
    {
        var fight = Fight.WithRelics(RelicEffects.BoneTea);
        int at = fight.State.Relics.FindIndex(relic => relic.DefId == RelicEffects.BoneTea);
        fight.State.Relics[at] = fight.State.Relics[at] with { Counter = RelicEffects.BoneTeaCombats };
        fight.State.Hand =
        [
            new CardInstance(IC.AscendersBane, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(Id("Clumsy"), false),
        ];

        RelicEffects.ApplyFirstTurnAfterSideTurnStart(fight.State, new Random(0));

        Assert.False(fight.State.Hand[0].Upgraded);
        Assert.True(fight.State.Hand[1].Upgraded);
        Assert.False(fight.State.Hand[2].Upgraded);
    }
}
