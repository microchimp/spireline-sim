using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Turn one's combat-start hooks run in the game's phase order, ahead of the opening draw, and
/// each phase walks the held relics in holding order (`Hook.AfterRoomEntered`, then
/// `BeforeCombatStart`, then the side's `BeforeSideTurnStart`; `AfterSideTurnStart` last).
/// </summary>
public class TurnOneHookPhaseOrderTests
{
    private static int[] PlayerPowerOrder(Fight fight) =>
        fight.State.PlayerBuffs.Select(buff => (int)buff.Id).ToArray();

    /// <summary>
    /// Vajra's Strength is `AfterRoomEntered`, Snecko Eye's Confused is `BeforeCombatStart`: the
    /// room-entered phase finishes first whichever relic is held first.
    /// </summary>
    [Fact]
    public void AfterRoomEnteredPowersComeBeforeBeforeCombatStartPowers()
    {
        var fight = Fight.WithRelics(RelicEffects.SneckoEye, RelicEffects.Vajra);

        var order = PlayerPowerOrder(fight);
        Assert.True(
            System.Array.IndexOf(order, (int)BuffId.Strength) < System.Array.IndexOf(order, (int)BuffId.Confused)
        );
    }

    /// <summary>Within one phase the relics run in the order the player holds them.</summary>
    [Fact]
    public void OnePhaseWalksTheHeldRelicsInHoldingOrder()
    {
        var gorgetFirst = PlayerPowerOrder(Fight.WithRelics(RelicEffects.Gorget, RelicEffects.OddlySmoothStone));
        var stoneFirst = PlayerPowerOrder(Fight.WithRelics(RelicEffects.OddlySmoothStone, RelicEffects.Gorget));

        Assert.True(
            System.Array.IndexOf(gorgetFirst, (int)BuffId.Plating) < System.Array.IndexOf(gorgetFirst, (int)BuffId.Dexterity)
        );
        Assert.True(
            System.Array.IndexOf(stoneFirst, (int)BuffId.Dexterity) < System.Array.IndexOf(stoneFirst, (int)BuffId.Plating)
        );
    }

    /// <summary>
    /// The split keeps every opener on turn one: Anchor's block (`BeforeCombatStart`) and Bread's
    /// two-energy price (`AfterSideTurnStart`, after the reset) both land.
    /// </summary>
    [Fact]
    public void TurnOneOpenersStillLandOnTurnOne()
    {
        var fight = Fight.WithRelics(RelicEffects.Anchor, RelicEffects.Bread);

        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.Equal(fight.State.MaxEnergy - RelicEffects.BreadTurnOneEnergyLoss, fight.State.Energy);
    }
}
