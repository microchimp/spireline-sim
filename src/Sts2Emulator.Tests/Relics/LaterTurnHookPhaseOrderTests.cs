using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A later turn's start follows `SetupPlayerTurn`'s phases: the powers' `AfterEnergyReset` walk
/// the power list in one pass, and the relics' `AfterSideTurnStart` (Candelabra, Chandelier)
/// pay once the hand is drawn.
/// </summary>
public class LaterTurnHookPhaseOrderTests
{
    [Fact]
    public void CandelabraAndChandelierStillPayOnTheirTurnsAfterTheMove()
    {
        var fight = Fight.WithRelics(RelicEffects.Candelabra, RelicEffects.Chandelier);
        fight.State.PlayerHp = 999;
        int baseline = fight.State.MaxEnergy;

        fight.EndTurn();
        Assert.Equal(baseline + 2, fight.State.Energy);
        fight.EndTurn();
        Assert.Equal(baseline + 3, fight.State.Energy);
        fight.EndTurn();
        Assert.Equal(baseline, fight.State.Energy);
    }

    [Fact]
    public void EnergyResetPowersPayOnePassAndExpire()
    {
        var fight = Fight.WithRelics();
        fight.State.PlayerHp = 999;
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Radiance, 2);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NextTurnEnergy, 3);
        int baseline = fight.State.MaxEnergy;

        fight.EndTurn();

        Assert.Equal(baseline + 1 + 3, fight.State.Energy);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Radiance));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NextTurnEnergy));
    }

    /// <summary>
    /// `Candelabra.AfterSideTurnStart` (TurnNumber 2) and `Chandelier.AfterSideTurnStart`
    /// (TurnNumber 3) pay after `SetupPlayerTurn`'s hand draw. `Void.AfterCardDrawn` calls
    /// `PlayerCmd.LoseEnergy`, which `PlayerCombatState.LoseEnergy` clamps at zero, so a hand of
    /// Voids drains the reset energy and the relic's gain survives. Paid at the reset, the
    /// Voids ate it.
    /// </summary>
    [Fact]
    public void CandelabraAndChandelierPayAfterTheHandDraw()
    {
        var fight = Fight.WithRelics(RelicEffects.Candelabra, RelicEffects.Chandelier);
        fight.State.PlayerHp = 999;
        fight.Draw([.. Enumerable.Repeat(new CardInstance(ST.Void, false), 5)]);

        fight.EndTurn();
        Assert.Equal(2, fight.State.Energy);

        fight.Draw([.. Enumerable.Repeat(new CardInstance(ST.Void, false), 5)]);
        fight.EndTurn();
        Assert.Equal(3, fight.State.Energy);
    }

    /// <summary>
    /// `Hook.AfterEnergyReset` walks the listeners in list order, so a Genesis applied before a
    /// Lightning Rod pays its stars first: `BlackHolePower.AfterStarsGained` kills the 1-HP enemy,
    /// and the Dark orb the full slot evokes (`DarkOrb.Evoke` hits the weakest hittable enemy)
    /// lands on the survivor. With a fixed Radiance, Lightning Rod, Genesis order the evoke
    /// killed the 1-HP enemy and the survivor took only the Black Hole hit.
    /// </summary>
    [Fact]
    public void EnergyResetPowersFireInTheOrderTheyWereApplied()
    {
        var fight = Fight.WithRelics().Enemy(hp: 1).Enemy(hp: 100);
        fight.State.PlayerHp = 999;
        fight.State.BaseOrbSlots = 1;
        fight.State.OrbCapacity = 1;
        fight.State.Orbs.Add(new OrbState(OrbType.Dark, EvokeValue: 6));
        fight.PlayerBuff(BuffId.BlackHole, 3)
            .PlayerBuff(BuffId.Genesis, 1)
            .PlayerBuff(BuffId.LightningRod, 1);

        fight.EndTurn();

        // The dead enemy leaves the roster. The survivor took Black Hole's 3 and the Dark
        // orb's evoke (6, plus the 6 its end-of-turn passive added).
        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(100 - 3 - 12, survivor.Hp);
    }
}
