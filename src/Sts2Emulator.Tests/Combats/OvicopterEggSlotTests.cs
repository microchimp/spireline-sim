using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Ovicopter's eggs stand in the encounter's slots: each egg takes the last slot of
/// egg1..egg5 that no living creature holds, and the roster reads in slot order with the
/// Ovicopter last. The copy put every new egg in front of the first egg, which is the same
/// order only while no egg has died: the nights' Ovicopter records are the same HPs in a
/// different order (e.g. [19,17,2,18,88] in the game, [19,17,18,2,88] in the copy).
/// </summary>
public class OvicopterEggSlotTests
{
    private static (Fight Fight, EnemyState Ovicopter) LaidOnce()
    {
        var fight = HiveNormal.At(CombatFactory.ActOneEncounter.Ovicopter);
        var ovicopter = fight.State.Enemies[0];
        Lay(fight, ovicopter);
        return (fight, ovicopter);
    }

    private static void Lay(Fight fight, EnemyState ovicopter)
    {
        fight.State.PlayerHp = 999;
        ovicopter.Hp = 999;
        ovicopter.CurrentIntent = new Intent(IntentType.Buff, 0);
        fight.EndTurn();
    }

    [Fact]
    public void TheFirstClutchFillsEggFiveFourAndThree()
    {
        var (fight, ovicopter) = LaidOnce();

        Assert.Equal([2, 3, 4, CombatFactory.OvicopterOwnSlot], fight.State.Enemies.Select(e => e.Slot));
        Assert.Same(ovicopter, fight.State.Enemies[^1]);
    }

    /// <summary>
    /// The middle egg of the first clutch dies and leaves the fight. The next clutch takes
    /// egg4 (its slot, free again), then egg2 and egg1: the newest egg of the second clutch
    /// stands BETWEEN the two survivors, not in front of them.
    /// </summary>
    [Fact]
    public void ASecondClutchRefillsTheFreedSlotBetweenTheSurvivors()
    {
        var (fight, ovicopter) = LaidOnce();
        var first = fight.State.Enemies.Where(e => e.DefId == KE.ToughEgg).ToList();
        fight.State.Enemies.Remove(first[1]);

        Lay(fight, ovicopter);

        var roster = fight.State.Enemies;
        Assert.Equal([0, 1, 2, 3, 4, CombatFactory.OvicopterOwnSlot], roster.Select(e => e.Slot));
        Assert.Same(first[0], roster[2]);
        Assert.Same(first[2], roster[4]);
        Assert.Same(ovicopter, roster[5]);
    }

    /// <summary>A corpse the copy still keeps gives its place to the egg laid into its slot.</summary>
    [Fact]
    public void AKeptCorpseGivesItsPlaceUp()
    {
        var (fight, ovicopter) = LaidOnce();
        var first = fight.State.Enemies.Where(e => e.DefId == KE.ToughEgg).ToList();
        first[1].Hp = 0;

        Lay(fight, ovicopter);

        var roster = fight.State.Enemies;
        Assert.Equal([0, 1, 2, 3, 4, CombatFactory.OvicopterOwnSlot], roster.Select(e => e.Slot));
        Assert.DoesNotContain(first[1], roster);
        Assert.Same(first[0], roster[2]);
        Assert.Same(first[2], roster[4]);
    }

    /// <summary>Only the first clutch's survivor at egg3 left: the next takes egg5, egg4, egg2.</summary>
    [Fact]
    public void ALoneSurvivorKeepsItsSlot()
    {
        var (fight, ovicopter) = LaidOnce();
        var first = fight.State.Enemies.Where(e => e.DefId == KE.ToughEgg).ToList();
        fight.State.Enemies.Remove(first[1]);
        fight.State.Enemies.Remove(first[2]);

        Lay(fight, ovicopter);

        var roster = fight.State.Enemies;
        Assert.Equal([1, 2, 3, 4, CombatFactory.OvicopterOwnSlot], roster.Select(e => e.Slot));
        Assert.Same(first[0], roster[1]);
    }
}
