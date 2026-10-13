using System.Collections.Generic;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A relic tally the game keeps as a <c>[SavedProperty]</c> and never resets at a fight's
/// start or end is run state: the next fight goes on counting from it. Pen Nib's attacks,
/// Joss Paper's exhausts, Nunchaku's attacks, Tuning Fork's skills and the turn counters of
/// Happy Flower, its fake, Pendulum and Pollinous Core were per-combat here, so every fight
/// started them at zero (run referee: Joss Paper and Pendulum differed after fights that
/// matched at every step, once relic counters were compared).
/// </summary>
public class SavedRelicTallyTests
{
    [Theory]
    [InlineData(RelicEffects.PenNib)]
    [InlineData(RelicEffects.JossPaper)]
    [InlineData(RelicEffects.Nunchaku)]
    [InlineData(RelicEffects.TuningFork)]
    [InlineData(RelicEffects.HappyFlower)]
    [InlineData(RelicEffects.FakeHappyFlower)]
    [InlineData(RelicEffects.Pendulum)]
    [InlineData(RelicEffects.PollinousCore)]
    public void TheTallySurvivesTheFight(int relicId)
    {
        var combat = new CombatState { Relics = [new RelicInstance(relicId, 2)] };
        var run = new List<RelicInstance> { new(relicId) };

        RelicEffects.CarryRunCountersBack(combat, run);

        Assert.Equal(2, run[0].Counter);
    }

    /// <summary>Shuriken's count is per turn and not saved: it stays behind with the fight.</summary>
    [Fact]
    public void APerCombatCountStaysBehind()
    {
        var combat = new CombatState { Relics = [new RelicInstance(RelicEffects.Shuriken, 2)] };
        var run = new List<RelicInstance> { new(RelicEffects.Shuriken) };

        RelicEffects.CarryRunCountersBack(combat, run);

        Assert.Equal(0, run[0].Counter);
    }
}
