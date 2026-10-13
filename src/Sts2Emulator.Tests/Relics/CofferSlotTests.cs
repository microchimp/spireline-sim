using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Alchemical Coffer: CreateRandomPotionsOutOfCombat(4) never repeats a potion within the call,
/// and each goes to originalSlotCount + i -- the new slots -- so an old empty slot stays empty
/// (run referee, seed RXVFG7FZ86).
/// </summary>
public class CofferSlotTests
{
    [Fact]
    public void FourDistinctPotionsInTheNewSlots()
    {
        var engine = new RunEngine();
        engine.Reset("COFFER0001");
        var s = engine.State;
        s.PotionSlots[0] = RunNonCombatEffects.NamedPotion("FirePotion");

        RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicAlchemicalCoffer);

        Assert.Equal(6, s.MaxPotionSlots);
        Assert.Equal(0, s.PotionSlots[1]);
        var added = s.PotionSlots.Skip(2).Take(4).ToList();
        Assert.All(added, p => Assert.NotEqual(0, p));
        Assert.Equal(4, added.Distinct().Count());
    }
}
