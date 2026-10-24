using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>`PotionModel.Usage` for every potion the game has (ids 1-63), as the emulator's usage table holds it.</summary>
public class PotionUsageTableTests
{
    [Fact]
    public void UsageTable_MatchesTheGame_ForEveryPotion()
    {
        // The game's overrides of PotionModel.Usage: four AnyTime potions, one Automatic, the rest CombatOnly.
        string[] anyTime = ["BloodPotion", "EntropicBrew", "FoulPotion", "FruitJuice"];
        for (int id = 1; id <= 63; id++)
        {
            string name = Potions.Get(id).Name;
            var expected =
                name == "FairyInABottle" ? PotionUsage.Automatic
                : anyTime.Contains(name) ? PotionUsage.AnyTime
                : PotionUsage.CombatOnly;
            Assert.True(expected == PotionEffects.UsageOf(id), $"{name}: expected {expected}");
            Assert.Equal(expected != PotionUsage.Automatic, PotionEffects.CanUseInCombat(id));
        }
    }
}
