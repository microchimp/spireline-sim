using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The merchant's potion row prices like the game's <c>MerchantPotionEntry.CalcCost</c>: the
/// rarity's base times <c>Rng.NextFloat(0.95f, 1.05f)</c>, the product taken in float and
/// rounded half to even (<c>Mathf.Round</c>). The oracle below is that rule written out on
/// its own; the test walks seeds until the old double-and-half-up price and the game's differ.
/// </summary>
public class ShopPotionPriceTests
{
    private static int Oracle(int baseCost, GameRng rng)
    {
        float jitter = (float)(rng.NextDouble() * (double)(1.05f - 0.95f) + 0.95f);
        return (int)Math.Round((float)baseCost * jitter);
    }

    [Theory]
    [InlineData(50)]
    [InlineData(75)]
    [InlineData(100)]
    public void ThePotionPriceIsTheGamesFloatProductRoundedHalfToEven(int baseCost)
    {
        int checkedSeeds = 0;
        for (uint seed = 0; seed < 3_000_000u; seed++)
        {
            int expected = Oracle(baseCost, new GameRng(seed, "shop"));
            int actual = RunRewardGenerator.PotionPrice(baseCost, new GameRng(seed, "shop"));
            Assert.True(expected == actual, $"base {baseCost}, seed {seed}: game {expected}, emulator {actual}");
            checkedSeeds++;
        }

        Assert.Equal(3_000_000, checkedSeeds);
    }
}
