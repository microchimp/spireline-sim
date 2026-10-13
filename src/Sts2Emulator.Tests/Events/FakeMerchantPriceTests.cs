using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Fake Merchant's six entries are priced as they are built (MerchantRelicEntry.SetModel ->
/// CalcCost: round(50 * Shops.NextFloat(0.85, 1.15))), on entering. The emulator charged a flat
/// 50 and drew nothing (run referee, seed CNV3YDEJXD: Shops six behind after the stall).
/// </summary>
public class FakeMerchantPriceTests
{
    [Fact]
    public void TheStallIsPricedOnEntering()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        s.CurrentActIndex = 1;
        s.Gold = 500;
        int shops = s.PlayerRng.Shops.CallCount;

        RunNonCombatEffects.BeginEvent(s, RunConstants.EventFakeMerchant);

        Assert.Equal(shops + 6, s.PlayerRng.Shops.CallCount);
        var prices = Enumerable.Range(0, 6).Select(i => RunNonCombatEffects.FakeMerchantPrice(s, i)).ToList();
        Assert.All(prices, p => Assert.InRange(p, 42, 58));
        s.Phase = RunPhase.FakeMerchant;
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(500 - prices[0], s.Gold);
    }
}
