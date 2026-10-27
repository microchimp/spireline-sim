using System.Text.Json.Nodes;
using Spireline.Referee;
using Xunit;

namespace Spireline.Referee.Tests;

// The shape of the per-card and per-relic holdings in "anchor_state" (AnchorHoldings). The lockstep's reader
// (referee/lockstep.py, tests/python/test_lockstep_anchor_state.py) is written against exactly these keys.
public class AnchorHoldingsTests
{
    [Fact]
    public void AnEnchantedCardCarriesItsEnchantmentAndAmount()
    {
        var c = AnchorHoldings.DeckCard("INFERNO", upgraded: true, "SWIFT", 2);
        Assert.Equal("{\"id\":\"INFERNO\",\"up\":1,\"ench\":\"SWIFT\",\"amount\":2}", c.ToJsonString());
    }

    [Fact]
    public void APlainCardCarriesOnlyItsIdAndUpgrade()
    {
        var c = AnchorHoldings.DeckCard("STRIKE_IRONCLAD", upgraded: false, null, 0);
        Assert.Equal("{\"id\":\"STRIKE_IRONCLAD\",\"up\":0}", c.ToJsonString());
    }

    [Fact]
    public void MadSciencesTinkerFieldsAreCarried()
    {
        var c = AnchorHoldings.DeckCard("MAD_SCIENCE", false, null, 0, tinkerType: 2, tinkerRider: 5);
        Assert.Equal(2, (int)c["tinker_type"]!);
        Assert.Equal(5, (int)c["tinker_rider"]!);
        Assert.Null(c["ench"]);
    }

    [Fact]
    public void ARelicCarriesItsSavedCountersByTheGamesNames()
    {
        var r = AnchorHoldings.Relic("PENDULUM", new[] { new KeyValuePair<string, int>("TurnsSeen", 2) },
            new[] { new KeyValuePair<string, bool>("IsWax", false) });
        Assert.Equal("{\"id\":\"PENDULUM\",\"ints\":{\"TurnsSeen\":2},\"bools\":{\"IsWax\":false}}", r.ToJsonString());
    }

    [Fact]
    public void ARelicThatSavesNothingIsJustItsId()
    {
        Assert.Equal("{\"id\":\"BURNING_BLOOD\"}", AnchorHoldings.Relic("BURNING_BLOOD", null, null).ToJsonString());
        Assert.Equal("{\"id\":\"BURNING_BLOOD\"}", AnchorHoldings.Relic("BURNING_BLOOD", [], []).ToJsonString());
    }
}
