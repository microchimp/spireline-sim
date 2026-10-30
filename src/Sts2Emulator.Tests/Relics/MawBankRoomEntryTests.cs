using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Maw Bank's two hooks as the game runs them. The bank pays its 12 gold in
/// <c>AfterRoomEntered</c>, and only for the run's base room, so one walk onto a map point is
/// one payment whatever the point turns out to be: a "?" is resolved to its room first and
/// that room is entered once. It closes in <c>AfterItemPurchased</c> with gold spent, which
/// every merchant entry raises when a purchase succeeds, the Fake Merchant's stall included.
/// </summary>
/// <remarks>
/// The nightly referee's four gold groups of 2026-10-29 (combat, event, rest and shop, 2,430
/// of 18,076 weight) were all the copy holding 12, 24 or 36 more gold than the game: the copy
/// paid the bank twice on every "?" (once for the map point, once more for the room it
/// resolved to).
/// </remarks>
public class MawBankRoomEntryTests
{
    private static readonly string[] Seeds = ["NXV45HW43K", "ABCDEF", "UNS55LCMKP"];

    private static RunEngine OnTheMapWithTheBank(string seed) => OnTheMap(seed, bank: true);

    private static RunEngine OnTheMap(string seed, bool bank)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        if (bank)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.MawBank);
        }

        engine.State.Phase = RunPhase.Map;
        return engine;
    }

    /// <summary>
    /// The gold the bank adds to one walk: the same walk with and without it, so what the room gives
    /// of its own (a chest's gold) cancels out.
    /// </summary>
    private static int BankPaysOn(string seed, int nodeType, int unknownsVisited = 0)
    {
        int GoldAfter(bool bank)
        {
            var engine = OnTheMap(seed, bank);
            engine.State.UnknownMapPointsVisited = unknownsVisited;
            int option = FirstOption(engine);
            engine.State.MapNodeTypes[option] = nodeType;
            int before = engine.State.Gold;
            Assert.Equal(0, engine.Step(option, -1, out _, out _, out _));
            return Gold(engine) - before;
        }

        return GoldAfter(bank: true) - GoldAfter(bank: false);
    }

    private static int FirstOption(RunEngine engine)
    {
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] != RunConstants.NodeNone)
            {
                return i;
            }
        }

        throw new InvalidOperationException("no map option");
    }

    /// <summary>The gold the player holds wherever the step left them: a fight keeps its own copy.</summary>
    private static int Gold(RunEngine engine) => engine.State.ActiveCombat?.PlayerGold ?? engine.State.Gold;

    [Theory]
    [InlineData(RunConstants.NodeEvent)]
    [InlineData(RunConstants.NodeNormal)]
    [InlineData(RunConstants.NodeRest)]
    [InlineData(RunConstants.NodeShop)]
    [InlineData(RunConstants.NodeRelic)]
    public void EnteringAnyRoomPaysTheBankOnce(int nodeType)
    {
        foreach (string seed in Seeds)
        {
            Assert.Equal(12, BankPaysOn(seed, nodeType));
        }
    }

    /// <summary>A "?" pays once whatever it resolves to.</summary>
    [Fact]
    public void AnUnknownPaysOnceWhateverItBecomes()
    {
        foreach (string seed in new[] { "NXV45HW43K", "ABCDEF", "UNS55LCMKP", "AAB", "HEADLESS1", "6C6CCTGSMZ", "Z5KW9Y5YA9" })
        {
            for (int visited = 0; visited < 6; visited++)
            {
                Assert.Equal(12, BankPaysOn(seed, RunConstants.NodeEvent, visited));
            }
        }
    }

    /// <summary>
    /// The Fake Merchant's stall is a merchant inventory: a relic bought there raises
    /// <c>AfterItemPurchased</c> with the gold spent, and the bank closes as at a shop.
    /// </summary>
    [Fact]
    public void ABuyAtTheFakeMerchantClosesTheBank()
    {
        var engine = OnTheMapWithTheBank("NXV45HW43K");
        engine.State.EventId = RunConstants.EventFakeMerchant;
        engine.State.Phase = RunPhase.FakeMerchant;
        engine.State.Gold = 500;
        var stall = RunNonCombatEffects.FakeMerchantStock(engine.State);
        int slot = stall.FindIndex(relic => relic != 0);
        int price = RunNonCombatEffects.FakeMerchantPrice(engine.State, slot);

        Assert.Equal(0, engine.Step(slot, -1, out _, out _, out _));
        Assert.Equal(500 - price, engine.State.Gold);

        RunNonCombatEffects.PayMawBank(engine.State);

        Assert.Equal(500 - price, engine.State.Gold);
    }
}
