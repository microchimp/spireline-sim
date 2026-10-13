using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Future of Potions: the card TYPE per potion is rolled off the event's stream as the
/// options are built (PotionToCardType, every potion in the belt in slot order); the three
/// CARDS come off PlayerRng.Rewards, as any CardReward's do (run referee, seed 6HMCK2BXTJ).
/// </summary>
public class FutureOfPotionsStreamTests
{
    [Fact]
    public void TypesAtEntryCardsOffTheRewardsStream()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        s.PotionSlots[0] = RunNonCombatEffects.NamedPotion("FirePotion");
        s.PotionSlots[1] = RunNonCombatEffects.NamedPotion("FruitJuice");
        RunNonCombatEffects.BeginEvent(s, RunConstants.EventTheFutureOfPotions);

        Assert.Equal(2, s.EventRandomOffer.Length);
        int rewards = s.PlayerRng.Rewards.CallCount;
        var type = (CardType)s.EventRandomOffer[1];

        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));

        Assert.Equal(rewards + 3, s.PlayerRng.Rewards.CallCount);
        Assert.All(s.RewardCards, c => Assert.Equal(type, Sts2Emulator.GeneratedData.Cards.Get(c).Type));
        Assert.All(s.RewardUpgraded, Assert.True);
    }
}
