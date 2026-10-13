using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Potions on one reward screen are rolled together, when the screen is offered:
/// <c>RewardsSet.GenerateWithoutOffering</c> populates every reward, and
/// <c>PotionReward.Populate</c> is where a potion is rolled. The emulator rolled a pair's
/// second potion only once the first was answered -- so leaving the screen never drew it, and
/// the Rewards stream fell behind the game's for the rest of the run. Found by the run referee
/// (seed 6BJ7MYTYCK, Whispering Hollow: the game's rewards counter two ahead after the option).
/// </summary>
public class PotionSetRollTests
{
    private static RunEngine AtHollow()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventWhisperingHollow;
        engine.State.Phase = RunPhase.Event;
        return engine;
    }

    [Fact]
    public void BothPotionsAreRolledWhenTheScreenOpens()
    {
        var engine = AtHollow();
        int before = engine.State.PlayerRng.Rewards.CallCount;
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        int afterOffer = engine.State.PlayerRng.Rewards.CallCount;
        Assert.NotEqual(0, engine.State.RewardPotion);
        Assert.Single(engine.State.PendingPotionRewards);
        Assert.NotEqual(0, engine.State.PendingPotionRewards[0]);

        // Leaving the screen draws nothing more: both were drawn as it opened.
        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));
        Assert.Equal(afterOffer, engine.State.PlayerRng.Rewards.CallCount);

        // Each potion is the same number of draws, so the pair is twice one.
        var single = new RunEngine();
        single.Reset("ABCDEF");
        RunRewardGenerator.NextPotion(single.State, single.State.PlayerRng.Rewards);
        Assert.Equal(2 * (single.State.PlayerRng.Rewards.CallCount - before), afterOffer - before);
    }
}
