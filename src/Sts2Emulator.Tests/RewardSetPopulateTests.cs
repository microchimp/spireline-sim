using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A RewardsSet populates every reward as it is offered (`RewardsSet.GenerateWithoutOffering`),
/// including the CardReward Prayer Wheel adds through `TryModifyRewards`. The emulator rolled
/// that extra reward only when the previous card screen was answered.
/// </summary>
public class RewardSetPopulateTests
{
    private static RunEngine FightWithRelic(string seed, int relic)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        engine.State.Relics.Add(new RelicInstance(relic));
        int action = Array.FindIndex(engine.State.MapNodeTypes, t => t == RunConstants.NodeNormal);
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(0, engine.DebugResolveCombat());
        return engine;
    }

    [Theory]
    [InlineData("QS2GYXRKWN")]
    [InlineData("QEY9JXP3KR")]
    public void PrayerWheelsExtraCardRewardIsRolledWhenTheSetIsOffered(string seed)
    {
        var engine = FightWithRelic(seed, RelicEffects.PrayerWheel);
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);

        // Already on the queue, three cards, before any card screen is answered.
        var queued = Assert.Single(engine.State.PendingCardOffers);
        Assert.Equal(3, queued.Cards.Count(c => c != 0));

        // Open the room's own card screen and answer it: the extra reward makes no draw.
        int draws = engine.State.PlayerRng.Rewards.CallCount;
        for (int a = 0; a < 8 && engine.State.Phase != RunPhase.CardReward; a++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.CardReward, engine.State.Phase);
        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));
        Assert.Equal(draws, engine.State.PlayerRng.Rewards.CallCount);
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        Assert.Equal(queued.Cards, Assert.Single(engine.State.PendingCardOffers).Cards);
    }

    [Fact]
    public void TheExtraRewardCostsTheSameDrawsWheneverItIsRolled()
    {
        var with = FightWithRelic("QS2GYXRKWN", RelicEffects.PrayerWheel);
        var without = FightWithRelic("QS2GYXRKWN", RelicEffects.Anchor);
        // Three cards of three draws each are the whole difference.
        Assert.Equal(
            without.State.PlayerRng.Rewards.CallCount + 9,
            with.State.PlayerRng.Rewards.CallCount
        );
    }
}
