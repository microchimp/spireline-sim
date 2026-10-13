using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Leaving a rewards screen drops everything still on it. Each of the game's RewardsSets is a
/// screen of its own; the emulator kept a skipped fight's card reward pending, and it came back
/// on the next screen the run opened (run referee, seed 6BJ7MYTYCK: an event's potion screen
/// offered the previous fight's three cards).
/// </summary>
public class RewardScreenLeaveTests
{
    [Theory]
    [InlineData("QS2GYXRKWN")]
    [InlineData("QEY9JXP3KR")]
    public void ASkippedCardRewardDoesNotFollowTheRun(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        int action = Array.FindIndex(engine.State.MapNodeTypes, t => t == RunConstants.NodeNormal);
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(0, engine.DebugResolveCombat());
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        Assert.True(engine.State.RewardCardPending);

        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Map, engine.State.Phase);
        Assert.False(engine.State.RewardCardPending);
        Assert.False(RunRewardGenerator.HasPendingRewards(engine.State));
    }
}
