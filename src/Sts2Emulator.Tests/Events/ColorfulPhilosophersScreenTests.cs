using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Colorful Philosophers offers its three card rewards with RewardsCmd.OfferCustom: one rewards
/// screen holding all three. Answering the first must come back to it for the other two (run
/// referee, seed 6HMCK2BXTJ: the game took two of the three, the emulator lost the rest).
/// </summary>
public class ColorfulPhilosophersScreenTests
{
    [Fact]
    public void AllThreeCardRewardsCanBeTaken()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventColorfulPhilosophers;
        engine.State.Phase = RunPhase.Event;
        int deck = engine.State.Deck.Count;

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));  // the next card reward
            Assert.Equal(RunPhase.CardReward, engine.State.Phase);
            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));  // its first card
        }

        Assert.Equal(deck + 3, engine.State.Deck.Count);
        Assert.False(RunRewardGenerator.HasPendingRewards(engine.State));
    }
}
