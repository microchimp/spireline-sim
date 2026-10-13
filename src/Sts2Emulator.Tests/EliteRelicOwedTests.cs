using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// An elite's rewards screen carries its relic beside the gold, potion and card, and the player may
/// claim them in any order. The emulator also kept a "relic owed after the card" flag
/// (<c>PendingRelicReward</c>) that claiming the relic first, or leaving the screen, never cleared;
/// the next card reward that ended without returning to a screen (an event's, such as The Future of
/// Potions') then pulled ANOTHER relic -- a rarity roll and a relic off the Rewards stream the game
/// never makes (Spire Codex runs: 68 Future of Potions floors after an elite).
/// </summary>
public class EliteRelicOwedTests
{
    private static RunEngine OnTheMap()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        return engine;
    }

    private static void OpenEliteRewards(RunEngine engine)
    {
        var s = engine.State;
        s.CurrentNodeType = RunConstants.NodeElite;
        s.LastResolvedRoomType = RunConstants.NodeElite;
        RunRewardGenerator.GenerateCombatRewards(s);
        Assert.NotEqual(0, s.RelicReward);
    }

    [Fact]
    public void ClaimingTheEliteRelicFirstClearsTheDebt()
    {
        var engine = OnTheMap();
        OpenEliteRewards(engine);
        var s = engine.State;
        int relicAction = (s.RewardGold != 0 ? 1 : 0) + (s.RewardPotion != 0 ? 1 : 0);

        Assert.Equal(0, engine.Step(relicAction, -1, out _, out _, out _));

        Assert.False(s.PendingRelicReward);
    }

    [Fact]
    public void LeavingTheScreenClearsTheDebt()
    {
        var engine = OnTheMap();
        OpenEliteRewards(engine);

        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.False(engine.State.PendingRelicReward);
    }

    [Fact]
    public void AnEventCardRewardAfterAnEliteDoesNotPayAnotherRelic()
    {
        var engine = OnTheMap();
        OpenEliteRewards(engine);
        var s = engine.State;
        int relicAction = (s.RewardGold != 0 ? 1 : 0) + (s.RewardPotion != 0 ? 1 : 0);
        engine.Step(relicAction, -1, out _, out _, out _);
        engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);

        s.PotionSlots[0] = RunNonCombatEffects.NamedPotion("FirePotion");
        s.PotionSlots[1] = RunNonCombatEffects.NamedPotion("FlexPotion");
        RunNonCombatEffects.BeginEvent(s, RunConstants.EventTheFutureOfPotions);
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(RunPhase.CardReward, s.Phase);
        int rewards = s.PlayerRng.Rewards.CallCount;

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.NotEqual(RunPhase.RelicReward, s.Phase);
        Assert.Equal(0, s.RelicReward);
        Assert.Equal(rewards, s.PlayerRng.Rewards.CallCount);
    }
}
