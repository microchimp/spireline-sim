using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Lost Coffer's AfterObtained offers a CardReward made with CardCreationSource.Other (base odds:
/// the rare offset is neither read nor grown; rarity, card and upgrade roll per card) and a
/// PotionReward, however the coffer is obtained. Found by the run referee: as Neow's option it
/// moved the offset (UK2BJLWEV1); claimed off Neow's Bones' screen it gave nothing (FHTYPREMZ9).
/// </summary>
public class LostCofferTests
{
    [Fact]
    public void ClaimedOffAScreenItOffersACardRewardAndAPotion()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        RunRewardGenerator.ClearRewardScreen(s);
        s.RelicReward = RunConstants.RelicLostCoffer;
        s.Phase = RunPhase.RelicReward;
        double offset = s.CardRarityOffset;
        int rewards = s.PlayerRng.Rewards.CallCount;

        Assert.True(RunRewardGenerator.ClaimRewardAtIndex(s, 0));

        Assert.Contains(s.Relics, r => r.DefId == RunConstants.RelicLostCoffer);
        Assert.True(s.RewardCardPending);
        Assert.NotEqual(0, s.RewardPotion);
        Assert.Equal(offset, s.CardRarityOffset);
        // Three draws a card (rarity, card, upgrade), then the potion's two.
        Assert.Equal(rewards + 3 * 3 + 2, s.PlayerRng.Rewards.CallCount);
    }
}
