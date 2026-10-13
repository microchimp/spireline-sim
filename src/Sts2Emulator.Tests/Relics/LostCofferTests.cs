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

/// <summary>
/// Kaleidoscope builds both card rewards as it is obtained: Niche shuffled once per reward and
/// three Rewards draws per card, before either is claimed (run referee, seed K99RQF4V67).
/// </summary>
public class KaleidoscopeDrawTimingTests
{
    [Fact]
    public void BothRewardsAreRolledAtPickup()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        var s = engine.State;
        int rewards = s.PlayerRng.Rewards.CallCount, niche = s.Rng.Niche.CallCount;

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // Kaleidoscope (this seed's first option)

        Assert.Equal(rewards + 2 * 3 * 3, s.PlayerRng.Rewards.CallCount);
        Assert.True(s.Rng.Niche.CallCount > niche);
    }
}
