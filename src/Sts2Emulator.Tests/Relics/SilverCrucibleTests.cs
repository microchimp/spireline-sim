using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Silver Crucible: the first three card rewards of the run come upgraded.
/// </summary>
/// <remarks>
/// It is <c>TryModifyCardRewardOptionsLate</c> — the relic edits the three cards AFTER
/// <c>CardFactory.CreateForReward</c> has finished rolling them, so it changes what the
/// screen shows and nothing about what was rolled. The emulator had the upgrade written as
/// the first term of a <c>||</c> chain whose second term was the upgrade ROLL, and C#
/// short-circuits: a run holding the Crucible spent two rewards-stream values per card
/// where the game spends three. The first card survived that, and every card after it in
/// the run was somebody else's.
/// </remarks>
public class SilverCrucibleTests
{
    private static RunEngine AtFirstCardReward(string seed, bool withCrucible)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        if (withCrucible)
        {
            engine.State.Relics.Add(new RelicInstance(RunConstants.RelicSilverCrucible, 3));
        }

        RunRewardGenerator.PopulateCardReward(engine.State);
        return engine;
    }

    /// <summary>
    /// The whole point: holding the relic must not move the stream, so the three cards
    /// offered are the same three either way — only their upgrade flags differ.
    /// </summary>
    [Theory]
    [InlineData("RRRR6WR3C4")]
    [InlineData("J09SPL8Y3V")]
    [InlineData("NXV45HW43K")]
    public void ItChangesTheUpgradesWithoutChangingTheCards(string seed)
    {
        var without = AtFirstCardReward(seed, withCrucible: false).State;
        var with = AtFirstCardReward(seed, withCrucible: true).State;

        Assert.Equal(without.RewardCards, with.RewardCards);
        Assert.All(with.RewardUpgraded, upgraded => Assert.True(upgraded));
    }

    /// <summary>
    /// Three rewards and then it is spent, so the fourth screen rolls its own upgrades.
    /// </summary>
    [Fact]
    public void ItUpgradesOnlyThreeRewards()
    {
        var engine = new RunEngine();
        engine.Reset("RRRR6WR3C4");
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicSilverCrucible, 3));

        for (int screen = 0; screen < 3; screen++)
        {
            RunRewardGenerator.PopulateCardReward(engine.State);
            Assert.All(engine.State.RewardUpgraded, upgraded => Assert.True(upgraded));
        }

        RunRewardGenerator.PopulateCardReward(engine.State);
        Assert.All(engine.State.RewardUpgraded, upgraded => Assert.False(upgraded));
    }

    // ---- Lost Coffer ----------------------------------------------------------------------

    private static RunEngine AtBonesCrucibleThenCoffer(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.Phase = RunPhase.Ancient;
        engine.State.NeowOptions[0] = RunConstants.RelicNeowsBones;
        // Choose the Bones, then claim the two relics it offers: on this seed the Crucible, then the Coffer.
        engine.Step(0, -1, out _, out _, out _);
        engine.Step(0, -1, out _, out _, out _);
        engine.Step(0, -1, out _, out _, out _);
        return engine;
    }

    /// <summary>
    /// Lost Coffer's card reward is a CardReward like any other: IsCardReward is on, so the Crucible's late hook
    /// upgrades it and counts a use. It used to be rolled by a path of its own that never asked the hook (run
    /// referee, W4CHJP64E3: the game's first card of the run was BLOOD_WALL+).
    /// </summary>
    [Fact]
    public void ALostCoffersCardRewardTakesTheCruciblesUse()
    {
        var state = AtBonesCrucibleThenCoffer("W4CHJP64E3").State;

        Assert.Contains(state.Relics, relic => relic.DefId == RunConstants.RelicLostCoffer);
        Assert.True(state.RewardCardPending);
        Assert.All(state.RewardUpgraded, upgraded => Assert.True(upgraded));
        Assert.Equal(
            2,
            state.Relics.Single(relic => relic.DefId == RunConstants.RelicSilverCrucible).Counter
        );
    }

    private static string CoffersOffer(string seed, bool egg)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        if (egg)
        {
            engine.State.Relics.Add(new RelicInstance(RunConstants.RelicMoltenEgg));
        }

        RunRewardGenerator.OfferLostCofferRewards(engine.State);
        var s = engine.State;
        // The offer, the potion, and the next value of the rewards stream: the draws are part of the contract.
        return string.Join(",", s.RewardCards)
            + "|" + string.Join(",", s.RewardUpgraded.Select(u => u ? 1 : 0))
            + "|" + s.RewardPotion
            + "|" + s.PlayerRng.Rewards.NextFloat().ToString("R");
    }

    /// <summary>With no Crucible (and no Wing Charm) the coffer's cards, upgrades, potion and stream position are what they were.</summary>
    [Theory]
    [InlineData("RRRR6WR3C4", false, "517,189,50|0,0,0|5|0.41242906")]
    [InlineData("J09SPL8Y3V", false, "349,238,404|0,0,0|6|0.84741414")]
    [InlineData("NXV45HW43K", false, "538,155,358|0,0,0|53|0.7359883")]
    [InlineData("J09SPL8Y3V", true, "349,238,404|1,0,0|6|0.84741414")]
    public void APlainCoffersOfferIsUnchanged(string seed, bool egg, string expected)
    {
        Assert.Equal(expected, CoffersOffer(seed, egg));
    }
}
