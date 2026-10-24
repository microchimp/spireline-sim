using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two powers that hang an extra reward on the room the fight was in: Royalties' gold
/// and Forbidden Grimoire's card removals.
/// </summary>
/// <remarks>
/// Both were applied by their cards and asked about by NOTHING, so neither reward ever
/// arrived. They are read off the COMBAT at the end of the fight, before the sync drops it,
/// and paid on the RUN -- which is why the owed amounts live there.
/// </remarks>
public class CombatEndRewardPowerTests
{
    private static RunEngine Run()
    {
        var engine = TestSwitches.Run(new RunEngine());
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>
    /// `RoyaltiesPower.AfterCombatEnd` adds a `GoldReward(Amount)` to the room: a row on the
    /// screen, not gold in the purse. It was paid at generation and queued again, twice over.
    /// </summary>
    [Fact]
    public void RoyaltiesIsOneRowOnTheScreenPaidWhenClaimed()
    {
        var engine = Run();
        int before = engine.State.Gold;
        engine.State.RoyaltiesGoldOwed = 30;

        RunRewardGenerator.GenerateCombatRewards(engine.State);

        Assert.Equal(before, engine.State.Gold);
        Assert.Equal(0, engine.State.RoyaltiesGoldOwed);
        Assert.Equal(new[] { 30 }, engine.State.PendingGoldRewards);
        int piles = 0;
        while (engine.State.RewardGold != 0)
        {
            Assert.True(RunRewardGenerator.ClaimRewardAtIndex(engine.State, 0));
            piles++;
        }

        Assert.Equal(2, piles);
        Assert.True(engine.State.Gold >= before + 30);
    }

    /// <summary>Ectoplasm zeroes it at the claim, which a bare `Gold +=` would have walked past.</summary>
    [Fact]
    public void EctoplasmStillZeroesIt()
    {
        var engine = Run();
        engine.State.Relics.Add(
            new RelicInstance(Sts2Emulator.Core.Effects.RelicEffects.Ectoplasm)
        );
        int before = engine.State.Gold;
        engine.State.RoyaltiesGoldOwed = 30;

        RunRewardGenerator.GenerateCombatRewards(engine.State);
        while (engine.State.RewardGold != 0)
        {
            RunRewardGenerator.ClaimRewardAtIndex(engine.State, 0);
        }

        Assert.Equal(before, engine.State.Gold);
    }

    /// <summary>
    /// `GenerateWithoutOffering` populates every reward in the room's list, and the Royalties
    /// row is one of its ExtraRewards: `GoldReward.Populate` draws `NextInt(30, 31)`, one
    /// Rewards value after the room's own -- so the room's own rewards are unchanged and the
    /// stream is one further on.
    /// </summary>
    [Fact]
    public void TheRoyaltiesRowDrawsOneRewardsValueAfterTheRoomsOwn()
    {
        var plain = Run();
        var owed = Run();
        owed.State.RoyaltiesGoldOwed = 30;

        RunRewardGenerator.GenerateCombatRewards(plain.State);
        RunRewardGenerator.GenerateCombatRewards(owed.State);

        Assert.Equal(plain.State.RewardGold, owed.State.RewardGold);
        Assert.Equal(plain.State.RewardCards, owed.State.RewardCards);
        Assert.Equal(
            plain.State.PlayerRng.Rewards.CallCount + 1,
            owed.State.PlayerRng.Rewards.CallCount
        );
    }

    /// <summary>
    /// The Grimoire's removal is answered as the reward screen is LEFT, one at a time, and
    /// the count is what decides how many screens the run sees.
    /// </summary>
    [Fact]
    public void TheGrimoireOpensARemovalOnTheWayOut()
    {
        var engine = Run();
        engine.State.CardRemovalsOwed = 2;
        engine.State.Phase = RunPhase.RelicReward;
        int deckBefore = engine.State.Deck.Count;

        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.Equal(RunPhase.TransformSelect, engine.State.Phase);
        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.Equal(1, engine.State.CardRemovalsOwed);

        Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, 0));
        Assert.Equal(deckBefore - 1, engine.State.Deck.Count);
    }

    /// <summary>Owing none, the screen is left the way it always was.</summary>
    [Fact]
    public void OwingNoneLeavesTheScreenUnchanged()
    {
        var engine = Run();
        engine.State.Phase = RunPhase.RelicReward;

        engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);

        Assert.NotEqual(RunPhase.TransformSelect, engine.State.Phase);
    }
}
