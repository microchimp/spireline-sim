using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Calling Bell's three relics are <c>RelicReward(rarity, owner)</c>, and a relic reward
/// populates with <c>PullNextRelicFromFront(player, rarity)</c> -- the front of that rarity's
/// queue, no filter. The emulator asked for a relic the MERCHANT may sell, which is Welcome to
/// Wongos's own filter (<c>r.IsAllowedInShops</c>), so a relic that cannot be sold (Amethyst
/// Aubergine, Bowler Hat, Lucky Fysh and two more) at the front of its queue was passed over:
/// the wrong relic on the screen, and a queue one position off for every relic after it. The
/// runs that took the Bell were over-represented 3.5 times among the first card-reward
/// mismatches.
/// </summary>
public class CallingBellRewardTests
{
    private static bool Sellable(int relicId) => GeneratedData.Relics.Get(relicId).IsAllowedInShops;

    [Fact]
    public void ItOffersTheFrontOfEachRarityWhetherOrNotAMerchantCouldSellIt()
    {
        int checkedSeeds = 0;
        int unsellableAtTheFront = 0;
        for (int i = 0; i < 400; i++)
        {
            var engine = new RunEngine();
            engine.Reset($"CB{i:D8}");
            var allowed = RelicGrabBag.AllowedInSoloRun(engine.State.Floor);
            var rarities = new[] { RelicRarity.Common, RelicRarity.Uncommon, RelicRarity.Rare };
            var expected = rarities
                .Select(rarity => engine.State.RelicBag.Clone().Pull(rarity, true, allowed, null))
                .ToArray();

            RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicCallingBell);

            // Queued for the rewards screen in the game's order: Common, Uncommon, Rare.
            Assert.Equal(
                expected.Select(id => id!.Value),
                engine.State.PendingBonusRelicRewards.Take(3)
            );
            checkedSeeds++;
            unsellableAtTheFront += expected.Count(id => !Sellable(id!.Value));
        }

        // The sweep reached the case it exists for: some queue had an unsellable relic at its front.
        Assert.True(checkedSeeds == 400 && unsellableAtTheFront > 0);
    }
}
