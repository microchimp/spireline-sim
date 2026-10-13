using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A card reward picks by position in its pool, so the pool's order is part of the result.
/// The game's order: the player's pool, then (Prismatic Gem) the other characters in
/// ModelDb.AllCharacters order, then (Dingy Rug) colourless, whichever relic's hook runs first.
/// </summary>
public class CardRewardPoolOrderTests
{
    private static int[] Concat(params int[][] pools) => pools.SelectMany(p => p).ToArray();

    [Fact]
    public void WithoutEitherRelicItIsTheIroncladPool() =>
        Assert.Equal(GeneratedData.CardPools.Ironclad.ToArray(), RunRewardGenerator.CardRewardPool(false, false));

    [Fact]
    public void PrismaticGemAddsTheOtherCharactersInTheGamesOrder() =>
        Assert.Equal(
            Concat(GeneratedData.CardPools.Ironclad.ToArray(), GeneratedData.CardPools.Silent.ToArray(), GeneratedData.CardPools.Regent.ToArray(),
                GeneratedData.CardPools.Necrobinder.ToArray(), GeneratedData.CardPools.Defect.ToArray()),
            RunRewardGenerator.CardRewardPool(false, true));

    [Fact]
    public void DingyRugsColourlessComesLastEvenWithPrismaticGem() =>
        Assert.Equal(
            Concat(GeneratedData.CardPools.Ironclad.ToArray(), GeneratedData.CardPools.Silent.ToArray(), GeneratedData.CardPools.Regent.ToArray(),
                GeneratedData.CardPools.Necrobinder.ToArray(), GeneratedData.CardPools.Defect.ToArray(), GeneratedData.CardPools.Colorless.ToArray()),
            RunRewardGenerator.CardRewardPool(true, true));
}
