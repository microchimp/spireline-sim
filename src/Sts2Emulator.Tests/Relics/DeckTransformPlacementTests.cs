using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// CardCmd.Transform on a deck card removes the original and adds the replacement to the deck
/// with no index -- the bottom of the deck. The emulator wrote the new card over the old one, so
/// the deck's order stayed wrong for the rest of the run (run referee, seed CNV3YDEJXD: Archaic
/// Tooth's Break sat where the Bash had been, the game's at the bottom).
/// </summary>
public class DeckTransformPlacementTests
{
    [Fact]
    public void ArchaicToothsCardLandsAtTheBottom()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var deck = engine.State.Deck;
        int bash = deck.FindIndex(c => c.DefId == RunNonCombatEffects.NamedCard("Bash"));
        Assert.True(bash >= 0 && bash < deck.Count - 1);
        int count = deck.Count;
        var after = deck.Skip(bash + 1).ToList();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicArchaicTooth);

        Assert.Equal(count, deck.Count);
        Assert.DoesNotContain(deck, c => c.DefId == RunNonCombatEffects.NamedCard("Bash"));
        Assert.Equal(RunNonCombatEffects.NamedCard("Break"), deck[^1].DefId);
        Assert.Equal(after, deck.Skip(bash).Take(after.Count).ToList());
    }

    /// <summary>
    /// GetTranscendenceTransformedCard carries the starter's upgrade and enchantment onto the
    /// ancient card (run referee, seed CNV3YDEJXD: an upgraded Bash became an upgraded Break).
    /// </summary>
    [Fact]
    public void ArchaicToothCarriesTheUpgradeAndTheEnchantment()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var deck = engine.State.Deck;
        int bash = deck.FindIndex(c => c.DefId == RunNonCombatEffects.NamedCard("Bash"));
        deck[bash] = deck[bash] with { Upgraded = true, Enchantment = Enchantment.Sharp, EnchantAmount = 2 };

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicArchaicTooth);

        Assert.Equal(RunNonCombatEffects.NamedCard("Break"), deck[^1].DefId);
        Assert.True(deck[^1].Upgraded);
        Assert.Equal(Enchantment.Sharp, deck[^1].Enchantment);
        Assert.Equal(2, deck[^1].EnchantAmount);
    }
}
