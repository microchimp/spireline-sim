using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Which block gains the `Move` modifiers see. `UnmovablePower`, `Vambrace` and `PaelsLegion`
/// each open `ModifyBlockMultiplicative` with `if (!props.IsCardOrMonsterMove()) return 1`:
/// a card's block (`BlockProps.card`, `cardUnpowered`) is theirs, a power's, relic's or
/// orb's (`BlockProps.nonCardUnpowered`, `Unpowered` alone) is not -- and `Unmovable` counts
/// only the `Move` gains in the history when it decides whether its doubling is spent.
/// </summary>
/// <remarks>
/// The emulator doubled every gain while a charge was left and used one up on each. Plating
/// under Unmovable paid 14 where the game pays 7, and night04's W3LQBXMEG1 had the Corpse
/// Slug's Whip Slap (8 x 2) leave the player alive on 4 HP where the game killed him.
/// </remarks>
public class BlockMoveFlagTests
{
    private static Fight WithUnmovable(int charges = 1) =>
        Fight.Hand().Enemy().PlayerBuff(BuffId.UnmovablePower, charges);

    /// <summary>The bug: Plating's end-of-turn block is `Unpowered` alone, so it is not doubled.</summary>
    [Fact]
    public void AnUnpoweredPowerGainIsNotDoubledByUnmovable()
    {
        var fight = WithUnmovable();

        int gained = CardEffects.GainUnpoweredBlock(fight.State, 7);

        Assert.Equal(7, gained);
        Assert.Equal(7, fight.State.PlayerBlock);
    }

    /// <summary>
    /// And it does not spend the charge either: the card played after Plating (or after a
    /// Frost orb's passive) is still the "first" block of the turn as the game counts it.
    /// </summary>
    [Fact]
    public void AnUnpoweredGainDoesNotUseUpAnUnmovableCharge()
    {
        var fight = WithUnmovable();

        CardEffects.GainUnpoweredBlock(fight.State, 7);
        int card = CardEffects.GainBlock(fight.State, 5);

        Assert.Equal(10, card);
        Assert.Equal(17, fight.State.PlayerBlock);
    }

    /// <summary>
    /// The history counts every `Move` gain this turn, not just the ones Unmovable was
    /// standing for: a Defend played BEFORE the Unmovable card has already used the
    /// charge, so the next Defend is not doubled.
    /// </summary>
    [Fact]
    public void ACardGainBeforeUnmovableWasPlayedStillCounts()
    {
        var fight = Fight.Hand().Enemy();

        CardEffects.GainBlock(fight.State, 5);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.UnmovablePower, 1);
        int later = CardEffects.GainBlock(fight.State, 5);

        Assert.Equal(5, later);
        Assert.Equal(10, fight.State.PlayerBlock);
    }

    /// <summary>Unchanged: two charges double the first two card gains and not the third.</summary>
    [Fact]
    public void EachChargeDoublesOneCardGain()
    {
        var fight = WithUnmovable(charges: 2);

        int a = CardEffects.GainBlock(fight.State, 5);
        int b = CardEffects.GainBlock(fight.State, 5);
        int c = CardEffects.GainBlock(fight.State, 5);

        Assert.Equal((10, 10, 5), (a, b, c));
    }

    /// <summary>
    /// Entrench is `Unpowered | Move` (`BlockProps.cardUnpowered`): Dexterity stays out of
    /// it, but it IS a card's block, so Unmovable doubles it.
    /// </summary>
    [Fact]
    public void EntrenchIsUnpoweredButStillACardGain()
    {
        var fight = WithUnmovable().PlayerBuff(BuffId.Dexterity, 3);
        fight.State.PlayerBlock = 8;

        int gained = CardEffects.GainCardUnpoweredBlock(fight.State, 8);

        Assert.Equal(16, gained);
        Assert.Equal(24, fight.State.PlayerBlock);
    }

    /// <summary>
    /// Vambrace reads the same flag: Entrench, a card's block, is doubled by it; Plating's
    /// tick, which lands first, neither is nor uses the relic up.
    /// </summary>
    [Fact]
    public void VambraceDoublesACardsUnpoweredBlockAndSkipsAPowers()
    {
        var fight = Fight.Hand().Enemy();
        fight.State.Relics.Add(new RelicInstance(RelicEffects.Vambrace));

        int plating = CardEffects.GainUnpoweredBlock(fight.State, 4);
        int entrench = CardEffects.GainCardUnpoweredBlock(fight.State, 8);

        Assert.Equal((4, 16), (plating, entrench));
    }

    /// <summary>
    /// The same split on the power side. Pillar of Creation and Shroud are `Unpowered` in
    /// the game, and the emulator's comments said so while its calls were the powered
    /// ones: Dexterity added to them and Frail took from them.
    /// </summary>
    [Fact]
    public void PillarOfCreationAndShroudIgnoreDexterityAndFrail()
    {
        var fight = Fight.Hand().Enemy().PlayerBuff(BuffId.Dexterity, 4).PlayerBuff(BuffId.Frail, 2);
        fight.PlayerBuff(BuffId.PillarOfCreation, 3).PlayerBuff(BuffId.Shroud, 2);

        CardEffects.NoteGeneratedCard(fight.State);
        CardEffects.ApplyPowerDoomRiders(fight.State, fight.Enemy0, 5);

        Assert.Equal(5, fight.State.PlayerBlock);
    }

    /// <summary>
    /// Danse Macabre pays when a card of resolved cost two or more is played, and its block
    /// is `Unpowered`: Bash (2) under Dexterity 3 still gains exactly the power's amount.
    /// </summary>
    [Fact]
    public void DanseMacabreBlockIgnoresDexterity()
    {
        var fight = Fight
            .Hand(new CardInstance(IC.Bash, false))
            .Enemy()
            .Energy(3)
            .PlayerBuff(BuffId.Dexterity, 3)
            .PlayerBuff(BuffId.DanseMacabre, 4);

        fight.Play(index: 0, target: 0);

        Assert.Equal(4, fight.State.PlayerBlock);
    }
}
