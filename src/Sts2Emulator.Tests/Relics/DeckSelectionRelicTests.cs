using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// The five shop relics whose `AfterObtained` raises a DECK-SELECTION screen. All five go
// through the machinery Empty Cage and the events already use; what differs is the kind,
// the count, and the enchantment applied. Three of those enchantments did not exist.

public class DollysMirrorTests
{
    private static RunState WithDeck(params int[] cardIds)
    {
        var state = new RunState { Deck = [.. cardIds.Select(id => new CardInstance(id, false))] };
        return state;
    }

    [Fact]
    public void ItAsksWhichCardToCopy()
    {
        var state = WithDeck(SI.Slice, SI.Backstab);

        var followUp = RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.DollysMirror);

        Assert.Equal(RunFollowUp.TransformSelect, followUp);
        Assert.Equal(DeckSelection.Duplicate, state.PendingSelectionKind);
    }

    [Fact]
    public void TheChosenCardIsCopiedIntoTheDeck()
    {
        var state = WithDeck(SI.Slice, SI.Backstab);
        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.DollysMirror);

        RunNonCombatEffects.ApplyDeckSelection(state, 1);

        Assert.Equal(3, state.Deck.Count);
        Assert.Equal(2, state.Deck.Count(c => c.DefId == SI.Backstab));
    }

    /// <summary>Only Quest cards are excluded — a curse is a legal thing to copy.</summary>
    [Fact]
    public void ACurseCanBeCopied()
    {
        var state = WithDeck(10001); // Ascender's Bane
        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.DollysMirror);

        Assert.True(RunNonCombatEffects.CanSelectCard(state, 0));
    }
}

public class GnarledHammerTests
{
    [Fact]
    public void ItEnchantsUpToThreeCardsWithSharpThree()
    {
        var state = new RunState
        {
            Deck = [.. Enumerable.Range(0, 4).Select(_ => new CardInstance(SI.Slice, false))],
        };

        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.GnarledHammer);
        Assert.Equal(DeckSelection.Enchant, state.PendingSelectionKind);
        Assert.Equal(3, state.PendingSelectionCount);

        RunNonCombatEffects.ApplyDeckSelection(state, 0);

        var enchanted = state.Deck[0];
        Assert.Equal(Enchantment.Sharp, enchanted.Enchantment);
        // Three, not the two Self-Help Book applies -- the relic passes its own amount.
        Assert.Equal(3, enchanted.EnchantAmount);
    }
}

/// <summary>
/// Kifuda, Gnarled Hammer and Claws ask with `CardSelectorPrefs(prompt, 0, max)`: a minimum of
/// zero, so the screen can be closed with fewer picks, none included. The emulator took exactly
/// the maximum, with no way to stop (13 dense fights logged it as a mirror gap, FINDINGS).
/// </summary>
public class OptionalDeckSelectionTests
{
    private static RunEngine Picked(int relicId)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Relics.Add(new RelicInstance(relicId, 0));
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(engine.State, relicId));
        engine.State.Phase = RunPhase.TransformSelect;
        return engine;
    }

    private static int[] Mask(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return mask;
    }

    [Theory]
    [InlineData(RelicEffects.Kifuda, Enchantment.Adroit)]
    [InlineData(RelicEffects.GnarledHammer, Enchantment.Sharp)]
    public void DoneClosesTheScreenWithTheOnePickMade(int relicId, Enchantment enchantment)
    {
        var engine = Picked(relicId);
        int done = engine.State.Deck.Count;
        Assert.Equal(1, Mask(engine)[done]);
        int first = Enumerable.Range(0, done).First(i => RunNonCombatEffects.CanSelectCard(engine.State, i));

        engine.Step(first, -1, out _, out _, out _);
        Assert.Equal(1, Mask(engine)[done]);
        engine.Step(done, -1, out _, out _, out _);

        Assert.Equal(DeckSelection.None, engine.State.PendingSelectionKind);
        Assert.NotEqual(RunPhase.TransformSelect, engine.State.Phase);
        Assert.Equal(1, engine.State.Deck.Count(c => c.Enchantment == enchantment));
    }

    [Fact]
    public void DoneBeforeAnyPickTakesNone()
    {
        var engine = Picked(RelicEffects.Kifuda);

        engine.Step(engine.State.Deck.Count, -1, out _, out _, out _);

        Assert.Equal(DeckSelection.None, engine.State.PendingSelectionKind);
        Assert.DoesNotContain(engine.State.Deck, c => c.Enchantment == Enchantment.Adroit);
    }

    [Fact]
    public void ClawsTransformsAsFewAsThePlayerPicks()
    {
        // `Claws.AfterObtained`: `CardSelectorPrefs(prompt, 0, Cards)` with a manual confirm --
        // up to six, any number. One pick and DONE transforms that one card.
        var engine = Picked(RunConstants.RelicClaws);
        var before = engine.State.Deck.ToList();
        int done = before.Count;
        Assert.True(engine.State.PendingSelectionOptional);
        int first = Enumerable.Range(0, done).First(i => RunNonCombatEffects.CanSelectCard(engine.State, i));

        engine.Step(first, -1, out _, out _, out _);
        engine.Step(engine.State.Deck.Count, -1, out _, out _, out _);

        Assert.Equal(DeckSelection.None, engine.State.PendingSelectionKind);
        Assert.Equal(before.Count, engine.State.Deck.Count);
        // One card left the deck and one came in for it; the rest are the cards they were.
        var after = engine.State.Deck.Select(c => c.DefId).ToList();
        var gone = before.Select(c => c.DefId).ToList();
        foreach (int id in after.ToList())
        {
            if (gone.Remove(id))
            {
                after.Remove(id);
            }
        }

        Assert.Single(gone);
        Assert.Single(after);
    }

    [Fact]
    public void ACompulsoryOneOffersNoDone()
    {
        // Punch Dagger: `CardSelectorPrefs(prompt, 1)`.
        var engine = Picked(RelicEffects.PunchDagger);

        Assert.Equal(0, Mask(engine)[engine.State.Deck.Count]);
        Assert.False(engine.State.PendingSelectionOptional);
    }
}

/// <summary>
/// Kifuda's Adroit: block equal to its amount whenever the card is played, any card type.
/// </summary>
public class KifudaTests
{
    [Fact]
    public void ItEnchantsWithAdroitThree()
    {
        var state = new RunState { Deck = [new CardInstance(SI.Slice, false)] };

        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.Kifuda);
        RunNonCombatEffects.ApplyDeckSelection(state, 0);

        Assert.Equal(Enchantment.Adroit, state.Deck[0].Enchantment);
        Assert.Equal(3, state.Deck[0].EnchantAmount);
    }

    /// <summary>
    /// And Adroit blocks on play — unlike Nimble it is not restricted to Skills, so an
    /// Adroit attack blocks too.
    /// </summary>
    [Fact]
    public void AnAdroitCardBlocksWhenPlayed()
    {
        var fight = Fight
            .Hand(Card(SI.Slice) with { Enchantment = Enchantment.Adroit, EnchantAmount = 3 })
            .Energy(3)
            .Enemy(hp: 60);

        fight.Play();

        Assert.Equal(3, fight.State.PlayerBlock);
        Assert.Equal(60 - 6, fight.Enemy0.Hp);
    }
}

/// <summary>
/// Punch Dagger's Momentum: Attacks only, and it GROWS.
/// </summary>
/// <remarks>
/// `OnPlay` adds the amount to a running bonus and `EnchantDamageAdditive` pays it — and
/// the damage is read BEFORE `OnPlay` runs, so the play that adds the amount does not
/// benefit from it. A freshly enchanted card hits for its printed damage the first time.
/// </remarks>
public class PunchDaggerTests
{
    [Fact]
    public void ItEnchantsOneAttackWithMomentumFive()
    {
        var state = new RunState { Deck = [new CardInstance(SI.Slice, false)] };

        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.PunchDagger);
        RunNonCombatEffects.ApplyDeckSelection(state, 0);

        Assert.Equal(Enchantment.Momentum, state.Deck[0].Enchantment);
        Assert.Equal(5, state.Deck[0].EnchantAmount);
    }

    /// <summary>Attacks only — `CanEnchantCardType` returns Attack alone.</summary>
    [Fact]
    public void ASkillCannotTakeIt()
    {
        var state = new RunState
        {
            Deck = [new CardInstance(SI.DefendSilent, false), new CardInstance(SI.Slice, false)],
        };
        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.PunchDagger);

        Assert.False(RunNonCombatEffects.CanSelectCard(state, 0));
        Assert.True(RunNonCombatEffects.CanSelectCard(state, 1));
    }

    [Fact]
    public void TheFirstPlayIsPlainAndTheSecondIsFiveHigher()
    {
        var fight = Fight
            .Hand(Card(SI.Slice) with { Enchantment = Enchantment.Momentum, EnchantAmount = 5 })
            .Energy(9)
            .Enemy(hp: 200);

        int before = fight.Enemy0.Hp;
        fight.Play();
        Assert.Equal(before - 6, fight.Enemy0.Hp);

        var grown = fight.State.DiscardPile.Single(c => c.DefId == SI.Slice);
        Assert.Equal(5, grown.BonusDamage);

        fight.State.Hand.Add(grown);
        before = fight.Enemy0.Hp;
        fight.Play(0);

        Assert.Equal(before - 11, fight.Enemy0.Hp);
    }
}

/// <summary>
/// Royal Stamp's RoyallyApproved: `OnEnchant` adds Innate AND Retain, and it has no
/// play-time behaviour of its own.
/// </summary>
public class RoyalStampTests
{
    [Fact]
    public void ItEnchantsOneCard()
    {
        var state = new RunState { Deck = [new CardInstance(SI.Slice, false)] };

        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.RoyalStamp);
        RunNonCombatEffects.ApplyDeckSelection(state, 0);

        Assert.Equal(Enchantment.RoyallyApproved, state.Deck[0].Enchantment);
    }

    /// <summary>Attacks and Skills only — a Power cannot take it.</summary>
    [Fact]
    public void APowerCannotTakeIt()
    {
        var state = new RunState
        {
            Deck = [new CardInstance(IC.Inflame, false), new CardInstance(SI.Slice, false)],
        };
        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.RoyalStamp);

        Assert.False(RunNonCombatEffects.CanSelectCard(state, 0));
        Assert.True(RunNonCombatEffects.CanSelectCard(state, 1));
    }

    /// <summary>Both keywords, from one enchantment.</summary>
    [Fact]
    public void ItGrantsInnateAndRetain()
    {
        var stamped = Card(SI.Slice) with { Enchantment = Enchantment.RoyallyApproved };

        Assert.True(stamped.IsInnate());
        Assert.True(stamped.IsRetained());
    }

    [Fact]
    public void AStampedCardSurvivesTheEndOfTheTurn()
    {
        var fight = Fight
            .Hand(Card(SI.Slice) with { Enchantment = Enchantment.RoyallyApproved })
            .Energy(0);
        fight.State.PlayerHp = 999;

        fight.EndTurn();

        Assert.Contains(fight.State.Hand, c => c.DefId == SI.Slice);
    }
}
