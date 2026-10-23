using System.Text.Json;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

/// <summary>The per-combat card uid (docs/rebuild/CARD_UID.md) and its invariant.</summary>
public class CardUidTests
{
    private static CardInstance Card(int id) => new(id, false);

    private static List<int> Uids(CombatState state) => CardUids.LiveCards(state).Select(c => c.UidOf()).ToList();

    /// <summary>A builder fight's piles are set by hand, so they are numbered as a combat start numbers them.</summary>
    private static Fight Numbered(Fight fight)
    {
        CardUids.NumberAll(fight.State);
        return fight;
    }

    [Fact]
    public void UidsAtCombatStartAreDistinctAndCounted()
    {
        var state = CombatFactory.NewCombat(7);
        var uids = Uids(state);

        Assert.NotEmpty(uids);
        Assert.DoesNotContain(0, uids);
        Assert.Equal(uids.Count, uids.Distinct().Count());
        Assert.Equal(uids.Count, state.LastCardUid);
        Assert.True(CardUidInvariant.Holds(state));
    }

    [Fact]
    public void APlayedCardKeepsItsUidThroughHandPlayDiscardAndExhaust()
    {
        var fight = Numbered(Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Enemy(hp: 60)
            .Energy(3));
        int strike = fight.State.Hand[0].UidOf();
        int counter = fight.State.LastCardUid;

        fight.Play(0, 0);

        var discarded = Assert.Single(fight.State.DiscardPile);
        Assert.Equal(IC.StrikeIronclad, discarded.DefId);
        Assert.Equal(strike, discarded.UidOf());

        fight.State.DiscardPile.RemoveAt(0);
        CardEffects.ExhaustCard(fight.State, discarded);

        Assert.Equal(strike, Assert.Single(fight.State.ExhaustPile).UidOf());
        Assert.Equal(counter, fight.State.LastCardUid);
        Assert.True(CardUidInvariant.Holds(fight.State));
    }

    [Fact]
    public void AnUpgradeOrACostChangeKeepsTheUid()
    {
        var card = Card(IC.StrikeIronclad).WithUid(5);

        Assert.Equal(5, (card with { Upgraded = true }).UidOf());
        Assert.Equal(5, (card with { CostForCombat = 0 }).UidOf());
        Assert.Equal(5, (card with { CostThisTurn = 1, FreeThisTurn = true }).UidOf());
    }

    [Fact]
    public void AGeneratedCardGetsANewUid()
    {
        var fight = Numbered(Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 60).Energy(3));
        int before = fight.State.LastCardUid;

        CardEffects.AddGeneratedCardToHand(fight.State, IC.DefendIronclad);
        CardEffects.AddGeneratedCardToHand(fight.State, IC.DefendIronclad);

        Assert.Equal([before + 1, before + 2], fight.State.Hand.Skip(1).Select(c => c.UidOf()));
        Assert.True(CardUidInvariant.Holds(fight.State));
    }

    [Fact]
    public void ACopyGetsANewUidAndTheOriginalKeepsItsOwn()
    {
        // Anger: CreateClone puts a copy of the played card into the discard pile.
        var fight = Numbered(Fight.Hand(Card(IC.Anger)).Enemy(hp: 60).Energy(3));
        int anger = fight.State.Hand[0].UidOf();
        int before = fight.State.LastCardUid;

        fight.Play(0, 0);

        var angers = fight.State.DiscardPile.Where(c => c.DefId == IC.Anger).Select(c => c.UidOf()).ToList();
        Assert.Equal(2, angers.Count);
        Assert.Contains(anger, angers);
        Assert.Contains(before + 1, angers);
        Assert.True(CardUidInvariant.Holds(fight.State));
    }

    [Fact]
    public void ACloneKeepsTheUidsAndContinuesTheCounter()
    {
        var state = CombatFactory.NewCombat(11);
        var clone = state.Clone();

        Assert.Equal(Uids(state), Uids(clone));
        Assert.Equal(state.LastCardUid, clone.LastCardUid);
        int next = state.LastCardUid + 1;
        Assert.Equal(next, clone.NewCardUid());
        Assert.Equal(next, state.NewCardUid());  // the two combats count on independently
    }

    [Fact]
    public void EqualityHashingPrintingAndSerializingIgnoreTheUid()
    {
        var a = new CardInstance(IC.Anger, true, CostBump: 1).WithUid(1);
        var b = a.WithUid(2);

        Assert.Equal(1, a.UidOf());
        Assert.Equal(2, b.UidOf());
        Assert.True(a == b);
        Assert.True(a.Equals(b));
        Assert.True(a.Equals((object)b));
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.Equal(a.GetHashCode(), a.WithUid(0).GetHashCode());
        Assert.Equal(a.ToString(), b.ToString());
        Assert.DoesNotContain("Uid", a.ToString());
        Assert.Equal(JsonSerializer.Serialize(a), JsonSerializer.Serialize(b));
        Assert.Equal(0, new List<CardInstance> { b }.IndexOf(a));
        Assert.Contains(a, new HashSet<CardInstance> { b });
        Assert.False(a == (b with { Upgraded = false }));
    }

    [Fact]
    public void TheInvariantCatchesAForcedDuplicateAndNormaliseRenumbersTheLaterCard()
    {
        var state = CombatFactory.NewCombat(3);
        var pile = state.Hand.Count >= 2 ? state.Hand : state.DrawPile;
        int first = pile[0].UidOf();
        pile[1] = pile[1].WithUid(first);
        long before = CardUidInvariant.NormalisedCards;

        var (kind, card) = CardUidInvariant.Check(state);
        Assert.Equal(CardUidInvariant.Breach.Duplicate, kind);
        Assert.Equal(first, card.UidOf());

        var log = new List<(CardUidInvariant.Breach, CardInstance)>();
        Assert.Equal(1, CardUidInvariant.Normalise(state, log));
        Assert.Single(log);
        Assert.Equal(first, pile[0].UidOf());
        Assert.Equal(state.LastCardUid, pile[1].UidOf());
        Assert.True(CardUidInvariant.Holds(state));
        Assert.True(CardUidInvariant.NormalisedCards >= before + 1);
        Assert.Equal(0, CardUidInvariant.Normalise(state));
    }

    [Fact]
    public void TheInvariantCatchesAMissingUidAndOneAboveTheCounter()
    {
        var state = CombatFactory.NewCombat(5);
        state.DrawPile.Add(Card(IC.StrikeIronclad));
        Assert.Equal(CardUidInvariant.Breach.NoUid, CardUidInvariant.Check(state).Kind);
        CardUidInvariant.Normalise(state);
        Assert.True(CardUidInvariant.Holds(state));

        state.HeldPlayedCard = Card(IC.DefendIronclad).WithUid(state.LastCardUid + 10);
        Assert.Equal(CardUidInvariant.Breach.AboveCounter, CardUidInvariant.Check(state).Kind);
        Assert.Equal(1, CardUidInvariant.Normalise(state));
        Assert.Equal(state.LastCardUid, state.HeldPlayedCard!.Value.UidOf());
    }
}
