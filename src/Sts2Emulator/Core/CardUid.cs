using System.Globalization;

namespace Sts2Emulator.Core;

/// <summary>
/// Marks a field that holds identity bookkeeping for the Tier 1 kernel (a card's per-combat uid, the
/// combat's uid counter) and nothing an output reads. The full-state walk (tools/emudump/StateWalk.cs)
/// leaves these fields out, so the state hash and every dump stay what they were before the uid existed.
/// </summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class IdentityOnlyAttribute : Attribute;

/// <summary>
/// A card's per-combat uid as <see cref="CardInstance" /> stores it. Every slot equals every other and
/// hashes to 0, so the record's synthesized equality and hash ignore the uid: two cards with the same
/// values and different uids stay equal wherever the old engine compares them.
/// </summary>
public readonly struct CardUidSlot(int value) : IEquatable<CardUidSlot>
{
    public int Value { get; } = value;

    public bool Equals(CardUidSlot other) => true;

    public override bool Equals(object? obj) => obj is CardUidSlot;

    public override int GetHashCode() => 0;

    public override string ToString() => Value.ToString(CultureInfo.InvariantCulture);
}

/// <summary>
/// Hands out per-combat card uids (docs/rebuild/CARD_UID.md). A card gets a fresh uid when it enters
/// the combat: every deck card when the combat starts (<see cref="NumberAll" />), and a card created
/// during the combat where it is created (<see cref="Fresh" />). A <c>with</c> change keeps the uid, so
/// a card keeps it through moves, upgrades and cost changes. The counter lives on the combat
/// (<see cref="CombatState.LastCardUid" />), so a clone's cards keep their uids and its new cards
/// continue the sequence.
/// </summary>
public static class CardUids
{
    /// <summary>The card's per-combat uid (0: none yet).</summary>
    public static int UidOf(this CardInstance card) => card.Uid;

    /// <summary>The card with the given uid, every other value kept.</summary>
    public static CardInstance WithUid(this CardInstance card, int uid) => card with { Uid = uid };

    /// <summary>A uid no card of this combat has had.</summary>
    public static int NewCardUid(this CombatState state) => ++state.LastCardUid;

    /// <summary>
    /// <paramref name="card" /> as a new card of this combat: a generated card, a copy of another card, a
    /// card transformed into a new one. Its values are kept, its uid is new.
    /// </summary>
    public static CardInstance Fresh(this CombatState state, CardInstance card) =>
        card with { Uid = state.NewCardUid() };

    /// <summary>
    /// Numbers every card the combat holds from 1 in walk order and sets the counter to the last: a
    /// combat's start (the deck cloned into the draw pile) and a combat read from a document.
    /// </summary>
    public static void NumberAll(CombatState state)
    {
        state.LastCardUid = 0;
        Rewrite(state, card => state.Fresh(card));
    }

    /// <summary>
    /// Gives a fresh uid to every live card that has none, keeping every other card's: a combat whose
    /// piles were partly rewritten from a document (EmuDump reanchor, setup items).
    /// </summary>
    public static void NumberNew(CombatState state) =>
        Rewrite(state, card => card.Uid == 0 ? state.Fresh(card) : card);

    /// <summary>
    /// The combat's live card slots in the kernel's walk order: Hand, Draw, Discard, Exhaust (the
    /// game's pile order), then the cards queued to auto-play; the played card held while its screen
    /// resolves follows them (<see cref="LiveCards" />). Not live: cards remembered from earlier
    /// (LastAttackOrSkillThisTurn), stolen ones, notes of a card still in a pile (ReturnToHandBeforeDraw:
    /// Bolas stays in the discard until it returns), and templates of cards not yet created
    /// (CopiesToHandBeforeDraw, a selection's AfterSelectionToHand).
    /// </summary>
    public static IEnumerable<List<CardInstance>> LivePiles(CombatState state)
    {
        yield return state.Hand;
        yield return state.DrawPile;
        yield return state.DiscardPile;
        yield return state.ExhaustPile;
        yield return state.AutoPlayQueue;
    }

    /// <summary>Every live card of the combat, in walk order (see <see cref="LivePiles" />).</summary>
    public static IEnumerable<CardInstance> LiveCards(CombatState state)
    {
        foreach (var pile in LivePiles(state))
        {
            foreach (var card in pile) yield return card;
        }

        if (state.HeldPlayedCard is { } held) yield return held;
    }

    /// <summary>Replaces every live card of the combat with <paramref name="map" /> of it, in walk order.</summary>
    public static void Rewrite(CombatState state, Func<CardInstance, CardInstance> map)
    {
        foreach (var pile in LivePiles(state))
        {
            for (int i = 0; i < pile.Count; i++) pile[i] = map(pile[i]);
        }

        if (state.HeldPlayedCard is { } held) state.HeldPlayedCard = map(held);
    }
}
