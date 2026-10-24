// The old state's view of one listener's own model (docs/rebuild/TIER1_KERNEL.md section 4, "the handler context"): what a
// handler reads about the model it runs for, found by its ListenerRef. This and OldState.cs are the only places that know
// where a power's amount, a relic or a card lives in the old representation. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>Where a card is, as a handler sees it: the game's piles (<see cref="OldPile" />'s values) and the run deck.
/// None: in no pile the combat lists (the auto-play queue, a stale held card).</summary>
public enum CardPile : sbyte { None = -1, Hand, Draw, Discard, Exhaust, Play, Deck }

/// <summary>A card of the combat as a handler reads it: its handle (the per-combat uid), its model id (the old def id,
/// <see cref="ListenerRef.ModelId" />'s space), the pile it is in and its position there (the card in play: Play, 0).</summary>
public readonly record struct CardView(CardHandle Handle, int ModelId, CardPile Pile, int Index);

/// <summary>A relic as a handler reads it: its model id, its position in the obtain order (the listener's instance) and
/// the counter the old state keeps on it (<c>RelicInstance.Counter</c>).</summary>
public readonly record struct RelicView(int ModelId, int Index, int Counter);

public static partial class OldState
{
    /// <summary>
    /// The listener's entry in the lists now (<see cref="Combat" />: the creature it sits on, its index, its pile, the buff
    /// that stands for a power, the enemy), or null when it is not there (a power removed from its owner mid-pass is still
    /// visited by the game, and the old state keeps no copy of it). Each kind is read straight off the list its instance
    /// field names (a power off its owner's buff list, a relic, potion or orb by position, a card by uid), allocating
    /// nothing; the rest (monsters, run models) from a listing of the combat. Needs a combat.
    /// </summary>
    public static OldListener? Locate(OldWorld w, ListenerRef who)
    {
        if (w.Combat is not { } s)
            return null;
        int at = who.Instance;
        switch ((ListenerKind)who.Kind)
        {
            case ListenerKind.Power:
            {
                if (OldCreatures.TryAt(w, new CreatureRef(at >> 8)) is not { } owner
                    || OldCreatures.Buffs(s, owner) is not { } buffs)
                    return null;
                int i = PowerEntry(w, buffs, who);
                return i < 0 ? null : new(who, at >> 8, i, OldPile.None, buffs[i].Id, owner.Enemy);
            }
            case ListenerKind.Relic:
                return PlayerActive(w) && (uint)at < (uint)s.Relics.Count && s.Relics[at].DefId == who.ModelId
                    ? new(who, 0, at, OldPile.None, default, null)
                    : null;
            case ListenerKind.Potion:
                return PlayerActive(w) && (uint)at < (uint)s.PotionSlots.Length && s.PotionSlots[at] != 0
                       && s.PotionSlots[at] == who.ModelId
                    ? new(who, 0, at, OldPile.None, default, null)
                    : null;
            case ListenerKind.Orb:
                return PlayerActive(w) && (uint)at < (uint)s.Orbs.Count && (int)s.Orbs[at].Type == who.ModelId
                    ? new(who, 0, at, OldPile.None, default, null)
                    : null;
            case ListenerKind.Card:
            case ListenerKind.Affliction:
            case ListenerKind.Enchantment:
            {
                if (!PlayerActive(w) || LocateCard(w, at) is not { } c)
                    return null;
                bool listed = (ListenerKind)who.Kind switch
                {
                    ListenerKind.Affliction => who.ModelId == (int)GameAffliction.Bound && c.Card.Bound,
                    ListenerKind.Enchantment => c.Card.Enchantment != Enchantment.None
                                                && (int)c.Card.Enchantment == who.ModelId,
                    _ => c.Card.DefId == who.ModelId,
                };
                return listed ? new(who, 0, c.Index, c.Pile, default, null) : null;
            }
        }

        var all = new List<OldListener>();
        Combat(w, all, []);
        foreach (var e in all)
            if (e.Who == who)
                return e;
        return null;
    }

    /// <summary>
    /// The index in <paramref name="buffs" /> of the entry that stands for the power listener <paramref name="who" />, or
    /// -1. <see cref="Powers" /> lists one power per (model, k): the k-th entry of any of the model's buffs, at its first
    /// entry, so the entry is the first whose model is <c>who.ModelId</c> and that is the k-th of its own BuffId (k is the
    /// instance's low byte). No list or dictionary is built.
    /// </summary>
    internal static int PowerEntry(OldWorld w, List<BuffState> buffs, ListenerRef who)
    {
        int k = who.Instance & 0xFF;
        for (int i = 0; i < buffs.Count; i++)
        {
            var m = w.Ids.Power(buffs[i].Id);
            if (m.ModelId != who.ModelId
                || m.Kind is PowerMapKind.EngineOnly or PowerMapKind.Folded or PowerMapKind.NotModelled)
                continue;
            int nth = 0;
            for (int j = 0; j < i; j++)
                if (buffs[j].Id == buffs[i].Id)
                    nth++;
            if (nth == k)
                return i;
        }

        return -1;
    }

    /// <summary>
    /// The live card with this uid, with the pile and position the listing gives it (<see cref="Combat" />): the card in
    /// flight (or the held card of an open screen, <see cref="PlayCard" />) in Play at 0, else its place in Hand, Draw,
    /// Discard or Exhaust; a card only the auto-play queue or a stale held field has is in pile None. Null for no card.
    /// Does not look at whether the player is active for hooks (a handle in an argument names a card either way).
    /// </summary>
    public static (CardInstance Card, OldPile Pile, int Index)? LocateCard(OldWorld w, int uid)
    {
        if (uid <= 0 || w.Combat is not { } s)
            return null;
        if (w.InFlight is { } flying && flying.Uid == uid)
            return (flying, OldPile.Play, 0);
        int i;
        if ((i = IndexOf(s.Hand, uid)) >= 0)
            return (s.Hand[i], OldPile.Hand, i);
        if ((i = IndexOf(s.DrawPile, uid)) >= 0)
            return (s.DrawPile[i], OldPile.Draw, i);
        if ((i = IndexOf(s.DiscardPile, uid)) >= 0)
            return (s.DiscardPile[i], OldPile.Discard, i);
        if ((i = IndexOf(s.ExhaustPile, uid)) >= 0)
            return (s.ExhaustPile[i], OldPile.Exhaust, i);
        if (PlayCard(w, s, []) is { } held && held.Uid == uid)
            return (held, OldPile.Play, 0);
        if ((i = IndexOf(s.AutoPlayQueue, uid)) >= 0)
            return (s.AutoPlayQueue[i], OldPile.None, i);
        return FindLive(w, uid) is { } stale ? (stale, OldPile.None, 0) : null;
    }

    private static int IndexOf(List<CardInstance> pile, int uid)
    {
        for (int i = 0; i < pile.Count; i++)
            if (pile[i].Uid == uid)
                return i;
        return -1;
    }

    /// <summary>
    /// The old buff that stands for a power listener, with its index in its owner's buff list; null when the power is not
    /// on its owner now (<see cref="Locate" />). Its <c>Magnitude</c> is the power's amount where the engine keeps the
    /// game's amount in it (the model's handler knows). Allocates nothing.
    /// </summary>
    public static (int Index, BuffState Buff)? PowerBuff(OldWorld w, ListenerRef who)
    {
        if ((ListenerKind)who.Kind != ListenerKind.Power || w.Combat is not { } s
            || OldCreatures.TryAt(w, new CreatureRef(who.Instance >> 8)) is not { } owner
            || OldCreatures.Buffs(s, owner) is not { } buffs)
            return null;
        int i = PowerEntry(w, buffs, who);
        return i < 0 ? null : (i, buffs[i]);
    }

    /// <summary>A card's handler view (<see cref="CardView" />) by handle, or null for no live card.</summary>
    public static CardView? CardOf(OldWorld w, CardHandle h) =>
        LocateCard(w, h.Uid) is { } c ? new(h, c.Card.DefId, (CardPile)(sbyte)c.Pile, c.Index) : null;

    /// <summary>
    /// The card model a listener runs for (a Card, or the Affliction or Enchantment on it: their instance is the card's
    /// uid): its view, or null when the listener is not a card kind, is not on the combat's lists now or (a Card) names
    /// another model than the card has now. A run deck card (negative instance) is the deck's: pile Deck, its index.
    /// </summary>
    public static CardView? CardOf(OldWorld w, ListenerRef who)
    {
        if ((ListenerKind)who.Kind is not (ListenerKind.Card or ListenerKind.Affliction or ListenerKind.Enchantment))
            return null;
        if (who.Instance < 0)
        {
            int d = -who.Instance - 1;
            return w.Run?.Deck is { } deck && (uint)d < (uint)deck.Count
                && (ListenerKind)who.Kind == ListenerKind.Card && deck[d].DefId == who.ModelId
                ? new(new CardHandle(who.Instance), who.ModelId, CardPile.Deck, d)
                : null;
        }

        return Locate(w, who) is { } e && LocateCard(w, who.Instance) is { } c
            ? new(new CardHandle(who.Instance), c.Card.DefId, (CardPile)(sbyte)e.Pile, e.Index)
            : null;
    }

    /// <summary>A relic listener's handler view, or null when it is not held now (the combat's relics, else the run's).</summary>
    public static RelicView? RelicOf(OldWorld w, ListenerRef who)
    {
        if ((ListenerKind)who.Kind != ListenerKind.Relic)
            return null;
        var relics = w.Combat?.Relics ?? w.Run?.Relics;
        int at = who.Instance;
        return PlayerActive(w) && relics is not null && (uint)at < (uint)relics.Count && relics[at].DefId == who.ModelId
            ? new(who.ModelId, at, relics[at].Counter)
            : null;
    }
}
