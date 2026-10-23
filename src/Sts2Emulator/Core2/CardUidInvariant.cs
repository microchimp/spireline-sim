using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>
/// The card identity invariant (docs/rebuild/CARD_UID.md): every live card of a combat has a uid, no
/// two share one, and none is above the combat's counter. <see cref="Normalise" /> is the safety net for
/// a creation site that forgot to hand out a fresh uid; the kernel calls it when it snapshots a pass's
/// listeners. Uids are read by nothing the old engine writes, so normalising changes no output.
/// </summary>
public static class CardUidInvariant
{
    private static long _normalised;
    private static long _actingCalls;

    /// <summary>Cards <see cref="Normalise" /> has renumbered in this process.</summary>
    public static long NormalisedCards => Interlocked.Read(ref _normalised);

    /// <summary>Calls of <see cref="Normalise" /> in this process that had to renumber at least one card.</summary>
    public static long ActingCalls => Interlocked.Read(ref _actingCalls);

    /// <summary>Why a live card breaks the invariant.</summary>
    public enum Breach
    {
        None,
        NoUid,
        Duplicate,
        AboveCounter,
    }

    /// <summary>The first breach in walk order, with the card, or <see cref="Breach.None" />.</summary>
    public static (Breach Kind, CardInstance Card) Check(CombatState state)
    {
        var seen = new HashSet<int>();
        foreach (var card in CardUids.LiveCards(state))
        {
            var kind = Classify(card, seen, state.LastCardUid);
            if (kind != Breach.None) return (kind, card);
        }

        return (Breach.None, default);
    }

    /// <summary>Whether the combat's live cards keep the invariant.</summary>
    public static bool Holds(CombatState state) => Check(state).Kind == Breach.None;

    /// <summary>
    /// Gives a fresh uid to every live card with none, above the counter, or sharing one with a card
    /// earlier in walk order; returns how many it renumbered and, when <paramref name="log" /> is given,
    /// adds each with its breach.
    /// </summary>
    public static int Normalise(CombatState state, List<(Breach Kind, CardInstance Card)>? log = null)
    {
        var seen = new HashSet<int>();
        int fixedCount = 0;
        int counter = state.LastCardUid;
        CardUids.Rewrite(state, card =>
        {
            var kind = Classify(card, seen, counter);
            if (kind == Breach.None) return card;
            log?.Add((kind, card));
            fixedCount++;
            var fresh = state.Fresh(card);
            seen.Add(fresh.UidOf());
            return fresh;
        });

        if (fixedCount > 0)
        {
            Interlocked.Add(ref _normalised, fixedCount);
            Interlocked.Increment(ref _actingCalls);
        }

        return fixedCount;
    }

    private static Breach Classify(CardInstance card, HashSet<int> seen, int counter)
    {
        int uid = card.UidOf();
        if (uid == 0) return Breach.NoUid;
        if (uid > counter) return Breach.AboveCounter;
        return seen.Add(uid) ? Breach.None : Breach.Duplicate;
    }
}
