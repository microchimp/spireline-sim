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
        // The safety net finds nothing to do almost always, and it runs at the start of every pass: read the cards where they lie
        // (no copy of the card, no closure, no set) and rewrite only when one breaks the invariant. The rewrite of an intact combat
        // put every card back as it was, so skipping it changes no card; it did bump each pile's version, which nothing reads.
        if (Intact(state))
        {
            return 0;
        }

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

    /// <summary>
    /// Whether <see cref="Classify" /> would say <see cref="Breach.None" /> for every live card of
    /// <see cref="CardUids.LivePiles" /> and the held card: each uid in 1 to the counter, none twice. Reads only.
    /// </summary>
    private static bool Intact(CombatState state)
    {
        int counter = state.LastCardUid;
        if (counter < 0 || counter >= MaxBitSet)
        {
            return false;
        }

        Span<ulong> seen = stackalloc ulong[(counter >> 6) + 1];
        if (!IntactPile(state.Hand, seen, counter) || !IntactPile(state.DrawPile, seen, counter)
            || !IntactPile(state.DiscardPile, seen, counter) || !IntactPile(state.ExhaustPile, seen, counter)
            || !IntactPile(state.AutoPlayQueue, seen, counter))
        {
            return false;
        }

        return state.HeldPlayedCard is not { } held || Take(held.Uid, seen, counter);
    }

    /// <summary>Counters at or above this take the general path (a bit set of this many uids is on the stack: 2 KB).</summary>
    private const int MaxBitSet = 16384;

    private static bool IntactPile(List<CardInstance> pile, Span<ulong> seen, int counter)
    {
        foreach (ref readonly var card in System.Runtime.InteropServices.CollectionsMarshal.AsSpan(pile))
        {
            if (!Take(card.Uid, seen, counter))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Marks <paramref name="uid" /> seen; false when it is none, above the counter, or already seen.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static bool Take(int uid, Span<ulong> seen, int counter)
    {
        if (uid <= 0 || uid > counter)
        {
            return false;
        }

        ulong bit = 1UL << (uid & 63);
        ref ulong word = ref seen[uid >> 6];
        if ((word & bit) != 0)
        {
            return false;
        }

        word |= bit;
        return true;
    }

    private static Breach Classify(CardInstance card, HashSet<int> seen, int counter)
    {
        int uid = card.UidOf();
        if (uid == 0) return Breach.NoUid;
        if (uid > counter) return Breach.AboveCounter;
        return seen.Add(uid) ? Breach.None : Breach.Duplicate;
    }
}
