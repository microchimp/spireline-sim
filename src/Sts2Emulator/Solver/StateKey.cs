// The fight solver's state key (docs/rebuild/FIGHT_SOLVER.md section 2, piece S1): the canonical bytes of a combat
// position, so that two positions get equal keys only when they play the same. Whole state by default: every field the
// full-state walk reaches is in tools/solver/key_fields.json, classed, and a field leaves the key only through an
// exclusion with its reason (KeyFieldCoverageTests). The per-field encoder is generated (StateKey.g.cs, by
// tools/solver/gen_key.py); this half holds the writer and the canonical forms:
//
//   multisets  the hand, discard and exhaust piles and the potion slots are written as their elements' encodings sorted,
//              so the order of equal-rule cards and a card's uid never reach the key; the hand and the slots stay ordered
//              while the combat holds an order reader (tools/solver/order_readers.json), and every pile stays ordered
//              while a screen is open (its candidates are pile indices);
//   regions    the draw pile is its known top (ordered), its unknown middle (a multiset) and its known bottom (ordered);
//              in hindsight mode it is written whole, in order;
//   card_ref   a field holding a card uid (EnemyState.CurlUpCardUid) is written as the pile the card is in, and the card
//              itself carries a mark in its pile, so two equal cards stay distinct only while something points at one;
//   streams    in expected mode no stream is in the key (chance is fresh at every node, the player cannot see seeds); in
//              hindsight mode each is written as (seed, position).
//
// New code only: nothing on an existing path calls it, and it reads the state without changing it (a kernel's queue is
// read through its field, never through the getter that would make one). A position whose kernel queue is busy has no key
// (StateKeyRefusedException): a keyed node needs an idle queue (FIGHT_SOLVER.md section 1).
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;

namespace Sts2Emulator.Solver;

/// <summary>Which key: expected mode (chance at the game's odds, nothing hidden keyed) or hindsight (streams and the whole
/// draw order fixed, FIGHT_SOLVER.md section 9).</summary>
public enum KeyMode
{
    Expected,
    Hindsight,
}

/// <summary>A position the key refuses (a busy kernel queue), with the reason.</summary>
public sealed class StateKeyRefusedException(string reason) : InvalidOperationException($"state key refused: {reason}")
{
    public string Reason { get; } = reason;
}

public static class StateKey
{
    /// <summary>The sha256 of key_fields.json and order_readers.json the encoder was generated from.</summary>
    public static string FieldsSha256 => StateKeyCodec.FieldsSha256;

    /// <summary>The key's canonical bytes. Throws <see cref="StateKeyRefusedException" /> on a busy kernel queue.</summary>
    public static byte[] Encode(CombatState state, KeyMode mode = KeyMode.Expected)
    {
        var w = new KeyWriter(mode);
        w.Combat(state);
        return w.ToArray();
    }

    /// <summary>As <see cref="Encode" />, returning the refusal's reason instead of throwing.</summary>
    public static bool TryEncode(CombatState state, KeyMode mode, out byte[]? key, out string? refusal)
    {
        try
        {
            key = Encode(state, mode);
            refusal = null;
            return true;
        }
        catch (StateKeyRefusedException e)
        {
            key = null;
            refusal = e.Reason;
            return false;
        }
    }

    /// <summary>The table's 128-bit key: the first 16 bytes of the canonical bytes' SHA-256.</summary>
    public static UInt128 Hash(CombatState state, KeyMode mode = KeyMode.Expected) => HashOf(Encode(state, mode));

    public static UInt128 HashOf(ReadOnlySpan<byte> key)
    {
        Span<byte> h = stackalloc byte[32];
        SHA256.HashData(key, h);
        return new UInt128(BitConverter.ToUInt64(h[..8]), BitConverter.ToUInt64(h[8..16]));
    }

    /// <summary>Whether the hand keeps its order in this position's key (an order reader, or an open screen).</summary>
    public static bool HandOrdered(CombatState state) => state.PendingSelection is not null || Readers.Holds(state, Readers.Hand);

    /// <summary>Whether the potion slots keep their order (a slot reader, or an open screen).</summary>
    public static bool PotionOrdered(CombatState state) => state.PendingSelection is not null || Readers.Holds(state, Readers.Potion);

    /// <summary>
    /// Interchangeable enemies (FIGHT_SOLVER.md section 5): for each enemy, the lowest index of an enemy whose state is
    /// equal to it except for <see cref="EnemyState.Slot" />. Every enemy is its own class while the combat holds an
    /// enemy-order reader (order_readers.json; today always), so aiming at either copy is merged only where it is exact.
    /// </summary>
    public static int[] EnemyClasses(CombatState state)
    {
        var n = state.Enemies.Count;
        var classes = Enumerable.Range(0, n).ToArray();
        if (StateKeyCodec.EnemyOrderAlways || Readers.Holds(state, Readers.Enemy))
            return classes;
        var bytes = new byte[n][];
        for (int i = 0; i < n; i++)
        {
            var e = state.Enemies[i].Clone();
            e.Slot = 0;
            var w = new KeyWriter(KeyMode.Expected);
            w.Enter(state);
            StateKeyCodec.Enc_EnemyState(w, e);
            bytes[i] = w.ToArray();
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (classes[j] == j && bytes[i].AsSpan().SequenceEqual(bytes[j]))
                {
                    classes[i] = j;
                    break;
                }
            }
        }

        return classes;
    }

    /// <summary>order_readers.json's cards by id, resolved once.</summary>
    internal static class Readers
    {
        internal static readonly HashSet<int> Hand = Resolve(StateKeyCodec.HandPositionCards);
        internal static readonly HashSet<int> Potion = Resolve(StateKeyCodec.PotionSlotCards);
        internal static readonly HashSet<int> Enemy = Resolve(StateKeyCodec.EnemyOrderCards);

        /// <summary>Names that name no card (KeyFieldCoverageTests requires none).</summary>
        internal static IEnumerable<string> Unresolved() =>
            StateKeyCodec.HandPositionCards.Concat(StateKeyCodec.PotionSlotCards).Concat(StateKeyCodec.EnemyOrderCards)
                .Where(n => GeneratedData.Cards.FindId(n) is null);

        static HashSet<int> Resolve(string[] names) =>
            [.. names.Select(GeneratedData.Cards.FindId).Where(i => i is not null).Select(i => i!.Value)];

        internal static bool Holds(CombatState s, HashSet<int> ids)
        {
            if (ids.Count == 0) return false;
            foreach (var c in AllCards(s))
            {
                if (ids.Contains(c.DefId)) return true;
            }

            return s.PendingSelection is { } p && p.GeneratedCandidates.Any(ids.Contains);
        }
    }

    /// <summary>Every card the combat holds, in any pile, queue, hold or screen (the reader scan).</summary>
    internal static IEnumerable<CardInstance> AllCards(CombatState s)
    {
        foreach (var pile in Piles(s))
        {
            foreach (var c in pile.Cards) yield return c;
        }

        foreach (var c in new[] { s.HeldPlayedCard, s.LastAttackOrSkillThisTurn, s.LastAttackOrSkillLastTurn })
        {
            if (c is { } v) yield return v;
        }

        foreach (var e in s.Enemies)
        {
            if (e.StolenCard is { } v) yield return v;
        }

        if (s.PendingSelection is { } p)
        {
            foreach (var c in p.AfterSelectionToHand) yield return c;
        }
    }

    /// <summary>The card lists a card_ref can point into, with the code the key writes for each (0 is "in none").</summary>
    internal static IEnumerable<(int Code, List<CardInstance> Cards)> Piles(CombatState s)
    {
        yield return (1, s.Hand);
        yield return (2, s.DrawPile);
        yield return (3, s.DiscardPile);
        yield return (4, s.ExhaustPile);
        yield return (5, s.ReturnToHandBeforeDraw);
        yield return (6, s.CopiesToHandBeforeDraw);
        yield return (7, s.AutoPlayQueue);
        yield return (8, s.StolenCards);
        yield return (9, s.StolenBackCards);
    }
}

/// <summary>
/// The key's writer: fixed-width little-endian values in the json's field order, collections length-prefixed (-1 for
/// null), a presence byte before a nullable value. A frame per combat being written (a stopped step's Before is a combat
/// inside the combat) holds that combat's forms and its card references.
/// </summary>
public sealed class KeyWriter
{
    sealed class Frame
    {
        public required CombatState State;
        public required bool HandOrdered;
        public required bool PotionOrdered;
        public required bool SelectionOpen;
        public required int[] Refs;  // the uids card_ref fields point at, in field order (0 is none)
        public Frame? Outer;
    }

    readonly KeyMode _mode;
    byte[] _buf;
    int _len;
    Frame? _frame;

    public KeyWriter(KeyMode mode)
    {
        _mode = mode;
        _buf = new byte[1024];
    }

    KeyWriter(KeyWriter parent)
    {
        _mode = parent._mode;
        _frame = parent._frame;
        _buf = new byte[64];
    }

    public bool Hindsight => _mode == KeyMode.Hindsight;
    public bool HandOrdered => _frame?.HandOrdered ?? true;
    public bool PotionOrdered => _frame?.PotionOrdered ?? true;
    public bool SelectionOpen => _frame?.SelectionOpen ?? true;

    public byte[] ToArray() => _buf.AsSpan(0, _len).ToArray();

    Span<byte> Grow(int n)
    {
        if (_len + n > _buf.Length) Array.Resize(ref _buf, Math.Max(_buf.Length * 2, _len + n));
        var s = _buf.AsSpan(_len, n);
        _len += n;
        return s;
    }

    public void P(int v) => BitConverter.TryWriteBytes(Grow(4), v);
    public void P(uint v) => BitConverter.TryWriteBytes(Grow(4), v);
    public void P(long v) => BitConverter.TryWriteBytes(Grow(8), v);
    public void P(ulong v) => BitConverter.TryWriteBytes(Grow(8), v);
    public void P(short v) => BitConverter.TryWriteBytes(Grow(2), v);
    public void P(byte v) => Grow(1)[0] = v;
    public void P(bool v) => Grow(1)[0] = v ? (byte)1 : (byte)0;
    public void P(float v) => BitConverter.TryWriteBytes(Grow(4), v);
    public void P(double v) => BitConverter.TryWriteBytes(Grow(8), v);
    public void Has(bool present) => Grow(1)[0] = present ? (byte)1 : (byte)0;

    public void S(string? v)
    {
        if (v is null)
        {
            P(-1);
            return;
        }

        var b = Encoding.UTF8.GetBytes(v);
        P(b.Length);
        b.CopyTo(Grow(b.Length));
    }

    /// <summary>A collection in its order.</summary>
    public void Seq<T>(IEnumerable<T>? xs, Action<KeyWriter, T> enc)
    {
        if (xs is null)
        {
            P(-1);
            return;
        }

        var items = xs as IReadOnlyCollection<T> ?? xs.ToList();
        P(items.Count);
        foreach (var x in items) enc(this, x);
    }

    /// <summary>A collection as a multiset: its elements' encodings, sorted.</summary>
    public void Bag<T>(IEnumerable<T>? xs, Action<KeyWriter, T> enc)
    {
        if (xs is null)
        {
            P(-1);
            return;
        }

        var parts = new List<byte[]>();
        foreach (var x in xs)
        {
            var sub = new KeyWriter(this);
            enc(sub, x);
            parts.Add(sub.ToArray());
        }

        parts.Sort(static (a, b) => a.AsSpan().SequenceCompareTo(b));
        P(parts.Count);
        foreach (var p in parts)
        {
            P(p.Length);
            p.CopyTo(Grow(p.Length));
        }
    }

    /// <summary>A card: its mark (1 + the card_ref pointing at it, 0 for none) and its value. The uid is never written.</summary>
    public void Card(CardInstance c)
    {
        int mark = 0;
        if (_frame is { } f && c.Uid != 0)
        {
            int at = Array.IndexOf(f.Refs, c.Uid);
            mark = at + 1;
        }

        P(mark);
        StateKeyCodec.Enc_CardInstance(this, c);
    }

    /// <summary>A card uid field: the pile its card is in (the card carries the mark), never the number.</summary>
    public void CardRef(int uid)
    {
        if (uid == 0 || _frame is null)
        {
            P(-1);
            return;
        }

        int code = 0;
        foreach (var (c, cards) in StateKey.Piles(_frame.State))
        {
            if (cards.Exists(x => x.Uid == uid))
            {
                code = c;
                break;
            }
        }

        P(code);
    }

    /// <summary>The draw pile: whole and in order in hindsight mode or under an open screen; else three regions.</summary>
    public void Regions(List<CardInstance> pile, int knownTop, int knownBottom)
    {
        if (Hindsight || SelectionOpen)
        {
            P((byte)1);
            Seq(pile, static (w, c) => w.Card(c));
            return;
        }

        int n = pile.Count;
        int top = Math.Clamp(knownTop, 0, n);
        int bottom = Math.Clamp(knownBottom, 0, n - top);
        P((byte)0);
        Seq(pile.GetRange(0, top), static (w, c) => w.Card(c));
        Bag(pile.GetRange(top, n - top - bottom), static (w, c) => w.Card(c));
        Seq(pile.GetRange(n - bottom, bottom), static (w, c) => w.Card(c));
    }

    public void Stream(CountingRandom? r)
    {
        Has(r is not null);
        if (r is null) return;
        P(r.Seed);
        P(r.CallCount);
    }

    public void Stream(Random? r)
    {
        if (r is CountingRandom cr)
        {
            Stream(cr);
            return;
        }

        Has(r is not null);
        if (r is not null) P(-1);  // a plain Random has no readable position
    }

    public void Stream(IEnumerable<CountingRandom?>? rs) => Seq(rs, static (w, r) => w.Stream(r));

    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_queue")]
    static extern ref ActionQueueSet? QueueOf(KernelState kernel);

    /// <summary>Opens <paramref name="s" />'s frame: its forms and card references, after the idle-queue check.</summary>
    internal void Enter(CombatState s)
    {
        if (s.Kernel is { AnyOn: true } k && QueueOf(k) is { IsEmpty: false })
            throw new StateKeyRefusedException("kernel-queue-busy");
        var refs = new int[s.Enemies.Count];
        for (int i = 0; i < refs.Length; i++) refs[i] = s.Enemies[i].CurlUpCardUid;
        _frame = new Frame
        {
            State = s,
            HandOrdered = StateKey.HandOrdered(s),
            PotionOrdered = StateKey.PotionOrdered(s),
            SelectionOpen = s.PendingSelection is not null,
            Refs = refs,
            Outer = _frame,
        };
    }

    /// <summary>A whole combat in its own frame.</summary>
    public void Combat(CombatState s)
    {
        Enter(s);
        try
        {
            StateKeyCodec.Enc_CombatState(this, s);
        }
        finally
        {
            _frame = _frame!.Outer;
        }
    }
}
