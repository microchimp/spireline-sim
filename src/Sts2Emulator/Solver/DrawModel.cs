namespace Sts2Emulator.Solver;

/// <summary>
/// The draw pile as the player knows it (docs/rebuild/FIGHT_SOLVER.md sections 4 and 5): a known
/// top run (top first), an unknown middle held as a multiset, a known bottom run (top first), and the
/// discard as a multiset. A shuffle does not roll here: the cards it shuffles join the middle, because
/// to the player a shuffled order is uniform over the orderings of the multiset (the engine's shuffle
/// sorts and then runs a Fisher-Yates on the Shuffle stream). A draw takes the known top if there is
/// one, else is a chance node over the middle's distinct values at count / size, and an empty pile
/// shuffles the discard in first, as the engine's draw does.
/// </summary>
/// <remarks>
/// Cards are values (<typeparamref name="T" />): the caller gives each card the value its key gives it
/// (equal values are interchangeable copies). A random placement into a pile with a known run would
/// leave the player not knowing where the known cards now are, a hidden branch, so it refuses.
/// </remarks>
public sealed class DrawModel<T>
    where T : IComparable<T>
{
    public List<T> Top { get; private init; } = [];

    public SortedDictionary<T, int> Middle { get; private init; } = [];

    public List<T> Bottom { get; private init; } = [];

    public SortedDictionary<T, int> Discard { get; private init; } = [];

    public int MiddleCount => Middle.Values.Sum();

    public int Count => Top.Count + MiddleCount + Bottom.Count;

    public int DiscardCount => Discard.Values.Sum();

    /// <summary>A pile whose order nobody knows (after a shuffle), and a discard.</summary>
    public static DrawModel<T> Shuffled(IEnumerable<T> pile, IEnumerable<T>? discard = null)
    {
        var m = new DrawModel<T>();
        foreach (var v in pile)
        {
            Add(m.Middle, v, 1);
        }

        foreach (var v in discard ?? [])
        {
            Add(m.Discard, v, 1);
        }

        return m;
    }

    public DrawModel<T> Clone() =>
        new()
        {
            Top = [.. Top],
            Middle = new SortedDictionary<T, int>(Middle),
            Bottom = [.. Bottom],
            Discard = new SortedDictionary<T, int>(Discard),
        };

    /// <summary><c>ShufflePile</c> on the draw pile: every card joins the middle.</summary>
    public void ForgetDrawOrder()
    {
        foreach (var v in Top.Concat(Bottom))
        {
            Add(Middle, v, 1);
        }

        Top.Clear();
        Bottom.Clear();
    }

    /// <summary><c>ShuffleDiscardIntoDraw</c>: the discard and the draw pile merge and shuffle into one middle.</summary>
    public void ShuffleDiscardIntoDraw()
    {
        ForgetDrawOrder();
        foreach (var (v, c) in Discard)
        {
            Add(Middle, v, c);
        }

        Discard.Clear();
    }

    public void PlaceTop(T v) => Top.Insert(0, v);

    public void PlaceBottom(T v)
    {
        if (MiddleCount == 0 && Bottom.Count == 0)
        {
            Top.Add(v);
        }
        else
        {
            Bottom.Add(v);
        }
    }

    /// <summary>
    /// A placement at a uniformly random position (the <c>placementRng</c> sites): into a pile with no
    /// known run it joins the middle with no roll (a uniform slot in a uniform order is a uniform order).
    /// </summary>
    public void PlaceRandom(T v)
    {
        if (Top.Count > 0 || Bottom.Count > 0)
        {
            throw new InvalidOperationException("hidden-branch: a random placement into a pile with a known run");
        }

        Add(Middle, v, 1);
    }

    public void AddToDiscard(T v) => Add(Discard, v, 1);

    /// <summary>The next draw's distribution (with the discard shuffled in when the pile is empty); empty when there is nothing to draw.</summary>
    public List<(T Value, double P)> NextDraw()
    {
        var m = Count == 0 ? WithDiscardShuffled() : this;
        if (m.Top.Count > 0)
        {
            return [(m.Top[0], 1d)];
        }

        int mid = m.MiddleCount;
        if (mid > 0)
        {
            return [.. m.Middle.Select(kv => (kv.Key, kv.Value / (double)mid))];
        }

        return m.Bottom.Count > 0 ? [(m.Bottom[0], 1d)] : [];
    }

    /// <summary>The pile after <paramref name="v" /> is drawn (one of NextDraw's outcomes).</summary>
    public DrawModel<T> AfterDraw(T v)
    {
        var m = Count == 0 ? WithDiscardShuffled() : Clone();
        if (m.Top.Count > 0)
        {
            Expect(m.Top[0], v);
            m.Top.RemoveAt(0);
        }
        else if (m.MiddleCount > 0)
        {
            Add(m.Middle, v, -1);
        }
        else
        {
            Expect(m.Bottom[0], v);
            m.Bottom.RemoveAt(0);
        }

        if (m.MiddleCount == 0 && m.Top.Count == 0 && m.Bottom.Count > 0)
        {
            // With the middle gone the bottom run is the top run.
            m.Top.AddRange(m.Bottom);
            m.Bottom.Clear();
        }

        return m;
    }

    /// <summary>
    /// Draws of <paramref name="k" /> cards one at a time (capped at <paramref name="room" />, the hand's
    /// free slots), merged by the multiset drawn: the reference for <see cref="MacroDraw" />.
    /// </summary>
    public List<(T[] Drawn, double P)> SequentialDraw(int k, int room = int.MaxValue)
    {
        var layer = new Dictionary<string, (T[] Drawn, DrawModel<T> Model, double P)> { [""] = ([], this, 1d) };
        for (int i = 0; i < Math.Min(k, room); i++)
        {
            var next = new Dictionary<string, (T[] Drawn, DrawModel<T> Model, double P)>();
            foreach (var (drawn, model, p) in layer.Values)
            {
                var outcomes = model.NextDraw();
                if (outcomes.Count == 0)
                {
                    Merge(next, drawn, model, p);
                    continue;
                }

                foreach (var (v, q) in outcomes)
                {
                    T[] d = [.. drawn, v];
                    Array.Sort(d);
                    Merge(next, d, model.AfterDraw(v), p * q);
                }
            }

            layer = next;
        }

        return [.. layer.Values.Select(e => (e.Drawn, e.P))];
    }

    /// <summary>
    /// The macro draw: k cards as one chance node over the distinct sub-multisets drawn, with the
    /// multivariate hypergeometric probability for each part taken from an unknown middle (the known
    /// runs are taken as they lie, and a pile that runs out shuffles the discard in and goes on).
    /// Used only while nothing observes the order between draws.
    /// </summary>
    public List<(T[] Drawn, double P)> MacroDraw(int k, int room = int.MaxValue)
    {
        int need = Math.Min(k, room);
        var taken = new List<T>();
        var m = Clone();
        TakeKnown(m.Top, taken, ref need);
        int mid = m.MiddleCount;
        if (need < mid)
        {
            return [.. Hypergeometric(m.Middle, need).Select(h => (Sorted(taken, h.Part), h.P))];
        }

        foreach (var (v, c) in m.Middle)
        {
            taken.AddRange(Enumerable.Repeat(v, c));
        }

        need -= mid;
        TakeKnown(m.Bottom, taken, ref need);
        if (need == 0 || m.DiscardCount == 0)
        {
            return [(Sorted(taken, []), 1d)];
        }

        need = Math.Min(need, m.DiscardCount);
        return [.. Hypergeometric(m.Discard, need).Select(h => (Sorted(taken, h.Part), h.P))];
    }

    private DrawModel<T> WithDiscardShuffled()
    {
        var m = Clone();
        m.ShuffleDiscardIntoDraw();
        return m;
    }

    private static void TakeKnown(List<T> run, List<T> taken, ref int need)
    {
        int n = Math.Min(need, run.Count);
        taken.AddRange(run.Take(n));
        need -= n;
    }

    /// <summary>Every sub-multiset of size <paramref name="r" /> of <paramref name="pool" />, with prod C(c_v, x_v) / C(size, r).</summary>
    private static List<(T[] Part, double P)> Hypergeometric(SortedDictionary<T, int> pool, int r)
    {
        var values = pool.ToArray();
        int size = values.Sum(kv => kv.Value);
        double total = Choose(size, r);
        var result = new List<(T[] Part, double P)>();
        var counts = new int[values.Length];

        void Walk(int i, int left, double ways)
        {
            if (i == values.Length)
            {
                if (left == 0)
                {
                    var part = new List<T>();
                    for (int j = 0; j < values.Length; j++)
                    {
                        part.AddRange(Enumerable.Repeat(values[j].Key, counts[j]));
                    }

                    result.Add(([.. part], ways / total));
                }

                return;
            }

            for (int x = 0; x <= Math.Min(left, values[i].Value); x++)
            {
                counts[i] = x;
                Walk(i + 1, left - x, ways * Choose(values[i].Value, x));
            }

            counts[i] = 0;
        }

        Walk(0, r, 1d);
        return result;
    }

    private static double Choose(int n, int k)
    {
        double c = 1d;
        for (int i = 1; i <= k; i++)
        {
            c = c * (n - k + i) / i;
        }

        return c;
    }

    private static T[] Sorted(List<T> a, T[] b)
    {
        T[] all = [.. a, .. b];
        Array.Sort(all);
        return all;
    }

    private static void Merge(Dictionary<string, (T[] Drawn, DrawModel<T> Model, double P)> into, T[] drawn, DrawModel<T> model, double p)
    {
        string key = string.Join("|", drawn);
        into[key] = into.TryGetValue(key, out var e) ? (drawn, e.Model, e.P + p) : (drawn, model, p);
    }

    private static void Expect(T known, T v)
    {
        if (known.CompareTo(v) != 0)
        {
            throw new ArgumentException($"{v} is not the known top card {known}");
        }
    }

    private static void Add(SortedDictionary<T, int> set, T v, int n)
    {
        int c = set.GetValueOrDefault(v) + n;
        if (c < 0)
        {
            throw new ArgumentException($"{v} is not in the pile");
        }

        if (c == 0)
        {
            set.Remove(v);
        }
        else
        {
            set[v] = c;
        }
    }
}
