namespace Sts2Emulator.Solver;

/// <summary>
/// Expectimax values by S1's expected-mode key (FIGHT_SOLVER.md section 5): transposition merging within a solve and
/// subtree reuse across the decisions of one fight. A value is a function of the key and the turns left in the
/// solve's horizon, so an entry answers a lookup with the same turns left; an exact entry (every leaf under it a
/// fight's end, no horizon cut) answers any lookup with at least its turns left, which see the same tree. A table is
/// bound to one engine (the first position it is used on); it never persists across processes. Eviction drops the
/// oldest generation first and costs only recomputation.
/// </summary>
public sealed class TranspositionTable(long maxEntries = 4_000_000)
{
    internal readonly record struct Entry(NodeValue Node, int Remaining, int Generation)
    {
        public bool Exact => Node.Exact;
    }

    readonly Dictionary<UInt128, Entry> _map = [];
    string? _engine;

    public int Generation { get; private set; }

    public int Count => _map.Count;

    public long MaxEntries { get; } = maxEntries;

    /// <summary>Binds the table to <paramref name="engine" /> on first use; a different engine is refused.</summary>
    public void Bind(string engine)
    {
        _engine ??= engine;
        if (_engine != engine)
        {
            throw new SolveRefused($"table-engine: the table holds {_engine} values, the position is {engine}");
        }
    }

    public void NextGeneration() => Generation++;

    public void Clear()
    {
        _map.Clear();
        _engine = null;
    }

    internal bool TryGet(UInt128 key, int remaining, out Entry entry)
    {
        int prof = SolverProf.Enter(Phase.Table);
        bool hit = _map.TryGetValue(key, out entry) && (entry.Exact ? remaining >= entry.Remaining : remaining == entry.Remaining);
        SolverProf.Exit(prof);
        if (hit)
        {
            return true;
        }

        entry = default;
        return false;
    }

    internal void Put(UInt128 key, NodeValue node, int remaining)
    {
        int prof = SolverProf.Enter(Phase.Table);
        try
        {
            PutCore(key, node, remaining);
        }
        finally
        {
            SolverProf.Exit(prof);
        }
    }

    void PutCore(UInt128 key, NodeValue node, int remaining)
    {
        if (_map.Count >= MaxEntries && !_map.ContainsKey(key))
        {
            Evict();
        }

        if (_map.TryGetValue(key, out var old) && old.Exact && !node.Exact)
        {
            return;
        }

        _map[key] = new Entry(node, remaining, Generation);
    }

    void Evict()
    {
        int oldest = _map.Values.Min(e => e.Generation);
        var drop = _map.Where(kv => kv.Value.Generation == oldest).Select(kv => kv.Key).ToList();
        if (drop.Count == _map.Count)
        {
            drop = drop.Take(drop.Count / 2 + 1).ToList();
        }

        foreach (var k in drop)
        {
            _map.Remove(k);
        }
    }
}
