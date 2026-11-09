namespace Sts2Emulator.Core;

/// <summary>
/// The position of the first definition with a given id in a generated table (<c>GeneratedData.Cards</c> and its siblings), so
/// <c>Get(id)</c> is an array read where it was a scan of the whole table and a closure allocation. "First" is what
/// <c>Array.Find</c> returned, so a table that repeats an id answers as it did. Built once per table, at type initialisation, from
/// an array nothing writes to afterwards.
/// </summary>
internal sealed class DefIndex
{
    /// <summary>Ids below this are read from an array; the generated tables' ids are well under it (cards run to 10,042).</summary>
    private const int Dense = 1 << 14;

    private readonly int[] _dense;
    private readonly Dictionary<int, int>? _sparse;

    private DefIndex(int[] dense, Dictionary<int, int>? sparse)
    {
        _dense = dense;
        _sparse = sparse;
    }

    public static DefIndex Of<T>(T[] all, Func<T, int> idOf)
    {
        var dense = new int[Dense];
        Dictionary<int, int>? sparse = null;
        for (int i = 0; i < all.Length; i++)
        {
            int id = idOf(all[i]);
            if ((uint)id < Dense)
            {
                if (dense[id] == 0)
                {
                    dense[id] = i + 1;
                }
            }
            else
            {
                (sparse ??= []).TryAdd(id, i);
            }
        }

        return new DefIndex(dense, sparse);
    }

    /// <summary>The position in the table of the first definition with this id, or -1 when none has it.</summary>
    public int First(int id)
    {
        if ((uint)id < Dense)
        {
            return _dense[id] - 1;
        }

        return _sparse is not null && _sparse.TryGetValue(id, out int at) ? at : -1;
    }
}
