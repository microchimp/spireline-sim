using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>RunMapGenerator.FindMatchingSegments</c> (map pruning's duplicate-segment search) looks only
/// at each segment's first meeting; it must give exactly what listing every path and every segment
/// gives: the same groups in the same order, each with the same segments in the same order, point
/// for point. Checked on every search of every act's map of many runs, against a copy of the plain
/// search it replaced.
/// </summary>
public class MapSegmentSearchTests
{
    [Fact]
    public void FirstMeetingSearchMatchesThePlainSearchOnEveryMap()
    {
        int searches = 0, groups = 0, segments = 0;
        RunMapGenerator.OnSegmentSearch = state =>
        {
            var want = PlainSearch(state);
            var hook = RunMapGenerator.OnSegmentSearch;
            RunMapGenerator.OnSegmentSearch = null;
            List<List<RunMapNode[]>> got;
            try
            {
                got = RunMapGenerator.FindMatchingSegments(state);
            }
            finally
            {
                RunMapGenerator.OnSegmentSearch = hook;
            }

            Assert.Equal(want.Count, got.Count);
            for (int g = 0; g < want.Count; g++)
            {
                Assert.Equal(want[g].Count, got[g].Count);
                for (int m = 0; m < want[g].Count; m++)
                {
                    Assert.Equal(want[g][m].Length, got[g][m].Length);
                    for (int k = 0; k < want[g][m].Length; k++)
                    {
                        Assert.Same(want[g][m][k], got[g][m][k]);
                    }
                }

                segments += want[g].Count;
            }

            searches++;
            groups += want.Count;
        };
        try
        {
            var rng = new Random(20261014);
            const string chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
            for (int run = 0; run < 40; run++)
            {
                string seed = new([.. Enumerable.Range(0, 10).Select(_ => chars[rng.Next(chars.Length)])]);
                var engine = new RunEngine();
                engine.Reset(seed, run % 2 == 0 ? 10 : 0);
                while (RunMapGenerator.AdvanceToNextAct(engine.State))
                {
                }
            }
        }
        finally
        {
            RunMapGenerator.OnSegmentSearch = null;
        }

        // Every map searches several times, and the groups it prunes are there to compare.
        Assert.True(searches >= 40 * 3 * 3, $"{searches} searches");
        Assert.True(groups > searches, $"{groups} groups in {searches} searches");
        Assert.True(segments > 2 * groups - 1, $"{segments} segments in {groups} groups");
    }

    // ── the plain search, as the emulator had it ────────────────────────────────────

    private static List<List<RunMapNode[]>> PlainSearch(RunState state)
    {
        var paths = AllPaths(state, state.MapNodes[state.CurrentMapCoord]);
        var segments = new SortedDictionary<string, List<RunMapNode[]>>(StringComparer.Ordinal);
        foreach (var path in paths)
        {
            for (int i = 0; i < path.Count - 1; i++)
            {
                if (!(path[i].Children.Count <= 1 ? path[i].Row == 0 : true))
                {
                    continue;
                }

                for (int j = 2; j < path.Count - i; j++)
                {
                    if (path[i + j].Parents.Count < 2)
                    {
                        continue;
                    }

                    var segment = path.Skip(i).Take(j + 1).ToArray();
                    string key = Key(segment);
                    if (!segments.TryGetValue(key, out var existing))
                    {
                        segments[key] = [segment];
                    }
                    else if (!existing.Any(other => Overlapping(other, segment)))
                    {
                        existing.Add(segment);
                    }
                }
            }
        }

        return segments.Values.Where(list => list.Count > 1).ToList();
    }

    private static List<List<RunMapNode>> AllPaths(RunState state, RunMapNode current)
    {
        if (current.NodeType == RunConstants.NodeBoss)
        {
            return [[current]];
        }

        var paths = new List<List<RunMapNode>>();
        foreach (var child in current.Children)
        {
            foreach (var path in AllPaths(state, state.MapNodes[child]))
            {
                path.Insert(0, current);
                paths.Add(path);
            }
        }

        return paths;
    }

    private static string Key(RunMapNode[] segment)
    {
        var start = segment[0];
        var end = segment[^1];
        string prefix = start.Row == 0
            ? $"{start.Row}-{end.Col},{end.Row}-"
            : $"{start.Col},{start.Row}-{end.Col},{end.Row}-";
        return prefix + string.Join(",", segment.Select(node => node.NodeType switch
        {
            RunConstants.NodeEvent => 1,
            RunConstants.NodeShop => 2,
            RunConstants.NodeRelic => 3,
            RunConstants.NodeRest => 4,
            RunConstants.NodeNormal => 5,
            RunConstants.NodeElite => 6,
            RunConstants.NodeBoss => 7,
            RunConstants.NodeNone => 8,
            _ => 0,
        }));
    }

    private static bool Overlapping(RunMapNode[] left, RunMapNode[] right)
    {
        if (left.Length < 3 || right.Length < 3)
        {
            return false;
        }

        for (int i = 1; i <= left.Length - 2; i++)
        {
            if (ReferenceEquals(left[i], right[i]))
            {
                return true;
            }
        }

        return false;
    }
}
