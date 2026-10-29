// The table's entry kinds (docs/rebuild/FIGHT_SOLVER.md section 19): an entry holds a value or only a bound left by a cut
// subtree, and a lookup reads each as what it is; a bound is never returned as a value.
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class TableEntryKindTests
{
    [Fact]
    public void ABoundEntryIsNeverReturnedAsAValue()
    {
        var t = new TranspositionTable();
        UInt128 k = 42;
        var bound = new Rational(3, 4);
        Assert.True(t.PutBound(k, bound, 1));
        Assert.False(t.TryGet(k, 1, out _, out bool sawBound));
        Assert.True(sawBound);
        Assert.False(t.TryGet(k, 5, out _));
        Assert.True(t.TryGetBound(k, out var got));
        Assert.Equal(bound, got);

        // A value replaces the bound; a bound never replaces a value, and a value is not read as a bound.
        var value = new NodeValue(new Rational(1, 2), true, Rational.One, new Rational(1, 2));
        t.Put(k, value, 1);
        Assert.True(t.TryGet(k, 1, out var e));
        Assert.Equal(value, e.Node);
        Assert.False(t.TryGetBound(k, out _));
        Assert.False(t.PutBound(k, bound, 1));
        Assert.True(t.TryGet(k, 3, out e));
        Assert.Equal(value, e.Node);
    }
}
