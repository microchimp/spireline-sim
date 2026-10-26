using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Solver;

/// <summary>
/// The draw pile as a multiset (FIGHT_SOLVER.md sections 4 and 5): the macro draw equals the
/// sequential draws, and both equal the engine's draws over many shuffle seeds, through reshuffles
/// of the discard, a full hand and a random placement.
/// </summary>
public class DrawModelTests
{
    private const int Samples = 4000;

    public static TheoryData<int[], int[], int[], int[], int, int> Models() =>
        new()
        {
            // top run, middle, bottom run, discard, cards to draw, hand room
            { [], [1, 1, 1, 2, 2, 3, 4, 4, 4, 4], [], [], 5, 10 },
            { [3], [1, 1, 2, 2, 2], [], [], 4, 10 },
            { [], [1, 2], [5], [1, 1, 3, 3, 3, 4], 6, 10 },
            { [2, 7], [1, 1, 1], [6], [2, 2, 3], 8, 5 },
            { [], [], [], [1, 2, 2, 3], 3, 10 },
            { [], [1, 2, 3], [], [], 5, 10 },
        };

    [Theory]
    [MemberData(nameof(Models))]
    public void TheMacroDrawIsTheSequentialDrawsMergedByMultiset(int[] top, int[] middle, int[] bottom, int[] discard, int k, int room)
    {
        var model = DrawModel<int>.Shuffled(middle, discard);
        foreach (int v in top.Reverse())
        {
            model.PlaceTop(v);
        }

        foreach (int v in bottom)
        {
            model.PlaceBottom(v);
        }

        var macro = ByKey(model.MacroDraw(k, room));
        var sequential = ByKey(model.SequentialDraw(k, room));
        var after = model.SequentialDraw(k, room).ToDictionary(o => string.Join(",", o.Drawn), o => Describe(o.After));
        foreach (var (drawn, post, _) in model.MacroDraw(k, room))
        {
            Assert.Equal(after[string.Join(",", drawn)], Describe(post));
        }

        Assert.Equal(sequential.Keys.Order(), macro.Keys.Order());
        foreach (var (key, p) in sequential)
        {
            Assert.Equal(p, macro[key], 12);
        }

        Assert.Equal(1d, macro.Values.Sum(), 12);
    }

    [Fact]
    public void AKnownTopIsDrawnFirstAndTheMiddleAtCountOverSize()
    {
        var model = DrawModel<int>.Shuffled([1, 1, 2]);
        model.PlaceTop(9);
        Assert.Equal([(9, 1d)], model.NextDraw());
        var after = model.AfterDraw(9);
        Assert.Equal([(1, 2d / 3), (2, 1d / 3)], after.NextDraw());
        model.PlaceBottom(8);
        Assert.Throws<InvalidOperationException>(() => model.PlaceRandom(5));
    }

    [Fact]
    public void AFirstHandIsTheEnginesDrawOverShuffleSeeds()
    {
        int[] deck = [IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad,
            IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.Bash];
        var model = DrawModel<int>.Shuffled(deck);
        AssertEngineDraws("first hand", model.MacroDraw(5), seed =>
        {
            var state = Bare(Pile(deck), [], seed);
            CardEffects.ShufflePile(state.DrawPile, state.ShuffleRng!);
            CardEffects.DrawCards(state, 5, new CountingRandom(seed), fromHandDraw: true);
            return state.Hand;
        });
    }

    [Fact]
    public void ADrawThatEmptiesThePileShufflesTheDiscardInAndDrawsOn()
    {
        int[] pile = [IC.Anger, IC.Bash];
        int[] discard = [IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.Anger];
        var model = DrawModel<int>.Shuffled(pile, discard);
        AssertEngineDraws("reshuffle", model.MacroDraw(5), seed =>
        {
            var state = Bare(Pile(pile), Pile(discard), seed);
            CardEffects.DrawCards(state, 5, new CountingRandom(seed), fromHandDraw: true);
            return state.Hand;
        });
    }

    [Fact]
    public void AFullHandCapsTheDraw()
    {
        int[] pile = [IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Bash, IC.Anger];
        var model = DrawModel<int>.Shuffled(pile);
        AssertEngineDraws("hand cap", model.MacroDraw(5, room: 2), seed =>
        {
            var state = Bare(Pile(pile), [], seed, Pile(IC.Inflame, IC.Inflame, IC.Inflame, IC.Inflame, IC.Inflame, IC.Inflame, IC.Inflame, IC.Inflame));
            CardEffects.ShufflePile(state.DrawPile, state.ShuffleRng!);
            CardEffects.DrawCards(state, 5, new CountingRandom(seed));
            return state.Hand.Where(c => c.DefId != IC.Inflame).ToList();
        });
    }

    [Fact]
    public void ACardPlacedAtRandomJoinsTheUnknownMiddle()
    {
        int[] pile = [IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.DefendIronclad];
        var model = DrawModel<int>.Shuffled(pile);
        model.PlaceRandom(IC.Bash);
        AssertEngineDraws("random placement", model.MacroDraw(3), seed =>
        {
            var state = Bare(Pile(pile), [], seed);
            CardEffects.ShufflePile(state.DrawPile, state.ShuffleRng!);
            CardEffects.AddCardToDrawPileRandomly(state, IC.Bash, 1, new CountingRandom(seed));
            CardEffects.DrawCards(state, 3, new CountingRandom(seed), fromHandDraw: true);
            return state.Hand;
        });
    }

    private static CombatState Bare(List<CardInstance> pile, List<CardInstance> discard, int seed, List<CardInstance>? hand = null)
    {
        var state = Fight.Hand([.. hand ?? []]).Draw([.. pile]).Discard([.. discard]).Enemy(hp: 500).State;
        state.ShuffleRng = new CountingRandom(seed * 3 + 1);
        return state;
    }

    private static void AssertEngineDraws(string what, List<(int[] Drawn, DrawModel<int> After, double P)> odds, Func<int, List<CardInstance>> engine)
    {
        var exact = ByKey(odds);
        var counts = new Dictionary<string, int>();
        for (int seed = 1; seed <= Samples; seed++)
        {
            var hand = engine(seed);
            string key = string.Join("|", hand.Select(c => c.DefId).Order());
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        foreach (var key in counts.Keys)
        {
            Assert.True(exact.ContainsKey(key), $"{what}: the engine drew {key}, which the model gives no chance");
        }

        foreach (var (key, p) in exact)
        {
            double f = counts.GetValueOrDefault(key) / (double)Samples;
            Assert.True(Math.Abs(f - p) <= 5 * Math.Sqrt(p * (1 - p) / Samples) + 1.0 / Samples, $"{what}: {key} drawn {f:F4}, model {p:F4}");
        }
    }

    private static string Describe(DrawModel<int> m) =>
        $"top {string.Join(",", m.Top)} middle {string.Join(",", m.Middle.Select(kv => $"{kv.Key}x{kv.Value}"))} "
        + $"bottom {string.Join(",", m.Bottom)} discard {string.Join(",", m.Discard.Select(kv => $"{kv.Key}x{kv.Value}"))}";

    [Fact]
    public void PerfectFitCardsOfOneValueTopTheReshuffleAndOfTwoValuesRefuse()
    {
        var model = DrawModel<int>.Shuffled([], [1, 2, 9, 9, 3], fitted: v => v == 9);
        Assert.Equal([(9, 1d)], model.NextDraw());
        var after = model.AfterDraw(9);
        Assert.Equal([9], after.Top);
        Assert.Equal(3, after.MiddleCount);
        var macro = model.MacroDraw(3);
        Assert.Equal(3, macro.Count);
        Assert.All(macro, o => Assert.Equal(1d / 3, o.P, 12));
        Assert.All(macro, o => Assert.Equal(2, o.Drawn.Count(v => v == 9)));
        Assert.All(macro, o => Assert.Equal(2, o.After.MiddleCount));

        var two = DrawModel<int>.Shuffled([], [1, 8, 9], fitted: v => v >= 8);
        Assert.Throws<InvalidOperationException>(() => two.NextDraw());
        Assert.Throws<InvalidOperationException>(() => two.MacroDraw(2));
    }

    private static Dictionary<string, double> ByKey(List<(int[] Drawn, DrawModel<int> After, double P)> odds)
    {
        var by = new Dictionary<string, double>();
        foreach (var (drawn, _, p) in odds)
        {
            string key = string.Join("|", drawn.Order());
            by[key] = by.GetValueOrDefault(key) + p;
        }

        return by;
    }
}
