// The chance-model test (docs/rebuild/FIGHT_SOLVER.md section 6). Successors.Expand is shared by the reference and the
// pruned solver, so the equal-value harness cannot see a wrong chance model: both sides would agree on it. Here each
// root move of the harness positions is also stepped on plain copies, unhooked, with real seeded streams and the draw
// pile's unknown middle in a seeded uniform order (what the player knows of it), and the outcomes' frequencies by
// expected-mode key must match Expand's odds: every counted outcome in Expand's support, every frequency within five
// standard errors (plus two samples) of its odds. Both sides' outcomes are labelled with the unknown middle left by the
// step put in one order (by key value, then uid): the multiset model's state. The key itself keeps the pile's order
// where an order reader is in the fight, and Expand only ever leaves the middle canonical.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Solver;

public class ChanceModelTests(ITestOutputHelper output)
{
    const int Samples = 200;

    [Fact]
    public void ExpandMatchesAnIndependentCountOnTheHarnessPositions()
    {
        var corpus = SolverPositions.Corpus();
        int positions = 0, moves = 0, refused = 0, branching = 0, screens = 0;
        var bad = new List<string>();
        foreach (var recipe in SolverPositions.Recipes())
        {
            var pos = SolverPositions.Build(recipe, corpus);
            if (pos is null)
            {
                continue;
            }

            positions++;
            int k = 0;
            foreach (var m in Actions.Legal(pos.State, Pruning.None).Take(6))
            {
                Dictionary<string, Rational> byValue, byCard;
                try
                {
                    byValue = Model(pos.State, pos.Rng, m, byCard: false);
                    byCard = Model(pos.State, pos.Rng, m, byCard: true);
                }
                catch (SolveRefused)
                {
                    refused++;
                    continue;
                }

                moves++;
                branching += byValue.Count > 1 ? 1 : 0;
                screens += pos.State.PendingSelection is not null ? 1 : 0;
                if (byValue.Count != byCard.Count || byValue.Any(kv => byCard.GetValueOrDefault(kv.Key, Rational.Zero) != kv.Value))
                {
                    bad.Add($"#{recipe.Id} move {m}: by value and by card differ");
                }

                var count = Count(pos.State, m, Samples, unchecked(recipe.Id * 31 + k++));
                bad.AddRange(Compare(byValue, count, Samples).Select(x => $"#{recipe.Id} move {m}: {x}"));
            }
        }

        output.WriteLine($"positions {positions}, moves counted {moves} ({branching} with more than one outcome, {screens} on an open screen), {Samples} samples each; refused {refused}");
        foreach (var d in bad.Take(30))
        {
            output.WriteLine(d);
        }

        Assert.Empty(bad);
        Assert.True(moves >= 1000 && branching >= 200, $"only {moves} moves, {branching} branching");
    }

    /// <summary>
    /// Uproar picks one of the draw pile's Attacks through StableShuffle, which sorts by (id, upgraded) first: two
    /// Rampages tie there, and which comes first is the pile's order. With one grown, the count is half and half;
    /// Expand laid the rest out canonically and conditioned on the front, which gave a quarter and three quarters, on
    /// both solvers alike. It refuses now (draw-tie); with no such tie (a Rampage and a Strike) it matches the count.
    /// </summary>
    [Fact]
    public void ASortedPickOverTiedCardsOfDifferentValueIsRefusedAndCountedEven()
    {
        int rampage = Cards.FindId("Rampage")!.Value, uproar = Cards.FindId("Uproar")!.Value;
        var grown = new CardInstance(rampage, false) { BonusDamage = 5 };
        var (tie, tieRng) = UproarOver(grown, new CardInstance(rampage, false));
        var m = Actions.Legal(tie, Pruning.None).First(x => tie.PendingSelection is null && x.Action < tie.Hand.Count && Math.Abs(tie.Hand[x.Action].DefId) == uproar);
        // Which Rampage Uproar played, by the one left in the draw pile (the fight's two enemies add target rolls).
        var count = Count(tie, m, 2000, 11, (s, _) => s.DrawPile.Where(c => Math.Abs(c.DefId) == rampage).Sum(c => c.BonusDamage) == 0 ? "grown played" : "fresh played");
        Assert.Equal(2, count.Count);
        Assert.All(count.Values, n => Assert.InRange(n / 2000.0, 0.44, 0.56));
        var e = Assert.Throws<SolveRefused>(() => Model(tie, tieRng, m, byCard: false));
        Assert.StartsWith("draw-tie", e.Reason);
        Assert.StartsWith("draw-tie", Reference.Solve(tie, tieRng, 0).Refused);

        var (plain, plainRng) = UproarOver(new CardInstance(rampage, false), new CardInstance(IC.StrikeIronclad, false));
        var pm = Actions.Legal(plain, Pruning.None).First(x => x.Action < plain.Hand.Count && Math.Abs(plain.Hand[x.Action].DefId) == uproar);
        var model = Model(plain, plainRng, pm, byCard: false);
        Assert.True(model.Count >= 2);
        Assert.Empty(Compare(model, Count(plain, pm, 2000, 12), 2000));
        output.WriteLine($"tie counted {string.Join(", ", count.Values.Select(n => (n / 2000.0).ToString("0.000")))}; no tie: model {string.Join(", ", model.Values.Select(p => p.ToDouble().ToString("0.000")))}");
    }

    internal static (CombatState State, CountingRandom Rng) UproarOver(CardInstance a, CardInstance b)
    {
        var state = Fight
            .Hand(new CardInstance(Cards.FindId("Uproar")!.Value, false), new CardInstance(IC.DefendIronclad, false))
            .Energy(3)
            .Draw(a, b, new CardInstance(IC.DefendIronclad, false))
            .Discard()
            .State;
        foreach (var pile in new[] { state.Hand, state.DrawPile })
        {
            for (int i = 0; i < pile.Count; i++)
            {
                pile[i] = pile[i].Uid == 0 ? state.Fresh(pile[i]) : pile[i];
            }
        }

        state.KnownTopCount = 0;
        state.KnownBottomCount = 0;
        foreach (var e in state.Enemies)
        {
            e.Hp = Math.Max(e.Hp, 200);
        }

        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            StateCloning.SetStream(state, i, new CountingRandom(300 + i));
        }

        return (state, new CountingRandom(9));
    }

    /// <summary>An outcome's label: the expected-mode key, or the fight's end with its score.</summary>
    static string Label(CombatState s, StepResult r)
    {
        if (r.Terminal)
        {
            return $"end {r.PlayerWon} {Utility.Terminal(s, r.PlayerWon)}";
        }

        try
        {
            var c = s.Clone();
            int kt = Math.Clamp(c.KnownTopCount, 0, c.DrawPile.Count);
            int kb = Math.Clamp(c.KnownBottomCount, 0, c.DrawPile.Count - kt);
            var middle = c.DrawPile.GetRange(kt, c.DrawPile.Count - kt - kb);
            middle.Sort((a, b) =>
            {
                int v = DrawSteer.CardValue(c, a).AsSpan().SequenceCompareTo(DrawSteer.CardValue(c, b));
                return v != 0 ? v : a.Uid.CompareTo(b.Uid);
            });
            for (int i = 0; i < middle.Count; i++)
            {
                c.DrawPile[kt + i] = middle[i];
            }

            return StateKey.Hash(c, KeyMode.Expected).ToString();
        }
        catch (StateKeyRefusedException e)
        {
            return "refused " + e.Reason.Split(':')[0];
        }
    }

    /// <summary>Expand's odds by label.</summary>
    static Dictionary<string, Rational> Model(CombatState s, CountingRandom rng, Move m, bool byCard)
    {
        var d = new Dictionary<string, Rational>();
        foreach (var o in Successors.Expand(s, rng, m, byCard, 32, new SolveCounters()))
        {
            var k = Label(o.State, o.Result);
            d[k] = d.GetValueOrDefault(k, Rational.Zero) + o.P;
        }

        return d;
    }

    /// <summary>
    /// The independent count: the move stepped on <paramref name="samples" /> copies, unhooked, each with every stream
    /// and the step's stream seeded afresh and the unknown middle in a seeded uniform order (a draw-pile screen's chosen
    /// card stays at the index the screen shows it at; the known top and bottom stay).
    /// </summary>
    static Dictionary<string, int> Count(CombatState s, Move m, int samples, int seed, Func<CombatState, StepResult, string>? label = null)
    {
        var d = new Dictionary<string, int>();
        for (int n = 0; n < samples; n++)
        {
            var seeds = new Random(unchecked(seed * 7919 + n));
            var copy = s.Clone();
            for (int i = 0; i < StateCloning.StreamCount; i++)
            {
                StateCloning.SetStream(copy, i, new CountingRandom(seeds.Next()));
            }

            var r = new CountingRandom(seeds.Next());
            ShuffleMiddle(copy, m, seeds);
            string k;
            try
            {
                k = (label ?? Label)(copy, Successors.Step(copy, r, m));
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                k = "error " + e.GetType().Name;
            }

            d[k] = d.GetValueOrDefault(k) + 1;
        }

        return d;
    }

    static void ShuffleMiddle(CombatState s, Move m, Random order)
    {
        int kt = Math.Clamp(s.KnownTopCount, 0, s.DrawPile.Count);
        int kb = Math.Clamp(s.KnownBottomCount, 0, s.DrawPile.Count - kt);
        int mid = s.DrawPile.Count - kt - kb;
        int keep = s.PendingSelection is { } open && open.Kind.ChoosesFromDrawPile() && m.Action < open.Candidates.Count
            && open.Candidates[m.Action] is int at && at >= kt && at < kt + mid ? at - kt : -1;
        var middle = s.DrawPile.GetRange(kt, mid);
        CardInstance? kept = keep >= 0 ? middle[keep] : null;
        if (kept is not null)
        {
            middle.RemoveAt(keep);
        }

        for (int i = middle.Count - 1; i > 0; i--)
        {
            int j = order.Next(i + 1);
            (middle[i], middle[j]) = (middle[j], middle[i]);
        }

        if (kept is not null)
        {
            middle.Insert(keep, kept.Value);
        }

        for (int i = 0; i < mid; i++)
        {
            s.DrawPile[kt + i] = middle[i];
        }
    }

    static List<string> Compare(Dictionary<string, Rational> model, Dictionary<string, int> count, int samples)
    {
        var bad = new List<string>();
        foreach (var k in model.Keys.Union(count.Keys))
        {
            double p = model.TryGetValue(k, out var q) ? q.ToDouble() : 0;
            double f = count.GetValueOrDefault(k) / (double)samples;
            string name = k.Length > 14 ? k[..14] : k;
            if (p == 0)
            {
                bad.Add($"outcome {name} counted at {f:0.000}, not in the model");
            }
            else if (Math.Abs(f - p) > 5 * Math.Sqrt(p * (1 - p) / samples) + 2.0 / samples)
            {
                bad.Add($"outcome {name}: model {p:0.0000}, counted {f:0.0000}");
            }
        }

        return bad;
    }
}
