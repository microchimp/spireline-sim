namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;

/// <summary>
/// The draw observers (FIGHT_SOLVER.md section 17): what acts on each drawn card in a way that depends on the order of the
/// cards drawn, or makes the count drawn depend on their values. With one present the draws from the unknown middle are
/// sequential; without, a run of plain draws is one chance node over the sub-multisets drawn. The names are the
/// <c>observer</c> entries of <c>tools/solver/draw_observers.json</c>, which <c>scan_draw_observers.py</c> writes from
/// the engine's draw loop, its draw-result readers and the hook listeners (<c>DrawObserverTests</c> holds the two equal).
/// </summary>
public static class DrawObservers
{
    static readonly HashSet<int> PillageIds = Ids("Pillage");
    static readonly HashSet<int> ScrapeIds = Ids("Scrape");

    /// <summary>Each observer with the test of whether it can act in this combat.</summary>
    public static readonly (string Name, Func<CombatState, bool> On)[] All =
    [
        ("Hellraiser", s => Buff(s, BuffId.Hellraiser)),
        ("ChainsOfBinding", s => Buff(s, BuffId.ChainsOfBinding)),
        ("Speedster", SpeedsterCanKill),
        ("Iteration", s => Buff(s, BuffId.Iteration)),
        ("Pagestorm", s => Buff(s, BuffId.Pagestorm)),
        ("Stratagem", s => Buff(s, BuffId.StratagemPower)),
        ("Pillage", s => StateKey.Readers.Holds(s, PillageIds)),
        ("Scrape", s => StateKey.Readers.Holds(s, ScrapeIds)),
        ("kernel-listener", s => s.Kernel is { AnyOn: true } k && (k.HasAny(Core2.HookId.AfterCardDrawn) || k.HasAny(Core2.HookId.AfterCardDrawnEarly))),
    ];

    /// <summary>Why the macro draw is off at <paramref name="s" />: the hand kept in order, an observer able to act, or null.</summary>
    public static string? Off(CombatState s)
    {
        if (StateKey.Readers.OrderedPiles(s, KeyMode.Expected).Contains("Hand"))
        {
            return "hand-ordered";
        }

        foreach (var (name, on) in All)
        {
            if (on(s))
            {
                return "observer:" + name;
            }
        }

        return null;
    }

    static bool Buff(CombatState s, BuffId id) => BuffSystem.Get(s.PlayerBuffs, id) > 0;

    /// <summary>
    /// Speedster hits every enemy per card drawn mid-turn: it acts on the order when a hit can kill (the draw stops at a
    /// fight's end; a death's effects see the hand drawn so far). Able to kill: a living enemy within ten hits.
    /// </summary>
    static bool SpeedsterCanKill(CombatState s)
    {
        int amount = BuffSystem.Get(s.PlayerBuffs, BuffId.Speedster);
        return amount > 0 && s.Enemies.Exists(e => e.Hp > 0 && e.Hp <= (long)amount * CardEffects.MaxCardsInHand);
    }

    static HashSet<int> Ids(string name) => Cards.FindId(name) is int id ? [id] : [];
}

/// <summary>
/// A run of plain draws taken as one chance node (FIGHT_SOLVER.md section 17): <see cref="Uids" /> were laid out at the
/// front of epoch <see cref="Epoch" />'s rest and must be taken again, by one draw call's loop, between the same two
/// chance calls (<see cref="Call" />: the solver's chance calls seen before the check that found them; -1 for the step's
/// end). A replay that takes them otherwise drew a count that depends on the cards: refused (<c>macro-draw-mismatch</c>).
/// </summary>
internal sealed record MacroBlock(int Epoch, int[] Uids, int Call);

/// <summary>The sub-multisets of a k-card draw from the rest, with their multivariate hypergeometric probabilities.</summary>
internal static class MacroDraw
{
    /// <summary>
    /// Every way to take <paramref name="k" /> cards from groups of interchangeable cards of sizes <paramref name="sizes" />:
    /// the count taken from each group and the probability prod C(c_g, x_g) / C(n, k), exact. In order: the first group's
    /// count highest first, then the next's (any fixed order would do; this one is the sequential draw's first branch first).
    /// </summary>
    public static List<(int[] Counts, Rational P)> Parts(IReadOnlyList<int> sizes, int k)
    {
        int n = sizes.Sum();
        if (k < 0 || k > n)
        {
            throw new ArgumentOutOfRangeException(nameof(k));
        }

        var total = Choose(n, k);
        var result = new List<(int[] Counts, Rational P)>();
        var counts = new int[sizes.Count];
        var suffix = new int[sizes.Count + 1];
        for (int g = sizes.Count - 1; g >= 0; g--)
        {
            suffix[g] = suffix[g + 1] + sizes[g];
        }

        void Walk(int g, int left, System.Numerics.BigInteger ways)
        {
            if (g == sizes.Count)
            {
                if (left == 0)
                {
                    result.Add(([.. counts], new Rational(ways, total)));
                }

                return;
            }

            int hi = Math.Min(left, sizes[g]);
            int lo = Math.Max(0, left - suffix[g + 1]);
            for (int x = hi; x >= lo; x--)
            {
                counts[g] = x;
                Walk(g + 1, left - x, ways * Choose(sizes[g], x));
            }

            counts[g] = 0;
        }

        Walk(0, k, System.Numerics.BigInteger.One);
        return result;
    }

    public static System.Numerics.BigInteger Choose(int n, int k)
    {
        if (k < 0 || k > n)
        {
            return System.Numerics.BigInteger.Zero;
        }

        System.Numerics.BigInteger c = System.Numerics.BigInteger.One;
        k = Math.Min(k, n - k);
        for (int i = 1; i <= k; i++)
        {
            c = c * (n - k + i) / i;
        }

        return c;
    }
}
