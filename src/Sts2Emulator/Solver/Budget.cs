namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

/// <summary>The solve ran past its node, step or time budget: the iteration in progress is dropped, never reported as exact.</summary>
public sealed class BudgetExceeded(string what) : Exception($"budget reached: {what}")
{
    public string What { get; } = what;
}

/// <summary>
/// The in-search size estimator: the week-3 sizing measurement's estimate (W4-S1), used to skip a horizon that cannot
/// finish within the budget before anything is spent on it (FIGHT_SOLVER.md section 7). For a horizon of h turns past
/// this one, S = P0 x (hands x sequences)^h, where P0 is the number of distinct ordered play sequences from the hand
/// with the energy left (identical cards one kind, each kind played up to its count, costs within the energy, ending
/// now included), hands is the number of distinct 5-card multisets the deck's kinds can deal, and sequences is the
/// same count for this hand at the turn's full energy. The measurement put the rest-of-fight estimate under 10^7 for
/// 58% of self-play decisions, for every decision with one turn left and for about 1% with four or more: a
/// turn-by-turn deepening with this check stops where the whole fight is out of reach and leaves the cut to L0.
/// It ignores targets, potions, screens and enemy rolls, so it undercounts where those are many.
/// </summary>
public static class SizeEstimate
{
    public const int HandSize = 5;

    /// <summary>log10 of the estimated state count of a solve <paramref name="horizon" /> turns past this one.</summary>
    public static double Log10(CombatState s, int horizon)
    {
        var hand = Kinds(s.Hand.Where(c => CombatEngine.CanPlayFromHand(s, c)));
        double p0 = Math.Log10(Sequences(hand, s.Energy));
        if (horizon <= 0)
        {
            return p0;
        }

        var deck = Kinds(s.Hand.Concat(s.DrawPile).Concat(s.DiscardPile));
        double hands = Math.Log10(Math.Max(1d, Hands(deck.Select(k => k.Count).ToArray(), HandSize)));
        double seq = Math.Log10(Sequences(hand, Math.Max(s.Energy, s.MaxEnergy)));
        return Math.Min(100d, p0 + horizon * (hands + seq));
    }

    static List<(int Cost, int Count)> Kinds(IEnumerable<CardInstance> cards) =>
        [.. cards.GroupBy(c => (Math.Abs(c.DefId), c.Upgraded)).Select(g => (Cards.Get(g.Key.Item1).Cost, g.Count()))];

    /// <summary>Distinct ordered play sequences (ending now included); an X cost (negative) takes all the energy.</summary>
    public static double Sequences(List<(int Cost, int Count)> kinds, int energy)
    {
        var memo = new Dictionary<string, double>();
        var counts = kinds.Select(k => k.Count).ToArray();

        double F(int e)
        {
            string key = string.Join(",", counts) + "|" + e;
            if (memo.TryGetValue(key, out double hit))
            {
                return hit;
            }

            double n = 1;
            for (int i = 0; i < counts.Length; i++)
            {
                int cost = kinds[i].Cost;
                int pay = cost < 0 ? e : cost;
                if (counts[i] == 0 || pay > e)
                {
                    continue;
                }

                counts[i]--;
                n += F(e - pay);
                counts[i]++;
            }

            memo[key] = n;
            return n;
        }

        return F(Math.Max(0, energy));
    }

    /// <summary>The coefficient of x^h in the product over kinds of (1 + x + ... + x^count): distinct h-card hands.</summary>
    public static double Hands(int[] counts, int h)
    {
        var poly = new double[h + 1];
        poly[0] = 1;
        foreach (int c in counts)
        {
            var next = new double[h + 1];
            for (int i = 0; i <= h; i++)
            {
                if (poly[i] == 0)
                {
                    continue;
                }

                for (int j = 0; j <= c && i + j <= h; j++)
                {
                    next[i + j] += poly[i];
                }
            }

            poly = next;
        }

        return poly[h];
    }
}
