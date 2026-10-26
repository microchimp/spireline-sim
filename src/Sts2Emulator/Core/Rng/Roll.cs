namespace Sts2Emulator.Core.Rng;

/// <summary>
/// The one door for a weighted roll in the combat rules (docs/rebuild/FIGHT_SOLVER.md section 4).
/// The game rolls an enemy's move in <c>RandomBranchState.GetNextState</c>: the branches' effective
/// weights summed, one float drawn uniformly below the sum, the branches walked in order subtracting
/// each weight until the remainder is not positive. The emulator's sites spell that draw out each in
/// their own arithmetic; through here each site keeps its arithmetic (<c>decide</c>, which maps the
/// draw to a branch exactly as the inline code did) and also names its weights.
/// </summary>
/// <remarks>
/// Unhooked, a roll is one <c>NextDouble</c> and <c>decide</c>: the bytes of the inline code. Hooked
/// (a solver copy), the chance source answers the branch from the weights, and the solver's odds are
/// weight / sum, exact as rationals of the weights. The game's float draw differs from those odds by
/// at most one float step at each branch boundary; that is the only rounding the exact layer accepts.
/// The rng log and the trace name a draw by the first engine frames above the stream, so they skip
/// this class: a draw routed through here is labelled with the site it was before.
/// </remarks>
public static class Roll
{
    /// <summary>
    /// A weighted branch: the index of the branch <paramref name="decide" /> picks for the draw, or,
    /// hooked, the source's pick. <paramref name="weights" /> are the effective weights in branch
    /// order (a blocked branch weighs 0); an all-zero list has the single outcome 0, which is the
    /// game's walk with nothing eligible (it draws and takes the first declared branch).
    /// </summary>
    public static int Branch(Random rng, string site, double[] weights, Func<double, int> decide) =>
        rng is CountingRandom counting ? counting.Branch(site, weights, decide) : decide(rng.NextDouble());

    /// <summary>
    /// <c>RandomBranchState.GetNextState</c>'s walk over float weights: draw <c>NextFloat(sum)</c>, then
    /// subtract in order until the remainder is not positive (the first branch when every weight is 0).
    /// </summary>
    public static int Walk(Random rng, string site, float[] weights) => Walk(rng, site, weights, static w => w);

    /// <summary>
    /// <see cref="Walk(Random, string, float[])" /> over <paramref name="items" /> weighed by
    /// <paramref name="weight" />. Unhooked it is the draw and the walk with nothing allocated (a
    /// static <paramref name="weight" /> is cached); the weights are copied out for the source only.
    /// </summary>
    public static int Walk<T>(Random rng, string site, IReadOnlyList<T> items, Func<T, float> weight)
    {
        // Enumerable.Sum over floats accumulates in double and casts once: the same total.
        double sum = 0d;
        for (int i = 0; i < items.Count; i++)
        {
            sum += weight(items[i]);
        }

        float total = (float)sum;
        return rng is CountingRandom { Chance: not null } hooked
            ? HookedWalk(hooked, site, items, weight, total)
            : WalkIndex(rng.NextDouble(), total, items, weight);
    }

    // Apart from Walk, so its closure is allocated only when hooked: a lambda capturing Walk's
    // parameters would allocate its display class on every call.
    private static int HookedWalk<T>(CountingRandom hooked, string site, IReadOnlyList<T> items, Func<T, float> weight, float total)
    {
        var odds = new double[items.Count];
        for (int i = 0; i < odds.Length; i++)
        {
            odds[i] = weight(items[i]);
        }

        return hooked.Branch(site, odds, d => WalkIndex(d, total, items, weight));
    }

    private static int WalkIndex<T>(double draw, float total, IReadOnlyList<T> items, Func<T, float> weight)
    {
        if (total <= 0f)
        {
            return 0;
        }

        float roll = (float)(draw * total);
        for (int i = 0; i < items.Count; i++)
        {
            roll -= weight(items[i]);
            if (roll <= 0f)
            {
                return i;
            }
        }

        return items.Count - 1;
    }

    /// <summary>A uniform pick of one of <paramref name="n" />: <c>Next(n)</c>, which hooked is a chance node of n equal branches.</summary>
    public static int Uniform(Random rng, int n) => rng.Next(n);
}
