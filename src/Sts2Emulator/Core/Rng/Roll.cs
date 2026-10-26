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
    public static int Walk(Random rng, string site, float[] weights)
    {
        float total = 0f;
        double sum = 0d;
        var odds = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            odds[i] = weights[i];
            sum += weights[i];
        }

        // Enumerable.Sum over floats accumulates in double and casts once: the same total.
        total = (float)sum;
        return Branch(rng, site, odds, d => WalkIndex(d, total, weights));
    }

    private static int WalkIndex(double draw, float total, float[] weights)
    {
        if (total <= 0f)
        {
            return 0;
        }

        float roll = (float)(draw * total);
        for (int i = 0; i < weights.Length; i++)
        {
            roll -= weights[i];
            if (roll <= 0f)
            {
                return i;
            }
        }

        return weights.Length - 1;
    }

    /// <summary>A uniform pick of one of <paramref name="n" />: <c>Next(n)</c>, which hooked is a chance node of n equal branches.</summary>
    public static int Uniform(Random rng, int n) => rng.Next(n);
}
