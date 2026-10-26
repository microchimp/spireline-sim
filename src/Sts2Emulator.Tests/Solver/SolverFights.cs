using Sts2Emulator.Core;
using Sts2Emulator.Interop;

namespace Sts2Emulator.Tests.Solver;

/// <summary>Real encounters with every combat stream a CountingRandom, as the run's entry points make them.</summary>
internal static class SolverFights
{
    /// <summary>A spread of the encounter list: every <paramref name="stride" />-th encounter.</summary>
    public static IEnumerable<CombatFactory.ActOneEncounter> Sample(int stride) =>
        Enum.GetValues<CombatFactory.ActOneEncounter>().Where((_, i) => i % stride == 0);

    public static (CombatState State, CountingRandom Rng) Encounter(CombatFactory.ActOneEncounter encounter, int seed)
    {
        var state = new CombatState { AscensionLevel = Ascension.DefaultLevel };
        SetStreams(state, seed);
        var rng = new CountingRandom(unchecked(seed * 7919 + 13));
        CombatFactory.Reset(state, rng, TestDeck.StarterDeckIds, (int)encounter, []);
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(state, i) is null)
            {
                StateCloning.SetStream(state, i, new CountingRandom(unchecked(seed * 131 + i)));
            }
        }

        return (state, rng);
    }

    public static void SetStreams(CombatState state, int seed)
    {
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            StateCloning.SetStream(state, i, new CountingRandom(unchecked(seed * 131 + i)));
        }
    }

    /// <summary>A seeded random legal action: the same choice for two equal states.</summary>
    public static int RandomAction(CombatState state, Random policy)
    {
        int[] actions = CombatEngine.ValidActions(state);
        return actions[policy.Next(actions.Length)];
    }

    public static string Bytes(CombatState state) => Convert.ToBase64String(CombatSnapshot.Write(state));
}
