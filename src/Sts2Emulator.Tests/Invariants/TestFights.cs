using Sts2Emulator.Core;
using Sts2Emulator.Tests.Invariants;

namespace Sts2Emulator.Tests;

/// <summary>The fights the invariant tests play: the suite's builder, given the streams a run gives a fight.</summary>
internal static class TestFights
{
    public static Combat Open(CombatFactory.ActOneEncounter encounter, int seed)
    {
        var state = Fight.Encounter(encounter, seed: seed).State;
        // The builder leaves the shuffle, energy-cost and monster streams unset; a run sets all of them.
        state.ShuffleRng = new CountingRandom(seed + 101);
        state.EnergyCostRng = new CountingRandom(seed + 102);
        state.AiRng = new CountingRandom(seed + 103);
        return new Combat(state, new CountingRandom(seed));
    }
}
