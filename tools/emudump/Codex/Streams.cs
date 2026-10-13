// The run's named RNG streams, read by call count, so a codex mismatch can be put down to a stream
// the emulator drew while making the offer. Verification-only (review condition E12).
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

namespace EmuDump.Codex;

public static class Streams
{
    public static readonly string[] Names =
    [
        "UpFront", "Shuffle", "UnknownMapPoint", "CombatCardGeneration", "CombatPotionGeneration",
        "CombatCardSelection", "CombatEnergyCosts", "CombatTargets", "MonsterAi", "Niche", "CombatOrbs",
        "TreasureRoomRelics", "Rewards", "Shops", "Transformations",
    ];

    /// <summary>
    /// The streams a fight's own turns draw from (decompiled: cards, powers, potions, monsters and
    /// in-combat relic hooks reach these through `RunState.Rng`), so a fight that is resolved instead
    /// of played leaves each of them behind the game's. `Niche` is separate: the turns only draw it
    /// when a creature joins mid-fight (`CombatState.AddCreature` rolls its HP on it) or an egg
    /// hatches (`ToughEgg`), which the history shows as monsters beyond the opening roster.
    /// </summary>
    public static readonly string[] FightTurns =
    [
        "Shuffle", "CombatCardGeneration", "CombatPotionGeneration", "CombatCardSelection",
        "CombatEnergyCosts", "CombatTargets", "MonsterAi", "CombatOrbs",
    ];

    static GameRng Get(RunState s, int i) => i switch
    {
        0 => s.Rng.UpFront,
        1 => s.Rng.Shuffle,
        2 => s.Rng.UnknownMapPoint,
        3 => s.Rng.CombatCardGeneration,
        4 => s.Rng.CombatPotionGeneration,
        5 => s.Rng.CombatCardSelection,
        6 => s.Rng.CombatEnergyCosts,
        7 => s.Rng.CombatTargets,
        8 => s.Rng.MonsterAi,
        9 => s.Rng.Niche,
        10 => s.Rng.CombatOrbs,
        11 => s.Rng.TreasureRoomRelics,
        12 => s.PlayerRng.Rewards,
        13 => s.PlayerRng.Shops,
        _ => s.PlayerRng.Transformations,
    };

    public static void Snapshot(RunState s, int[] into)
    {
        for (int i = 0; i < Names.Length; i++)
        {
            into[i] = Get(s, i).CallCount;
        }
    }

    /// <summary>The streams whose call count moved between two snapshots.</summary>
    public static IEnumerable<string> Moved(int[] before, int[] after)
    {
        for (int i = 0; i < Names.Length; i++)
        {
            if (before[i] != after[i])
            {
                yield return Names[i];
            }
        }
    }
}
