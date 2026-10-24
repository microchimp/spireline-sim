// The migrated models (docs/rebuild/TIER1_KERNEL.md section 4): one `internal static class <GameClass>` per game model under
// Core2/Models/{Powers,Relics,Cards,Monsters,Potions}/ (namespace Sts2Emulator.Core2.Models.<Kind>), each with a
// `public static void Register(HandlerTable t)` that adds its handlers. A model is listed here when its family migrates; its
// handlers register only while that family's switch is on.
//
// One named block per handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3), and `All` spreads the blocks. A batch adds
// its block next to the others and one `.. <Block>` line to `All`; a textual conflict between two batches is resolved by
// keeping every block. The blocks to come, by the names the briefs and the rebuild branches use:
//   local: Monsters (monster-1 and monster-2, rebuild/f08-m1, f08-m2), L1 and L2 (the LOCAL non-monster models,
//          rebuild/f08-l1, f08-l2);
//   cloud: Power1, Power2, Power3, Power4 (C2-C5) and Relic1 (C6).
using Sts2Emulator.Core2.Models.Cards;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;

namespace Sts2Emulator.Core2.Models;

public static class Registry
{
    /// <summary>C1, the pilot (cloud/f08-pilot): the port list's power-5, relic-2 and card-1 rows.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Pilot =
    [
        PhantomBladesPower.Register,
        LethalityPower.Register,
        TrackingPower.Register,
        DoubleDamagePower.Register,
        VitruvianMinion.Register,
        TheBoot.Register,
        HandDrill.Register,
        Melancholy.Register,
    ];

    /// <summary>C2 (cloud/f08-power-1): the port list's power-1 rows that are CLOUD (CLOUD_BRIEFS section 1). Thorns and
    /// Flame Barrier are not here: their old bodies are private, the stop rule's case (the branch's PR lists them).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power1 =
    [
        StrengthPower.Register,
        VulnerablePower.Register,
        WeakPower.Register,
        VigorPower.Register,
        TankPower.Register,
        ColossusPower.Register,
        ShriekPower.Register,
    ];

    /// <summary>C3 (cloud/f08-power-2): the port list's power-2 rows of the models the batch ports.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power2 =
    [
        TaintedPower.Register,
        GigantificationPower.Register,
        IntangiblePower.Register,
        SlipperyPower.Register,
        BufferPower.Register,
        PossessSpeedPower.Register,
        PossessStrengthPower.Register,
        SwipePower.Register,
    ];

    /// <summary>C4 (cloud/f08-power-3): the port list's power-3 rows. Not here: Crab Rage and Surrounded, whose AfterDeath
    /// bodies (CombatEngine.TryEnrageCrabPartner, FacePast, TurnPlayerToFaceSurvivor) are private (the stop rule of
    /// CLOUD_BRIEFS section 3); Stock and Asleep, the local batch L2's.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power3 =
    [
        FlutterPower.Register,
        PersonalHivePower.Register,
        PlowPower.Register,
        SoarPower.Register,
    ];

    /// <summary>C5 (cloud/f08-power-4): the port list's power-4 rows that are CLOUD (Conqueror is ported locally, round R4c).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power4 =
    [
        CalcifyPower.Register,
        AccuracyPower.Register,
        SicEmPower.Register,
        ConstrictPower.Register,
        HeistPower.Register,
    ];

    // Declare every block above All: static fields initialise in textual order, so a block below All is still null when
    // All spreads it.
    /// <summary>Every migrated model's registration, one block per batch, in no particular order (the table is keyed,
    /// not ordered).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> All =
    [
        .. Pilot,
        .. Power1,
        .. Power2,
        .. Power3,
        .. Power4,
    ];
}
