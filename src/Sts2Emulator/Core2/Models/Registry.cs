// The migrated models (docs/rebuild/TIER1_KERNEL.md section 4): one class per game model under Core2/Models/<Kind>/,
// sealed with no instances (a static class cannot implement an interface), each an IMigratedModel whose Register adds
// its handlers. A model is listed here when its family migrates; its handlers
// register only while that family's switch is on. One block per handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3):
// a batch's branch adds its block and appends it to All; a textual conflict between two batches is resolved by keeping
// every block.
using Sts2Emulator.Core2.Models.Power;
using Sts2Emulator.Core2.Models.Relic;

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
    ];

    /// <summary>Every migrated model's registration, one block per batch, in no particular order (the table is keyed,
    /// not ordered).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> All =
    [
        // C1 pilot: power-5, relic-2, card-1
        .. Pilot,
    ];
}
