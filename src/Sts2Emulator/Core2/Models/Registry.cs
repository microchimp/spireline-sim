// The migrated models (docs/rebuild/TIER1_KERNEL.md section 4): one static class per game model under Core2/Models/<Kind>/,
// each an IMigratedModel whose Register adds its handlers. A model is listed here when its family migrates; its handlers
// register only while that family's switch is on.
namespace Sts2Emulator.Core2.Models;

public static class Registry
{
    /// <summary>Every migrated model's registration, in no particular order (the table is keyed, not ordered).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> All = [];
}
