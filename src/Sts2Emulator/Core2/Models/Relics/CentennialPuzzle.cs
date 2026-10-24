// game: CentennialPuzzle (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Centennial Puzzle: the first unblocked damage to its owner in a combat draws 3.</summary>
internal static class CentennialPuzzle
{
    /// <summary>game: CentennialPuzzle.AfterDamageReceived. While the combat is in progress, when the target is the owner,
    /// the unblocked damage is positive and the relic is not used this combat: UsedThisCombat set, then 3 draws.
    /// UsedThisCombat is the old relic counter (<c>RelicEffects.FiresOncePerCombat</c> sets it); the draws are the old body
    /// <c>RelicEffects.CentennialPuzzle_AfterDamageReceived</c>. The flash is presentation.</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (!c.CombatInProgress || !c.IsPlayer(a.Target) || c.Damage(a.Result).UnblockedDamage <= 0
            || !RelicEffects.FiresOncePerCombat(c.OldCombat, RelicEffects.CentennialPuzzle))
            return HookStep.Done;
        RelicEffects.CentennialPuzzle_AfterDamageReceived(c.OldCombat, null);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "CentennialPuzzle")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
