// game: BeatingRemnant (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Beating Remnant: its owner loses at most 20 HP a turn; this is the turn's tally the cap reads.</summary>
internal static class BeatingRemnant
{
    /// <summary>game: BeatingRemnant.AfterDamageReceived. While the combat is in progress, when the target is the owner:
    /// DamageReceivedThisTurn += the unblocked damage. The tally is the old <c>CombatState.UnblockedDamageThisTurn</c>
    /// (cleared by the old turn start, read by the old cap <c>RelicEffects.BeatingRemnant_ModifyHpLostAfterOsty</c>),
    /// through the old body <c>CardEffects.BeatingRemnant_AfterDamageReceived</c>. Core2 owned the line for every player
    /// hit before; as in the game, only a held relic counts now (the old field has no other reader).</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (!c.CombatInProgress || !c.IsPlayer(a.Target))
            return HookStep.Done;
        CardEffects.BeatingRemnant_AfterDamageReceived(c.OldCombat, c.Damage(a.Result).UnblockedDamage);
        return HookStep.Done;
    }

    /// <summary>game: BeatingRemnant.ModifyHpLostAfterOsty. While the combat is in progress and the target is the owner:
    /// min(amount, MaxHpLoss (20) - DamageReceivedThisTurn), in the decimal fold (the old int body is
    /// <c>RelicEffects.BeatingRemnant_ModifyHpLostAfterOsty</c>, the same 20 and tally); else the amount.</summary>
    internal static decimal ModifyHpLostAfterOsty(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current) =>
        c.CombatInProgress && c.IsPlayer(a.Target) ? Math.Min(current, 20m - c.OldCombat.UnblockedDamageThisTurn) : current;

    /// <summary>game: BeatingRemnant.AfterModifyingHpLostAfterOsty: a flash (presentation), nothing to change.</summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self,
        in AfterModifyingHpLostAfterOstyArgs a, ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "BeatingRemnant")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived)
            .On(HookId.ModifyHpLostAfterOsty, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOsty)
            .On(HookId.AfterModifyingHpLostAfterOsty,
                (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
