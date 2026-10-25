// game: HardenedShellPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-4;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Hardened Shell: its owner loses at most the power's amount of HP a turn; this is the turn's tally.</summary>
internal static class HardenedShellPower
{
    /// <summary>game: HardenedShellPower.AfterDamageReceived. When the target is the owner and the hit was not fully blocked:
    /// damageReceivedThisTurn += the unblocked damage (the HP display is presentation). The tally is the old
    /// <c>EnemyState.HardenedShellDamageThisTurn</c> (cleared by the old <c>CombatEngine.RefillHardenedShells</c>), through
    /// the old body <c>CardEffects.HardenedShell_AfterDamageReceived</c>, which adds the HP lost since the HP it is handed:
    /// the HP now plus the unblocked damage. The old engine gives the power to enemies only.</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner)
            return HookStep.Done;
        var r = c.Damage(a.Result);
        if (r.WasFullyBlocked || c.Enemy(p.Owner) is not { } e)
            return HookStep.Done;
        CardEffects.HardenedShell_AfterDamageReceived(e, e.Hp + r.UnblockedDamage);
        return HookStep.Done;
    }

    /// <summary>game: HardenedShellPower.ModifyHpLostBeforeOstyLate. When the target is the owner and the amount is not 0:
    /// min(amount, Amount - damageReceivedThisTurn), in the decimal fold (the tally is the old
    /// <c>EnemyState.HardenedShellDamageThisTurn</c>; the old int body is
    /// <c>CardEffects.HardenedShell_ModifyHpLostBeforeOstyLate</c>); else the amount.</summary>
    internal static decimal ModifyHpLostBeforeOstyLate(HookCtx c, ListenerRef self, in ModifyHpLostArgs a,
        decimal current) =>
        c.Power(self) is { } p && a.Target == p.Owner && current != 0m && c.Enemy(p.Owner) is { } e
            ? Math.Min(current, p.Amount - e.HardenedShellDamageThisTurn)
            : current;

    /// <summary>game: HardenedShellPower.AfterModifyingHpLostBeforeOsty: a flash (presentation), nothing to change.</summary>
    internal static HookStep AfterModifyingHpLostBeforeOsty(HookCtx c, ListenerRef self,
        in AfterModifyingHpLostBeforeOstyArgs a, ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "HardenedShellPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived)
            .On(HookId.ModifyHpLostBeforeOstyLate, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostBeforeOstyLate)
            .On(HookId.AfterModifyingHpLostBeforeOsty,
                (HookEvent<AfterModifyingHpLostBeforeOstyArgs>)AfterModifyingHpLostBeforeOsty);
}
