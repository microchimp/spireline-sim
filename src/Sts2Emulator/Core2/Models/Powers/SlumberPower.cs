// game: SlumberPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Slumber: the Slumbering Beetle sleeps until it has taken enough hits.</summary>
internal static class SlumberPower
{
    /// <summary>game: SlumberPower.AfterDamageReceived. When the target is the owner and the unblocked damage is not 0: the
    /// power decremented; at 0 or less the Slumbering Beetle is stunned into its wake-up move. The old body is
    /// <c>CardEffects.Slumber_AfterDamageReceived</c> (Stunned and the intent cleared).</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || c.Damage(a.Result).UnblockedDamage == 0
            || c.Enemy(p.Owner) is not { } e)
            return HookStep.Done;
        CardEffects.Slumber_AfterDamageReceived(e);
        return HookStep.Done;
    }

    /// <summary>game: SlumberPower.AfterRemoved (pseudo-hook, from the owner's death removal): the override does nothing
    /// (it returns at once), so neither does the handler.</summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SlumberPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived)
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
