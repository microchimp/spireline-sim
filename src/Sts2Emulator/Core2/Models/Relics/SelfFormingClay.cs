// game: SelfFormingClay (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Self-Forming Clay: unblocked damage to its owner gives block next turn.</summary>
internal static class SelfFormingClay
{
    /// <summary>game: SelfFormingClay.AfterDamageReceived. While the combat is in progress, when the target is the owner and
    /// the unblocked damage is positive: SelfFormingClayPower by BlockNextTurn (3) on the owner. The old body is
    /// <c>RelicEffects.SelfFormingClay_AfterDamageReceived</c> (the power is the old BlockNextTurn buff).</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (!c.CombatInProgress || !c.IsPlayer(a.Target) || c.Damage(a.Result).UnblockedDamage <= 0)
            return HookStep.Done;
        RelicEffects.SelfFormingClay_AfterDamageReceived(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "SelfFormingClay")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
