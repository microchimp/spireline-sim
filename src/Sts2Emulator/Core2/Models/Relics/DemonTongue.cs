// game: DemonTongue (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Demon Tongue: the first unblocked damage its owner takes on its own turn is healed back.</summary>
internal static class DemonTongue
{
    /// <summary>game: DemonTongue.AfterDamageReceived. When the current side is the owner's, the target is the owner, the
    /// unblocked damage is positive and the relic has not triggered this turn: triggered, then CreatureCmd.Heal of the owner
    /// by the unblocked damage. The turn flag is the old relic counter (cleared by the old turn start); the old body
    /// <c>RelicEffects.DemonTongue_AfterDamageReceived</c> sets it and heals through the old heal, which the switch routes
    /// to the Heal command. The flash is presentation.</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        int unblocked = c.Damage(a.Result).UnblockedDamage;
        if (c.Relic(self) is not { Counter: 0 } || !c.OldCombat.PlayerTurn || !c.IsPlayer(a.Target) || unblocked <= 0)
            return HookStep.Done;
        RelicEffects.DemonTongue_AfterDamageReceived(c.OldCombat, unblocked);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "DemonTongue")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
