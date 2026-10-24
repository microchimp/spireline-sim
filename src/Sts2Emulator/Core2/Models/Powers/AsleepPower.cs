// game: AsleepPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Asleep: the Lagavulin Matriarch sleeps until it takes unblocked damage.</summary>
internal static class AsleepPower
{
    /// <summary>game: AsleepPower.AfterDamageReceived. When the target is the owner and the unblocked damage is not 0: its
    /// Plating removed, IsAwake set, the owner stunned into its wake-up move, the power removed. The old body is
    /// <c>EnemyAI.WakeMatriarchIfStruck</c> (the monster batch left it to this power: LagavulinMatriarch's own
    /// AfterDamageReceived handler does not call it). The animation is presentation.</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || c.Damage(a.Result).UnblockedDamage == 0
            || c.Enemy(p.Owner) is not { } e)
            return HookStep.Done;
        EnemyAI.WakeMatriarchIfStruck(e);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "AsleepPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
