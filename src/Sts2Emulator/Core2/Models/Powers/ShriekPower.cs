// Shriek (power:SHRIEK_POWER), the port list's power-1 row: AfterDamageReceived. The old body is
// EnemyAI.TriggerShriekIfWounded (a function of its own before round R5; the row's two locations are its own lines),
// called from CardEffects.LandDamageOnEnemy under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ShriekPower
{
    /// <summary>
    /// game: ShriekPower.AfterDamageReceived. An event: the old body is called for a hit on the owner that cost HP (the
    /// old site, CardEffects.LandDamageOnEnemy: the HP lost is above 0, whatever the props; here the result's
    /// UnblockedDamage), with the owner's old EnemyState (the body takes one: the owner is an enemy). The body keeps its
    /// own tests (the threshold above 0, the owner at or below it and alive) and acts on that one enemy, so the
    /// dispatcher's one call per owner is the old site's one call. The body deals no damage: no re-entry.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is not { } owner
            || c.K.Commands.Result(a.Result).UnblockedDamage <= 0)
            return HookStep.Done;
        EnemyAI.TriggerShriekIfWounded(owner);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ShriekPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
