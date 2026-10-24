// Plow (power:PLOW_POWER), the port list's power-3 row: AfterDamageReceived. The old body is
// EnemyAI.TriggerPlowIfWounded, called from CardEffects.LandDamageOnEnemy under the condition the site keeps. Game code
// is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class PlowPower
{
    /// <summary>
    /// game: PlowPower.AfterDamageReceived. An event: the old body, which strips the owner's Plow, Strength and
    /// Temporary Strength and stuns it once a hit leaves its HP at or under the threshold (the body's own test, the
    /// amount being the threshold), is called for a hit on the owner that lost it HP (the old site,
    /// CardEffects.LandDamageOnEnemy: HP lost above 0, whatever the hit's props, so an unpowered tick counts; the HP lost
    /// is the result's UnblockedDamage). The old body takes the owner's EnemyState: the old site is the enemy side's
    /// landing of a hit.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is not { } owner
            || c.K.Commands.Result(a.Result).UnblockedDamage <= 0)
            return HookStep.Done;
        EnemyAI.TriggerPlowIfWounded(owner);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PlowPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
