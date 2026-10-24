// Slippery (power:SLIPPERY_POWER), the port list's power-2 rows: AfterDamageReceived and ModifyHpLostAfterOsty. The old
// bodies are CardEffects.Slippery_ModifyHpLostAfterOsty (the number the loss becomes) and
// CardEffects.Slippery_AfterModifyingHpLostAfterOsty (the stack spent), both called from CardEffects.LandDamageOnEnemy
// under the one condition the site keeps. The port list registers the stack's spending under AfterDamageReceived while
// the function is named for another hook: the handler is registered for the row's hook and calls what the row names.
// Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SlipperyPower
{
    /// <summary>
    /// game: SlipperyPower.ModifyHpLostAfterOsty. A fold-set over the HP lost after the pet redirect: the old function's
    /// number for a loss of 1 or more by the power's owner, an enemy, whatever the props and the dealer (the old site,
    /// CardEffects.LandDamageOnEnemy, lands powered and unpowered hits alike and asks that the target's Slippery is above
    /// 0 and the loss is at least 1); the value unchanged otherwise. The old site compares an int; the running value here
    /// is a decimal, cast once at the HP loss, so a loss under 1 is left as it is (it truncates to 0 either way).
    /// </summary>
    internal static decimal ModifyHpLostAfterOsty(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is null || me.Amount <= 0m
            || current < 1m)
            return current;
        return CardEffects.Slippery_ModifyHpLostAfterOsty();
    }

    /// <summary>
    /// game: SlipperyPower.AfterDamageReceived. An event: the stack spent by
    /// CardEffects.Slippery_AfterModifyingHpLostAfterOsty (the function the row names for this hook) when a hit on the
    /// power's owner, an enemy, took HP (the old site spends it beside the number above, under the same condition:
    /// Slippery above 0 and a loss of at least 1, which is the result's UnblockedDamage of 1 or more once the fold has
    /// run). The dispatcher runs the hook for a receiver the hit did not kill; a killed owner is stripped of its powers by
    /// the kill, as the old engine strips its corpse.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is not { } target || me.Amount <= 0m
            || c.K.Commands.Result(a.Result).UnblockedDamage < 1)
            return HookStep.Done;
        CardEffects.Slippery_AfterModifyingHpLostAfterOsty(target);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SlipperyPower")
            .On(HookId.ModifyHpLostAfterOsty, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOsty)
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
