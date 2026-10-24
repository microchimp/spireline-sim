// Flutter (power:FLUTTER_POWER), the port list's power-3 rows: ModifyDamageMultiplicative and AfterDamageReceived. The
// old bodies are BuffSystem.Flutter_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage, and
// CardEffects.Flutter_AfterDamageReceived (round R5), called from CardEffects.LandDamageOnEnemy, each under the
// condition its site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class FlutterPower
{
    /// <summary>
    /// game: FlutterPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack on the owner (the old
    /// site, BuffSystem.IncomingDamage, is the powered-attack path and asks that the defender's Flutter is above 0); 1
    /// otherwise. The dealer is not read. The number is the old function's flat one half: the amount is a count of hits
    /// to take, not a multiplier.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        return (decimal)BuffSystem.Flutter_ModifyDamageMultiplicative();
    }

    /// <summary>
    /// game: FlutterPower.AfterDamageReceived. An event: the old body, which takes one off the owner's Flutter and stuns
    /// the owner when none is left, is called for a powered attack on the owner that lost it HP (the old site,
    /// CardEffects.LandDamageOnEnemy: a powered hit, HP lost above 0, the target's Flutter above 0; the HP lost is the
    /// result's UnblockedDamage). The old body takes the owner's EnemyState: the old site is the enemy side's landing of
    /// a hit, and the old player-side body has no Flutter, so the handler acts for an enemy owner only, as the old
    /// engine does.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m
            || c.Enemy(me.Owner) is not { } owner || c.K.Commands.Result(a.Result).UnblockedDamage <= 0)
            return HookStep.Done;
        CardEffects.Flutter_AfterDamageReceived(owner);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "FlutterPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative)
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
