// Lethality (power:LETHALITY_POWER), the port list's power-5 row: ModifyDamageMultiplicative. The old body is
// CardEffects.Lethality_ModifyDamageMultiplicative (round R5), called from CardEffects.DealDamageToEnemy under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class LethalityPower
{
    /// <summary>
    /// game: LethalityPower.ModifyDamageMultiplicative. A fold-mul: the factor for the owner's powered attack while the
    /// combat's <c>LethalAttackPlay</c> flag is set (the old site, CardEffects.DealDamageToEnemy: the power's amount is
    /// above 0 and the flag the play set holds); 1 otherwise. The number is the old function's (a float, 1 + amount/100,
    /// taken as a decimal as the template takes Vulnerable's).
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m
            || !c.OldCombat.LethalAttackPlay)
            return 1m;
        return (decimal)CardEffects.Lethality_ModifyDamageMultiplicative((int)me.Amount);
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "LethalityPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
