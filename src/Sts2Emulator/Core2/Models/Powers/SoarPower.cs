// Soar (power:SOAR_POWER), the port list's power-3 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.Soar_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the condition the
// site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SoarPower
{
    /// <summary>
    /// game: SoarPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack on the owner (the old
    /// site, BuffSystem.IncomingDamage, is the powered-attack path and asks that the defender's Soar is above 0); 1
    /// otherwise. The dealer is not read. The number is the old function's flat one half.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        return (decimal)BuffSystem.Soar_ModifyDamageMultiplicative();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SoarPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
