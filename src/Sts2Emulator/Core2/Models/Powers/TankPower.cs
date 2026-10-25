// Tank (power:TANK_POWER), the port list's power-1 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.Tank_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the condition the site
// keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class TankPower
{
    /// <summary>
    /// game: TankPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack on the owner (the old site:
    /// the defender's Tank above 0; the target must be the owner, whoever deals); 1 otherwise. The number is the old
    /// function's flat 2: the amount is a stack count, not a multiplier.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        return (decimal)BuffSystem.Tank_ModifyDamageMultiplicative();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "TankPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
