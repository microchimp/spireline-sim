// Tainted (power:TAINTED_POWER), the port list's power-2 row: ModifyDamageAdditive. The old body is
// BuffSystem.Tainted_ModifyDamageAdditive (round R5), called from BuffSystem.IncomingDamage under the condition the site
// keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class TaintedPower
{
    /// <summary>
    /// game: TaintedPower.ModifyDamageAdditive. A fold-add: the addend for a powered attack on the power's owner, whoever
    /// deals it (the old site, BuffSystem.IncomingDamage, is the powered-attack path and adds the defender's Tainted to
    /// every hit it computes, with no test of the attacker); 0 otherwise. The number is the old function's: the owner's
    /// own amount, read off the owner's buff list as the function takes it (0 when the stack is at 0, as there).
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack())
            return 0m;
        return BuffSystem.Tainted_ModifyDamageAdditive(c.OldBuffs(me.Owner));
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "TaintedPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
