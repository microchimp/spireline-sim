// Vigor (power:VIGOR_POWER), the port list's power-1 row: ModifyDamageAdditive. The old body is
// BuffSystem.Vigor_ModifyDamageAdditive (round R5), called from BuffSystem.IncomingDamage with the attacker's buff list and
// no condition at the site (the fold adds the number as it is). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class VigorPower
{
    /// <summary>
    /// game: VigorPower.ModifyDamageAdditive. A fold-add: the addend for a powered attack dealt by the owner itself, the
    /// old function's number read off the owner's buff list; 0 otherwise. The dealer must be the owner, as for Strength
    /// (no pet case; an enemy's Vigor raises its own attacks the same way). The spending of the power after the attack is
    /// another hook's, not this fold's. The old engine's Osty path folds Osty's hit with the player's list (see
    /// StrengthPower): the same R7 check.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack())
            return 0m;
        return BuffSystem.Vigor_ModifyDamageAdditive(c.OldBuffs(me.Owner));
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "VigorPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
