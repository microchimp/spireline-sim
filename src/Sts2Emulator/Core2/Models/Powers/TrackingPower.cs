// Tracking (power:TRACKING_POWER), the port list's power-5 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.Tracking_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the condition
// the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class TrackingPower
{
    /// <summary>
    /// game: TrackingPower.ModifyDamageMultiplicative. A fold-mul: the factor for the owner's powered card attack on a
    /// target that has Weak (the old site, BuffSystem.IncomingDamage: the attacker's Tracking above 0 and the defender's
    /// Weak above 0; the card source the old fold cannot see is required as the old engine's comment on the site records
    /// the game's rule); 1 otherwise. The number is the old function's: the power's own amount.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m
            || a.CardSource is null || a.Target is not { } target
            || BuffSystem.Get(c.OldBuffs(target), BuffId.Weak) <= 0)
            return 1m;
        return BuffSystem.Tracking_ModifyDamageMultiplicative((int)me.Amount);
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "TrackingPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
