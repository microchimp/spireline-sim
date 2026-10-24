// Double Damage (power:DOUBLE_DAMAGE_POWER), the port list's power-5 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.DoubleDamage_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Power;

public sealed class DoubleDamagePower : IMigratedModel
{
    private DoubleDamagePower()
    {
    }

    /// <summary>
    /// game: DoubleDamagePower.ModifyDamageMultiplicative. A fold-mul: the factor for the owner's powered card attack
    /// (the old site, BuffSystem.IncomingDamage: the attacker's Double Damage above 0; the card source the old fold
    /// cannot see is required as the old engine's comment on the site records the game's rule); 1 otherwise. The number
    /// is the old function's flat 2: the amount is a stack count, not a multiplier.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m
            || a.CardSource is null)
            return 1m;
        return BuffSystem.DoubleDamage_ModifyDamageMultiplicative();
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DoubleDamagePower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
