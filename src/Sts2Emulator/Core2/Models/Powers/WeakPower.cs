// Weak (power:WEAK_POWER), the port list's power-1 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.Weak_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the condition the site
// keeps. The row's second location (BuffSystem.IncomingDamage, Tracking's read of the defender's Weak) is Tracking's own
// rule, ported by the pilot's TrackingPower; this handler ports the one fold. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class WeakPower
{
    /// <summary>
    /// game: WeakPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack dealt by the owner itself
    /// (the old site: the attacker's Weak above 0; the dealer must be the owner, no pet case, as for Strength); 1
    /// otherwise. The old function reads the owner's list (Debilitate on the attacker halves instead) and takes the
    /// Paper Krane shift the way the two old sites pass it: EnemyAI.DealAttackDamage, where the target is the player,
    /// passes RelicEffects.WeakMultiplierDeltaAgainstPlayer; CardEffects.DealDamageToEnemy, where the target is an
    /// enemy, passes 0; a hit with no target takes none. Osty is never the target of this fold (the pet redirect comes
    /// after it). The old function returns
    /// a float; the conversion at the return gives the game's decimal for every factor the old function makes.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        float delta = a.Target is { } target && c.Enemy(target) is null ? RelicEffects.WeakMultiplierDeltaAgainstPlayer(c.OldCombat) : 0f;
        return (decimal)BuffSystem.Weak_ModifyDamageMultiplicative(c.OldBuffs(me.Owner), delta);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "WeakPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
