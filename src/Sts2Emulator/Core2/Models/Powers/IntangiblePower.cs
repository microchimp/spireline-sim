// Intangible (power:INTANGIBLE_POWER), the port list's power-2 rows: ModifyDamageCap, ModifyHpLostAfterOsty and
// AfterModifyingHpLostAfterOsty (its AfterModifyingDamageAmount row is not modelled). The old bodies are
// BuffSystem.CapIncomingDamage (the damage cap, called from BuffSystem.IncomingDamage, CardEffects.DealDamageToPlayer and
// CardEffects.DealUnpoweredDamageToEnemy) and BuffSystem.CapHpLoss (the HP-loss cap, called from CardEffects.LosePlayerHp),
// each under the condition the function itself holds: the defender's Intangible above 0. Game code is cited by class and
// method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class IntangiblePower
{
    /// <summary>
    /// game: IntangiblePower.ModifyDamageCap. A cap: the ceiling on damage aimed at the power's owner, whatever the props
    /// and the dealer (the old sites pass every damage number to BuffSystem.CapIncomingDamage, powered or not, and the
    /// function reads the defender's buffs alone); no ceiling (decimal.MaxValue) otherwise. The number is the old
    /// function's: what it leaves of a damage above every cap, 1 while the owner's Intangible is above 0. The pass keeps
    /// the running minimum of the caps and lowers the damage to it.
    /// </summary>
    internal static decimal ModifyDamageCap(HookCtx c, ListenerRef self, in ModifyDamageArgs a)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner)
            return decimal.MaxValue;
        int cap = BuffSystem.CapIncomingDamage(int.MaxValue, c.OldBuffs(me.Owner));
        return cap == int.MaxValue ? decimal.MaxValue : cap;
    }

    /// <summary>
    /// game: IntangiblePower.ModifyHpLostAfterOsty. A fold-set over the HP lost after the pet redirect: the old function's
    /// cap on a loss of the power's owner, whatever the props and the dealer (the old site, CardEffects.LosePlayerHp,
    /// passes every loss to BuffSystem.CapHpLoss); the value unchanged otherwise. The old function takes and returns an
    /// int, the HP lost after the old engine's cast; here the running value is a decimal, cast once at the HP loss, so
    /// the cap is decided on the truncated value and a value the cap does not lower is returned as it was, decimals kept.
    /// </summary>
    internal static decimal ModifyHpLostAfterOsty(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner)
            return current;
        int lost = (int)Math.Min(current, int.MaxValue);
        int capped = BuffSystem.CapHpLoss(lost, c.OldBuffs(me.Owner));
        return capped == lost ? current : capped;
    }

    /// <summary>
    /// game: IntangiblePower.AfterModifyingHpLostAfterOsty. The dispatcher's filter-set pass reaches it only when this
    /// power is among the fold's modifiers (the cap lowered the loss). The old engine has no body at this point: the
    /// locations the port list names for the pair, BuffSystem.CapIncomingDamage and BuffSystem.CapHpLoss, are the numbers
    /// the two passes above apply. Nothing to do here.
    /// </summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self, in AfterModifyingHpLostAfterOstyArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "IntangiblePower")
            .On(HookId.ModifyDamageCap, (HookCap<ModifyDamageArgs, decimal>)ModifyDamageCap)
            .On(HookId.ModifyHpLostAfterOsty, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOsty)
            .On(HookId.AfterModifyingHpLostAfterOsty, (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
