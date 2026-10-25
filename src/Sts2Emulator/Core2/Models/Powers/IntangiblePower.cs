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
    /// game: IntangiblePower.ModifyHpLostAfterOsty. A fold-set over the HP lost after the pet redirect, with the game's
    /// value: for a loss of the power's owner, 1 for any amount of 1 or more and the amount itself below 1, whatever the
    /// props and the dealer (the old site, CardEffects.LosePlayerHp, passes every loss to BuffSystem.CapHpLoss); the value
    /// unchanged for another target. The old function takes an int, the HP lost after the old engine's cast, and agrees
    /// with this value on every int; here the running value is a decimal, cast once at the HP loss, so 1.5 becomes 1 and
    /// 0.5 stays 0.5. The dispatcher lists this power as a modifier only when the truncated value moved (1.5 to 1 does
    /// not). The stack test is the old function's (Intangible above 0): the game removes a power at 0. The game returns the
    /// amount unchanged while the combat is not in progress (CombatManager.IsInProgress, <c>HookCtx.CombatInProgress</c>,
    /// R7b2): a loss after the combat has ended is not capped. The old function has no such test.
    /// </summary>
    internal static decimal ModifyHpLostAfterOsty(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (!c.CombatInProgress || c.Power(self) is not { } me || a.Target != me.Owner || me.Amount <= 0m || current < 1m)
            return current;
        return 1m;
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
