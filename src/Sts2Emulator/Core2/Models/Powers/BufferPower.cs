// Buffer (power:BUFFER_POWER), the port list's power-2 rows: ModifyHpLostAfterOstyLate and AfterModifyingHpLostAfterOsty.
// The old bodies are CardEffects.Buffer_ModifyHpLostAfterOstyLate (the number the loss becomes) and
// CardEffects.Buffer_AfterModifyingHpLostAfterOsty (the stack spent), both called from CardEffects.LosePlayerHp under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class BufferPower
{
    /// <summary>
    /// game: BufferPower.ModifyHpLostAfterOstyLate. A fold-set over the HP lost after the pet redirect: the old function's
    /// number (0) for a loss of the power's owner, the player, whatever the props and the dealer (the old site,
    /// CardEffects.LosePlayerHp, is every HP loss of the player and asks that the player's Buffer is above 0); the value
    /// unchanged otherwise. The old site returns before Buffer when nothing would be lost; here the dispatcher lists this
    /// power as a modifier only when its 0 moved the truncated value, so a loss that truncates to 0 spends no stack, as
    /// there. The old function acts on the player's list, so the handler acts for an owner that is the player (a power on
    /// a creature that is no enemy; the old state's pet holds no powers) and leaves an enemy's loss alone, as the old
    /// enemy site (CardEffects.LandDamageOnEnemy) has no Buffer.
    /// </summary>
    internal static decimal ModifyHpLostAfterOstyLate(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is not null || me.Amount <= 0m)
            return current;
        return CardEffects.Buffer_ModifyHpLostAfterOstyLate();
    }

    /// <summary>
    /// game: BufferPower.AfterModifyingHpLostAfterOsty. An event the dispatcher's filter-set pass reaches only when this
    /// power is among the fold's modifiers (its 0 took the loss off): the stack spent by
    /// CardEffects.Buffer_AfterModifyingHpLostAfterOsty, which takes the stack count and removes the power at its last
    /// stack (the old site calls it with the player's Buffer, above 0, right before returning the 0 above). The owner is
    /// the player, as above.
    /// </summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self, in AfterModifyingHpLostAfterOstyArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || c.Enemy(me.Owner) is not null || me.Amount <= 0m)
            return HookStep.Done;
        CardEffects.Buffer_AfterModifyingHpLostAfterOsty(c.OldCombat, (int)me.Amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "BufferPower")
            .On(HookId.ModifyHpLostAfterOstyLate, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOstyLate)
            .On(HookId.AfterModifyingHpLostAfterOsty, (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
