// Buffer (power:BUFFER_POWER), the port list's power-2 rows: ModifyHpLostAfterOstyLate and AfterModifyingHpLostAfterOsty.
// The old bodies are CardEffects.Buffer_ModifyHpLostAfterOstyLate (the number the loss becomes) and
// CardEffects.Buffer_AfterModifyingHpLostAfterOsty (the stack spent), both called from CardEffects.LosePlayerHp under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class BufferPower
{
    /// <summary>
    /// game: BufferPower.ModifyHpLostAfterOstyLate. A fold-set over the HP lost after the pet redirect: 0 for a loss of the
    /// power's owner, whatever the props and the dealer, the value unchanged otherwise. The owner is any creature, an
    /// enemy too, as the game's (R7b2; the old functions read the player's list alone and the old enemy site,
    /// CardEffects.LandDamageOnEnemy, has no Buffer, so the old engine took an enemy's loss). The number is the old
    /// function's (0). The old site returns before Buffer when nothing would be lost; here the dispatcher lists this power
    /// as a modifier only when its 0 moved the truncated value, so a loss that truncates to 0 spends no stack, as there.
    /// </summary>
    internal static decimal ModifyHpLostAfterOstyLate(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || me.Amount <= 0m)
            return current;
        return CardEffects.Buffer_ModifyHpLostAfterOstyLate();
    }

    /// <summary>
    /// game: BufferPower.AfterModifyingHpLostAfterOsty (PowerCmd.Decrement). An event the dispatcher's filter-set pass
    /// reaches only when this power is among the fold's modifiers (its 0 took the loss off): one stack spent, the power
    /// removed at its last. The player's is the old function, CardEffects.Buffer_AfterModifyingHpLostAfterOsty; an
    /// enemy's is the same step on that enemy's list.
    /// </summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self, in AfterModifyingHpLostAfterOstyArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || me.Amount <= 0m)
            return HookStep.Done;
        if (c.Enemy(me.Owner) is { } enemy)
        {
            if (me.Amount == 1m)
                BuffSystem.Remove(enemy.Buffs, BuffId.Buffer);
            else
                BuffSystem.Apply(enemy.Buffs, BuffId.Buffer, -1);
            return HookStep.Done;
        }
        CardEffects.Buffer_AfterModifyingHpLostAfterOsty(c.OldCombat, (int)me.Amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "BufferPower")
            .On(HookId.ModifyHpLostAfterOstyLate, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOstyLate)
            .On(HookId.AfterModifyingHpLostAfterOsty, (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
