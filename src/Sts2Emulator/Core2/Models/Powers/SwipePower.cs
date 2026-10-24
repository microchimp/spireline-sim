// Swipe (power:SWIPE_POWER), the port list's power-2 row: BeforeDeath. The old body is CombatEngine.Swipe_BeforeDeath
// (round R5), called from CombatEngine.AnswerDeaths for the fallen creature under the condition the site keeps. Game
// code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SwipePower
{
    /// <summary>
    /// game: SwipePower.BeforeDeath. An event: before the death of the power's owner, an enemy, that holds a stolen card
    /// and did not escape (the old site runs inside CombatEngine.AnswerDeaths' loop over the fallen and asks that the
    /// fallen creature's StolenCard is set and that it did not escape; the dispatcher calls every listener for every
    /// death, so the handler asks that the dying creature is its owner), the old body gives the card back (the combat's
    /// StolenBackCards), clears the owner's stolen card and removes the power. The old engine keeps one stolen card per
    /// thief, so the second of two Swipe instances on one owner finds the power gone and does nothing.
    /// </summary>
    internal static HookStep BeforeDeath(HookCtx c, ListenerRef self, in BeforeDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Creature != me.Owner || c.Enemy(me.Owner) is not { } fallen
            || fallen.StolenCard is not { } swiped || fallen.Escaped)
            return HookStep.Done;
        CombatEngine.Swipe_BeforeDeath(c.OldCombat, fallen, swiped);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SwipePower").On(HookId.BeforeDeath, (HookEvent<BeforeDeathArgs>)BeforeDeath);
}
