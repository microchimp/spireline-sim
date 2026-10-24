// Possess Speed (power:POSSESS_SPEED_POWER), the port list's power-2 row: AfterDeath. The old body is
// CombatEngine.PossessSpeed_AfterDeath (round R5), called from CombatEngine.AnswerDeaths for the fallen creature under
// the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class PossessSpeedPower
{
    /// <summary>
    /// game: PossessSpeedPower.AfterDeath. An event: on the death of the power's owner, an enemy, that was not prevented
    /// (the old site runs inside CombatEngine.AnswerDeaths' loop over the fallen, the answer to a real death, for a
    /// fallen creature whose Possess Speed is above 0; the dispatcher calls every listener for every death, so the
    /// handler asks that the dead creature is its owner), the old body gives the player the Dexterity the owner took and
    /// removes the power from the corpse. The corpse is still listed when the hook runs (the kill marks it dying), so the
    /// old body finds it.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || a.Creature != me.Owner
            || c.Enemy(me.Owner) is not { } fallen || me.Amount <= 0m)
            return HookStep.Done;
        CombatEngine.PossessSpeed_AfterDeath(c.OldCombat, fallen);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PossessSpeedPower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
