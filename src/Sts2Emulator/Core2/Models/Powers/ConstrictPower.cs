// Constrict (power:CONSTRICT_POWER), the port list's power-4 row: AfterDeath. The old body is
// CombatEngine.Constrict_AfterDeath (round R5), called from CombatEngine.AnswerDeaths' real-death loop under the condition
// the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ConstrictPower
{
    /// <summary>
    /// game: ConstrictPower.AfterDeath. An event: the old body, which removes the player's Constrict, is called on a death
    /// that was not prevented (the old site runs for a real death) of a Slithering Strangler (the old site's test of the
    /// fallen: its def id; the game's test is that the dead creature is the power's Applier) while the
    /// handler's owner is on the player's side (the old function acts on the player's buffs, and the old state's one power
    /// holder on that side is the player; a holder on an enemy is not what the old function removes, so its handler does
    /// nothing). The debuff is a survivor's, not the fallen's: the per-owner condition is "my owner is the player and the
    /// fallen is a Strangler", not "the fallen is my owner".
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || c.Enemy(me.Owner) is not null
            || c.Enemy(a.Creature) is not { } fallen || fallen.DefId != KE.SlitheringStrangler)
            return HookStep.Done;
        CombatEngine.Constrict_AfterDeath(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ConstrictPower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
