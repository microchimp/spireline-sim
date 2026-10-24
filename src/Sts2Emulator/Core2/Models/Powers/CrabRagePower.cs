// Crab Rage (power:CRAB_RAGE_POWER), the port list's power-3 row: AfterDeath. The old body is
// CombatEngine.TryEnrageCrabPartner, called from CombatEngine.AnswerDeaths' death loop after the revive chain, made internal
// for this handler (batch L3a). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class CrabRagePower
{
    /// <summary>
    /// game: CrabRagePower.AfterDeath. An event for a survivor: when the dead creature is not the owner and is on the
    /// owner's side (an enemy, as the owner), the owner gains Strength and Block and the power is removed. The handler acts
    /// for a living owner (the old body's test). The old body enrages every other living Crab Rage holder when the fallen
    /// holds Crab Rage (the old test, kept; the game's is any ally's death) and removes each one's power: the rage is spent,
    /// so a second holder's handler finds no partner left to enrage and a dispatch with N holders equals one old call (the
    /// N-holder test). The game does not read wasRemovalPrevented; the old call sits after the revive chain, which no Crab
    /// Rage holder takes. No damage: no re-entry. The old site's turn after a rage (TurnPlayerToFaceSurvivor) is
    /// SurroundedPower's AfterDeath.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Owner(self) is not { } me || a.Creature == me || !c.IsAlive(me) || c.Enemy(me) is null
            || c.Enemy(a.Creature) is not { } fallen)
            return HookStep.Done;
        CombatEngine.TryEnrageCrabPartner(c.OldCombat, fallen);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "CrabRagePower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
