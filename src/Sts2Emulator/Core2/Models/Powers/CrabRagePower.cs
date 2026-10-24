// Crab Rage (power:CRAB_RAGE_POWER), the port list's power-3 row: AfterDeath. The old body is
// CombatEngine.TryEnrageCrabPartner, called from CombatEngine.AnswerDeaths' death loop after the revive chain; its step for
// one holder is CombatEngine.EnrageCrab (extracted in R7b2), which this handler calls. Game code is cited by class and
// method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class CrabRagePower
{
    /// <summary>
    /// game: CrabRagePower.AfterDeath. An event for a survivor: when the dead creature is not the owner and is on the
    /// owner's side, the owner gains Strength and Block and the power is removed. The death is any ally's, as the game's
    /// (R7b2; the old site, CombatEngine.TryEnrageCrabPartner, answers only a Crab Rage holder's death and enrages every
    /// other living holder): the handler calls the old body's step for its own owner, CombatEngine.EnrageCrab (Strength,
    /// the unpowered Block, the power removed, so the rage is spent once). The handler acts for a living enemy owner (the
    /// old body's test; the old state's player never holds Crab Rage), so an ally is an enemy. The game does not read
    /// wasRemovalPrevented; the old call sits after the revive chain, which no Crab Rage holder takes. No damage: no
    /// re-entry. The old site's turn after a rage (TurnPlayerToFaceSurvivor) is SurroundedPower's AfterDeath.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Owner(self) is not { } me || a.Creature == me || !c.IsAlive(me) || c.Enemy(me) is not { } owner
            || c.Enemy(a.Creature) is null || BuffSystem.Get(owner.Buffs, BuffId.CrabRage) <= 0)
            return HookStep.Done;
        CombatEngine.EnrageCrab(owner);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "CrabRagePower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
