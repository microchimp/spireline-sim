// Ravenous (power:RAVENOUS_POWER), the port list's power-1 row: AfterDeath. The old body is CombatEngine.Ravenous_AfterDeath
// (round R5), a loop over every living holder; batch L3a split it per holder into CombatEngine.Ravenous_AfterDeathOf, which
// the loop and this handler both call. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class RavenousPower
{
    /// <summary>
    /// game: RavenousPower.AfterDeath. An event for a survivor: on a death that was not prevented, of a creature that is
    /// not the owner and is on the owner's side (an enemy, as the owner), with the owner alive, the owner is stunned and
    /// gains Strength by the power's amount (the devour animation is not modelled). The old site calls the body after the
    /// revive chain (a real death) and the body's loop takes the living holders; the handler calls the per-holder part for
    /// its own owner alone, so N holders give N owners' effects, not N squared. No damage: no re-entry.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || a.Creature == me || !c.IsAlive(me)
            || c.Enemy(me) is not { } owner || c.Enemy(a.Creature) is null)
            return HookStep.Done;
        CombatEngine.Ravenous_AfterDeathOf(owner);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "RavenousPower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
