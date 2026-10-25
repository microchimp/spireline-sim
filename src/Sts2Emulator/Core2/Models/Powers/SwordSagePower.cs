// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SwordSagePower
{
    /// <summary>
    /// game: SwordSagePower.AfterRemoved (pseudo-hook, the kill's power sweep). Every Sovereign Blade of the old owner's
    /// player loses the power's amount of replays; an owner with no player has no cards. The old engine never removes the
    /// power (it gives the replays in CardEffects.GiveSovereignBladesReplays, the body here with the amount negated);
    /// under F10 the sweep calls this after the player's death.
    /// </summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx)
    {
        int amount = a.Removed.Where(b => b.Id == BuffId.SwordSage).Sum(b => b.Magnitude);
        if (a.OldOwner.Kind != CreatureKind.Player || amount == 0)
            return HookStep.Done;
        CardEffects.GiveSovereignBladesReplays(c.OldCombat, -amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SwordSagePower").On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
