// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class BookRepairKnife
{
    /// <summary>
    /// game: BookRepairKnife.AfterDiedToDoom. An event: the relic's heal times the count of the Doom-killed creatures that
    /// are not the owner and whose powers all let the death trigger fatal. The old site, CardEffects.KillDoomedEnemies,
    /// counts the same (CardEffects.TriggersFatal) and calls RelicEffects.ApplyAfterDiedToDoom ungated, under F08 too:
    /// no Core2 command dispatches AfterDiedToDoom yet, so the heal is the old site's, once; this handler is the one a
    /// Core2 Doom kill will dispatch (it replaces the old call then).
    /// </summary>
    internal static HookStep AfterDiedToDoom(HookCtx c, ListenerRef self, in AfterDiedToDoomArgs a, ChoiceCtx? ctx)
    {
        if (c.Relic(self) is null)
            return HookStep.Done;
        var s = c.OldCombat;
        int fatal = a.Creatures.Count(r => !c.IsPlayer(r) && c.Enemy(r) is { } e && CardEffects.TriggersFatal(s, e));
        RelicEffects.ApplyAfterDiedToDoom(s, fatal);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "BookRepairKnife")
            .On(HookId.AfterDiedToDoom, (HookEvent<AfterDiedToDoomArgs>)AfterDiedToDoom);
}
