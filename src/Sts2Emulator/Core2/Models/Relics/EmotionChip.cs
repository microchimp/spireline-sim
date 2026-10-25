// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
namespace Sts2Emulator.Core2.Models.Relics;

internal static class EmotionChip
{
    /// <summary>
    /// game: EmotionChip.AfterDamageReceived. While the combat is in progress, for unblocked damage on the owner, the
    /// relic's status turns Active: presentation only. The effect (the orbs' passives at the next turn start) reads the
    /// combat history (LostHpInPreviousTurn), which the old engine plays in RelicEffects.ApplyAfterPlayerTurnStart; this
    /// hook changes no state.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "EmotionChip")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
