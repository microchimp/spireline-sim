// game: WaterfallGiant (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch
// monster-2; docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Waterfall Giant: its death and room-removal overrides (F10). Presentation only.</summary>
internal static class WaterfallGiant
{
    /// <summary>
    /// game: WaterfallGiant.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When the dead creature is
    /// this giant and it no longer holds Steam Eruption: a music parameter and its ambient sound loop stopped; otherwise
    /// nothing. Presentation only, so the handler changes nothing. Standing up to blow is SteamEruptionPower.AfterDeath's
    /// (WaterfallGiant.TriggerAboutToBlowState, the old <c>EnemyAI.TriggerAboutToBlow</c>): that power's pair (batch
    /// power-4) calls it, not this handler.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    /// <summary>
    /// game: WaterfallGiant.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving monster only). Stops its ambient sound loop. Presentation
    /// only, so the handler changes nothing.
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "WaterfallGiant")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
