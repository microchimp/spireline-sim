// game: OwlMagistrate (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Owl Magistrate: its kill-sequence override (F10). Presentation only.</summary>
internal static class OwlMagistrate
{
    /// <summary>
    /// game: OwlMagistrate.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving monster only). While it is flying, stops its flight sound loop. Presentation only: no combat state,
    /// so the handler changes nothing; it is the game's call at the game's point (the room removal, before the move
    /// state reset).
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "OwlMagistrate")
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
