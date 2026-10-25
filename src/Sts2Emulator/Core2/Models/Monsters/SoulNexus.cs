// game: SoulNexus (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The SoulNexus monster: its kill-sequence overrides (F10).</summary>
internal static class SoulNexus
{
    /// <summary>
    /// game: SoulNexus.AfterDeath, the handler it subscribes to its creature's Died event in AfterAddedToRoom (pseudo-hook
    /// Died, from Creature.InvokeDiedEvent through the kill, to the dying monster only, before the removal query). Unsubscribes itself and sets its node's animation track.
    /// No combat state, so the handler changes nothing.
    /// </summary>
    internal static HookStep Died(HookCtx c, ListenerRef self, in DiedArgs a, ChoiceCtx? ctx) => HookStep.Done;

    /// <summary>
    /// game: SoulNexus.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving monster only). Outside a game over, sets its node's animation track. No combat state, so the handler
    /// changes nothing; it is the game's call at the game's point (the room removal, before the move state reset).
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "SoulNexus")
            .On(HookId.Died, (HookEvent<DiedArgs>)Died)
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
