// game: Queen (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Queen: her answer to the Torch Head Amalgam's death, and her room removal.</summary>
internal static class Queen
{
    /// <summary>
    /// game: Queen.AfterDeath (Hook.AfterDeath, once per Queen in the combat). When the dead creature is a Torch Head
    /// Amalgam and this Queen is alive: she notes the amalgam is gone (the old engine reads that off the roster in
    /// EnemyAI.SelectIntent, so it keeps no flag), says her line (visual), and when her announced move is
    /// BURN_BRIGHT_FOR_ME it becomes the enrage there and then (SetMoveImmediate): the old body
    /// <see cref="CombatEngine.Queen_AfterDeath" />, split per Queen from EnrageQueenIfWaitingToBurnBright so a handler
    /// acts on its own Queen only. When the dead creature is this Queen: music and animation only. The game does not
    /// read <c>wasRemovalPrevented</c>.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Owner(self) is not { } me)
            return HookStep.Done;
        if (c.IsMonster(a.Creature, "TorchHeadAmalgam") && c.IsAlive(me) && c.Enemy(me) is { } queen)
            CombatEngine.Queen_AfterDeath(queen);
        return HookStep.Done;
    }

    /// <summary>
    /// game: Queen.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving Queen only). Outside a game over, sets her node's
    /// animation track. No combat state, so the handler changes nothing.
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "Queen")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
