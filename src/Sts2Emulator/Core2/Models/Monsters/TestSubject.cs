// game: TestSubject (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch
// monster-2; docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Test Subject: its death and room-removal overrides (F10). Presentation only.</summary>
internal static class TestSubject
{
    /// <summary>
    /// game: TestSubject.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When the dead creature is this
    /// monster and it no longer holds Adaptable (its last form): its colour reset and a music parameter; otherwise
    /// nothing. Presentation only, so the handler changes nothing. The respawn is AdaptablePower.AfterDeath's (the
    /// dead-state move), whose body is the port list's old location for this pair too
    /// (<c>CombatEngine.TryRespawnTestSubject</c>): that power's pair (batch power-4) calls it, not this handler.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    /// <summary>
    /// game: TestSubject.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving monster only). Unsubscribes the two creature events it
    /// subscribed in AfterAddedToRoom (PowerApplied and PowerRemoved, whose handlers only set its colour while Intangible
    /// comes and goes). Presentation only, so the handler changes nothing.
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "TestSubject")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
