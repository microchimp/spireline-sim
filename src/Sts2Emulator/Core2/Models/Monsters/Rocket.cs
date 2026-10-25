// game: Rocket (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Kaiser Crab's right arm: its HP-change, death and room-removal overrides (F08, F10). Presentation only.</summary>
internal static class Rocket
{
    /// <summary>
    /// game: Rocket.AfterCurrentHpChanged (Hook.AfterCurrentHpChanged, run for every listener of the combat). When the
    /// creature is this arm and the delta is negative, the boss background plays the right arm's hurt animation; otherwise
    /// nothing. Presentation only: no combat state, so the handler changes nothing.
    /// </summary>
    internal static HookStep AfterCurrentHpChanged(HookCtx c, ListenerRef self, in AfterCurrentHpChangedArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    /// <summary>
    /// game: Rocket.BeforeDeath (Hook.BeforeDeath, run for every listener of the combat). When the dying creature is this
    /// arm: a death sound, the right arm's death animation, and (by CombatManager.IsOverOrEnding) the body's death animation
    /// or a music parameter. Presentation only: no combat state, so the handler changes nothing. The partner's enrage is
    /// CrabRagePower.AfterDeath's, a power-batch pair.
    /// </summary>
    internal static HookStep BeforeDeath(HookCtx c, ListenerRef self, in BeforeDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    /// <summary>
    /// game: Rocket.BeforeRemovedFromRoom (pseudo-hook, from CombatManager.RemoveCreature through
    /// <see cref="CreatureCmd.RemoveCreature" />, to the leaving monster only). Stops its buff sound loop. Presentation
    /// only: no combat state, so the handler changes nothing.
    /// </summary>
    internal static HookStep BeforeRemovedFromRoom(HookCtx c, ListenerRef self, in BeforeRemovedFromRoomArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "Rocket")
            .On(HookId.AfterCurrentHpChanged, (HookEvent<AfterCurrentHpChangedArgs>)AfterCurrentHpChanged)
            .On(HookId.BeforeDeath, (HookEvent<BeforeDeathArgs>)BeforeDeath)
            .On(HookId.BeforeRemovedFromRoom, (HookEvent<BeforeRemovedFromRoomArgs>)BeforeRemovedFromRoom);
}
