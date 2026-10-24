// game: DecimillipedeSegment (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Decimillipede segments' shared logic: the game's three segment models (Back, Front, Middle) inherit it and
/// override visuals only; the old state lists all three under the one enemy id DecimillipedeSegment, so one
/// registration serves the three (<see cref="ModelRegistration.InheritedBy" />).</summary>
internal static class DecimillipedeSegment
{
    /// <summary>
    /// game: DecimillipedeSegment.AfterDeath (Hook.AfterDeath, run for every listener of the combat, each segment its own copy). When
    /// the dead creature is not this monster's: nothing. When it is: outside test mode, swaps the phobia-mode body texture. No combat state in either branch, so the
    /// handler changes nothing.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "DecimillipedeSegment")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .InheritedBy("DecimillipedeSegmentBack", "DecimillipedeSegmentFront", "DecimillipedeSegmentMiddle");
}
