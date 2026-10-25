// game: Vantom (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-1;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Vantom monster: its kill-sequence overrides (F10).</summary>
internal static class Vantom
{
    /// <summary>
    /// game: Vantom.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When
    /// the dead creature is not this monster's: nothing. When it is: sets a music parameter. No combat state in either branch, so the
    /// handler changes nothing.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "Vantom")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
