// game: TheInsatiable (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-2;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Insatiable: its kill-sequence override (F10). Presentation only.</summary>
internal static class TheInsatiable
{
    /// <summary>
    /// game: TheInsatiable.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When the dead creature is this monster: sets a music parameter; otherwise nothing. Presentation only: no combat
    /// state in any branch, so the handler changes nothing.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "TheInsatiable")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
