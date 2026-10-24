// game: KinPriest (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch monster-2;
// docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Kin Priest: its kill-sequence override (F10). Presentation only.</summary>
internal static class KinPriest
{
    /// <summary>
    /// game: KinPriest.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When the dead creature is a Kin
    /// Follower and this priest is not dead: a music parameter, and when no living follower is left on the priest's side,
    /// the first living priest there says its all-followers-dead line (KinPriest.AllFollowerDeathResponse, a talk
    /// bubble). When the dead creature is this priest: a music parameter. Presentation only: no combat state in any
    /// branch, so the handler changes nothing.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "KinPriest")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
