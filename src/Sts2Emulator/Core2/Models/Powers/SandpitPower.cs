// game: SandpitPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Sandpit: when it runs out, the creatures it holds are eaten.</summary>
internal static class SandpitPower
{
    /// <summary>
    /// game: SandpitPower.AfterRemoved (pseudo-hook). When the old owner or the target is dead: nothing; else the held
    /// players and Osty are killed (force). Core2 dispatches AfterRemoved only from the owner's death removal, where the
    /// old owner is dead, so the game returns at once and so does the handler. Not modelled: a removal outside a death
    /// (PowerCmd.Remove, not part of F08/F10).
    /// </summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a,
        ChoiceCtx? ctx)
    {
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SandpitPower")
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
