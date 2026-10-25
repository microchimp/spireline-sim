// game: TangledPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-4;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Tangled: its owner's cards are Entangled while it lasts.</summary>
internal static class TangledPower
{
    /// <summary>
    /// game: TangledPower.AfterRemoved (pseudo-hook, from the owner's death removal): CardCmd.ClearAffliction of the old
    /// owner's Entangled cards. The old engine keeps no Entangled affliction on the cards: it reads the cost off the
    /// Tangled buff (<c>CombatEngine.EffectiveCost</c>), so the removed buff leaves nothing to clear and the handler
    /// changes nothing.
    /// </summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a,
        ChoiceCtx? ctx)
    {
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "TangledPower")
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
