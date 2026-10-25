// game: RingingPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-4;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Ringing: its owner may play one Ringing card a turn while it lasts.</summary>
internal static class RingingPower
{
    /// <summary>
    /// game: RingingPower.AfterRemoved (pseudo-hook, from the owner's death removal): CardCmd.ClearAffliction of the old
    /// owner's Ringing cards. The old engine keeps no Ringing affliction on the cards: it reads the block off the Ringing
    /// buff (<c>CombatEngine.IsBlockedByRinging</c>), so the removed buff leaves nothing to clear and the handler changes
    /// nothing.
    /// </summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a,
        ChoiceCtx? ctx)
    {
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "RingingPower")
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
