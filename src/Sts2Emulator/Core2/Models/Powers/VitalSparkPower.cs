// game: VitalSparkPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Vital Spark: Skills carry its Tainted affliction while it lives; its removal clears them.</summary>
internal static class VitalSparkPower
{
    /// <summary>
    /// game: VitalSparkPower.AfterRemoved (pseudo-hook, from the owner's death removal): CardCmd.ClearAffliction of every
    /// Tainted card of every player. The old engine keeps no affliction on the cards: it reads the Tainted stamp off the
    /// living enemies' Vital Spark when a Skill is played (<c>CombatEngine.ApplyAfterCardPlayedPowers</c>), so the dead
    /// owner's power leaves nothing to clear and the handler changes nothing.
    /// </summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a,
        ChoiceCtx? ctx)
    {
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "VitalSparkPower")
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
