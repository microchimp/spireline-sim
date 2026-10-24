// SurprisePower (power:SURPRISE_POWER), batch L3b (docs/rebuild/MIGRATION_F08.md): AfterDeath and
// ShouldStopCombatFromEnding, over the old bodies CombatEngine.Surprise_AfterDeath and Surprise_ShouldStopCombatFromEnding.
// Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>game: SurprisePower: the Gremlin Merc's death sends in a Sneaky and a Fat Gremlin, holding the fight open.</summary>
internal static class SurprisePower
{
    /// <summary>
    /// game: SurprisePower.AfterDeath. When the dead creature is the owner and its removal was not prevented: a Sneaky
    /// Gremlin and a Fat Gremlin are added, the Fat one holding Heist for the gold the owner's Thievery took. The old body is
    /// <c>CombatEngine.Surprise_AfterDeath</c> (old site: CombatEngine.AnswerDeaths, for a fallen holding Surprise), handed
    /// the stream the old site hands it (<see cref="HookCtx.Niche" />); its HP rolls read the combat's NicheHpRng first.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || a.Creature != me.Owner || me.Amount <= 0
            || c.Enemy(me.Owner) is not { } fallen)
            return HookStep.Done;
        CombatEngine.Surprise_AfterDeath(c.OldCombat, c.Niche, fallen);
        return HookStep.Done;
    }

    /// <summary>game: SurprisePower.ShouldStopCombatFromEnding: true while the power is on its owner (the old
    /// <c>CombatEngine.Surprise_ShouldStopCombatFromEnding</c>).</summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Owner(self) is { } me && c.Enemy(me) is { } e && CombatEngine.Surprise_ShouldStopCombatFromEnding(e);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SurprisePower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding);
}
