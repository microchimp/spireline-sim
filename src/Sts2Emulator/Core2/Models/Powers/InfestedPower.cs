// InfestedPower (power:INFESTED_POWER), batch L3b (docs/rebuild/MIGRATION_F08.md): AfterDeath and
// ShouldStopCombatFromEnding, over the old bodies CombatEngine.Infested_AfterDeath and Infested_ShouldStopCombatFromEnding.
// Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>game: InfestedPower: the Phrog Parasite's death sends in four Wrigglers, holding the fight open.</summary>
internal static class InfestedPower
{
    /// <summary>
    /// game: InfestedPower.AfterDeath. When the dead creature is the owner and its removal was not prevented (after the death animation's wait),
    /// four stunned Wrigglers are added, in the Phrog Parasite's Wriggler slots. The old body is
    /// <c>CombatEngine.Infested_AfterDeath</c> (old site: CombatEngine.AnswerDeaths, for a fallen holding Infested), handed
    /// the stream the old site hands it (<see cref="HookCtx.Niche" />); its HP rolls read the combat's NicheHpRng first.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || a.Creature != me.Owner || me.Amount <= 0
            || c.Enemy(me.Owner) is not { } fallen)
            return HookStep.Done;
        CombatEngine.Infested_AfterDeath(c.OldCombat, c.Niche, fallen);
        return HookStep.Done;
    }

    /// <summary>game: InfestedPower.ShouldStopCombatFromEnding: true while the power is on its owner (the old
    /// <c>CombatEngine.Infested_ShouldStopCombatFromEnding</c>).</summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Owner(self) is { } me && c.Enemy(me) is { } e && CombatEngine.Infested_ShouldStopCombatFromEnding(e);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "InfestedPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding);
}
