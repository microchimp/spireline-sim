// Infested (power:INFESTED_POWER), the port list's power-4 rows: AfterDeath and ShouldStopCombatFromEnding. The old bodies
// are CombatEngine.Infested_AfterDeath (called from CombatEngine.AnswerDeaths' real-death loop under the condition the
// site keeps) and CombatEngine.Infested_ShouldStopCombatFromEnding (called from CombatEngine.HoldsTheFightOpen, which ors
// it with the other holds for every enemy of the roster), both round R5. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class InfestedPower
{
    /// <summary>
    /// game: InfestedPower.AfterDeath. An event: the old body, which spawns the Wrigglers after the fallen and takes the
    /// power off it, is called on a death that was not prevented (the old site runs for a real death) of the handler's
    /// owner (the power sits on the dying creature) while the power's amount is above 0 (the old site's test of the
    /// fallen's buff). The body takes the old engine's combat rng, which the kernel does not hold: CreateEnemy rolls a
    /// summon's HP on the combat's Niche stream (EnemyAI.RollSummonedHp) and reads that rng only as the fallback of a
    /// combat without one, a case the game has no counterpart for. The handler passes the Niche stream, which the roll
    /// reads anyway, and refuses a combat without one rather than draw another stream (an accessor for the combat rng is
    /// asked of the local session in the batch's PR; this line is what it replaces).
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || a.Creature != me.Owner || me.Amount <= 0m
            || c.Enemy(a.Creature) is not { } fallen)
            return HookStep.Done;
        var s = c.OldCombat;
        var niche = s.NicheHpRng ?? throw new InvalidOperationException(
            "InfestedPower.AfterDeath: the combat has no Niche HP stream, and the kernel holds no combat rng to fall back on");
        CombatEngine.Infested_AfterDeath(s, niche, fallen);
        return HookStep.Done;
    }

    /// <summary>
    /// game: InfestedPower.ShouldStopCombatFromEnding. A predicate (any): the old function's answer for the handler's owner
    /// (an enemy holding the power above 0 keeps the fight open; the old site, CombatEngine.HoldsTheFightOpen, asks it of
    /// every enemy of the roster with no condition of its own, and the dispatcher asks only while no primary enemy lives).
    /// The power leaves the corpse with the death's answer (Infested_AfterDeath removes it), which ends the hold.
    /// </summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Power(self) is { } me && c.Enemy(me.Owner) is { } owner && CombatEngine.Infested_ShouldStopCombatFromEnding(owner);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "InfestedPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding);
}
