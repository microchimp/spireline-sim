// Hand Drill (relic:HAND_DRILL), the port list's relic-2 row: AfterDamageGiven. The old body is
// RelicEffects.ApplyAfterBlockBroken (a function of its own before round R5, Hand Drill's alone), called from
// CardEffects.LandDamageOnEnemy under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class HandDrill
{
    /// <summary>
    /// game: HandDrill.AfterDamageGiven. An event: the old body, which applies Vulnerable to the enemy hit, is called
    /// when the hit broke the target's block (the old site, CardEffects.LandDamageOnEnemy: block that was there and is
    /// now gone, which is the result's WasBlockBroken), the hit is the player's side's (a dealer that is no enemy: the
    /// site is the player's side landing a hit) and the target is an enemy (the old body takes its EnemyState and asks
    /// that it is alive and that the relic is held).
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Relic(self) is null || a.Dealer is not { } dealer || c.Enemy(dealer) is not null
            || c.Enemy(a.Target) is not { } target || !c.K.Commands.Result(a.Results).WasBlockBroken)
            return HookStep.Done;
        RelicEffects.ApplyAfterBlockBroken(c.OldCombat, target);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "HandDrill")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
