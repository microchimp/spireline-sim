// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class MonarchsGazePower
{
    /// <summary>
    /// game: MonarchsGazePower.AfterDamageGiven. An event: when the dealer is the owner and the hit is a powered attack,
    /// the power's amount of MonarchsGazeStrengthDownPower on the target, whatever the damage. The old site is inline in
    /// CardEffects.LandDamageOnEnemy after the switch's early return; its body, CardEffects.ApplyTemporaryStrengthDownTo
    /// (Strength down until the owner's turn ends, Artifact eats it), keeps the old site's living-target test.
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Dealer != p.Owner || !a.Props.IsPoweredAttack()
            || c.Enemy(a.Target) is not { Hp: > 0 } target)
            return HookStep.Done;
        CardEffects.ApplyTemporaryStrengthDownTo(target, (int)p.Amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "MonarchsGazePower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
