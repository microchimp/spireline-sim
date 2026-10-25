// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class EnvenomPower
{
    /// <summary>
    /// game: EnvenomPower.AfterDamageGiven. An event: when the dealer is the owner, the hit is a powered attack and its
    /// unblocked damage is positive, Poison of the power's amount on the target (PowerCmd.Apply). The old site is inline in
    /// CardEffects.LandDamageOnEnemy (powered, HP lost), after the switch's early return, so under F08 only this handler
    /// applies it. The body is the old chokepoint CardEffects.ApplyEnemyDebuffToTarget (Snecko Skull, the Lamp, Artifact),
    /// which skips a target already at 0 HP; the old site reads no dealer, the handler takes the game's dealer test.
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Dealer != p.Owner || !a.Props.IsPoweredAttack()
            || c.Damage(a.Results).UnblockedDamage <= 0 || c.Enemy(a.Target) is not { } target)
            return HookStep.Done;
        var s = c.OldCombat;
        CardEffects.ApplyEnemyDebuffToTarget(s, target, BuffId.Poison, (int)p.Amount, s.ShuffleRng ?? new Random(0));
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "EnvenomPower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
