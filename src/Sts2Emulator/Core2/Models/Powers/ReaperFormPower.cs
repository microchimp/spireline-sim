// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ReaperFormPower
{
    /// <summary>
    /// game: ReaperFormPower.AfterDamageGiven. An event: when the dealer is the owner or the owner's pet, the hit is a
    /// powered attack and its total damage (blocked plus unblocked) is positive, Doom of total times the power's amount on
    /// the target, from no card. The old site is inline in CardEffects.LandDamageOnEnemy after the switch's early return:
    /// it multiplies the damage asked (the handler takes the game's result total) and skips a dead target (kept), and
    /// applies the Doom raw with CardEffects.ApplyPowerDoomRiders by hand (no card source, so no Lamp), as here.
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || !c.IsOwnerOrPet(a.Dealer, p.Owner) || !a.Props.IsPoweredAttack()
            || c.Enemy(a.Target) is not { Hp: > 0 } target)
            return HookStep.Done;
        var r = c.Damage(a.Results);
        int total = r.BlockedDamage + r.UnblockedDamage;
        if (total <= 0)
            return HookStep.Done;
        int doom = total * (int)p.Amount;
        BuffSystem.Apply(target.Buffs, BuffId.Doom, doom);
        CardEffects.ApplyPowerDoomRiders(c.OldCombat, target, doom);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ReaperFormPower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
