// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ReflectPower
{
    /// <summary>
    /// game: ReflectPower.AfterDamageReceived. An event: when the target is the owner, the hit's blocked damage is positive,
    /// it is a powered attack and has a dealer, the dealer takes the blocked damage back, unpowered, from the owner, no card
    /// (CreatureCmd.Damage: a re-entry). The old site, EnemyAI.ApplyPlayerReflect, sits in EnemyAI.DealAttackDamage after
    /// the switch's early return (EnemyHit), so under F08 only this handler deals it; the old body is the same unpowered
    /// hit on the attacking enemy.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || a.Dealer is not { } dealer || !a.Props.IsPoweredAttack())
            return HookStep.Done;
        int blocked = c.Damage(a.Result).BlockedDamage;
        if (blocked <= 0 || Old(c, dealer) is not { } hit || Old(c, p.Owner) is not { } owner)
            return HookStep.Done;
        CreatureCmd.Damage(c.K, [hit], blocked, OldEntry.Unpowered, owner, null);
        return HookStep.Done;
    }

    /// <summary>The old creature a reference names: an enemy, the player, or Osty (a creature with a pet owner).</summary>
    private static OldCreature? Old(HookCtx c, CreatureRef r) =>
        c.Enemy(r) is { } e ? OldCreature.Of(e) : c.IsPlayer(r) ? OldCreature.Player : c.PetOwner(r) is not null ? OldCreature.Osty : null;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ReflectPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
