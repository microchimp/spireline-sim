// Flame Barrier (power:FLAME_BARRIER_POWER), the port list's power-1 row: AfterDamageReceived. The old body is
// EnemyAI.ApplyPlayerFlameBarrier, called per hit from an enemy's attack on the player, made internal for this handler
// (batch L3a). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class FlameBarrierPower
{
    /// <summary>
    /// game: FlameBarrierPower.AfterDamageReceived. An event: when the target is the power's owner, the dealer is known and
    /// the hit was a powered attack, the dealer takes the power's amount back, whether block absorbed the hit or not. The
    /// old site is the enemy attack on the player, per hit; the old body keeps its own tests (the player's Flame Barrier
    /// above 0, the player alive) and acts on that one attacker. The old engine gives the power to the player only, so the
    /// handler acts for a player owner and an enemy dealer. A re-entry: under the switch the body's
    /// DealUnpoweredDamageToEnemy is gated into CreatureCmd.Damage, unpowered, with the owner (the player) as dealer and no
    /// card source, as the game's call.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !c.IsPlayer(me.Owner) || !a.Props.IsPoweredAttack()
            || a.Dealer is not { } dealer || c.Enemy(dealer) is not { } attacker)
            return HookStep.Done;
        EnemyAI.ApplyPlayerFlameBarrier(attacker, c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "FlameBarrierPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
