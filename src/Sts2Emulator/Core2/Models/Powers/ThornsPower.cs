// Thorns (power:THORNS_POWER), the port list's power-1 row: BeforeDamageReceived. The old bodies are two, one per side:
// CardEffects.TriggerEnemyThorns (an enemy's Thorns, called from the player's hit on that enemy: DealDamageToEnemy's powered
// path and DealUnpoweredDamageToEnemy when Omnislice asks) and EnemyAI.ApplyPlayerThorns (the player's Thorns, called from
// an enemy's attack on the player), both made internal for this handler (batch L3a). Game code is cited by class and method
// only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ThornsPower
{
    /// <summary>
    /// game: ThornsPower.BeforeDamageReceived. An event: when the target is the power's owner, the dealer is known and the
    /// hit is a powered attack or comes from Omnislice (the game's test; the old sites are the player's powered hit and
    /// Omnislice's splash on an enemy, and an enemy's attack on the player), the dealer takes the power's amount back. The
    /// handler chooses the old body by the owner's side, as the two call sites do: an enemy owner runs
    /// CardEffects.TriggerEnemyThorns for that enemy, the player runs EnemyAI.ApplyPlayerThorns against the dealing enemy.
    /// Each body reads its own owner's Thorns, so the dispatcher's one call per owner is one old call per owner.
    /// A re-entry: under the switch both bodies reach CreatureCmd.Damage (DealDamageToPlayer and DealUnpoweredDamageToEnemy
    /// are gated), unpowered, with the owner as dealer and no card source, as the game's call. The game also sets
    /// SkipHurtAnim, a presentation flag. Where the old body and the game differ: the enemy-owned body hits the player
    /// whoever dealt (the game hits the dealer, Osty when Osty struck), and the player-owned body needs an enemy dealer.
    /// </summary>
    internal static HookStep BeforeDamageReceived(HookCtx c, ListenerRef self, in BeforeDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || a.Dealer is not { } dealer
            || !(a.Props.IsPoweredAttack() || IsOmnislice(c, a.CardSource)))
            return HookStep.Done;
        if (c.Enemy(me.Owner) is { } owner)
            CardEffects.TriggerEnemyThorns(c.OldCombat, owner);
        else if (c.IsPlayer(me.Owner) && c.Enemy(dealer) is { } attacker)
            EnemyAI.ApplyPlayerThorns(attacker, c.OldCombat);
        return HookStep.Done;
    }

    /// <summary>The game's test that the card source is Omnislice: the card source's def is Omnislice.</summary>
    private static bool IsOmnislice(HookCtx c, CardHandle? source) =>
        source is { } h && c.Card(h) is { } card && Math.Abs(card.ModelId) == CL.Omnislice;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ThornsPower")
            .On(HookId.BeforeDamageReceived, (HookEvent<BeforeDamageReceivedArgs>)BeforeDamageReceived);
}
