// Sic 'Em (power:SIC_EM_POWER), the port list's power-4 row: AfterDamageGiven. The old body is
// CardEffects.SicEm_AfterDamageGiven (round R5), called from CardEffects.DealOstyDamage, the one site that lands an Osty
// attack, under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SicEmPower
{
    /// <summary>
    /// game: SicEmPower.AfterDamageGiven. An event: the old body, which summons Osty for the power's amount (a living pet
    /// grows by it), is called when the creature hit is the handler's owner (the debuff sits on the enemy hit; the old
    /// site reads the target's SicEm and asks that it is above 0) and the dealer is a pet of the hit's card owner and not
    /// that owner (the old site, CardEffects.DealOstyDamage, is Osty's attack, reached from card plays alone, so the hit's
    /// card source is the player's card and the dealer the player's pet; the game's rule as the old engine records it on
    /// the buff: when Osty damages the holder). HookCtx has no pet accessor, and a model may not read the old creatures
    /// itself, so the pet is found as c.IsOwnerOrPet of the card owner (asked for in the batch's PR: a pet accessor would
    /// drop the card test). The old site asks nothing of the result, and the game's hook runs for every result, so a
    /// blocked hit summons too; a hit that kills the holder still summons, as the dying creature stays listed until the
    /// kill and the old site reads the corpse's buff before the death answer.
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || me.Amount <= 0m || a.Dealer is not { } dealer
            || a.CardSource is not { } card)
            return HookStep.Done;
        var player = c.CardOwner(card);
        if (dealer == player || !c.IsOwnerOrPet(dealer, player))
            return HookStep.Done;
        CardEffects.SicEm_AfterDamageGiven(c.OldCombat, (int)me.Amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SicEmPower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
