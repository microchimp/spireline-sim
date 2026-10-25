// Calcify (power:CALCIFY_POWER), the port list's power-4 row: ModifyDamageAdditive. The old body is
// CardEffects.Calcify_ModifyDamageAdditive (round R5), called from CardEffects.OstyAttackDamage, the one site that costs
// an Osty attack, under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class CalcifyPower
{
    /// <summary>
    /// game: CalcifyPower.ModifyDamageAdditive. A fold-add: the addend for a powered attack dealt by one of the owner's
    /// pets (the game's rule as the old engine records it on the buff: a powered attack whose dealer is Osty; the player's
    /// own attacks get nothing); 0 otherwise. The old site, CardEffects.OstyAttackDamage, is reached only for an Osty
    /// attack, so its condition in the handler's terms is the dealer test: a pet of the owner and not the owner itself
    /// (c.IsOwnerOrPet admits both; the old state's one pet is Osty, whose owner is the player). No card source is asked
    /// for: the old site costs every Osty attack, card or not. The number is the old function's, read off the player's
    /// buff (so the owner is the player; a holder on an enemy has no pet and adds nothing).
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || !a.Props.IsPoweredAttack() || a.Dealer is not { } dealer || dealer == me.Owner
            || !c.IsOwnerOrPet(dealer, me.Owner))
            return 0m;
        return CardEffects.Calcify_ModifyDamageAdditive(c.OldCombat);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "CalcifyPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
