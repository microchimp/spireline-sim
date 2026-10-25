// Gigantification (power:GIGANTIFICATION_POWER), the port list's power-2 row: ModifyDamageMultiplicative. The old body
// is CardEffects.Gigantification_ModifyDamageMultiplicative (round R5), called from CardEffects.DmgFrom under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class GigantificationPower
{
    /// <summary>
    /// game: GigantificationPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack whose card source
    /// is an Attack card of the power's owner while the power's amount is above 0 (the old site, CardEffects.DmgFrom, is
    /// the player's card damage: the def's type is Attack and the player's Gigantification is above 0); 1 otherwise. The
    /// number is the old function's flat 3. The game triples the one card its BeforeAttack latched (the first powered
    /// attack card) and its AfterAttack clears the latch and spends a stack: that latch is the play site's
    /// (CombatEngine), not this fold's, so the handler reads the card's type and the amount as the old site does. The
    /// dealer is not read (the old site reads none; the factor follows the card), so the owner's card dealt by its pet
    /// counts.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || !a.Props.IsPoweredAttack() || a.CardSource is not { } source
            || c.CardOwner(source) != me.Owner || c.Card(source) is not { } card
            || GeneratedData.Cards.Get(card.ModelId).Type != CardType.Attack || me.Amount <= 0m)
            return 1m;
        return CardEffects.Gigantification_ModifyDamageMultiplicative();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "GigantificationPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
