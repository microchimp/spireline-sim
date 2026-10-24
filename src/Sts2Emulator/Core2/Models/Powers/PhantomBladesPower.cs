// Phantom Blades (power:PHANTOM_BLADES_POWER), the port list's power-5 row: ModifyDamageAdditive. The old body is
// CardEffects.PhantomBlades_ModifyDamageAdditive (round R5), called from CardEffects.ShivDamage under the condition the
// site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class PhantomBladesPower
{
    /// <summary>
    /// game: PhantomBladesPower.ModifyDamageAdditive. A fold-add: the addend for the owner's powered attack with a Shiv
    /// as its card source while no Shiv play has finished this turn (the old site, CardEffects.ShivDamage: the card is
    /// named Shiv and <c>ShivsPlayedThisTurn</c> is 0); 0 otherwise. The number is the old function's, read off the
    /// owner's buff (the old function reads the player's list, so the owner is the player).
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack()
            || a.CardSource is not { } source || c.Card(source) is not { } card || !IsShiv(card.ModelId)
            || c.OldCombat.ShivsPlayedThisTurn != 0)
            return 0m;
        return CardEffects.PhantomBlades_ModifyDamageAdditive(c.OldCombat);
    }

    /// <summary>The old site's test of the card: its def is named Shiv. The game tests the Shiv tag on the card source
    /// (CardTag.Shiv), not the name; the old card data carries the tag too (<c>CardDef.ShivTag</c>), which the old site does
    /// not read. A card tagged Shiv under another name would count in the game and not here. Kept as the old site has it
    /// (the handler must equal the old function), and recorded as an old-site divergence in docs/rebuild/MIGRATION_F08.md
    /// (R7 checks).</summary>
    private static bool IsShiv(int defId) => GeneratedData.Cards.Get(defId).Name == "Shiv";

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PhantomBladesPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
