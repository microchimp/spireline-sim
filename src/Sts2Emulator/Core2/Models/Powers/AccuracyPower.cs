// Accuracy (power:ACCURACY_POWER; the old engine's ShivDamage buff, which OldModelIds maps to it), the port list's power-4
// row: ModifyDamageAdditive. The old body is CardEffects.Accuracy_ModifyDamageAdditive (round R5), called from
// CardEffects.ShivDamage, which the two Shiv sites (the play arm named Shiv, and ApplyBaseDamageAndBlock for a card named
// Shiv or HiddenDaggers) reach. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class AccuracyPower
{
    /// <summary>
    /// game: AccuracyPower.ModifyDamageAdditive. A fold-add: the addend for the owner's powered attack whose card source is
    /// one of the old engine's Shivs (the old sites cost a card named Shiv or HiddenDaggers through CardEffects.ShivDamage,
    /// which adds Accuracy_ModifyDamageAdditive to every card it costs, first Shiv of the turn or not); 0 otherwise. The
    /// number is the old function's, read off the player's buff (so the owner is the player).
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack()
            || a.CardSource is not { } source || c.Card(source) is not { } card || !IsShiv(card.ModelId))
            return 0m;
        return CardEffects.Accuracy_ModifyDamageAdditive(c.OldCombat);
    }

    /// <summary>The old sites' test of the card: its def is named Shiv or HiddenDaggers. The game tests the Shiv tag on the
    /// card source (CardTag.Shiv), not the name; the old card data carries the tag too (<c>CardDef.ShivTag</c>), which the
    /// old sites do not read. A card tagged Shiv under another name would count in the game and not here. Kept as the old
    /// sites have it (the handler must equal the old function), and recorded as an old-site divergence for
    /// docs/rebuild/MIGRATION_F08.md (R7 checks), as Phantom Blades' is.</summary>
    private static bool IsShiv(int defId) => GeneratedData.Cards.Get(defId).Name is "Shiv" or "HiddenDaggers";

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "AccuracyPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
