// Mystic Lighter (relic:MYSTIC_LIGHTER), the port list's relic-1 row: ModifyDamageAdditive. The old body is
// RelicEffects.EnchantedCardDamageBonus, called from CardEffects.DmgFrom under the condition the site keeps. Game code is
// cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class MysticLighter
{
    /// <summary>
    /// game: MysticLighter.ModifyDamageAdditive. A fold-add: the addend for a powered attack (ValueProp.IsPoweredAttack)
    /// whose card source carries an enchantment, any enchantment, and is the owner's card (the dealer is not read); 0
    /// otherwise. The old site, CardEffects.DmgFrom: the copy's Enchantment is not None and the def is an Attack (the
    /// handler keeps the Attack test, which the game's override does not have); the old function asks that the relic is
    /// held, and the site reads neither the dealer nor the props, as the computation is only reached for the player's card
    /// attacks; the handler follows the game (see StrikeDummy.ModifyDamageAdditive). The old state holds one card owner,
    /// the player (<see cref="HookCtx.CardOwner" />), so the owner test is met by every live card. The enchantment's own bonus is not this
    /// one: the dispatcher's enchant-add step (Steps.EnchantAdd, before the additive pass) adds the card's
    /// EnchantmentModel.EnchantDamageAdditive through CardEffects.EnchantDamageAdditive, which reads the enchantment alone
    /// (Sharp, Vigorous, TezcatarasEmber, Momentum, Inky) and never this relic, as the old DmgFrom adds the two apart; so
    /// the relic's bonus lands once, here. The copy's enchantment is read off the live card the handle names
    /// (<see cref="Copy" />). The number is the old function's flat 9.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || c.Owner(self) is not { } owner || !a.Props.IsPoweredAttack() || a.CardSource is not { } source
            || Copy(c, source) is not { Enchantment: not Enchantment.None } card
            || GeneratedData.Cards.Get(card.DefId).Type != CardType.Attack || c.CardOwner(source) != owner)
            return 0m;
        return RelicEffects.EnchantedCardDamageBonus(c.OldCombat);
    }

    /// <summary>
    /// The live copy the card source names, through the context alone: its view (pile and position, by the K2b uid) read
    /// back off the old combat's lists, or the card in flight for Play (OldWorld.InFlight, else the held played card);
    /// null for no live card. The CardView carries the model id, pile and position, not the copy's enchantment; Tier 1 reads
    /// of OldCombat and World until the view (or a context member) carries it.
    /// </summary>
    private static CardInstance? Copy(HookCtx c, CardHandle? source)
    {
        if (source is not { } h || c.Card(h) is not { } view)
            return null;
        var s = c.OldCombat;
        return view.Pile switch
        {
            CardPile.Hand => s.Hand[view.Index],
            CardPile.Draw => s.DrawPile[view.Index],
            CardPile.Discard => s.DiscardPile[view.Index],
            CardPile.Exhaust => s.ExhaustPile[view.Index],
            CardPile.Play => c.World.InFlight ?? s.HeldPlayedCard,
            _ => null,
        };
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "MysticLighter")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
