// Mystic Lighter (relic:MYSTIC_LIGHTER), the port list's relic-1 row: ModifyDamageAdditive. The old body is
// RelicEffects.EnchantedCardDamageBonus, called from CardEffects.DmgFrom under the condition the site keeps. Game code is
// cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class MysticLighter
{
    /// <summary>
    /// game: MysticLighter.ModifyDamageAdditive. A fold-add: the addend for damage whose card source is an Attack carrying
    /// an enchantment, any enchantment (the old site, CardEffects.DmgFrom: the copy's Enchantment is not None and the def
    /// is an Attack; the old function asks that the relic is held; neither the dealer nor the props are read, as the
    /// computation is only reached for the player's card attacks); 0 otherwise. The enchantment's own bonus is not this
    /// one: the dispatcher's enchant-add step (Steps.EnchantAdd, before the additive pass) adds the card's
    /// EnchantmentModel.EnchantDamageAdditive through CardEffects.EnchantDamageAdditive, which reads the enchantment alone
    /// (Sharp, Vigorous, TezcatarasEmber, Momentum, Inky) and never this relic, as the old DmgFrom adds the two apart; so
    /// the relic's bonus lands once, here. The copy's enchantment is read off the live card the handle names
    /// (<see cref="Copy" />). The number is the old function's flat 9.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || Copy(c, a.CardSource) is not { Enchantment: not Enchantment.None } card
            || GeneratedData.Cards.Get(card.DefId).Type != CardType.Attack)
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
