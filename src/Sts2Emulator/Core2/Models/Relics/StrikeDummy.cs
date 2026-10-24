// Strike Dummy (relic:STRIKE_DUMMY), the port list's relic-1 row: ModifyDamageAdditive. The old body is
// RelicEffects.StrikeDummy_ModifyDamageAdditive (round R5), called from RelicEffects.CardDamageBonus under the condition
// the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class StrikeDummy
{
    /// <summary>
    /// game: StrikeDummy.ModifyDamageAdditive. A fold-add: the addend for a powered attack (ValueProp.IsPoweredAttack)
    /// whose card source is an Attack carrying the Strike tag, when the dealer is the owner's creature or the card is the
    /// owner's (the game answers 0 when both fail: a dealer that is not the owner and a card of another owner); 0
    /// otherwise. The old site, RelicEffects.CardDamageBonus, is the card-damage computation of CardEffects.DmgFrom: it
    /// answers 0 for a def that is no Attack, then asks the def's StrikeTag and that the relic is held, and reads neither
    /// the dealer nor the props, as the computation is only reached for the player's card attacks; the handler follows the
    /// game, so an unpowered hit with a card source (an upgraded Attack's self-damage) gets no bonus here. The old state
    /// holds one card owner, the player (<see cref="HookCtx.CardOwner" />), so the card test is met by every live card and
    /// the dealer test cannot refuse a card hit. The number is the old function's flat 3.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || c.Owner(self) is not { } owner || !a.Props.IsPoweredAttack() || a.CardSource is not { } source
            || c.Card(source) is not { } card || GeneratedData.Cards.Get(card.ModelId) is not { Type: CardType.Attack, StrikeTag: true }
            || (a.Dealer != owner && c.CardOwner(source) != owner))
            return 0m;
        return RelicEffects.StrikeDummy_ModifyDamageAdditive();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "StrikeDummy")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
