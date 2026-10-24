// Fake Strike Dummy (relic:FAKE_STRIKE_DUMMY), the port list's relic-1 row: ModifyDamageAdditive. The old body is
// RelicEffects.FakeStrikeDummy_ModifyDamageAdditive (round R5), called from RelicEffects.CardDamageBonus under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class FakeStrikeDummy
{
    /// <summary>
    /// game: FakeStrikeDummy.ModifyDamageAdditive. A fold-add: the addend for a powered attack (ValueProp.IsPoweredAttack)
    /// whose card source is an Attack carrying the Strike tag, when the dealer is the owner's creature or the card is the
    /// owner's (the same rule as Strike Dummy's, at the other number); 0 otherwise. The old site,
    /// RelicEffects.CardDamageBonus, is 0 for a def that is no Attack, then the def's StrikeTag and that the relic is held,
    /// and reads neither the dealer nor the props, as the computation is only reached for the player's card attacks; the
    /// handler follows the game (see StrikeDummy.ModifyDamageAdditive). The number is the old function's flat 1.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || c.Owner(self) is not { } owner || !a.Props.IsPoweredAttack() || a.CardSource is not { } source
            || c.Card(source) is not { } card || GeneratedData.Cards.Get(card.ModelId) is not { Type: CardType.Attack, StrikeTag: true }
            || (a.Dealer != owner && c.CardOwner(source) != owner))
            return 0m;
        return RelicEffects.FakeStrikeDummy_ModifyDamageAdditive();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "FakeStrikeDummy")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
