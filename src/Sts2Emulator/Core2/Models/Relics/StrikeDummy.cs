// Strike Dummy (relic:STRIKE_DUMMY), the port list's relic-1 row: ModifyDamageAdditive. The old body is
// RelicEffects.StrikeDummy_ModifyDamageAdditive (round R5), called from RelicEffects.CardDamageBonus under the condition
// the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class StrikeDummy
{
    /// <summary>
    /// game: StrikeDummy.ModifyDamageAdditive. A fold-add: the addend for damage whose card source is an Attack carrying
    /// the Strike tag (the old site, RelicEffects.CardDamageBonus, is the card-damage computation of CardEffects.DmgFrom:
    /// it answers 0 for a def that is no Attack, then asks the def's StrikeTag and that the relic is held; it reads
    /// neither the dealer nor the props, as the computation is only reached for the player's card attacks); 0 otherwise.
    /// The number is the old function's flat 3.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || a.CardSource is not { } source || c.Card(source) is not { } card
            || GeneratedData.Cards.Get(card.ModelId) is not { Type: CardType.Attack, StrikeTag: true })
            return 0m;
        return RelicEffects.StrikeDummy_ModifyDamageAdditive();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "StrikeDummy")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
