// Pen Nib (relic:PEN_NIB), the port list's relic-1 row: ModifyDamageMultiplicative. The old body is
// RelicEffects.PenNib_ModifyDamageMultiplicative (round R5), called from RelicEffects.CardDamageMultiplier under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class PenNib
{
    /// <summary>
    /// game: PenNib.ModifyDamageMultiplicative. A fold-mul: the factor for damage whose card source is an Attack while the
    /// relic is armed (the old site, RelicEffects.CardDamageMultiplier: the def is an Attack, the relic's counter is 0 and
    /// the combat's PenNibArmed flag is up; RelicEffects.BeforeCardPlayedRelics counts the owner's Attacks on the relic's
    /// counter, wraps it at ten and arms the flag on the wrap, so the tenth Attack is the one doubled, and a card that is
    /// no Attack lowers the flag); 1 otherwise. The counter is this listener's own (the RelicView's); the old site reads
    /// the first Pen Nib held, the same relic while one is held. The number is the old function's flat 2.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is not { Counter: 0 } || !c.OldCombat.PenNibArmed || a.CardSource is not { } source
            || c.Card(source) is not { } card || GeneratedData.Cards.Get(card.ModelId).Type != CardType.Attack)
            return 1m;
        return RelicEffects.PenNib_ModifyDamageMultiplicative();
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "PenNib")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
