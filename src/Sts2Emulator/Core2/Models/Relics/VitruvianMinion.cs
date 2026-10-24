// Vitruvian Minion (relic:VITRUVIAN_MINION), the port list's relic-2 row: ModifyDamageMultiplicative. The old body is
// RelicEffects.VitruvianMinion_ModifyDamageMultiplicative (round R5), called from RelicEffects.CardDamageMultiplier
// under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class VitruvianMinion
{
    /// <summary>
    /// game: VitruvianMinion.ModifyDamageMultiplicative. A fold-mul: the factor for damage whose card source carries the
    /// Minion tag, whatever the card's type (the old site, RelicEffects.CardDamageMultiplier, reads the tag before its
    /// Attack gate and asks that the relic is held); 1 otherwise. The number is the old function's flat 2.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || a.CardSource is not { } source || c.Card(source) is not { } card
            || !GeneratedData.Cards.Get(card.ModelId).MinionTag)
            return 1m;
        return RelicEffects.VitruvianMinion_ModifyDamageMultiplicative();
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "VitruvianMinion")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
