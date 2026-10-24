// Conqueror (power:CONQUEROR_POWER), a LOCAL model of round R4c: ModifyDamageMultiplicative. The old body is
// CardEffects.ConqueredDamage, which the old Sovereign Blade body applied to each enemy it hit (the F08 path sends the
// card's one AttackCmd at the base amount, so the doubling is this handler's). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ConquerorPower
{
    /// <summary>
    /// game: ConquerorPower.ModifyDamageMultiplicative. A fold-mul: the factor when the card source is Sovereign Blade,
    /// the damage is a powered attack and the target is the power's owner (the old site's condition: the Sovereign Blade
    /// body, each enemy it hits); 1 otherwise. The number is the old function's (ConqueredDamage of one).
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (a.CardSource is not { } source || c.Card(source) is not { } card
            || GeneratedData.Cards.Get(card.ModelId).Name != "SovereignBlade")
            return 1m;
        if (!a.Props.IsPoweredAttack())
            return 1m;
        if (c.Power(self) is not { } me || a.Target != me.Owner || c.Enemy(me.Owner) is not { } owner)
            return 1m;
        return CardEffects.ConqueredDamage(owner, 1);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ConquerorPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
