// HangPower (power:HANG_POWER), batch L3b (docs/rebuild/MIGRATION_F08.md): ModifyDamageMultiplicative. The old Hang card
// body (CardEffects.ApplyNecrobinderCard) multiplied its base by the stacks before the damage call; under the F08+F10
// switch that multiply is gated off and the factor is this fold's, so it is applied once. Game code is cited by class and
// method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class HangPower
{
    /// <summary>
    /// game: HangPower.ModifyDamageMultiplicative. A fold-mul: the power's amount when the target is its owner and the card
    /// source is Hang (any props: the game tests neither the dealer nor the attack kind); 1 otherwise. The old site read
    /// the stacks of the enemy Hang hits (its first enemy) and multiplied by at least 1, which equals the amount of a holder.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner)
            return 1m;
        if (a.CardSource is not { } source || c.Card(source) is not { } card
            || GeneratedData.Cards.Get(card.ModelId).Name != "Hang")
            return 1m;
        return me.Amount;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "HangPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
