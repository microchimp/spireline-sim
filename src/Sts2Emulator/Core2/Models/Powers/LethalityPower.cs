// Lethality (power:LETHALITY_POWER), the port list's power-5 row: ModifyDamageMultiplicative. The old body is
// CardEffects.Lethality_ModifyDamageMultiplicative (round R5), called from CardEffects.DealDamageToEnemy under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class LethalityPower
{
    /// <summary>
    /// game: LethalityPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack whose card source is a
    /// card of the power's owner, while the combat's <c>LethalAttackPlay</c> flag is set (the old site,
    /// CardEffects.DealDamageToEnemy: the power's amount is above 0 and the flag the play set holds); 1 otherwise. The game
    /// asks for a card source and for that card's owner to be the power's owner; it never looks at the dealer, so a hit by
    /// the owner's pet from the owner's card counts. The number is the old function's.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || !a.Props.IsPoweredAttack() || a.CardSource is not { } source
            || c.CardOwner(source) != me.Owner || me.Amount <= 0m || !c.OldCombat.LethalAttackPlay)
            return 1m;
        // The old function returns the float 1 + amount/100; the game's factor is the decimal 1 + amount/100. For an
        // integer amount the float-to-decimal conversion (seven significant digits) gives that decimal exactly (pinned for
        // amounts 1 to 500 by PilotTests).
        return (decimal)CardEffects.Lethality_ModifyDamageMultiplicative((int)me.Amount);
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "LethalityPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
