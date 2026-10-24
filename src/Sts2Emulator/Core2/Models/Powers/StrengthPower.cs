// Strength (power:STRENGTH_POWER), the port list's power-1 row: ModifyDamageAdditive. The old body is
// BuffSystem.Strength_ModifyDamageAdditive (round R5), called from BuffSystem.IncomingDamage with the attacker's buff list
// and no condition at the site (the fold adds the number as it is). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class StrengthPower
{
    /// <summary>
    /// game: StrengthPower.ModifyDamageAdditive. A fold-add: the addend for a powered attack dealt by the owner itself,
    /// the old function's number read off the owner's buff list; 0 otherwise. The dealer must be the owner: no pet case
    /// (the old engine's note at CardEffects.OstyAttackDamage records that the game makes Osty the attacker of its own
    /// hits, which the player's Strength does not raise). The old site has no condition of its own, so a negative
    /// Strength subtracts here as there. The old engine's Osty path (CardEffects.DealOstyDamage into DealDamageToEnemy)
    /// folds Osty's hit with the player's list and so adds the player's Strength to it; the game and this handler do
    /// not: an R7 check the PR lists.
    /// </summary>
    internal static decimal ModifyDamageAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack())
            return 0m;
        return BuffSystem.Strength_ModifyDamageAdditive(c.OldBuffs(me.Owner));
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "StrengthPower")
            .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageAdditive);
}
