// Diamond Diadem (power:DIAMOND_DIADEM_POWER): ModifyDamageMultiplicative. The old body is the halving in
// EnemyAI.DealAttackDamage (the power applied by the DiamondDiadem relic's turn end). The port list left it out: the
// manifest counts no record reaching the power itself (round R7f). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class DiamondDiademPower
{
    /// <summary>
    /// game: DiamondDiademPower.ModifyDamageMultiplicative. A fold-mul: one half for a powered attack on the owner; 1
    /// otherwise. The old site, EnemyAI.DealAttackDamage, halves the integer damage of an enemy's attack on the player while
    /// the power is above 0 (its path is the powered-attack path); the one cast in CreatureCmd.Damage gives its result
    /// (the floor of half the floor is the floor of the half), as for Colossus.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        return 0.5m;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DiamondDiademPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
