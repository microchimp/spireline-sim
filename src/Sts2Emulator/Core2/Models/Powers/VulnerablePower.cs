// Vulnerable (power:VULNERABLE_POWER), the port list's power-1 row: ModifyDamageMultiplicative. The old body is
// BuffSystem.Vulnerable_ModifyDamageMultiplicative (round R5), called from BuffSystem.IncomingDamage under the condition the
// site keeps. The row's second location (EnemyAI.DealAttackDamage, Colossus's read of the attacking enemy's Vulnerable) is
// Colossus's own rule, ported by ColossusPower; this handler ports the one fold. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class VulnerablePower
{
    /// <summary>
    /// game: VulnerablePower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack on the owner (the old
    /// site: the defender's Vulnerable above 0; the target must be the owner); 1 otherwise. The old function's inputs are
    /// passed the way the two old sites pass them: the attacker's list (Cruelty on it raises the factor) is the enemy's
    /// own on the enemy-attack path (EnemyAI.DealAttackDamage) and the player's on the card path
    /// (CardEffects.DealDamageToEnemy, where the dealer is the player or Osty, whose hits the old engine folds with the
    /// player's list; the game reads the dealer's own powers, and Osty has none in the old state: an R7 check the PR
    /// lists); the Paper Phrog shift, RelicEffects.VulnerableMultiplierDeltaAgainstEnemies, is passed on the card path
    /// alone (the target is an enemy), 0 when the owner is the player. The old function returns a float; the conversion
    /// at the return gives the game's decimal for every factor the old function makes.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m)
            return 1m;
        List<BuffState> attacker = a.Dealer is { } d ? c.Enemy(d) is { } enemy ? enemy.Buffs : c.OldCombat.PlayerBuffs : [];
        float delta = c.Enemy(me.Owner) is not null ? RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(c.OldCombat) : 0f;
        return (decimal)BuffSystem.Vulnerable_ModifyDamageMultiplicative(attacker, c.OldBuffs(me.Owner), delta);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "VulnerablePower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
