// Colossus (power:COLOSSUS_POWER), the port list's power-1 row: ModifyDamageMultiplicative. The old body is
// EnemyAI.Colossus_ModifyDamageMultiplicative (round R5), called from EnemyAI.DealAttackDamage under the condition the site
// keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ColossusPower
{
    /// <summary>
    /// game: ColossusPower.ModifyDamageMultiplicative. A fold-mul: the factor for a powered attack on the owner by a dealer
    /// that has Vulnerable (the old site, EnemyAI.DealAttackDamage: the player's Colossus above 0 and the attacking
    /// enemy's Vulnerable above 0); 1 otherwise. The old engine holds Colossus on the player alone, so its dealer is
    /// always an enemy and the handler reads an enemy dealer's list as the site does; the game asks the dealer's
    /// Vulnerable whoever deals (a player-side dealer against a Colossus holder never arises in the old engine).
    /// The old function halves the integer damage in place (int in, int out, after the old fold's cast), so it cannot
    /// be the fold's factor: the factor is the game's one half as a decimal, and the one cast in CreatureCmd.Damage gives
    /// the old function's result for every integer damage (the floor of half the floor is the floor of the half), which
    /// PowerOneTests pins against the old function and against the old enemy-attack path.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack() || me.Amount <= 0m
            || a.Dealer is not { } d || c.Enemy(d) is not { } dealer || BuffSystem.Get(dealer.Buffs, BuffId.Vulnerable) <= 0)
            return 1m;
        return Half;
    }

    /// <summary>The game's factor; the old function's integer halving (EnemyAI.Colossus_ModifyDamageMultiplicative) is
    /// this factor truncated once, which the damage command does.</summary>
    private const decimal Half = 0.5m;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ColossusPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
