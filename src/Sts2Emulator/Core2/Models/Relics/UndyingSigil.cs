// Undying Sigil (relic:UNDYING_SIGIL): ModifyDamageMultiplicative. The old body is RelicEffects.IncomingDamageMultiplierFromDoom,
// called from EnemyAI.DealAttackDamage. The port list left it out: no record in the manifest reaches it (round R7f).
// Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class UndyingSigil
{
    /// <summary>
    /// game: UndyingSigil.ModifyDamageMultiplicative. A fold-mul: the old function's factor (one half from an attacker at
    /// or below its own Doom) for a powered attack on the relic's owner, the player, by another creature; 1 otherwise.
    /// The old engine's dealers of an attack on the player are enemies, so the attacker is read as an enemy.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Relic(self) is null || a.Dealer is not { } d || !a.Props.IsPoweredAttack() || a.Target is not { } t
            || !c.IsPlayer(t) || c.Enemy(d) is not { } dealer)
            return 1m;
        return (decimal)RelicEffects.IncomingDamageMultiplierFromDoom(c.OldCombat, dealer);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "UndyingSigil")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
