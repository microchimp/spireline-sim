// Surrounded (power:SURROUNDED_POWER), the port list's power-3 rows: ModifyDamageMultiplicative and AfterDeath. The old
// bodies are BuffSystem.Surrounded_ModifyDamageMultiplicative (round R5, called from BuffSystem.IncomingDamage, the old
// powered fold) and CombatEngine.TurnPlayerToFaceSurvivor (with its FacePast), called from CombatEngine.AnswerDeaths after a
// Crab Rage, made internal for this handler (batch L3a). Its BeforeCardPlayed and BeforePotionUsed (the turn toward a
// target) are not F08/F10 pairs. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class SurroundedPower
{
    /// <summary>
    /// game: SurroundedPower.ModifyDamageMultiplicative. A fold-mul: 1.5 for a hit on the owner whose dealer is at the
    /// owner's back (the dealer holds Back Attack Left while the owner faces right, or Back Attack Right while it faces
    /// left), 1 otherwise. The old body reads the defender's facing and the attacker's Back Attack, and the old fold calls it
    /// for every powered hit with no other test, so the handler passes the dealer's and the owner's buffs and asks for a
    /// powered attack (the old fold's reach; the game has no props test, so it multiplies an unpowered hit from the back
    /// too). The body reads its owner's facing only, so it acts for its owner alone. The factor is the old function's
    /// float, exact as a decimal.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || a.Dealer is not { } dealer || !a.Props.IsPoweredAttack())
            return 1m;
        return (decimal)BuffSystem.Surrounded_ModifyDamageMultiplicative(c.OldBuffs(dealer), c.OldBuffs(me.Owner));
    }

    /// <summary>
    /// game: SurroundedPower.AfterDeath. An event for the player: on a death that was not prevented, of a creature on the
    /// other side from the owner, when every remaining hittable enemy is on one side (all Back Attack Left or all Back
    /// Attack Right), the owner turns to face the first of them (UpdateDirection). The condition is the game's: the old
    /// site calls the body only after a Crab Rage enraged a partner, which in the one fight with Surrounded (the Kaiser
    /// Crab's two halves) is the death of one half with the other alive, the case the game's test also picks. The old body
    /// reads the player's facing and the living enemies (the game's hittable enemies) and turns the player only, so the
    /// handler acts for a player owner; an enemy holder's handler does nothing. No damage: no re-entry.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || !c.IsPlayer(me) || c.Enemy(a.Creature) is null)
            return HookStep.Done;
        CombatEngine.TurnPlayerToFaceSurvivor(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SurroundedPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative)
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
