// Constrict (power:CONSTRICT_POWER), the port list's power-4 row: AfterDeath. The old body is
// CombatEngine.Constrict_AfterDeath (round R5), called from CombatEngine.AnswerDeaths' real-death loop under the condition
// the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class ConstrictPower
{
    /// <summary>
    /// game: ConstrictPower.AfterDeath. An event: the old body, which removes the player's Constrict, is called on a death
    /// that was not prevented of the power's Applier (the game's test, R7b2: <c>EnemyState.AppliedPlayersConstrict</c>,
    /// which the Strangler's Constrict move records under the switch when it creates the player's power), while the
    /// handler's owner is on the player's side (the old function acts on the player's buffs, and the old state's one power
    /// holder on that side is the player; a holder on an enemy is not what the old function removes, so its handler does
    /// nothing). The old site's test, any Slithering Strangler by def id, applies only when no enemy is recorded as the
    /// Applier (a state built without the record: a snapshot, a test). The debuff is a survivor's, not the fallen's.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || c.Enemy(me.Owner) is not null
            || c.Enemy(a.Creature) is not { } fallen || !IsApplier(c.OldCombat, fallen))
            return HookStep.Done;
        CombatEngine.Constrict_AfterDeath(c.OldCombat);
        return HookStep.Done;
    }

    /// <summary>Whether <paramref name="fallen" /> is the Applier of the player's Constrict: the recorded one, or with none
    /// recorded, the old site's test (a Slithering Strangler).</summary>
    private static bool IsApplier(CombatState s, EnemyState fallen) =>
        s.Enemies.Exists(e => e.AppliedPlayersConstrict)
            ? fallen.AppliedPlayersConstrict
            : fallen.DefId == KE.SlitheringStrangler;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ConstrictPower").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
