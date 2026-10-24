// Heist (power:HEIST_POWER), the port list's power-4 row: BeforeDeath. The old body is CombatEngine.Heist_BeforeDeath
// (round R5), called from CombatEngine.AnswerDeaths under the condition the site keeps. The row's second location,
// CombatEngine.SpawnGremlinMercReinforcements, is where the power is applied to the Fat Gremlin the Merc's death sends in
// (the buff and its gold), not a body of the hook: the handler calls Heist_BeforeDeath alone. Game code is cited by class
// and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class HeistPower
{
    /// <summary>
    /// game: HeistPower.BeforeDeath. An event: the old body, which banks the fallen's stolen gold for the reward
    /// (StolenBackGold), zeroes it and removes the power, is called when the dying creature is the handler's owner (the
    /// power sits on the thief) and the old site's condition holds: the fallen is a Fat Gremlin that has not escaped and
    /// holds stolen gold (the site's branch is the else of Infested's, which a Fat Gremlin never carries, so that exclusion
    /// is not repeated). The old site answers the death after it; the hook is the game's BeforeDeath, so the handler reads
    /// the fallen at 0 HP, still listed, before ShouldDie.
    /// </summary>
    internal static HookStep BeforeDeath(HookCtx c, ListenerRef self, in BeforeDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Creature != me.Owner || c.Enemy(a.Creature) is not { } fallen
            || fallen.DefId != KE.FatGremlin || fallen.Escaped || fallen.HeistGold <= 0)
            return HookStep.Done;
        CombatEngine.Heist_BeforeDeath(c.OldCombat, fallen);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "HeistPower").On(HookId.BeforeDeath, (HookEvent<BeforeDeathArgs>)BeforeDeath);
}
