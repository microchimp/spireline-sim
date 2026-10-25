// Gremlin Horn (relic:GREMLIN_HORN), the port list's relic-1 row: AfterDeath. The old body is
// CombatEngine.GremlinHorn_AfterDeath (round R5), called from CombatEngine.AnswerDeaths inside its loop over the fallen,
// under the condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class GremlinHorn
{
    /// <summary>
    /// game: GremlinHorn.AfterDeath. An event: the old body (one energy and one card for the player) runs for a death on
    /// the other side from the relic's owner. The old site, CombatEngine.AnswerDeaths, runs it once per fallen enemy
    /// while the relic is held, and reads neither the prevented flag (the game tests the side alone, so a death something
    /// else undoes still pays) nor the corpse. The dispatcher runs this once per death, with the dead creature as the
    /// argument, so the per-owner condition is: the relic is held and the dead creature is an enemy, the other side from
    /// the player, the one holder (the player's or Osty's death is on the owner's side and pays nothing). The old body
    /// takes a Random for the draw's reshuffle; CardEffects.ShuffleIfNecessary prefers the combat's Shuffle stream to it,
    /// and the kernel hands a handler no Random, so the stream is passed, with the old engine's own stand-in for a
    /// hand-made combat that has none (CardEffects.PersonalHive_AfterDamageReceived does the same). The draw can deal
    /// damage through a draw hook of the old engine (DamageAllForSpeedster): a re-entry the PR lists.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (c.Relic(self) is null || c.Enemy(a.Creature) is null)
            return HookStep.Done;
        var s = c.OldCombat;
        CombatEngine.GremlinHorn_AfterDeath(s, s.ShuffleRng ?? new Random(0));
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "GremlinHorn").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
