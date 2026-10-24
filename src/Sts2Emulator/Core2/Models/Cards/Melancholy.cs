// Melancholy (card:MELANCHOLY), the port list's card-1 row: AfterDeath. The old body is
// CardEffects.CheapenMelancholyForDeath, called from CombatEngine.HandleEnemyDeaths' real-death half; it cheapens every
// copy in every combat pile in one call, while the game runs the hook once per copy (MIGRATION_F08.md section 5, "One
// hook call per copy"), so the handler applies the old loop's per-copy statement to its own copy only. Game code is
// cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Cards;

internal static class Melancholy
{
    /// <summary>
    /// game: Melancholy.AfterDeath. An event: on a death that was not prevented (the old site calls the body from the
    /// half of CombatEngine.HandleEnemyDeaths that runs for a real death), this copy's cost bump goes down by one where it
    /// lies in one of the combat's four lists (Hand, Draw, Discard, Exhaust: the piles CardEffects.CheapenMelancholyForDeath
    /// walks). The old function cannot be called from here: it cheapens every copy at once, and N copies' handlers
    /// would cheapen each N times; the per-copy statement is the old loop's body, applied to the copy this listener runs
    /// for (<c>c.Card(self)</c>), so the dispatcher's one call per copy equals the old function's one call.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Card(self) is not { } me)
            return HookStep.Done;
        var s = c.OldCombat;
        if (!Cheapen(s.Hand, me.Handle.Uid) && !Cheapen(s.DrawPile, me.Handle.Uid) && !Cheapen(s.DiscardPile, me.Handle.Uid))
            Cheapen(s.ExhaustPile, me.Handle.Uid);
        return HookStep.Done;
    }

    /// <summary>The old loop's statement for one copy: found by uid in <paramref name="pile" />, its CostBump one lower.
    /// Returns whether the copy was there.</summary>
    private static bool Cheapen(List<CardInstance> pile, int uid)
    {
        for (int i = 0; i < pile.Count; i++)
        {
            if (pile[i].Uid == uid)
            {
                pile[i] = pile[i] with { CostBump = pile[i].CostBump - 1 };
                return true;
            }
        }

        return false;
    }

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Card, "Melancholy").On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
