// Tungsten Rod (relic:TUNGSTEN_ROD), the port list's relic-1 rows: ModifyHpLostAfterOsty and AfterModifyingHpLostAfterOsty.
// The old body is RelicEffects.TungstenRod_ModifyHpLostAfterOsty (round R5), called from RelicEffects.ModifyHpLost out
// of CardEffects.LosePlayerHp, the player's half of the damage command, under the condition the site keeps. The row's
// second location, RunNonCombatEffects.LoseHp, is the run-level HP loss outside the combat code, which R5 left: this
// handler covers the combat location only. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class TungstenRod
{
    /// <summary>
    /// game: TungstenRod.ModifyHpLostAfterOsty. A fold-set over the HP lost after the pet redirect: one less, floored at
    /// zero, for a loss of the relic's owner (the old site, CardEffects.LosePlayerHp, is the player's HP loss whatever its
    /// props, dealer or card: an attack, a card's own cost, a Burn, Poison and Thorns all come through it, and the relic
    /// must be held); the value unchanged for a loss of any other creature (an enemy, Osty). The old function takes and
    /// returns an int, the loss after the old engine's cast; here the running value is a decimal, cast once at the HP loss
    /// (CreatureCmd.Damage), so the handler moves it by the old function's difference, as the game takes one off the
    /// decimal (3.5 lost becomes 2.5, truncated to 2 at the loss), and a loss under 1, which the old engine's cast made 0
    /// before the function saw it, is floored to 0 as the game floors it.
    /// </summary>
    internal static decimal ModifyHpLostAfterOsty(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Relic(self) is null || c.Owner(self) is not { } owner || a.Target != owner)
            return current;
        int lost = (int)Math.Min(current, int.MaxValue);
        int reduced = RelicEffects.TungstenRod_ModifyHpLostAfterOsty(lost);
        return lost <= 0 ? 0m : current - (lost - reduced);
    }

    /// <summary>
    /// game: TungstenRod.AfterModifyingHpLostAfterOsty. The dispatcher's filter-set pass reaches it only when this relic
    /// is among the fold's modifiers (the reduction moved the truncated loss). The old engine has no body at this point:
    /// the location the port list names for the pair, RelicEffects.ModifyHpLost, is the fold above. Nothing to do here.
    /// </summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self, in AfterModifyingHpLostAfterOstyArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "TungstenRod")
            .On(HookId.ModifyHpLostAfterOsty, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOsty)
            .On(HookId.AfterModifyingHpLostAfterOsty, (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
