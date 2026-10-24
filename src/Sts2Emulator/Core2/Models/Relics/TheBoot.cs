// The Boot (relic:THE_BOOT), the port list's relic-2 rows: ModifyHpLostAfterOstyLate and AfterModifyingHpLostAfterOsty.
// The old body is RelicEffects.BootDamageFloor (the floor's number), called from CardEffects.LandDamageOnEnemy under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

internal static class TheBoot
{
    /// <summary>
    /// game: TheBoot.ModifyHpLostAfterOstyLate. A fold-set over the HP lost after the pet redirect: the old function's
    /// floor for a powered attack by the player's side (the player or their pet: a dealer that is no enemy) on an enemy
    /// (the old site, CardEffects.LandDamageOnEnemy, is the player's side landing a hit on an enemy, and passes the floor
    /// only when the hit is powered); the value unchanged otherwise. The old function takes and returns an int, the HP
    /// lost after the old engine's cast; here the running value is a decimal, cast once at the HP loss, so the floor is
    /// decided on the truncated value (1 to 4 lost becomes 5) and a value the floor does not raise is returned as it was,
    /// decimals kept.
    /// </summary>
    internal static decimal ModifyHpLostAfterOstyLate(HookCtx c, ListenerRef self, in ModifyHpLostArgs a, decimal current)
    {
        if (c.Relic(self) is null || !a.Props.IsPoweredAttack() || a.Dealer is not { } dealer || c.Enemy(dealer) is not null
            || c.Enemy(a.Target) is null)
            return current;
        int lost = (int)Math.Min(current, int.MaxValue);
        int floored = RelicEffects.BootDamageFloor(c.OldCombat, lost);
        return floored == lost ? current : floored;
    }

    /// <summary>
    /// game: TheBoot.AfterModifyingHpLostAfterOsty. The dispatcher's filter-set pass reaches it only when this relic is
    /// among the fold's modifiers (the floor raised the loss). The old engine has no body at this point: the location
    /// the port list names for the pair, RelicEffects.BootDamageFloor, is the number the fold above applies. Nothing to
    /// do here.
    /// </summary>
    internal static HookStep AfterModifyingHpLostAfterOsty(HookCtx c, ListenerRef self, in AfterModifyingHpLostAfterOstyArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    public static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "TheBoot")
            .On(HookId.ModifyHpLostAfterOstyLate, (HookFold<ModifyHpLostArgs, decimal>)ModifyHpLostAfterOstyLate)
            .On(HookId.AfterModifyingHpLostAfterOsty, (HookEvent<AfterModifyingHpLostAfterOstyArgs>)AfterModifyingHpLostAfterOsty);
}
