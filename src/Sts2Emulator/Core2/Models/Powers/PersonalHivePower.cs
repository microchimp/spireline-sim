// Personal Hive (power:PERSONAL_HIVE_POWER), the port list's power-3 row: AfterDamageReceived. The old body is
// CardEffects.PersonalHive_AfterDamageReceived (round R5), called from CardEffects.LandDamageOnEnemy under the
// condition the site keeps. Game code is cited by class and method only.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class PersonalHivePower
{
    /// <summary>
    /// game: PersonalHivePower.AfterDamageReceived. An event: the old body, which shuffles the owner's amount of Dazed
    /// into the player's draw pile (its places drawn from the combat's shuffle stream), is called for a powered attack on
    /// the owner with a dealer (the game's <c>dealer != null</c> test, R7b2; the old site, CardEffects.LandDamageOnEnemy,
    /// has none) that left the owner alive (the old site's HP test, kept: the target's HP above 0; no damage result is
    /// read, so a fully blocked hit counts). The game gives the Dazed to the dealer's player, Osty's owner when Osty
    /// dealt: the old state's one player. The old body takes the combat and the amount; the site is the enemy side's
    /// landing of a hit, so the owner is an enemy, as in the old engine.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || a.Dealer is null || !a.Props.IsPoweredAttack()
            || me.Amount <= 0m
            || c.Enemy(me.Owner) is not { Hp: > 0 })
            return HookStep.Done;
        CardEffects.PersonalHive_AfterDamageReceived(c.OldCombat, (int)me.Amount);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PersonalHivePower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
