// StockPower (power:STOCK_POWER), batch L3b (docs/rebuild/MIGRATION_F08.md): AfterDeath and ShouldStopCombatFromEnding.
// The game's AfterDeath adds a NEW Axebot in the dead one's slot (CreatureCmd.Add) and the dead one leaves; the old
// CombatEngine.TryRespawnAxebot revived the same creature in place, which Core2's Kill would take out of the roster, so the
// handler builds the new bot with the old respawn's parts and lists it with Core2's CreatureCmd.Add. Game code is cited by
// class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>game: StockPower: an Axebot comes back as a new Axebot while stock is left, holding the fight open.</summary>
internal static class StockPower
{
    /// <summary>
    /// game: StockPower.AfterDeath. When the dead creature is the owner, its removal was not prevented and the amount is
    /// above 0: a new Axebot with StockAmount = Amount - 1 is added on the owner's side (CreatureCmd.Add; its
    /// AfterAddedToRoom applies that stock). The old site's condition (CombatEngine.TryRespawnAxebot: an Axebot holding
    /// Stock) is kept; its parts build the bot: the HP draw on the stream the old site passes
    /// (<c>CombatEngine.RollAxebotRespawnHp</c> on <see cref="HookCtx.Niche" />) and the state the old respawn leaves
    /// (<c>CombatEngine.NewRespawnedAxebot</c>). The dead bot leaves the roster after this, through the Kill's removal, so
    /// the new one stands at the roster's end as in the game (the old engine kept the revived bot at its index).
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } me || a.Creature != me.Owner || me.Amount <= 0
            || !c.IsMonster(me.Owner, "Axebot"))
            return HookStep.Done;
        int ascension = c.OldCombat.AscensionLevel;
        int hp = CombatEngine.RollAxebotRespawnHp(c.Niche, ascension);
        CreatureCmd.Add(c.K, CombatEngine.NewRespawnedAxebot((int)me.Amount, hp, ascension));
        return HookStep.Done;
    }

    /// <summary>game: StockPower.ShouldStopCombatFromEnding: true while the power is on its owner (the old
    /// <c>CombatEngine.Stock_ShouldStopCombatFromEnding</c>).</summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Owner(self) is { } me && c.Enemy(me) is { } e && CombatEngine.Stock_ShouldStopCombatFromEnding(e);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "StockPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding);
}
