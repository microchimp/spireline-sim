// game: ShrinkPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Shrink: its owner's powered attacks deal 30% less; it goes when the creature that applied it dies.</summary>
internal static class ShrinkPower
{
    /// <summary>
    /// game: ShrinkPower.AfterDeath. When the dead creature is the power's Applier and its removal was not prevented:
    /// PowerCmd.Remove of the power. The old buff keeps no applier, so the applier is read the way the old engine reads
    /// it: a Shrink on the player was applied by the Shrinker Beetle (the one monster that applies it,
    /// <c>EnemyAI</c>'s Shrinker Beetle move), and its death removes it through the old body
    /// <c>CardEffects.RemoveDeathBoundPowers</c>. Not modelled: a Shrink on an enemy (the player applied it, by a potion)
    /// goes with the player's death, which ends the combat, and two Shrinker Beetles in one combat, which no encounter
    /// has.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } p || !c.IsPlayer(p.Owner)
            || !c.IsMonster(a.Creature, "ShrinkerBeetle") || c.Enemy(a.Creature) is not { } beetle)
            return HookStep.Done;
        CardEffects.RemoveDeathBoundPowers(c.OldCombat, beetle);
        return HookStep.Done;
    }

    /// <summary>game: ShrinkPower.AfterRemoved (pseudo-hook, from the owner's death removal): the owner's node scaled back
    /// (presentation). No combat state, so the handler changes nothing.</summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    /// <summary>game: ShrinkPower.ModifyDamageMultiplicative. When the owner is the dealer and the props are a powered
    /// attack: (100 - DamageDecrease (30)) / 100; else 1. The old factor is
    /// <c>BuffSystem.Shrink_ModifyDamageMultiplicative</c> (0.70).</summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a,
        decimal current) =>
        c.Power(self) is { } p && a.Dealer == p.Owner && a.Props.IsPoweredAttack()
            ? (decimal)BuffSystem.Shrink_ModifyDamageMultiplicative()
            : 1m;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ShrinkPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved)
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
