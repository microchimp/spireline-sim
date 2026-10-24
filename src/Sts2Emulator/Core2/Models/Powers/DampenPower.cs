// game: DampenPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Dampen: its owner's upgraded cards are downgraded while a caster lives.</summary>
internal static class DampenPower
{
    /// <summary>game: DampenPower.AfterDeath. When the removal was not prevented and the dead creature is one of the
    /// casters that applied it: that caster is dropped, and with none left the power is removed (its AfterRemoved gives
    /// the upgrades back). The old buff keeps no caster set; the old engine reads the casters as the living Magi Knights
    /// (the one source): the last one's death runs the old body <c>CombatEngine.Dampen_AfterDeath</c> (the removal and the
    /// upgrades back).</summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } p || !c.IsPlayer(p.Owner)
            || !c.IsMonster(a.Creature, "MagiKnight") || c.Enemy(a.Creature) is not { } fallen
            || c.OldCombat.Enemies.Any(o => o.DefId == fallen.DefId && o.Hp > 0 && !ReferenceEquals(o, fallen)))
            return HookStep.Done;
        CombatEngine.Dampen_AfterDeath(c.OldCombat);
        return HookStep.Done;
    }

    /// <summary>game: DampenPower.AfterRemoved (pseudo-hook, from the owner's death removal): every card it downgraded is
    /// upgraded back. The old engine marks those cards (<c>CardInstance.DampenedUpgrade</c>); the old body
    /// <c>CardEffects.RemoveDampen</c> gives the upgrades back (its own buff removal finds the buff already gone).</summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx)
    {
        CardEffects.RemoveDampen(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DampenPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
