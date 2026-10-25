// game: IllusionPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Illusion: its owner revives after death, keeping its buffs, and cannot be hit while reviving.</summary>
internal static class IllusionPower
{
    /// <summary>game: IllusionPower.AfterDeath. When the dead creature is the owner and its removal was not prevented:
    /// isReviving set and the REVIVE_MOVE made the owner's move at once. The old body <c>CombatEngine.TryReviveIllusion</c>
    /// does it over the old state (the Reviving marker, the announced intent, the block cleared) and also strips the
    /// debuffs the death removal below would (IllusionDeathStrips). The stun animation is presentation.</summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || a.Creature != me || c.Enemy(me) is not { } e)
            return HookStep.Done;
        CombatEngine.TryReviveIllusion(e);
        return HookStep.Done;
    }

    /// <summary>game: IllusionPower.ShouldAllowHitting. Another creature: true; the owner: false while it is reviving
    /// (isReviving is the old Reviving marker, as <c>EnemyAI.IsTargetable</c> reads it).</summary>
    internal static bool ShouldAllowHitting(HookCtx c, ListenerRef self, in ShouldAllowHittingArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me || BuffSystem.Get(c.OldBuffs(me), BuffId.Reviving) <= 0;

    /// <summary>game: IllusionPower.ShouldCreatureBeRemovedFromCombatAfterDeath. Another creature: true; the owner: false
    /// (the old <c>EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath</c>, false while it holds the power).</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me
        || (c.Enemy(me) is { } e && EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath(e));

    /// <summary>game: IllusionPower.ShouldPowerBeRemovedOnDeath. A debuff that is no ITemporaryPower: true (removed); any
    /// other power: false (kept), whoever holds it, as the game asks every listener. The temporary powers are the old
    /// folded TEMPORARY_* buffs, which the death removal does not list (Core2's StripCorpse takes them off an Illusion's
    /// corpse by the old IllusionDeathStrips), so none reaches this answer.</summary>
    internal static bool ShouldPowerBeRemovedOnDeath(HookCtx c, ListenerRef self, in ShouldPowerBeRemovedOnDeathArgs a) =>
        c.IsDebuff(a.Power);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "IllusionPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldAllowHitting, (HookPredicate<ShouldAllowHittingArgs>)ShouldAllowHitting)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath)
            .On(HookId.ShouldPowerBeRemovedOnDeath,
                (HookPredicate<ShouldPowerBeRemovedOnDeathArgs>)ShouldPowerBeRemovedOnDeath);
}
