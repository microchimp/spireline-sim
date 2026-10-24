// game: MinionPower, ReattachPower, AdaptablePower, SteamEruptionPower and PainfulStabsPower (MegaCrit.Sts2.Core.Models.Powers):
// their ShouldPowerBeRemovedAfterOwnerDeath pseudo-hook (F10; docs/rebuild/MIGRATION_F08.md section 5 "Pseudo-hooks", batch
// L1). Each override answers false, so the power stays on its dead owner (the revives and the fatal checks read it there).
// Batch L2 added the other F08/F10 pairs of Reattach, Adaptable, Steam Eruption and Painful Stabs (Minion has none), each
// calling the power-owned old body the monster batches left (the monsters' own handlers do not call them).
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>game: MinionPower.ShouldPowerBeRemovedAfterOwnerDeath: false.</summary>
internal static class MinionPower
{
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "MinionPower").On(HookId.ShouldPowerBeRemovedAfterOwnerDeath,
            (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath);
}

/// <summary>game: ReattachPower: a Decimillipede segment revives while another segment lives.</summary>
internal static class ReattachPower
{
    /// <summary>game: ReattachPower.ShouldPowerBeRemovedAfterOwnerDeath: false.</summary>
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    /// <summary>game: ReattachPower.AfterDeath. When the dead creature is the owner and its removal was not prevented:
    /// while another segment lives (or the owner is no longer dead) isReviving is set and the segment's dead state made
    /// its move; else every segment fades out (presentation). The old body <c>CombatEngine.TryReattachSegment</c> does the
    /// revive over the old state (the Reviving marker, 2 on the player's turn, the intent, the block cleared, the buffs
    /// the death removal strips) and answers no revive when no other segment lives.</summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || a.Creature != me || c.Enemy(me) is not { } e)
            return HookStep.Done;
        CombatEngine.TryReattachSegment(c.OldCombat, e);
        return HookStep.Done;
    }

    /// <summary>game: ReattachPower.ShouldAllowHitting. Another creature: true; the owner: false while it is reviving
    /// (the old Reviving marker).</summary>
    internal static bool ShouldAllowHitting(HookCtx c, ListenerRef self, in ShouldAllowHittingArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me || BuffSystem.Get(c.OldBuffs(me), BuffId.Reviving) <= 0;

    /// <summary>game: ReattachPower.ShouldCreatureBeRemovedFromCombatAfterDeath. Another creature: true; the owner: false
    /// (the old <c>EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath</c>).</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me
        || (c.Enemy(me) is { } e && EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath(e));

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ReattachPower")
            .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath)
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldAllowHitting, (HookPredicate<ShouldAllowHittingArgs>)ShouldAllowHitting)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath);
}

/// <summary>game: AdaptablePower: the Test Subject comes back after each death, holding the fight open.</summary>
internal static class AdaptablePower
{
    /// <summary>game: AdaptablePower.ShouldPowerBeRemovedAfterOwnerDeath: false.</summary>
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    /// <summary>game: AdaptablePower.AfterDeath. When the dead creature is the owner, a Test Subject, and its removal was
    /// not prevented: isReviving set and TestSubject.TriggerDeadState. The old body is
    /// <c>CombatEngine.TryRespawnTestSubject</c> (the Reviving marker, the intent, the block cleared); the respawn itself
    /// is the Test Subject's move (<c>CombatEngine.RespawnTestSubject</c>, which passes the combat).</summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || a.Creature != me || !c.IsMonster(me, "TestSubject")
            || c.Enemy(me) is not { } e)
            return HookStep.Done;
        CombatEngine.TryRespawnTestSubject(e);
        return HookStep.Done;
    }

    /// <summary>game: AdaptablePower.ShouldAllowHitting. Another creature: true; the owner: not while it is reviving (the
    /// old Reviving marker).</summary>
    internal static bool ShouldAllowHitting(HookCtx c, ListenerRef self, in ShouldAllowHittingArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me || BuffSystem.Get(c.OldBuffs(me), BuffId.Reviving) <= 0;

    /// <summary>game: AdaptablePower.ShouldStopCombatFromEnding: true while the power is on its owner (the old
    /// <c>CombatEngine.Adaptable_ShouldStopCombatFromEnding</c>).</summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Owner(self) is { } me && c.Enemy(me) is { } e && CombatEngine.Adaptable_ShouldStopCombatFromEnding(e);

    /// <summary>game: AdaptablePower.ShouldCreatureBeRemovedFromCombatAfterDeath. Another creature: true; the owner: false
    /// (the old <c>EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath</c>).</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me
        || (c.Enemy(me) is { } e && EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath(e));

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "AdaptablePower")
            .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath)
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldAllowHitting, (HookPredicate<ShouldAllowHittingArgs>)ShouldAllowHitting)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath);
}

/// <summary>game: SteamEruptionPower: the Waterfall Giant stands up to blow after its death, holding the fight open.</summary>
internal static class SteamEruptionPower
{
    /// <summary>game: SteamEruptionPower.ShouldPowerBeRemovedAfterOwnerDeath: false.</summary>
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    /// <summary>game: SteamEruptionPower.AfterDeath. When the dead creature is the owner and its removal was not
    /// prevented: WaterfallGiant.TriggerAboutToBlowState. The old body is <c>EnemyAI.TriggerAboutToBlow</c>, passed the
    /// combat so its HP write is the SetMaxAndCurrentHp command.</summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Owner(self) is not { } me || a.Creature != me || c.Enemy(me) is not { } e)
            return HookStep.Done;
        EnemyAI.TriggerAboutToBlow(e, c.OldCombat);
        return HookStep.Done;
    }

    /// <summary>game: SteamEruptionPower.ShouldStopCombatFromEnding: true while the power is on its owner (the old
    /// <c>CombatEngine.SteamEruption_ShouldStopCombatFromEnding</c>).</summary>
    internal static bool ShouldStopCombatFromEnding(HookCtx c, ListenerRef self, in ShouldStopCombatFromEndingArgs a) =>
        c.Owner(self) is { } me && c.Enemy(me) is { } e && CombatEngine.SteamEruption_ShouldStopCombatFromEnding(e);

    /// <summary>game: SteamEruptionPower.ShouldCreatureBeRemovedFromCombatAfterDeath. Another creature: true; the owner:
    /// false (the old <c>EnemyAI.SteamEruption_ShouldCreatureBeRemovedFromCombatAfterDeath</c>).</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me
        || (c.Enemy(me) is { } e && EnemyAI.SteamEruption_ShouldCreatureBeRemovedFromCombatAfterDeath(e));

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SteamEruptionPower")
            .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath)
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.ShouldStopCombatFromEnding, (HookPredicate<ShouldStopCombatFromEndingArgs>)ShouldStopCombatFromEnding)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath);
}

/// <summary>game: PainfulStabsPower: its owner's corpse stays in the combat.</summary>
internal static class PainfulStabsPower
{
    /// <summary>game: PainfulStabsPower.ShouldPowerBeRemovedAfterOwnerDeath: false.</summary>
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    /// <summary>game: PainfulStabsPower.ShouldCreatureBeRemovedFromCombatAfterDeath: true for any creature but the owner
    /// (the old <c>EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath</c>).</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) =>
        c.Owner(self) is not { } me || a.Creature != me
        || (c.Enemy(me) is { } e && EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath(e));

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PainfulStabsPower")
            .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath);
}
