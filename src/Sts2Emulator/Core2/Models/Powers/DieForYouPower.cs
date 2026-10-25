// DieForYouPower (power:DIE_FOR_YOU_POWER), batch R7a (docs/rebuild/MIGRATION_F08.md, R7 checks): ModifyUnblockedDamageTarget,
// the pet redirect. The game applies the power to Osty once, when OstyCmd.Summon first adds the pet, and never removes it
// (ShouldPowerBeRemovedAfterOwnerDeath): Summon's revive branch re-adds the same creature, which kept it, so a revived Osty
// holds it too. An Osty the old state holds (OstyMaxHp above 0, dead or revived) holds it, and OldState lists it on Osty. The old enemy-attack site (EnemyAI's player hit) moved the unblocked rest of a powered hit onto a living Osty
// itself; under the F08 switch that hit is AttackCmd's CreatureCmd.Damage, whose ModifyUnblockedDamageTarget fold this
// handler answers, and Damage deals the overkill back to the player. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>game: DieForYouPower: Osty takes the unblocked damage of a powered attack on its owner while it lives.</summary>
internal static class DieForYouPower
{
    /// <summary>
    /// game: DieForYouPower.ModifyUnblockedDamageTarget. A fold-set over the hit's target: the owner (Osty) when the target so
    /// far is the owner's pet owner, the owner is alive and the hit is a powered attack; the target unchanged otherwise. The
    /// dealer is not read.
    /// </summary>
    internal static CreatureRef ModifyUnblockedDamageTarget(HookCtx c, ListenerRef self, in ModifyUnblockedDamageTargetArgs a,
        CreatureRef current)
    {
        if (c.Owner(self) is not { } me || c.PetOwner(me) is not { } owner || current != owner)
            return current;
        if (!c.IsAlive(me) || !a.Props.IsPoweredAttack())
            return current;
        return me;
    }

    /// <summary>game: DieForYouPower.ShouldAllowHitting: the creature is alive (any creature, not only the owner). No
    /// Core2 command asks the hook yet; the old targeting (OldCreatures.HittableEnemies) already takes only the living.</summary>
    internal static bool ShouldAllowHitting(HookCtx c, ListenerRef self, in ShouldAllowHittingArgs a) => c.IsAlive(a.Creature);

    /// <summary>game: DieForYouPower.ShouldCreatureBeRemovedFromCombatAfterDeath: false for the owner (Osty), true for
    /// another creature. The old engine keeps a dead Osty (OstyHp 0, its max kept) to be summoned again.</summary>
    internal static bool ShouldCreatureBeRemovedFromCombatAfterDeath(HookCtx c, ListenerRef self,
        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) => c.Owner(self) is not { } me || a.Creature != me;

    /// <summary>game: DieForYouPower.ShouldPowerBeRemovedAfterOwnerDeath: false (a revived Osty holds it again).</summary>
    internal static bool ShouldPowerBeRemovedAfterOwnerDeath(HookCtx c, ListenerRef self, in PowerDeathArgs a) => false;

    /// <summary>Round R7g: the three pairs `port_list --unreached` listed, in block R7g (R7a's block keeps the fold).</summary>
    internal static void RegisterDeath(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DieForYouPower")
            .On(HookId.ShouldAllowHitting, (HookPredicate<ShouldAllowHittingArgs>)ShouldAllowHitting)
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)ShouldCreatureBeRemovedFromCombatAfterDeath)
            .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, (HookPredicate<PowerDeathArgs>)ShouldPowerBeRemovedAfterOwnerDeath);

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DieForYouPower")
            .On(HookId.ModifyUnblockedDamageTarget,
                (HookFold<ModifyUnblockedDamageTargetArgs, CreatureRef>)ModifyUnblockedDamageTarget);
}
