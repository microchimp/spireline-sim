// DieForYouPower (power:DIE_FOR_YOU_POWER), batch R7a (docs/rebuild/MIGRATION_F08.md, R7 checks): ModifyUnblockedDamageTarget,
// the pet redirect. The game applies the power to Osty once, when OstyCmd.Summon first adds the pet, and never removes it
// (ShouldPowerBeRemovedAfterOwnerDeath), so an Osty the old state holds (OstyMaxHp above 0) holds it, and OldState lists it
// on Osty. The old enemy-attack site (EnemyAI's player hit) moved the unblocked rest of a powered hit onto a living Osty
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

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "DieForYouPower")
            .On(HookId.ModifyUnblockedDamageTarget,
                (HookFold<ModifyUnblockedDamageTargetArgs, CreatureRef>)ModifyUnblockedDamageTarget);
}
