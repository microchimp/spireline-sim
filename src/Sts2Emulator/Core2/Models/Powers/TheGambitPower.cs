// game: TheGambitPower (MegaCrit.Sts2.Core.Models.Powers), its F08 override (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>The Gambit: the next unblocked powered attack on its owner kills it (F08; the kill is Core2's Kill).</summary>
internal static class TheGambitPower
{
    /// <summary>
    /// game: TheGambitPower.AfterDamageReceived. When the target is the power's owner, the props are a powered attack and
    /// the unblocked damage is positive: PowerCmd.Remove of the power, then CreatureCmd.Kill of the owner. The old body
    /// <c>EnemyAI.TheGambit_AfterDamageReceived</c> does both (the buff's removal, then the gated Kill); the old engine
    /// gives the power to the player only, as the game's cards do.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || !a.Props.IsPoweredAttack()
            || c.Damage(a.Result).UnblockedDamage <= 0 || !c.IsPlayer(a.Target))
            return HookStep.Done;
        EnemyAI.TheGambit_AfterDamageReceived(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "TheGambitPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
