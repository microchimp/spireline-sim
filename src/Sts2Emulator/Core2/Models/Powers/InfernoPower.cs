// game: InfernoPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Inferno: unblocked damage to its owner on the owner's own turn deals its amount to every hittable enemy.</summary>
internal static class InfernoPower
{
    /// <summary>
    /// game: InfernoPower.AfterDamageReceived. When the target is the power's owner, the unblocked damage is positive and
    /// the current side is the owner's (the player's turn): one CreatureCmd.Damage over CombatState.HittableEnemies, the
    /// power's amount, Unpowered, the owner as dealer, no card source. The old body
    /// (<c>CardEffects.TriggerInfernoAfterPlayerDamage</c>) hit the enemies with one damage call each; this is the game's
    /// one call, so the AfterDamageReceived passes and the one Kill follow all the hits. The burst's visuals are not
    /// modelled. The old engine gives the power to the player only.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || c.Damage(a.Result).UnblockedDamage <= 0
            || !c.IsPlayer(p.Owner) || !c.OldCombat.PlayerTurn)
            return HookStep.Done;
        var targets = new List<OldCreature>();
        foreach (var r in c.HittableEnemies())
        {
            if (c.Enemy(r) is { } e)
                targets.Add(OldCreature.Of(e));
        }

        CreatureCmd.Damage(c.K, targets, p.Amount, new ValueProp(ValueProps.Unpowered), OldCreature.Player, null,
            a.ChoiceContext);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "InfernoPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
