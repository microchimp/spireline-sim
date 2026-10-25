// game: PaperCutsPower (MegaCrit.Sts2.Core.Models.Powers), its F08 override (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Paper Cuts: the player loses max HP when its owner's attack gets through (F08; needs item 9's LoseMaxHp).</summary>
internal static class PaperCutsPower
{
    /// <summary>
    /// game: PaperCutsPower.AfterDamageGiven. When the dealer is the power's owner, the target is the player, the props are
    /// a powered attack and the hit's unblocked damage is positive: CreatureCmd.LoseMaxHp on the target by the power's
    /// amount, not from a card. The old engine ran it inline in <c>EnemyAI.DealAttackDamage</c>
    /// (<c>EnemyAI.PaperCuts_AfterDamageGiven</c>), which the switch skips.
    /// </summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Dealer != p.Owner || !c.IsPlayer(a.Target) || !a.Props.IsPoweredAttack()
            || c.Damage(a.Results).UnblockedDamage <= 0)
            return HookStep.Done;
        CreatureCmd.LoseMaxHp(c.K, OldCreature.Player, p.Amount, isFromCard: false, a.ChoiceContext);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "PaperCutsPower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
