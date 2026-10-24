// game: ImbalancedPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Imbalanced: an attack of its owner that is fully blocked knocks the owner off balance.</summary>
internal static class ImbalancedPower
{
    /// <summary>game: ImbalancedPower.AfterDamageGiven. When the dealer is the owner and the hit was fully blocked: a
    /// Bowlbug Rock goes off balance (IsOffBalance: the old <c>EnemyState.OffBalance</c>, set by the old body
    /// <c>EnemyAI.Imbalanced_AfterDamageGiven</c>), any other owner is stunned. Not modelled: the other owner's
    /// CreatureCmd.Stun, as the old engine gives the power to the Bowlbug Rock only
    /// (<c>CombatFactory.ApplyIntrinsicEnemyPowers</c>). The flash is presentation.</summary>
    internal static HookStep AfterDamageGiven(HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Dealer != p.Owner || !c.Damage(a.Results).WasFullyBlocked)
            return HookStep.Done;
        if (c.IsMonster(p.Owner, "BowlbugRock") && c.Enemy(p.Owner) is { } rock)
            EnemyAI.Imbalanced_AfterDamageGiven(rock);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ImbalancedPower")
            .On(HookId.AfterDamageGiven, (HookEvent<AfterDamageGivenArgs>)AfterDamageGiven);
}
