// game: RupturePower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Rupture: unblocked damage to its owner on its own turn gives Strength, banked while the card that dealt it is
/// being played.</summary>
internal static class RupturePower
{
    /// <summary>
    /// game: RupturePower.AfterDamageReceived. When the target is the power's owner, the unblocked damage is positive and
    /// the current side is the owner's: a card source the owner is playing now (its playedCards entry, made at
    /// BeforeCardPlayed on the owner's side) banks the amount, paid at that card's AfterCardPlayed; any other source applies
    /// Strength at once. The bank is the old <c>CombatState.RuptureOwed</c> (paid by the old <c>CombatEngine.PayRupture</c>)
    /// and a card being played is the old in-flight card on the player's turn: the old body
    /// <c>CardEffects.Rupture_AfterDamageReceived</c>, with the fromPlayedCard Core2's damage command passed it before.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || c.Damage(a.Result).UnblockedDamage <= 0
            || !c.IsPlayer(p.Owner) || !c.OldCombat.PlayerTurn)
            return HookStep.Done;
        bool playing = a.CardSource is { } h && c.World.InFlight is { } flying && flying.Uid == h.Uid;
        CardEffects.Rupture_AfterDamageReceived(c.OldCombat, (int)p.Amount, playing);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "RupturePower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
