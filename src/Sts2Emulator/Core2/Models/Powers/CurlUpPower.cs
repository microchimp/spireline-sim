// game: CurlUpPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-2;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Curl Up: the first card whose powered attack hits its owner gives the owner block once that card is
/// played.</summary>
internal static class CurlUpPower
{
    /// <summary>
    /// game: CurlUpPower.AfterDamageReceived. When the target is the owner, the props are a powered attack and there is a
    /// card source (the first one, or the same card again): the card is remembered (playedCard). No block here: the game
    /// gains it at that card's AfterCardPlayed (CreatureCmd.GainBlock by the amount, Unpowered, then the power removed).
    /// The old engine keeps one flag for the remembered card, <c>EnemyState.CurlUpArmed</c>, set by the old body
    /// <c>CardEffects.CurlUp_AfterDamageReceived</c>. Hybrid (Tier 1): the block is the old engine's, paid by
    /// <c>CombatEngine.PayCurlUp</c> through the old block code once the card has finished, because AfterCardPlayed and
    /// F09's GainBlock are not migrated; this handler only arms it.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || !a.Props.IsPoweredAttack() || a.CardSource is not { } card
            || c.Enemy(p.Owner) is not { } e)
            return HookStep.Done;
        // A card remembered already: another card's hit is not remembered (playedCard != null, cardSource != playedCard).
        if (e.CurlUpArmed && e.CurlUpCardUid != 0 && e.CurlUpCardUid != card.Uid)
            return HookStep.Done;
        CardEffects.CurlUp_AfterDamageReceived(e);
        e.CurlUpCardUid = card.Uid;
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "CurlUpPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
