// game: HexPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Hex: its owner's cards are Hexed (Ethereal) while it lasts; it goes with the creature that applied it.</summary>
internal static class HexPower
{
    /// <summary>game: HexPower.AfterDeath. When the dead creature is the power's Applier and its removal was not
    /// prevented: PowerCmd.Remove of the power. The old buff keeps no applier; the old engine reads it as the Spectral
    /// Knight (the one creature that applies Hex, one per Knights fight): the old body <c>CombatEngine.Hex_AfterDeath</c>.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx)
    {
        if (a.WasRemovalPrevented || c.Power(self) is not { } p || !c.IsPlayer(p.Owner)
            || !c.IsMonster(a.Creature, "SpectralKnight"))
            return HookStep.Done;
        CombatEngine.Hex_AfterDeath(c.OldCombat);
        return HookStep.Done;
    }

    /// <summary>game: HexPower.AfterRemoved (pseudo-hook, from the owner's death removal): CardCmd.ClearAffliction of the
    /// owner's Hexed cards. The old engine keeps no Hexed affliction on the cards: it reads Ethereal off the Hex buff
    /// (<c>CombatEngine.IsEtherealForPowers</c>), so the removed buff leaves nothing to clear and the handler changes
    /// nothing.</summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "HexPower")
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath)
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
