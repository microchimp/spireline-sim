// game: FairyInABottle (MegaCrit.Sts2.Core.Models.Potions), its F10 overrides (docs/rebuild/port-f08-f10.md, batch
// potion-1; docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Potions;

/// <summary>Fairy in a Bottle: the owner's death is refused while it is held; it is used up and heals 30% (F10).</summary>
internal static class FairyInABottle
{
    /// <summary>game: FairyInABottle.ShouldDie. Another creature: true (no say). The owner: false while the potion is held
    /// (a listener is a held potion: its slot).</summary>
    internal static bool ShouldDie(HookCtx c, ListenerRef self, in ShouldDieArgs a) => !c.IsPlayer(a.Creature);

    /// <summary>game: FairyInABottle.AfterPreventingDeath: OnUseWrapper, that is the potion used up and OnUse's
    /// CreatureCmd.Heal of the owner by max(max HP x 0.3, 1) (<c>CombatEngine.FairyInABottle_AfterPreventingDeath</c>:
    /// the slot emptied, the heal gated to the Heal command). Not modelled here: the potion-use hooks of OnUseWrapper
    /// (their family's, not F08/F10) and the use's visuals.</summary>
    internal static HookStep AfterPreventingDeath(HookCtx c, ListenerRef self, in AfterPreventingDeathArgs a,
        ChoiceCtx? ctx)
    {
        if (a.Preventer != self || !c.IsPlayer(a.Creature))
            return HookStep.Done;
        CombatEngine.FairyInABottle_AfterPreventingDeath(c.OldCombat, self.Instance);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Potion, "FairyInABottle")
            .On(HookId.ShouldDie, (HookPredicate<ShouldDieArgs>)ShouldDie)
            .On(HookId.AfterPreventingDeath, (HookEvent<AfterPreventingDeathArgs>)AfterPreventingDeath);
}
