// game: LizardTail (MegaCrit.Sts2.Core.Models.Relics), its F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Lizard Tail: once, the owner's death is refused and it heals half its max HP (F10, coupled to Kill).</summary>
internal static class LizardTail
{
    /// <summary>game: LizardTail.ShouldDieLate. Another creature, or the relic used: true (no say). The owner with the
    /// relic unused: false (the death is prevented). WasUsed is the old relic's counter
    /// (<c>RelicEffects.LizardTailUnspent</c>).</summary>
    internal static bool ShouldDieLate(HookCtx c, ListenerRef self, in ShouldDieArgs a) =>
        !(c.IsPlayer(a.Creature) && RelicEffects.LizardTailUnspent(c.OldCombat));

    /// <summary>game: LizardTail.AfterPreventingDeath. WasUsed set (the counter, <c>RelicEffects.FiresOncePerCombat</c>),
    /// then CreatureCmd.Heal of the creature by max(1, max HP x 50%) (<c>RelicEffects.LizardTail_AfterPreventingDeath</c>,
    /// gated to the Heal command). The flash and the disabled status are presentation.</summary>
    internal static HookStep AfterPreventingDeath(HookCtx c, ListenerRef self, in AfterPreventingDeathArgs a,
        ChoiceCtx? ctx)
    {
        if (a.Preventer != self || !c.IsPlayer(a.Creature))
            return HookStep.Done;
        RelicEffects.FiresOncePerCombat(c.OldCombat, RelicEffects.LizardTail);
        RelicEffects.LizardTail_AfterPreventingDeath(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "LizardTail")
            .On(HookId.ShouldDieLate, (HookPredicate<ShouldDieArgs>)ShouldDieLate)
            .On(HookId.AfterPreventingDeath, (HookEvent<AfterPreventingDeathArgs>)AfterPreventingDeath);
}
