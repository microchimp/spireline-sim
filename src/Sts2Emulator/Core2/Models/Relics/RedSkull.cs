// game: RedSkull (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Red Skull: 3 Strength while its owner is at or under half its max HP.</summary>
internal static class RedSkull
{
    /// <summary>game: RedSkull.AfterCurrentHpChanged. Any creature's HP change while the combat is in progress re-reads the
    /// owner (ModifyStrengthIfNecessary): over half its max HP with the Strength applied, -3 Strength; at or under half
    /// without it, +3. The applied flag is the old relic counter; the old body is
    /// <c>RelicEffects.RedSkull_AfterCurrentHpChanged</c>. The status and the flash are presentation.</summary>
    internal static HookStep AfterCurrentHpChanged(HookCtx c, ListenerRef self, in AfterCurrentHpChangedArgs a,
        ChoiceCtx? ctx)
    {
        if (c.CombatInProgress)
            RelicEffects.RedSkull_AfterCurrentHpChanged(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "RedSkull")
            .On(HookId.AfterCurrentHpChanged, (HookEvent<AfterCurrentHpChangedArgs>)AfterCurrentHpChanged);
}
