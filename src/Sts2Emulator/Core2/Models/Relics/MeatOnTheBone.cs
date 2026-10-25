// game: MeatOnTheBone (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Meat on the Bone: heals its owner after a won combat ended at or under half HP.</summary>
internal static class MeatOnTheBone
{
    /// <summary>game: MeatOnTheBone.AfterCurrentHpChanged. The owner's HP change while the combat is in progress sets the
    /// relic's status (whether it will heal): presentation only, so the handler changes nothing. The heal itself is
    /// AfterCombatVictoryEarly (F17, the old reward generation).</summary>
    internal static HookStep AfterCurrentHpChanged(HookCtx c, ListenerRef self, in AfterCurrentHpChangedArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "MeatOnTheBone")
            .On(HookId.AfterCurrentHpChanged, (HookEvent<AfterCurrentHpChangedArgs>)AfterCurrentHpChanged);
}
