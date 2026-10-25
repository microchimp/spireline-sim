// game: LagavulinMatriarch (MegaCrit.Sts2.Core.Models.Monsters), its F08/F10 overrides (docs/rebuild/port-f08-f10.md,
// batch monster-1; docs/rebuild/MIGRATION_F08.md section 5). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Monsters;

/// <summary>The Lagavulin Matriarch: her damage and death overrides (F08, F10). Presentation only.</summary>
internal static class LagavulinMatriarch
{
    /// <summary>
    /// game: LagavulinMatriarch.AfterDamageReceived (Hook.AfterDamageReceived, run for every listener of the combat). When
    /// the target is her: stops her sleeping effect, and at half HP or below opens her shell's eyes once (an animation and
    /// the IsShellAwake flag, which only that animation reads). Presentation only, so the handler changes nothing.
    /// The port list's old location for this pair, <c>EnemyAI.WakeMatriarchIfStruck</c>, is AsleepPower.AfterDamageReceived's
    /// body (the wake: Asleep and Plating gone, the stun): that power's pair (batch power-3) calls it, not this handler.
    /// </summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx) => HookStep.Done;

    /// <summary>
    /// game: LagavulinMatriarch.AfterDeath (Hook.AfterDeath, run for every listener of the combat). When the dead creature
    /// is her: a death animation track and her sleeping effect stopped. Presentation only, so the handler changes nothing.
    /// </summary>
    internal static HookStep AfterDeath(HookCtx c, ListenerRef self, in AfterDeathArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Monster, "LagavulinMatriarch")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived)
            .On(HookId.AfterDeath, (HookEvent<AfterDeathArgs>)AfterDeath);
}
