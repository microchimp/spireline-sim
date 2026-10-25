// game: LavaLamp (MegaCrit.Sts2.Core.Models.Relics), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch relic-1;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

namespace Sts2Emulator.Core2.Models.Relics;

/// <summary>Lava Lamp: a combat in which its owner took no unblocked, blockable damage upgrades the card rewards.</summary>
internal static class LavaLamp
{
    /// <summary>game: LavaLamp.AfterDamageReceived. In a combat room, when the target is the owner, the unblocked damage is
    /// positive and the props are not Unblockable: TookDamageThisCombat set. The latch is the old
    /// <c>CombatState.TookUnblockedDamage</c> (the run copies it at the combat's end; the reward upgrade reads it), through
    /// <c>RelicEffects.LavaLamp_AfterDamageReceived</c>. Core2 owned the line for every player hit before; as in the game,
    /// only a held lamp latches now.</summary>
    internal static HookStep AfterDamageReceived(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a,
        ChoiceCtx? ctx)
    {
        if (!c.IsPlayer(a.Target) || c.Damage(a.Result).UnblockedDamage <= 0 || a.Props.Has(ValueProps.Unblockable))
            return HookStep.Done;
        RelicEffects.LavaLamp_AfterDamageReceived(c.OldCombat);
        return HookStep.Done;
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Relic, "LavaLamp")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)AfterDamageReceived);
}
