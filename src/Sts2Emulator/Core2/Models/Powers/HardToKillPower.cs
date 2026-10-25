// game: HardToKillPower (MegaCrit.Sts2.Core.Models.Powers), its F08 override (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Hard to Kill: the damage its owner takes per hit is capped at the power's amount (F08, open pair).</summary>
internal static class HardToKillPower
{
    /// <summary>
    /// game: HardToKillPower.ModifyDamageCap. The target is the power's owner: the amount; else no cap
    /// (<c>decimal.MaxValue</c>). Any props, any dealer. The old engine has no location for it (the cap sat inline in
    /// <c>CardEffects.DealDamageToEnemy</c> and <c>DealUnpoweredDamageToEnemy</c>, after the int fold).
    /// AfterModifyingDamageAmount is a Flash only (not modelled).
    /// </summary>
    internal static decimal ModifyDamageCap(HookCtx c, ListenerRef self, in ModifyDamageArgs a) =>
        c.Power(self) is { } p && a.Target == p.Owner ? p.Amount : decimal.MaxValue;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "HardToKillPower")
            .On(HookId.ModifyDamageCap, (HookCap<ModifyDamageArgs, decimal>)ModifyDamageCap);
}
