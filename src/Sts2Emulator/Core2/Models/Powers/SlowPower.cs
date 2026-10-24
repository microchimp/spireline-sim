// game: SlowPower (MegaCrit.Sts2.Core.Models.Powers), its F08 override (docs/rebuild/port-f08-f10.md, batch power-3;
// docs/rebuild/MIGRATION_F08.md, batch L1). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Slow: powered attacks on its owner deal 10% more per card played this turn (F08, open pair: mechanism B).</summary>
internal static class SlowPower
{
    /// <summary>
    /// game: SlowPower.ModifyDamageMultiplicative. The target is the power's owner and the props are a powered attack:
    /// 1 + 0.1 x SlowAmount, a decimal factor inside the fold (the old engine multiplied the folded int by a float after
    /// the fold); else 1. SlowAmount is the game's dynamic var that AfterCardPlayed raises and the owner's side turn start
    /// resets; the old engine keeps it as the owner's <c>BuffId.SlowCount</c> buff (CombatEngine raises and clears it), so
    /// the handler reads that buff (Tier 1: <c>c.OldBuffs</c>). AfterModifyingDamageAmount is a Flash only.
    /// </summary>
    internal static decimal ModifyDamageMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a,
        decimal current)
    {
        if (c.Power(self) is not { } p || a.Target != p.Owner || !a.Props.IsPoweredAttack())
            return 1m;
        return 1m + 0.1m * BuffSystem.Get(c.OldBuffs(p.Owner), BuffId.SlowCount);
    }

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "SlowPower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)ModifyDamageMultiplicative);
}
