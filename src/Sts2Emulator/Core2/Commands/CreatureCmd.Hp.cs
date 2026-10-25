// game: CreatureCmd.Heal, SetCurrentHp, SetMaxHp, SetMaxAndCurrentHp, GainMaxHp, LoseMaxHp, and Creature.HealInternal,
// SetCurrentHpInternal and SetMaxHpInternal, over the old HP fields (docs/rebuild/MIGRATION_F08.md section 3 item 9; the
// spec per method is the batch L1 section). Round R4: the old write sites call these under the F08+F10 switch; batch L1:
// each is the game's method, with the hooks and commands it calls in its order. Not modelled: the heal's sound, animations
// and numbers, the revive animation, the map point's history entries and the scaled wait (presentation and run history).
// Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class CreatureCmd
{
    // construct: game CreatureCmd.Heal
    //   -    HealInternal
    //   if   Hook.AfterCurrentHpChanged
    // end construct
    /// <summary>game: CreatureCmd.Heal. While the combat is ending only the player heals. Creature.HealInternal (the HP up
    /// to the max; a dead player brought back has its hooks activated again), then AfterCurrentHpChanged with the
    /// asked-for amount whenever it is positive, even when the HP did not move (a creature at full HP). Returns the HP
    /// gained.</summary>
    public static int Heal(KernelState k, OldCreature c, decimal amount)
    {
        var s = Combat(k);
        if (CombatEnd.IsEnding(k) && c.Kind != CreatureKind.Player)
            return 0;
        int before = OldCreatures.Hp(s, c);
        HealInternal(k, c, amount);
        if (amount > 0m && OldCreatures.TryRef(k.World!, c) is { } r)
        {
            var a = new AfterCurrentHpChangedArgs(r, amount);
            Settle(k, Hooks.AfterCurrentHpChanged(k, a), st => Hooks.AfterCurrentHpChangedResume(k, a, st));
        }
        return OldCreatures.Hp(s, c) - before;
    }

    // construct: game CreatureCmd.SetCurrentHp
    //   -    SetCurrentHpInternal
    //   if   Hook.AfterCurrentHpChanged
    //   if   Kill
    // end construct
    /// <summary>game: CreatureCmd.SetCurrentHp. Creature.SetCurrentHpInternal (no floor; capped at the max), then, when
    /// the asked-for HP differs from the old HP, AfterCurrentHpChanged with that difference (asked-for minus old, not the
    /// capped change); a creature dead after it is killed. No hook activation: a revive through this command leaves a
    /// dead player's hooks off (only Heal turns them on). An enemy set to 0 HP stays in the combat (marked dying, as the
    /// old state keeps a dead enemy only inside a kill) until the kill removes it, so the hook still names it.</summary>
    public static void SetCurrentHp(KernelState k, OldCreature c, decimal hp)
    {
        var s = Combat(k);
        int before = OldCreatures.Hp(s, c);
        using var dying = c.Enemy is { } e && before > 0 && hp <= 0 ? k.World!.MarkDying(e) : null;
        SetCurrentHpInternal(s, c, hp);
        if (hp != before && OldCreatures.TryRef(k.World!, c) is { } r)
        {
            var a = new AfterCurrentHpChangedArgs(r, hp - before);
            Settle(k, Hooks.AfterCurrentHpChanged(k, a), st => Hooks.AfterCurrentHpChangedResume(k, a, st));
        }
        if (OldCreatures.IsDead(s, c))
            Kill(k, [c]);
    }

    // construct: game CreatureCmd.SetMaxHp
    //   -    SetMaxHpInternal
    //   if   Kill
    // end construct
    /// <summary>game: CreatureCmd.SetMaxHp. Creature.SetMaxHpInternal with the amount floored at 0 (the max, the HP lowered
    /// to it), no hook; a max of 0 kills. Returns the change of the max.</summary>
    public static int SetMaxHp(KernelState k, OldCreature c, decimal max)
    {
        var s = Combat(k);
        int old = OldCreatures.MaxHp(s, c);
        SetMaxHpInternal(s, c, Math.Max(0m, max));
        int now = OldCreatures.MaxHp(s, c);
        if (now <= 0)
            Kill(k, [c]);
        return now - old;
    }

    // construct: game CreatureCmd.SetMaxAndCurrentHp
    //   -    SetMaxHp
    //   -    SetCurrentHp
    // end construct
    /// <summary>game: CreatureCmd.SetMaxAndCurrentHp: SetMaxHp, then SetCurrentHp.</summary>
    public static void SetMaxAndCurrentHp(KernelState k, OldCreature c, decimal hp)
    {
        SetMaxHp(k, c, hp);
        SetCurrentHp(k, c, hp);
    }

    // construct: game CreatureCmd.GainMaxHp
    //   -    SetMaxHp
    //   -    Heal
    // end construct
    /// <summary>game: CreatureCmd.GainMaxHp. A negative amount is refused (the game throws). SetMaxHp to the raised max,
    /// then a Heal by the change SetMaxHp returns (so its hook and its ending gate are the heal's).</summary>
    public static void GainMaxHp(KernelState k, OldCreature c, decimal amount)
    {
        var s = Combat(k);
        if (amount < 0m)
            throw new ArgumentException("amount must be non-negative. Use LoseMaxHp for max HP loss.", nameof(amount));
        int gained = SetMaxHp(k, c, OldCreatures.MaxHp(s, c) + amount);
        Heal(k, c, gained);
    }

    // construct: game CreatureCmd.LoseMaxHp
    //   if   Damage
    //   -    SetMaxHp
    // end construct
    /// <summary>game: CreatureCmd.LoseMaxHp. A negative amount is refused (the game throws). When the lowered max is under
    /// the HP, the excess is real damage first (CreatureCmd.Damage: unblockable and unpowered, a move as well when it comes
    /// from a card; no dealer, no card source), then SetMaxHp to the lowered max, never under 1.</summary>
    public static void LoseMaxHp(KernelState k, OldCreature c, decimal amount, bool isFromCard, ChoiceCtx? ctx = null)
    {
        var s = Combat(k);
        if (amount < 0m)
            throw new ArgumentException("amount must be non-negative. Use GainMaxHp for max HP gain.", nameof(amount));
        decimal newMax = OldCreatures.MaxHp(s, c) - amount;
        if (newMax < OldCreatures.Hp(s, c))
        {
            var props = new ValueProp(ValueProps.Unblockable | ValueProps.Unpowered | (isFromCard ? ValueProps.Move : 0));
            Damage(k, [c], OldCreatures.Hp(s, c) - newMax, props, null, null, ctx);
        }
        SetMaxHp(k, c, Math.Max(1m, newMax));
    }

    /// <summary>game: Creature.HealInternal: SetCurrentHpInternal to the HP plus the amount; a dead creature brought back
    /// has its player's hooks activated again (Player.ActivateHooks; a monster has no player, and its Revived event has
    /// only node subscribers).</summary>
    private static void HealInternal(KernelState k, OldCreature c, decimal amount)
    {
        var s = Combat(k);
        bool wasDead = OldCreatures.IsDead(s, c);
        SetCurrentHpInternal(s, c, OldCreatures.Hp(s, c) + amount);
        if (wasDead && !OldCreatures.IsDead(s, c) && c.Kind == CreatureKind.Player)
            k.World!.ActivatePlayerHooks();
    }

    /// <summary>game: Creature.SetCurrentHpInternal: the HP set to the amount capped at the max, truncated.</summary>
    private static void SetCurrentHpInternal(CombatState s, OldCreature c, decimal amount) =>
        OldCreatures.SetHp(s, c, (int)Math.Min(amount, OldCreatures.MaxHp(s, c)));

    /// <summary>game: Creature.SetMaxHpInternal: the max set (truncated, at most 999999999), the HP lowered to it.</summary>
    private static void SetMaxHpInternal(CombatState s, OldCreature c, decimal amount)
    {
        if (amount < 0m)
            throw new ArgumentException("amount must be non-negative.", nameof(amount));
        OldCreatures.SetMaxHp(s, c, Math.Min((int)amount, 999999999));
        OldCreatures.SetHp(s, c, Math.Min(OldCreatures.Hp(s, c), OldCreatures.MaxHp(s, c)));
    }

    private static CombatState Combat(KernelState k) =>
        k.World?.Combat ?? throw new InvalidOperationException("the HP commands need a switched-on combat");
}
