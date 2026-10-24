// The old state's view of one listener's own model (docs/rebuild/TIER1_KERNEL.md section 4, "the handler context"): what a
// handler reads about the model it runs for, found by its ListenerRef. This and OldState.cs are the only places that know
// where a power's amount, a relic or a card lives in the old representation. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class OldState
{
    /// <summary>
    /// The listener's entry in the lists now (<see cref="Combat" />: the creature it sits on, its index, its pile, the buff
    /// that stands for a power, the enemy), or null when it is not there (a power removed from its owner mid-pass is still
    /// visited by the game, and the old state keeps no copy of it). A power is read straight off its owner's buff list (its
    /// instance field names the creature, <see cref="PowerInstance" />); any other kind from a listing of the combat.
    /// Needs a combat.
    /// </summary>
    public static OldListener? Locate(OldWorld w, ListenerRef who)
    {
        if (w.Combat is null)
            return null;
        if ((ListenerKind)who.Kind == ListenerKind.Power)
        {
            if (OldCreatures.TryAt(w, new CreatureRef(who.Instance >> 8)) is not { } owner)
                return null;
            foreach (var p in OldCreatures.Powers(w, owner))
                if (p.Who == who)
                    return p;
            return null;
        }

        var all = new List<OldListener>();
        Combat(w, all, []);
        foreach (var e in all)
            if (e.Who == who)
                return e;
        return null;
    }

    /// <summary>
    /// The old buff that stands for a power listener, with its index in its owner's buff list; null when the power is not
    /// on its owner now (<see cref="Locate" />). Its <c>Magnitude</c> is the power's amount where the engine keeps the
    /// game's amount in it (the model's handler knows).
    /// </summary>
    public static (int Index, BuffState Buff)? PowerBuff(OldWorld w, ListenerRef who)
    {
        if ((ListenerKind)who.Kind != ListenerKind.Power || Locate(w, who) is not { } p
            || OldCreatures.TryAt(w, new CreatureRef(who.Instance >> 8)) is not { } owner
            || OldCreatures.Buffs(w.Combat!, owner) is not { } buffs)
            return null;
        return (p.Index, buffs[p.Index]);
    }
}
