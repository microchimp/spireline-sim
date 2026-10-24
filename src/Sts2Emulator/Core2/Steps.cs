// Named steps of the hook dispatchers: the parts of a dispatcher outside its listener loops
// (tools/kernel/hooks.json "pre" / "post"). One function per step, and per dispatcher for the steps that read its
// arguments: those are declared in Hooks.g.cs as partial methods typed by the dispatcher's argument record (a new
// dispatcher using one fails to build until it is implemented here). Round F08-R2 implemented the steps of
// MIGRATION_F08.md section 3; those of other families (F07's cost passthrough, F09's block enchantment) still throw,
// naming the step and the game method it ports.
namespace Sts2Emulator.Core2;

public static partial class Steps
{
    /// <summary>netid-guard: no local net id, the dispatch does nothing. Returns whether it proceeds.
    /// game: Hook.AfterDeath, AfterDiedToDoom, BeforeFlush, BeforeSideTurnStart, BeforeTurnEnd, AfterTurnEnd,
    /// AfterAutoPrePlayPhaseEntered, AfterAutoPostPlayPhaseEntered.</summary>
    /// <remarks>The emulator is one local player, whose net id is always set: the dispatch proceeds. The generator puts
    /// the fast path first, so a dispatch with no listener does not reach this.</remarks>
    public static bool NetIdGuard(HookRuntime rt, HookId dispatcher) => true;

    /// <summary>negative-cost-passthrough: a negative cost is returned unmodified. Returns whether the dispatch
    /// proceeds. game: Hook.ModifyEnergyCostInCombat, Hook.ModifyStarCost.</summary>
    public static bool NegativeCostPassthrough(HookRuntime rt, HookId dispatcher, decimal cost) =>
        throw Missing("negative-cost-passthrough", dispatcher);

    /// <summary>owner-in-combat: a power whose owner is out of the combat is removed without asking (reads
    /// <c>a.Power</c>). Returns whether the dispatch proceeds. game: Hook.ShouldPowerBeRemovedOnDeath.</summary>
    /// <remarks>game: the power's Owner.CombatState is null once its creature has left the combat. Here: the owner's
    /// position names a creature in the combat (OldCreatures.At).</remarks>
    public static partial bool OwnerInCombat(HookRuntime rt, HookId dispatcher, in ShouldPowerBeRemovedOnDeathArgs a)
    {
        if (rt is not KernelState { World: { Combat: not null } w })
            return false;
        try
        {
            OldCreatures.At(w, a.Power.Owner);
            return true;
        }
        catch (ArgumentOutOfRangeException)
        {
            return false;
        }
    }

    /// <summary>enchant-add: the source card's enchantment adds to the value (reads <c>a.CardSource</c>,
    /// <c>a.Props</c>). game: Hook.ModifyBlock.</summary>
    public static partial decimal EnchantAdd(HookRuntime rt, HookId dispatcher, in ModifyBlockArgs a, decimal value) =>
        throw Missing("enchant-add", dispatcher);

    /// <summary>enchant-add of Hook.ModifyDamage (reads <c>a.CardSource</c>, <c>a.Props</c>).</summary>
    /// <remarks>game: Hook.ModifyDamage adds the card's EnchantmentModel.EnchantDamageAdditive(value, props) when the
    /// card has an enchantment (OldCreatures.EnchantDamageAdditive: the old body).</remarks>
    public static partial decimal EnchantAdd(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value) =>
        rt is KernelState { World: { } w } ? value + OldCreatures.EnchantDamageAdditive(w, a.CardSource, a.Props) : value;

    /// <summary>enchant-mul: the source card's enchantment multiplies the value (reads <c>a.CardSource</c>,
    /// <c>a.Props</c>). game: Hook.ModifyBlock.</summary>
    public static partial decimal EnchantMul(HookRuntime rt, HookId dispatcher, in ModifyBlockArgs a, decimal value) =>
        throw Missing("enchant-mul", dispatcher);

    /// <summary>enchant-mul of Hook.ModifyDamage (reads <c>a.CardSource</c>, <c>a.Props</c>).</summary>
    /// <remarks>game: Hook.ModifyDamage multiplies by the card's EnchantmentModel.EnchantDamageMultiplicative, after
    /// the additive one.</remarks>
    public static partial decimal EnchantMul(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value) =>
        rt is KernelState { World: { } w } ? value * OldCreatures.EnchantDamageMultiplicative(w, a.CardSource, a.Props) : value;

    /// <summary>preview-hittable: a multi-target preview runs the passes once per hittable enemy (each with
    /// <c>a with { Target = enemy }</c>) and keeps the result when they all agree. Returns whether it produced the
    /// result. game: Hook.ModifyDamage.</summary>
    /// <remarks>game: Hook.ModifyDamage with no target and CardPreviewMode.MultiCreatureTargeting, for an all-enemies
    /// or random-enemy card in Hand or Play: the passes once per CombatState.HittableEnemies entry; when every result
    /// agrees after an int cast, the first result and the distinct changed set; else (or with no hittable enemy) false,
    /// and the dispatcher runs the plain passes. A real hit (preview mode None) never takes it.</remarks>
    public static partial bool PreviewHittable(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value,
        HookPasses<ModifyDamageArgs> passes, out decimal result, out IReadOnlyList<ListenerRef> changed)
    {
        result = value;
        changed = Array.Empty<ListenerRef>();
        if (a.Target is not null || a.PreviewMode != PreviewModes.MultiCreatureTargeting
            || rt is not KernelState { World: { Combat: not null } w } || !OldCreatures.IsAllEnemiesCardInHandOrPlay(w, a.CardSource))
            return false;
        decimal? first = null;
        var all = new List<ListenerRef>();
        foreach (var enemy in OldCreatures.HittableEnemies(w))
        {
            decimal r = passes(rt, a with { Target = OldCreatures.Ref(w, enemy) }, value, out var ch);
            if (first is null)
                first = r;
            else if ((int)r != (int)first.Value)
                return false;
            all.AddRange(ch);
        }

        if (first is not { } v)
            return false;
        result = v;
        changed = all.Distinct().ToList();
        return true;
    }

    /// <summary>clamp-min0: the result is at least zero. game: Hook.ModifyBlock, Hook.ModifyDamage.</summary>
    public static decimal ClampMin0(HookId dispatcher, decimal value) => Math.Max(0m, value);

    private static NotImplementedException Missing(string step, HookId dispatcher) =>
        new($"kernel step {step} (game: Hook.{GameName(dispatcher)}) is implemented with its family's migration");

    /// <summary>The game method a dispatcher id names.</summary>
    public static string GameName(HookId id)
    {
        string s = id.ToString();
        return s.StartsWith("Dispatch_", StringComparison.Ordinal) ? s["Dispatch_".Length..] : s;
    }
}
