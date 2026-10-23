// Named steps of the hook dispatchers: the parts of a dispatcher outside its listener loops
// (tools/kernel/hooks.json "pre" / "post"). One function per step, and per dispatcher for the steps that read its
// arguments: those are declared in Hooks.g.cs as partial methods typed by the dispatcher's argument record (a new
// dispatcher using one fails to build until it is implemented here). Round K3 implements them; until then each
// throws, naming the step and the game method it ports.
namespace Sts2Emulator.Core2;

public static partial class Steps
{
    /// <summary>netid-guard: no local net id, the dispatch does nothing. Returns whether it proceeds.
    /// game: Hook.AfterDeath, AfterDiedToDoom, BeforeFlush, BeforeSideTurnStart, BeforeTurnEnd, AfterTurnEnd,
    /// AfterAutoPrePlayPhaseEntered, AfterAutoPostPlayPhaseEntered.</summary>
    public static bool NetIdGuard(HookRuntime rt, HookId dispatcher) => throw Missing("netid-guard", dispatcher);

    /// <summary>negative-cost-passthrough: a negative cost is returned unmodified. Returns whether the dispatch
    /// proceeds. game: Hook.ModifyEnergyCostInCombat, Hook.ModifyStarCost.</summary>
    public static bool NegativeCostPassthrough(HookRuntime rt, HookId dispatcher, decimal cost) =>
        throw Missing("negative-cost-passthrough", dispatcher);

    /// <summary>owner-in-combat: a power whose owner is out of the combat is removed without asking (reads
    /// <c>a.Power</c>). Returns whether the dispatch proceeds. game: Hook.ShouldPowerBeRemovedOnDeath.</summary>
    public static partial bool OwnerInCombat(HookRuntime rt, HookId dispatcher, in ShouldPowerBeRemovedOnDeathArgs a) =>
        throw Missing("owner-in-combat", dispatcher);

    /// <summary>enchant-add: the source card's enchantment adds to the value (reads <c>a.CardSource</c>,
    /// <c>a.Props</c>). game: Hook.ModifyBlock.</summary>
    public static partial decimal EnchantAdd(HookRuntime rt, HookId dispatcher, in ModifyBlockArgs a, decimal value) =>
        throw Missing("enchant-add", dispatcher);

    /// <summary>enchant-add of Hook.ModifyDamage (reads <c>a.CardSource</c>, <c>a.Props</c>).</summary>
    public static partial decimal EnchantAdd(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value) =>
        throw Missing("enchant-add", dispatcher);

    /// <summary>enchant-mul: the source card's enchantment multiplies the value (reads <c>a.CardSource</c>,
    /// <c>a.Props</c>). game: Hook.ModifyBlock.</summary>
    public static partial decimal EnchantMul(HookRuntime rt, HookId dispatcher, in ModifyBlockArgs a, decimal value) =>
        throw Missing("enchant-mul", dispatcher);

    /// <summary>enchant-mul of Hook.ModifyDamage (reads <c>a.CardSource</c>, <c>a.Props</c>).</summary>
    public static partial decimal EnchantMul(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value) =>
        throw Missing("enchant-mul", dispatcher);

    /// <summary>preview-hittable: a multi-target preview runs the passes once per hittable enemy (each with
    /// <c>a with { Target = enemy }</c>) and keeps the result when they all agree. Returns whether it produced the
    /// result. game: Hook.ModifyDamage.</summary>
    public static partial bool PreviewHittable(HookRuntime rt, HookId dispatcher, in ModifyDamageArgs a, decimal value,
        HookPasses<ModifyDamageArgs> passes, out decimal result, out IReadOnlyList<ListenerRef> changed) =>
        throw Missing("preview-hittable", dispatcher);

    /// <summary>clamp-min0: the result is at least zero. game: Hook.ModifyBlock, Hook.ModifyDamage.</summary>
    public static decimal ClampMin0(HookId dispatcher, decimal value) => throw Missing("clamp-min0", dispatcher);

    private static NotImplementedException Missing(string step, HookId dispatcher) =>
        new($"kernel step {step} (game: Hook.{GameName(dispatcher)}) is implemented in round K3");

    /// <summary>The game method a dispatcher id names.</summary>
    public static string GameName(HookId id)
    {
        string s = id.ToString();
        return s.StartsWith("Dispatch_", StringComparison.Ordinal) ? s["Dispatch_".Length..] : s;
    }
}
