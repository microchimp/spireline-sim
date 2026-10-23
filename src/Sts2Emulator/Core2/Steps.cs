// Named steps of the hook dispatchers: the parts of a dispatcher outside its listener loops
// (tools/kernel/hooks.json "pre" / "post"). One function each; round K3 implements them. Until then each throws,
// naming the step and the game method it ports.
namespace Sts2Emulator.Core2;

public static class Steps
{
    /// <summary>netid-guard: no local net id, the dispatch does nothing. Returns whether it proceeds.
    /// game: Hook.AfterDeath, AfterDiedToDoom, BeforeFlush, BeforeSideTurnStart, BeforeTurnEnd, AfterTurnEnd,
    /// AfterAutoPrePlayPhaseEntered, AfterAutoPostPlayPhaseEntered.</summary>
    public static bool NetIdGuard(HookRuntime rt, HookId dispatcher) => throw Missing("netid-guard", dispatcher);

    /// <summary>negative-cost-passthrough: a negative cost is returned unmodified. Returns whether the dispatch
    /// proceeds. game: Hook.ModifyEnergyCostInCombat, Hook.ModifyStarCost.</summary>
    public static bool NegativeCostPassthrough(HookRuntime rt, HookId dispatcher, decimal cost) =>
        throw Missing("negative-cost-passthrough", dispatcher);

    /// <summary>owner-in-combat: a power whose owner is out of the combat is removed without asking.
    /// Returns whether the dispatch proceeds. game: Hook.ShouldPowerBeRemovedOnDeath.</summary>
    public static bool OwnerInCombat<TArgs>(HookRuntime rt, HookId dispatcher, in TArgs a) where TArgs : struct =>
        throw Missing("owner-in-combat", dispatcher);

    /// <summary>enchant-add: the source card's enchantment adds to the value. game: Hook.ModifyBlock, Hook.ModifyDamage.</summary>
    public static decimal EnchantAdd<TArgs>(HookRuntime rt, HookId dispatcher, in TArgs a, decimal value)
        where TArgs : struct => throw Missing("enchant-add", dispatcher);

    /// <summary>enchant-mul: the source card's enchantment multiplies the value. game: Hook.ModifyBlock, Hook.ModifyDamage.</summary>
    public static decimal EnchantMul<TArgs>(HookRuntime rt, HookId dispatcher, in TArgs a, decimal value)
        where TArgs : struct => throw Missing("enchant-mul", dispatcher);

    /// <summary>preview-hittable: a multi-target preview runs the passes per hittable enemy and keeps the result
    /// when they all agree. Returns whether it produced the result. game: Hook.ModifyDamage.</summary>
    public static bool PreviewHittable<TArgs>(HookRuntime rt, HookId dispatcher, in TArgs a, decimal value,
        HookPasses<TArgs> passes, out decimal result, out IReadOnlyList<ListenerRef> changed) where TArgs : struct =>
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
