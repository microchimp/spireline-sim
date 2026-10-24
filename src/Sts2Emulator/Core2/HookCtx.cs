// The handler context (docs/rebuild/TIER1_KERNEL.md section 4): the first parameter of every handler delegate
// (HookEvent, HookFold, HookCap, HookTry, HookTryOrNull, HookTryInPlace, HookPredicate, HookMutate), and the only way a
// handler reaches the combat it acts in. A handler never takes the old state, an engine or a static; it asks the context.
// Tier 1 answers over the old state (OldState.cs, OldState.Models.cs); a Tier 2 runtime answers the same questions over
// its own state, so a handler written against the neutral members below ports without a change.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>A power listener as a handler reads it: the creature it sits on and its amount (the game's PowerModel.Amount).</summary>
public readonly record struct PowerView(CreatureRef Owner, decimal Amount);

/// <summary>
/// What a handler is handed (one per dispatcher call; no state of its own, so a handler cannot keep anything in it).
/// <list type="bullet">
/// <item>Engine-neutral members (the shape a Tier 2 context keeps): <see cref="Power" /> and <see cref="Owner" />.</item>
/// <item>Tier 1 members (the old state, the one view of it): <see cref="K" /> (what the commands take:
/// <c>CreatureCmd.Damage(c.K, ...)</c>), <see cref="World" /> and <see cref="OldBuffs" />, for a handler whose body is
/// the old engine's extracted function
/// (<c>BuffSystem.Strength_ModifyDamageAdditive(List&lt;BuffState&gt;)</c>). A Tier 2 context has none of these; a handler
/// that uses them is rewritten when the model moves.</item>
/// </list>
/// A context made by a runtime that is no <see cref="KernelState" /> (a test fake) answers none of them.
/// </summary>
public readonly struct HookCtx
{
    private readonly HookRuntime _rt;

    internal HookCtx(HookRuntime rt) => _rt = rt;

    /// <summary>The per-combat kernel state this dispatch runs on (Tier 1): the commands, the world, the marks.</summary>
    public KernelState K => _rt as KernelState
                            ?? throw new InvalidOperationException("this hook runtime is no kernel state (a test fake)");

    /// <summary>The old state, with its marks (Tier 1). Needs a switch on, as every dispatch that reaches a handler has.</summary>
    public OldWorld World => K.World ?? throw new InvalidOperationException("no switch is on: the kernel holds no world");

    /// <summary>
    /// The power listener <paramref name="self" /> as it is on its owner now; null when it is not there (a power removed
    /// from its owner mid-pass is still visited by the game, and the old state keeps no copy: such a handler returns the
    /// identity or is a NotModelled gap its batch names). The amount is the old buff's magnitude.
    /// </summary>
    public PowerView? Power(ListenerRef self) =>
        OldState.PowerBuff(World, self) is { } p
            ? new PowerView(new CreatureRef(self.Instance >> 8), p.Buff.Magnitude)
            : null;

    /// <summary>The creature the listener sits on (the game's Owner or Creature of the model); null for a listener with
    /// none (a modifier, a badge, a scaling model) or one that is not on the old state now.</summary>
    public CreatureRef? Owner(ListenerRef self) =>
        OldState.Locate(World, self) is { Creature: >= 0 } me ? new CreatureRef(me.Creature) : null;

    /// <summary>The old buff list of a creature in the combat (Tier 1: the argument of an extracted
    /// <c>Model_Hook</c> function of BuffSystem). Throws for a creature that has none (a pet).</summary>
    public List<BuffState> OldBuffs(CreatureRef creature) =>
        OldCreatures.Buffs(World.Combat!, OldCreatures.At(World, creature))
        ?? throw new InvalidOperationException($"creature {creature.Index} has no old buff list");
}
