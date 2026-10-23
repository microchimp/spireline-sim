// The kernel's hook runtime API (docs/rebuild/TIER1_KERNEL.md sections 2 to 5). Hooks.g.cs is written against it;
// round K3 implements it over the old engine's state (listener snapshots, handler table, hook actions).
namespace Sts2Emulator.Core2;

// ------------------------------------------------------------------------------------------- kernel types
// What a dispatcher's arguments are, in the kernel's own terms (tools/kernel/hooks.json "args"). Opaque handles:
// handlers never look inside them; Core2/OldState.cs (K2/K3) is the only code that maps them to the old engine.

/// <summary>A card in the combat, by its per-combat uid (K2b).</summary>
public readonly record struct CardHandle(int Uid);
/// <summary>A creature in the combat, by roster slot.</summary>
public readonly record struct CreatureRef(int Index);
/// <summary>A player, by seat.</summary>
public readonly record struct PlayerRef(int Index);
/// <summary>A power instance: its owner, its model and the instance among that owner's powers of the model.</summary>
public readonly record struct PowerRef(CreatureRef Owner, int ModelId, int Instance);
public readonly record struct OrbRef(PlayerRef Owner, int Slot);
public readonly record struct PotionRef(PlayerRef Owner, int Slot);
/// <summary>A card play in flight (its card, target and play index live in the runtime).</summary>
public readonly record struct CardPlayRef(int Id);
/// <summary>An attack command in flight.</summary>
public readonly record struct AttackRef(int Id);
/// <summary>The result of one damage event.</summary>
public readonly record struct DamageResultRef(int Id);
/// <summary>The value properties of an amount (bits mapped by OldState).</summary>
public readonly record struct ValueProp(int Bits);
public enum Side : byte { Player, Enemy }
public readonly record struct AutoPlay(int Kind);
public readonly record struct PreviewMode(int Kind);
public readonly record struct PileKind(int Kind);
public readonly record struct PilePos(int Kind);
/// <summary>Where a played card goes: pile and position.</summary>
public readonly record struct PileDest(PileKind Pile, PilePos Position);
/// <summary>A game-only object the old engine never models (map, rewards, merchant, events): passed through.</summary>
public readonly record struct Opaque(object? Value);

/// <summary>Run-time pass selector of ModifyDamage (a "when" of its passes and steps).</summary>
[Flags]
public enum ModifyDamageHookType { None = 0, Additive = 1, Multiplicative = 2, Cap = 4, All = Additive | Multiplicative | Cap }

/// <summary>Run-time pass selector of ModifyHpLost.</summary>
[Flags]
public enum HpLossHookPhase { None = 0, BeforeOsty = 1, AfterOsty = 2, All = BeforeOsty | AfterOsty }

/// <summary>One hook listener (a power, relic, card, monster ...) as K2's snapshot names it.</summary>
public readonly record struct ListenerRef(int Kind, int ModelId, int Instance);

/// <summary>A player choice context: the stack of models whose code is running inside it.</summary>
public sealed class ChoiceCtx
{
    private readonly List<ListenerRef> _models = new();

    public ChoiceCtx(HookId dispatcher = HookId.None, ListenerRef? owner = null)
    {
        Dispatcher = dispatcher;
        Owner = owner;
    }

    /// <summary>The dispatcher of a per-listener hook context; None for a caller's own context.</summary>
    public HookId Dispatcher { get; }
    /// <summary>The listener a per-listener hook context belongs to.</summary>
    public ListenerRef? Owner { get; }
    public IReadOnlyList<ListenerRef> Models => _models;

    public void PushModel(ListenerRef model) => _models.Add(model);

    public void PopModel(ListenerRef model)
    {
        if (_models.Count == 0 || _models[^1] != model)
            throw new InvalidOperationException($"choice context popped {model} out of order");
        _models.RemoveAt(_models.Count - 1);
    }
}

// ------------------------------------------------------------------------------------------- dispatch shapes

public enum HookIterator : byte { Combat, CombatDirect, RunCombat, Run }

public enum ContextMode : byte { None, Passed, PassedUnpushed, CallerHook, PerListener }

/// <summary>The typed continuation of a handler that stopped at a choice; one record type per resume site (K4).</summary>
public abstract record HookResume;

/// <summary>What an event handler returns: done, or paused at a choice with the request and its resume record.</summary>
public readonly record struct HookStep(object? Request, HookResume? Resume)
{
    public static HookStep Done => default;
    public static HookStep Paused(object request, HookResume resume) => new(request, resume);
    public bool IsPaused => Request is not null;
}

/// <summary>Where a pause happened: the dispatcher, the hook virtual and pass, the listener and its context.</summary>
public readonly record struct PauseSite(HookId Dispatcher, HookId Hook, int Pass, ListenerRef Who, ContextMode Context,
    ChoiceCtx? Ctx);

/// <summary>One entry of a pass's listener snapshot: the listener and its handler for the pass's hook.</summary>
public readonly record struct HookListener<THandler>(ListenerRef Who, THandler Fn) where THandler : Delegate;

// Handler shapes, one per pass op (section 4). TArgs is the dispatcher's generated argument record.
public delegate HookStep HookEvent<TArgs>(ListenerRef self, in TArgs a, ChoiceCtx? ctx) where TArgs : struct;
/// <summary>fold-add returns the addend, fold-mul the factor, fold-set the new value.</summary>
public delegate TValue HookFold<TArgs, TValue>(ListenerRef self, in TArgs a, TValue current) where TArgs : struct;
/// <summary>fold-cap: the listener's cap (the pass keeps the running minimum).</summary>
public delegate TValue HookCap<TArgs, TValue>(ListenerRef self, in TArgs a) where TArgs : struct;
/// <summary>try: whether the listener changed the value, and the value.</summary>
public delegate bool HookTry<TArgs, TValue>(ListenerRef self, in TArgs a, TValue current, out TValue next)
    where TArgs : struct;
/// <summary>try without a value: the listener changes its arguments in place and says whether it did.</summary>
public delegate bool HookTryInPlace<TArgs>(ListenerRef self, in TArgs a) where TArgs : struct;
public delegate bool HookPredicate<TArgs>(ListenerRef self, in TArgs a) where TArgs : struct;
public delegate void HookMutate<TArgs>(ListenerRef self, in TArgs a) where TArgs : struct;

/// <summary>A dispatcher's passes as one call, for a step that runs them more than once (preview-hittable).</summary>
public delegate decimal HookPasses<TArgs>(HookRuntime rt, in TArgs a, decimal value,
    out IReadOnlyList<ListenerRef> changed) where TArgs : struct;

/// <summary>
/// The runtime Hooks.g.cs calls. One per engine instance; round K3 implements it over the old state.
/// </summary>
public abstract class HookRuntime
{
    /// <summary>Whether any switched-on listener handles this hook virtual (the dispatcher's zero-listener fast path).</summary>
    public abstract bool HasAny(HookId hook);

    /// <summary>
    /// The pass's listener snapshot, in game order, of the listeners with a handler for <paramref name="hook"/>, taken
    /// at pass start (the combat iterator's guard read once; run+combat builds its combat part after the deck part).
    /// Membership at visit time is the snapshot's business: an entry that has left the combat is not yielded.
    /// </summary>
    public abstract IReadOnlyList<HookListener<THandler>> Listeners<THandler>(HookIterator iterator, HookId hook)
        where THandler : Delegate;

    /// <summary>A per-listener hook choice context (the game's HookPlayerChoiceContext).</summary>
    public abstract ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who);

    /// <summary>
    /// A handler paused at a choice. Per-listener contexts: record a hook action and return, the dispatch moves on.
    /// Passed contexts: pause the action that owns the context (old replay where the caller cannot resume).
    /// </summary>
    public abstract void OnPaused(in PauseSite site, HookStep step);

    /// <summary>A mark in the hook-action sequence, taken at dispatcher start when it has wait points.</summary>
    public abstract long HookActionMark();

    /// <summary>A wait_all point: drain the hook actions this dispatch started since the mark, in id order.</summary>
    public abstract void WaitAll(HookId dispatcher, long mark);

    public static bool Contains(IReadOnlyList<ListenerRef> set, ListenerRef who)
    {
        for (int i = 0; i < set.Count; i++)
            if (set[i] == who)
                return true;
        return false;
    }

    /// <summary>The changed set a pass collected (an empty one when nothing changed).</summary>
    public static IReadOnlyList<ListenerRef> Changed(List<ListenerRef>? changed) =>
        changed is null ? Array.Empty<ListenerRef>() : changed;
}
