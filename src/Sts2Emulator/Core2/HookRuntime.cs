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

    /// <summary>An independent context with the same owner and the same models pushed (a copy of the state holding it).</summary>
    internal ChoiceCtx Copy()
    {
        var c = new ChoiceCtx(Dispatcher, Owner);
        c._models.AddRange(_models);
        return c;
    }

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

/// <summary>
/// The typed continuation of a handler that stopped at a choice; one record type per resume site (K4). Plain data: it
/// holds what the rest of the body needs (handles, amounts), never a live object, so a copy of the state holding it can
/// share it. <see cref="Continue" /> runs the rest of the body with the player's answer (the candidate index, or the
/// candidate count for a decline); it may pause again at a later choice of the same body.
/// </summary>
public abstract record HookResume
{
    /// <summary>The resume site's name, part of a keyed answer's key (HookChoiceReplay); the record type by default.</summary>
    public virtual string ResumeSite => GetType().Name;

    /// <summary>The rest of the handler's body, given the answer.</summary>
    public virtual HookStep Continue(HookRuntime rt, int answer) =>
        throw new NotSupportedException($"{ResumeSite} has no continuation");
}

/// <summary>
/// A pass's listener snapshot as plain data (K4): the listeners of its first part, of its late part when that was built,
/// and the visit index of the entry that paused. A paused dispatch keeps this, never the live cursor, so a copy of the
/// state mid-pause carries no closure over the source's state; <see cref="HookRuntime.Reopen{THandler}" /> rebuilds the
/// cursor over the state it resumes on.
/// </summary>
public sealed record PassSnapshot(ListenerRef[] First, ListenerRef[]? Late, int Index);

/// <summary>A cursor that can say where it stands as plain data.</summary>
public interface IPassCursor
{
    PassSnapshot Snapshot();
}

/// <summary>What an event handler returns: done, or paused at a choice with the request and its resume record.</summary>
public readonly record struct HookStep(object? Request, HookResume? Resume)
{
    public static HookStep Done => default;
    public static HookStep Paused(object request, HookResume resume) => new(request, resume);
    public bool IsPaused => Request is not null;
}

/// <summary>
/// Where a pause happened: the dispatcher, the hook virtual, the pass and the listener's index in that pass's visit
/// order (<see cref="HookCursor{THandler}.Index"/>), the listener and its context. (Pass, Index) is where a dispatch
/// stopped by the pause continues (K4's flow pause and resume).
/// </summary>
public readonly record struct PauseSite(HookId Dispatcher, HookId Hook, int Pass, int Index, ListenerRef Who,
    ContextMode Context, ChoiceCtx? Ctx);

/// <summary>
/// What an event dispatcher whose context awaits each listener's whole task returns (every context but
/// per-listener): done, or stopped at the listener that paused, with the pass's cursor positioned after it. The
/// later listeners run only through the dispatcher's <c>Resume</c> method, called once the paused listener's resume
/// record has run to completion.
/// </summary>
public readonly record struct HookStatus(PauseSite? Site, HookStep Step, PassSnapshot? Snapshot)
{
    public static HookStatus Done => default;

    /// <summary>A pause at <paramref name="site" />; the pass's cursor is kept as its plain-data snapshot.</summary>
    public static HookStatus Paused(in PauseSite site, HookStep step, IPassCursor cursor) => new(site, step, cursor.Snapshot());

    public bool IsPaused => Site is not null;

    /// <summary>The pause site, checked to be a pause of <paramref name="dispatcher"/> (a Resume call's guard).</summary>
    public PauseSite At(HookId dispatcher)
    {
        if (Site is not { } s || s.Dispatcher != dispatcher || Snapshot is null)
            throw new InvalidOperationException($"{dispatcher} resumed from {(Site is { } x ? x.Dispatcher : "no pause")}");
        return s;
    }
}

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
/// <summary>try whose condition also needs the listener's new value to be present ("also": out-not-null): a true
/// answer with no value changes nothing.</summary>
public delegate bool HookTryOrNull<TArgs, TValue>(ListenerRef self, in TArgs a, TValue current, out TValue? next)
    where TArgs : struct where TValue : struct;
/// <summary>try without a value: the listener changes its arguments in place and says whether it did.</summary>
public delegate bool HookTryInPlace<TArgs>(ListenerRef self, in TArgs a) where TArgs : struct;
public delegate bool HookPredicate<TArgs>(ListenerRef self, in TArgs a) where TArgs : struct;
/// <summary>mutate: the listener changes a mutable argument in place (a list it may reorder, a set, a game object).</summary>
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
    /// The pass's listeners with a handler for <paramref name="hook"/>, as a cursor in game order (section 3): a
    /// snapshot taken at pass start (the combat iterator's guard read once), run+combat building its combat part only
    /// when the deck part ends, and each entry checked for membership when it is visited.
    /// </summary>
    public abstract HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook)
        where THandler : Delegate;

    /// <summary>
    /// The cursor of a paused pass, rebuilt over the state the dispatch resumes on (K4): the listeners of the pass's
    /// snapshot in its order, each taken from a fresh listing of the same hook (a listener the fresh listing lacks has
    /// left the combat and is skipped; one it adds was added mid-pass and is not visited), positioned after the paused
    /// entry. A late part not yet built is built when the first part ends, as in the first visit.
    /// </summary>
    public virtual HookCursor<THandler> Reopen<THandler>(HookIterator iterator, HookId hook, in HookStatus paused)
        where THandler : Delegate =>
        HookCursor<THandler>.Reopen(Listeners<THandler>(iterator, hook),
            paused.Snapshot ?? throw new InvalidOperationException($"{hook}: no paused pass to reopen"));

    /// <summary>A per-listener hook choice context (the game's HookPlayerChoiceContext).</summary>
    public abstract ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who);

    /// <summary>
    /// A handler paused at a choice. Per-listener contexts: record a hook action; the dispatch moves on to the next
    /// listener. Every other context: the dispatch stops after this call and returns a paused
    /// <see cref="HookStatus"/>; pause the action that owns the context (old replay where the caller cannot resume).
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

/// <summary>
/// One pass's listeners, visited in game order (TIER1_KERNEL.md section 3). The first part is the snapshot taken at
/// pass start; <c>late</c>, when given, builds the second part once the first is exhausted (run+combat: the run's deck
/// part, then the combat's list built when the deck part ends); <c>member</c> is the visit-time membership check (an
/// entry that has left the combat, or whose owner is no longer active for hooks, is skipped). A listener added
/// mid-pass is in neither part, so it is not visited.
/// </summary>
public sealed class HookCursor<THandler> : IPassCursor where THandler : Delegate
{
    private IReadOnlyList<HookListener<THandler>> _part;
    private Func<IReadOnlyList<HookListener<THandler>>>? _late;
    private readonly Func<ListenerRef, bool>? _member;
    private readonly IReadOnlyList<HookListener<THandler>> _first;
    private IReadOnlyList<HookListener<THandler>>? _second;
    private int _at = -1, _base;

    public HookCursor(IReadOnlyList<HookListener<THandler>> first,
        Func<IReadOnlyList<HookListener<THandler>>>? late = null, Func<ListenerRef, bool>? member = null)
    {
        _part = first;
        _first = first;
        _late = late;
        _member = member;
    }

    /// <summary>Where this cursor stands, as plain data (the listeners of each part built so far and the index).</summary>
    public PassSnapshot Snapshot() =>
        new(_first.Select(e => e.Who).ToArray(), _second?.Select(e => e.Who).ToArray(), Index);

    /// <summary>
    /// <paramref name="snap" />'s pass over <paramref name="fresh" />, an unvisited cursor of the same hook over the
    /// state now: the snapshot's listeners in its order with the fresh handlers and membership, positioned after the
    /// paused entry (see <see cref="HookRuntime.Reopen{THandler}" />).
    /// </summary>
    public static HookCursor<THandler> Reopen(HookCursor<THandler> fresh, PassSnapshot snap)
    {
        if (fresh._at != -1)
            throw new InvalidOperationException("reopen needs an unvisited cursor");
        var gone = new HashSet<ListenerRef>();
        var first = Map(snap.First, fresh._first, gone);
        IReadOnlyList<HookListener<THandler>>? second = null;
        var late = fresh._late;
        if (snap.Late is { } built)
        {
            second = Map(built, late?.Invoke() ?? [], gone);
            late = null;
        }

        var member = fresh._member;
        var c = new HookCursor<THandler>(first, second is null ? late : () => second,
            who => !gone.Contains(who) && (member is null || member(who)));
        c.Index = snap.Index;
        if (snap.Index < first.Count)
        {
            c._at = snap.Index;
        }
        else
        {
            c._second = second!;
            c._part = second!;
            c._late = null;
            c._base = first.Count;
            c._at = snap.Index - first.Count;
        }

        return c;
    }

    private static List<HookListener<THandler>> Map(ListenerRef[] refs, IReadOnlyList<HookListener<THandler>> now,
        HashSet<ListenerRef> gone)
    {
        var by = new Dictionary<ListenerRef, HookListener<THandler>>();
        foreach (var e in now)
            by.TryAdd(e.Who, e);
        var list = new List<HookListener<THandler>>(refs.Length);
        foreach (var r in refs)
        {
            if (by.TryGetValue(r, out var e))
            {
                list.Add(e);
            }
            else
            {
                gone.Add(r);
                list.Add(new HookListener<THandler>(r, null!));
            }
        }

        return list;
    }

    /// <summary>A cursor over no listeners.</summary>
    public static HookCursor<THandler> Empty() => new(Array.Empty<HookListener<THandler>>());

    /// <summary>The position of the entry <see cref="Next"/> last returned, in the pass's visit order (both parts,
    /// skipped entries counted); -1 before the first.</summary>
    public int Index { get; private set; } = -1;

    /// <summary>The next listener that is a member when visited; false at the end of the pass.</summary>
    public bool Next(out HookListener<THandler> listener)
    {
        while (true)
        {
            if (++_at >= _part.Count)
            {
                if (_late is null)
                {
                    _at = _part.Count;
                    listener = default;
                    return false;
                }
                _base += _part.Count;
                _part = _second = _late();
                _late = null;
                _at = -1;
                continue;
            }
            var e = _part[_at];
            if (_member is not null && !_member(e.Who))
                continue;
            Index = _base + _at;
            listener = e;
            return true;
        }
    }
}
