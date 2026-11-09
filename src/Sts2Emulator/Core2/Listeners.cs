// The listener cursor over the old state (docs/rebuild/TIER1_KERNEL.md section 3): a pass's snapshot in the game's
// order, filtered by the handler table, with the game's visit-time membership. HookRuntime.Listeners (K3) calls
// OldListeners.Cursor. The old representation is read only through OldState.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>The kinds of hook listener, as <see cref="ListenerRef.Kind" /> holds them.</summary>
public enum ListenerKind : byte
{
    Power,
    Monster,
    Relic,
    Potion,
    Orb,
    Card,
    Affliction,
    Enchantment,
    Modifier,
    Badge,
    Scaling,
}

/// <summary>The game's afflictions (MegaCrit.Sts2.Core.Models.Afflictions) by model id; the old state holds Bound only.</summary>
public enum GameAffliction
{
    None,
    Bound,
    Entangled,
    Galvanized,
    Hexed,
    Ringing,
    Smog,
    Tainted,
}

/// <summary>
/// Which listeners handle a hook virtual: the handler table K3 builds (per hook, per kind and model id). A listener
/// with no handler is not in the pass.
/// </summary>
public interface IHandlerSource
{
    /// <summary>The handler of this model for this hook virtual, or null.</summary>
    Delegate? Find(HookId hook, ListenerKind kind, int modelId);

    /// <summary>Whether any model of this kind handles the hook (a gap whose model is unknown is reported then).</summary>
    bool AnyOf(HookId hook, ListenerKind kind);
}

/// <summary>Every listener handles every hook with one no-op handler (tools and tests of the order itself).</summary>
public sealed class EveryListener : IHandlerSource
{
    public static readonly Action Noop = () => { };
    public static readonly EveryListener Instance = new();

    public Delegate? Find(HookId hook, ListenerKind kind, int modelId) => Noop;

    public bool AnyOf(HookId hook, ListenerKind kind) => true;
}

/// <summary>
/// The kernel's listener cursors over one <see cref="OldWorld" />. Each <see cref="Cursor{T}" /> call is one pass:
/// <list type="bullet">
/// <item>the snapshot is taken when the pass starts, in the game's order (<see cref="OldState.Combat" />), keeping the
/// listeners the handler source has a handler for;</item>
/// <item>the guarded combat iterator reads the old engine's own combat-over flag (<c>IsOverOrEnding</c>) once, at pass
/// start, and is empty when it is set (not during setup); the direct and run iterators have no guard;</item>
/// <item>run+combat lists the run deck first and builds the combat's part only when the deck part ends (game:
/// RunState.IterateHookListeners with a child combat state); with no combat it is the run iterator;</item>
/// <item>each entry is checked when it is visited (<see cref="OldState.IsMember" />, game: CombatState.Contains): a
/// power removed from its owner is still visited, a creature removed or a player no longer active is not, a card is
/// found by uid wherever it went;</item>
/// <item>a listener added during the pass is in no snapshot, so it is not visited;</item>
/// <item>gaps of the old state (<see cref="NotModelledKind" />) that a handler could stand for are reported per pass.</item>
/// </list>
/// </summary>
public sealed class OldListeners(OldWorld world, IHandlerSource handlers)
{
    private readonly List<NotModelled> _last = [];

    public OldWorld World { get; } = world;
    public IHandlerSource Handlers { get; } = handlers;

    /// <summary>Called once per gap per pass, as the part holding it is built.</summary>
    public Action<NotModelled>? Report { get; set; }

    /// <summary>Passes started so far (each cursor is one pass; the serial is the gap's <see cref="NotModelled.Pass" />).</summary>
    public int Passes { get; private set; }

    /// <summary>The gaps reported for the latest pass (both parts once the late one is built).</summary>
    public IReadOnlyList<NotModelled> LastPass => _last;

    /// <summary>One pass's listeners for <paramref name="hook" /> in the game's order.</summary>
    public HookCursor<T> Cursor<T>(HookIterator iterator, HookId hook) where T : Delegate
    {
        int pass = ++Passes;
        _last.Clear();
        // A kind with no handler for this hook has no listener in the pass and no gap reported (Find and AnyOf are false for
        // it), so the enumeration leaves it out. A source this does not know is asked for everything.
        int kinds = Handlers is HandlerTable table ? table.KindsOf(hook) : OldState.AllKinds;
        var scope = new PassScope(this, hook, pass, kinds);
        var combat = World.Combat;
        Func<IReadOnlyList<HookListener<T>>>? late = null;
        IReadOnlyList<HookListener<T>> first;
        switch (iterator)
        {
            case HookIterator.Combat when combat is not null && !World.InSetup && CombatEngine.IsOverOrEnding(combat):
                if (combat.Turn == 0)
                {
                    // The guard is set before the first turn: the combat may be starting with its setup not declared
                    // (OldWorld.Setup), which the game exempts. The empty pass is not trusted.
                    scope.Flush(Enum.GetValues<ListenerKind>()
                        .Select(k => new OldGap(NotModelledKind.SetupUndeclared, k, -1, 0, "guard set before turn 1"))
                        .ToList());
                }

                return HookCursor<T>.Empty();
            case HookIterator.Combat:
            case HookIterator.CombatDirect:
                first = combat is null ? [] : scope.Build<T>(Part.Combat);
                break;
            case HookIterator.RunCombat when combat is not null:
                first = scope.Build<T>(Part.RunDeck);
                late = () => World.Combat is null ? [] : scope.Build<T>(Part.Combat);
                break;
            default:
                first = scope.Build<T>(Part.RunWithItems);
                break;
        }

        return new HookCursor<T>(first, late, scope.MemberCheck);
    }

    /// <summary>The parts a pass lists: the combat's, the run's deck alone, the run's deck with its relics and potions.</summary>
    private enum Part : byte
    {
        Combat,
        RunDeck,
        RunWithItems,
    }

    // The two scratch lists of a part's enumeration. A part takes them out of the field while it uses them (a pass built from
    // inside a Report callback finds the field empty and allocates its own) and puts them back cleared.
    private List<OldListener>? _rawScratch;
    private List<OldGap>? _gapScratch;

    private List<OldListener> TakeRaw()
    {
        var l = _rawScratch ?? new List<OldListener>(64);
        _rawScratch = null;
        return l;
    }

    private List<OldGap> TakeGaps()
    {
        var l = _gapScratch ?? new List<OldGap>(16);
        _gapScratch = null;
        return l;
    }

    private void Give(List<OldListener> raw, List<OldGap> gaps)
    {
        raw.Clear();
        gaps.Clear();
        _rawScratch ??= raw;
        _gapScratch ??= gaps;
    }

    /// <summary>
    /// One pass's state: its hook, serial and kinds, the listeners its parts kept (<see cref="PassEntries" />, for the
    /// membership check at a visit) and the view that notes what a visit finds changed. Not generic: only the listing of a part
    /// (<see cref="Build{T}" />) knows the handler type.
    /// </summary>
    private sealed class PassScope
    {
        private readonly OldListeners _o;
        private readonly HookId _hook;
        private readonly int _pass;
        private readonly int _kinds;
        private PassEntries? _entries;
        private PassView? _view;

        public PassScope(OldListeners owner, HookId hook, int pass, int kinds)
        {
            (_o, _hook, _pass, _kinds) = (owner, hook, pass, kinds);
            MemberCheck = IsMember;
        }

        /// <summary>The visit-time membership check the cursor calls (one delegate for the pass).</summary>
        public Func<ListenerRef, bool> MemberCheck { get; }

        private bool IsMember(ListenerRef who) =>
            _entries is not null && _entries.TryGet(who, out var e) && OldState.IsMember(_o.World, e, _view);

        /// <summary>The kept listeners of a part, in the game's order, each with the handler of the pass's hook.</summary>
        public IReadOnlyList<HookListener<T>> Build<T>(Part part) where T : Delegate
        {
            var o = _o;
            var world = o.World;
            if (world.Combat is { } c)
            {
                // The uid invariant's safety net (CardUidInvariant): a card some creation site left without a uid
                // gets one before it is named by a listener.
                CardUidInvariant.Normalise(c);
            }

            var raw = o.TakeRaw();
            var gaps = o.TakeGaps();
            List<HookListener<T>>? list = null;
            try
            {
                switch (part)
                {
                    case Part.Combat:
                        OldState.Combat(world, raw, gaps, _kinds);
                        break;
                    case Part.RunDeck:
                        OldState.Run(world, raw, gaps, withRunItems: false, _kinds);
                        break;
                    default:
                        OldState.Run(world, raw, gaps, withRunItems: true, _kinds);
                        break;
                }

                bool orbs = false, deck = false;
                for (int i = 0; i < raw.Count; i++)
                {
                    var e = raw[i];
                    switch (o.Handlers.Find(_hook, e.Kind, e.Who.ModelId))
                    {
                        case null:
                            continue;
                        case T fn:
                            if (!(_entries ??= new()).TryAdd(e))
                            {
                                throw new InvalidOperationException($"{_hook}: listener {e.Who} listed twice in one pass");
                            }

                            (list ??= new List<HookListener<T>>(4)).Add(new HookListener<T>(e.Who, fn));
                            orbs |= e.Kind == ListenerKind.Orb;
                            deck |= e.Pile == OldPile.Deck;
                            break;
                        case var other:
                            throw new InvalidOperationException(
                                $"{_hook}: the handler of {e.Kind} {e.Who.ModelId} is {other.GetType().Name}, not {typeof(T).Name}");
                    }
                }

                // The snapshot's change probes: orbs and deck cards have no identity, so any later change is reported. (The
                // view is made with the first kept listener: a visit is the only reader, and a pass that kept none has none.)
                if (list is not null)
                {
                    _view ??= new PassView { Gap = FlushOne };
                    _view.Snapshot(world, orbs, deck);
                }

                Flush(gaps);
            }
            finally
            {
                o.Give(raw, gaps);
            }

            return list ?? (IReadOnlyList<HookListener<T>>)Array.Empty<HookListener<T>>();
        }

        private void FlushOne(OldGap gap) => Flush([gap]);

        /// <summary>
        /// Reports the gaps a handler could stand for. A gap seen at a visit can belong to a pass that is no longer the
        /// latest (a handler ran another pass): Report always gets it, LastPass only while the pass is the latest.
        /// </summary>
        public void Flush(List<OldGap> gaps)
        {
            var o = _o;
            foreach (var gap in gaps)
            {
                bool needed = gap.ModelId != 0
                    ? o.Handlers.Find(_hook, gap.Kind, gap.ModelId) is not null
                    : o.Handlers.AnyOf(_hook, gap.Kind);
                if (needed)
                {
                    var nm = new NotModelled(_pass, _hook, gap);
                    if (_pass == o.Passes)
                    {
                        o._last.Add(nm);
                    }

                    o.Report?.Invoke(nm);
                }
            }
        }
    }

    /// <summary>
    /// The listeners a pass kept, found by who. A pass keeps a handful, so up to <see cref="LinearUpTo" /> are scanned in an
    /// array; past that a dictionary takes over. Adding one already there is refused, as the dictionary's TryAdd refused it.
    /// </summary>
    private sealed class PassEntries
    {
        private const int LinearUpTo = 16;
        private OldListener[]? _items = new OldListener[8];
        private int _count;
        private Dictionary<ListenerRef, OldListener>? _map;

        public bool TryAdd(in OldListener e)
        {
            if (_map is not null)
            {
                return _map.TryAdd(e.Who, e);
            }

            for (int i = 0; i < _count; i++)
            {
                if (_items![i].Who == e.Who)
                {
                    return false;
                }
            }

            if (_count == LinearUpTo)
            {
                _map = new Dictionary<ListenerRef, OldListener>(LinearUpTo * 2);
                for (int i = 0; i < _count; i++)
                {
                    _map.Add(_items![i].Who, _items[i]);
                }

                _items = null;
                return _map.TryAdd(e.Who, e);
            }

            if (_count == _items!.Length)
            {
                Array.Resize(ref _items, _count * 2);
            }

            _items[_count++] = e;
            return true;
        }

        public bool TryGet(ListenerRef who, out OldListener e)
        {
            if (_map is not null)
            {
                return _map.TryGetValue(who, out e);
            }

            for (int i = 0; i < _count; i++)
            {
                if (_items![i].Who == who)
                {
                    e = _items[i];
                    return true;
                }
            }

            e = default;
            return false;
        }
    }

    /// <summary>
    /// Every listener of one pass in the game's order, unfiltered and with no membership applied (the run+combat
    /// iterator's two parts back to back): for tools and the order tests.
    /// </summary>
    public static List<OldListener> Enumerate(OldWorld w, HookIterator iterator, List<OldGap>? gaps = null)
    {
        var into = new List<OldListener>();
        gaps ??= [];
        if (w.Combat is { } c)
        {
            CardUidInvariant.Normalise(c);
        }

        switch (iterator)
        {
            case HookIterator.Combat:
            case HookIterator.CombatDirect:
                if (w.Combat is not null)
                {
                    OldState.Combat(w, into, gaps);
                }

                break;
            case HookIterator.RunCombat when w.Combat is not null:
                OldState.Run(w, into, gaps, withRunItems: false);
                OldState.Combat(w, into, gaps);
                break;
            default:
                OldState.Run(w, into, gaps, withRunItems: true);
                break;
        }

        return into;
    }
}
