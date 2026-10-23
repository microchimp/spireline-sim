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
        var entries = new Dictionary<ListenerRef, OldListener>();
        var view = new PassView { Gap = g => Flush([g]) };
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
                    Flush(Enum.GetValues<ListenerKind>()
                        .Select(k => new OldGap(NotModelledKind.SetupUndeclared, k, -1, 0, "guard set before turn 1"))
                        .ToList());
                }

                return HookCursor<T>.Empty();
            case HookIterator.Combat:
            case HookIterator.CombatDirect:
                first = combat is null ? [] : Build((into, gaps) => OldState.Combat(World, into, gaps));
                break;
            case HookIterator.RunCombat when combat is not null:
                first = Build((into, gaps) => OldState.Run(World, into, gaps, withRunItems: false));
                late = () => World.Combat is null ? [] : Build((into, gaps) => OldState.Combat(World, into, gaps));
                break;
            default:
                first = Build((into, gaps) => OldState.Run(World, into, gaps, withRunItems: true));
                break;
        }

        return new HookCursor<T>(first, late,
            who => entries.TryGetValue(who, out var e) && OldState.IsMember(World, e, view));

        List<HookListener<T>> Build(Action<List<OldListener>, List<OldGap>> part)
        {
            if (World.Combat is { } c)
            {
                // The uid invariant's safety net (CardUidInvariant): a card some creation site left without a uid
                // gets one before it is named by a listener.
                CardUidInvariant.Normalise(c);
            }

            var raw = new List<OldListener>();
            var gaps = new List<OldGap>();
            part(raw, gaps);
            var list = new List<HookListener<T>>(raw.Count);
            bool orbs = false, deck = false;
            foreach (var e in raw)
            {
                switch (Handlers.Find(hook, e.Kind, e.Who.ModelId))
                {
                    case null:
                        continue;
                    case T fn:
                        if (!entries.TryAdd(e.Who, e))
                        {
                            throw new InvalidOperationException($"{hook}: listener {e.Who} listed twice in one pass");
                        }

                        list.Add(new HookListener<T>(e.Who, fn));
                        orbs |= e.Kind == ListenerKind.Orb;
                        deck |= e.Pile == OldPile.Deck;
                        break;
                    case var other:
                        throw new InvalidOperationException(
                            $"{hook}: the handler of {e.Kind} {e.Who.ModelId} is {other.GetType().Name}, not {typeof(T).Name}");
                }
            }

            // The snapshot's change probes: orbs and deck cards have no identity, so any later change is reported.
            if (orbs && World.Combat is { } cs)
            {
                view.Orbs = new ChangeProbe<OrbState>(cs.Orbs);
            }

            if (deck && World.Run is { } rs)
            {
                view.Deck = new ChangeProbe<CardInstance>(rs.Deck);
            }

            Flush(gaps);
            return list;
        }

        // Reports the gaps a handler could stand for. A gap seen at a visit can belong to a pass that is no longer the
        // latest (a handler ran another pass): Report always gets it, LastPass only while the pass is the latest.
        void Flush(List<OldGap> gaps)
        {
            foreach (var gap in gaps)
            {
                bool needed = gap.ModelId != 0
                    ? Handlers.Find(hook, gap.Kind, gap.ModelId) is not null
                    : Handlers.AnyOf(hook, gap.Kind);
                if (needed)
                {
                    var nm = new NotModelled(pass, hook, gap);
                    if (pass == Passes)
                    {
                        _last.Add(nm);
                    }

                    Report?.Invoke(nm);
                }
            }
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
