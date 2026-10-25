// The handler table (docs/rebuild/TIER1_KERNEL.md section 4): per hook virtual, the handlers of the migrated models,
// keyed by the game's model (its class name, as data/id_map.json names it). The old state's model id is resolved through
// IModelIds (OldModelIds, Core2/OldState.cs) and nowhere else. A handler is registered only when the switch of its hook's
// family is on, so a switched-off group has no handler for any listener and its dispatchers take the fast path.
namespace Sts2Emulator.Core2;

/// <summary>Resolves a game model (kind and class name) to the model id the old state lists (OldModelIds).</summary>
public interface IModelIds
{
    int Resolve(ListenerKind kind, string gameId);
}

/// <summary>A migrated model: one static class under Core2/Models/ whose Register adds its handlers. No model implements it
/// now: the house layout is a static class with a <c>public static void Register(HandlerTable t)</c> (a static class cannot
/// implement an interface), listed in Registry.cs as a method group. It stays for <see cref="HandlerTable.Register{TModel}" />,
/// which has no caller either.</summary>
public interface IMigratedModel
{
    static abstract void Register(HandlerTable table);
}

/// <summary>The hooks one model handles, as a bit set over <see cref="HookId" />.</summary>
public readonly struct HookMask
{
    private readonly ulong[]? _bits;

    internal HookMask(ulong[]? bits) => _bits = bits;

    public bool Has(HookId hook) =>
        _bits is not null && (int)hook / 64 < _bits.Length && (_bits[(int)hook / 64] & (1UL << ((int)hook % 64))) != 0;

    public bool IsEmpty => _bits is null || Array.TrueForAll(_bits, b => b == 0);

    public IEnumerable<HookId> Hooks
    {
        get
        {
            if (_bits is null)
                yield break;
            for (int i = 0; i < _bits.Length * 64; i++)
                if ((_bits[i / 64] & (1UL << (i % 64))) != 0)
                    yield return (HookId)i;
        }
    }
}

public sealed class HandlerTable : IHandlerSource
{
    private static readonly int Words = (Enum.GetValues<HookId>().Max(h => (int)h) / 64) + 1;
    private readonly IModelIds _ids;
    private readonly Func<Switch, bool> _on;
    private readonly Dictionary<(HookId Hook, ListenerKind Kind, int Model), Delegate> _handlers = new();
    private readonly Dictionary<(ListenerKind Kind, int Model), ulong[]> _masks = new();
    private readonly Dictionary<(ListenerKind Kind, int Model), string> _names = new();
    private readonly HashSet<(HookId Hook, ListenerKind Kind)> _kinds = new();
    private readonly HashSet<(HookId Hook, ListenerKind Kind, int Model)> _stubs = new();
    private readonly Dictionary<(ListenerKind Kind, int Model), List<string>> _inheritors = new();
    private readonly bool[] _any = new bool[Words * 64];

    /// <param name="ids">The game-model to old-id resolution (OldModelIds).</param>
    /// <param name="on">Whether a family's switch is on (Kernel.On).</param>
    public HandlerTable(IModelIds ids, Func<Switch, bool> on)
    {
        _ids = ids;
        _on = on;
    }

    /// <summary>Handlers registered (switched-on ones only).</summary>
    public int Count => _handlers.Count;

    /// <summary>Registers a migrated model's handlers (each only if its hook's switch is on).</summary>
    public HandlerTable Register<TModel>() where TModel : IMigratedModel
    {
        TModel.Register(this);
        return this;
    }

    /// <summary>The registration scope of one game model: <c>t.Model(kind, "Akabeko").On(hook, handler)...</c>.</summary>
    public ModelRegistration Model(ListenerKind kind, string gameId) => new(this, kind, gameId, _ids.Resolve(kind, gameId));

    /// <summary>
    /// Adds a handler for <paramref name="hook" /> (a hook virtual) of a model. Registered only when the switch of the
    /// hook's family is on; a hook no family holds, a dispatcher id, or a second handler for the same (hook, model) is
    /// an error. Returns whether it was registered.
    /// </summary>
    internal bool Add(ListenerKind kind, string gameId, int model, HookId hook, Delegate fn)
    {
        ArgumentNullException.ThrowIfNull(fn);
        if (hook.ToString().StartsWith("Dispatch_", StringComparison.Ordinal))
            throw new ArgumentException($"{gameId}: {hook} is a dispatcher, not a hook virtual");
        var family = SwitchGroups.FamilyOf(hook);
        if (family == Switch.None)
            throw new ArgumentException($"{gameId}: hook {hook} is in no family (tools/fidelity/hook_families.json)");
        if (!_on(family))
            return false;
        if (_names.TryGetValue((kind, model), out var other) && other != gameId)
            throw new ArgumentException($"{kind} {gameId} and {other} resolve to the same old id {model}");
        if (!_handlers.TryAdd((hook, kind, model), fn))
            throw new ArgumentException($"{kind} {gameId} already has a handler for {hook}");
        if (fn.Method.IsDefined(typeof(StubHandlerAttribute), false))
            _stubs.Add((hook, kind, model));
        _names[(kind, model)] = gameId;
        if (!_masks.TryGetValue((kind, model), out var bits))
            _masks[(kind, model)] = bits = new ulong[Words];
        bits[(int)hook / 64] |= 1UL << ((int)hook % 64);
        _kinds.Add((hook, kind));
        _any[(int)hook] = true;
        return true;
    }

    /// <summary>Whether any registered handler handles this hook virtual (the dispatchers' fast path).</summary>
    public bool HasAny(HookId hook) => _any[(int)hook];

    public Delegate? Find(HookId hook, ListenerKind kind, int modelId) =>
        _handlers.GetValueOrDefault((hook, kind, modelId));

    public bool AnyOf(HookId hook, ListenerKind kind) => _kinds.Contains((hook, kind));

    /// <summary>The hooks a model handles (by its old-state id, as a listener names it).</summary>
    public HookMask Mask(ListenerKind kind, int modelId) => new(_masks.GetValueOrDefault((kind, modelId)));

    /// <summary>The hooks a model handles (by its game name).</summary>
    public HookMask Mask(ListenerKind kind, string gameId) => Mask(kind, _ids.Resolve(kind, gameId));

    /// <summary>The game name a model was registered under.</summary>
    public string? GameId(ListenerKind kind, int modelId) => _names.GetValueOrDefault((kind, modelId));

    /// <summary>
    /// Declares game models that inherit every override of the model registered under <paramref name="model" /> and that
    /// the old state lists under that model's id (the three Decimillipede segments are one old enemy id; the game's logic is
    /// in their base class). Their listeners are the base's for dispatch; <see cref="Registered" /> names each of them too.
    /// </summary>
    internal void AddInheritors(ListenerKind kind, int model, IEnumerable<string> gameIds)
    {
        if (!_inheritors.TryGetValue((kind, model), out var names))
            _inheritors[(kind, model)] = names = new();
        foreach (var g in gameIds)
            if (!names.Contains(g))
                names.Add(g);
    }

    /// <summary>
    /// Every registered (model, hook) pair, in kind, name, hook order, with whether the handler is a stub; a model's
    /// inheritors (<see cref="AddInheritors" />) are listed with each of its pairs. This is what
    /// tools/kernel/port_list.py reads (EmuDump kernel-handlers): the table itself, not the source text.
    /// </summary>
    public IReadOnlyList<RegisteredHandler> Registered() =>
        _handlers.Keys
            .SelectMany(k => new[] { _names[(k.Kind, k.Model)] }
                .Concat(_inheritors.GetValueOrDefault((k.Kind, k.Model)) ?? [])
                .Select(name => new RegisteredHandler(k.Kind, name, k.Hook, _stubs.Contains(k))))
            .OrderBy(r => r.Kind).ThenBy(r => r.GameId, StringComparer.Ordinal).ThenBy(r => r.Hook.ToString(), StringComparer.Ordinal)
            .ToList();
}

/// <summary>One registered handler: the game model, the hook virtual, and whether the handler is a stub.</summary>
public readonly record struct RegisteredHandler(ListenerKind Kind, string GameId, HookId Hook, bool Stub);

/// <summary>
/// Marks a handler method that is a placeholder: registered so the model's shape is in place, its body only throws
/// NotImplementedException. The port list does not count a stub as ported (HandlerTable.Registered says which are).
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class StubHandlerAttribute : Attribute;

/// <summary>One model's registrations (see <see cref="HandlerTable.Model" />).</summary>
public readonly struct ModelRegistration(HandlerTable table, ListenerKind kind, string gameId, int model)
{
    public ListenerKind Kind => kind;
    public string GameId => gameId;
    /// <summary>The old state's id of the model (for tests and reports; handlers never read it).</summary>
    public int OldId => model;

    /// <summary>Adds the handler of one hook virtual (when its switch is on).</summary>
    public ModelRegistration On(HookId hook, Delegate handler)
    {
        table.Add(kind, gameId, model, hook, handler);
        return this;
    }

    /// <summary>The game models that inherit this model's overrides and that the old state does not tell apart from it
    /// (one old id): each is listed as handled by this model's handlers (<see cref="HandlerTable.AddInheritors" />).</summary>
    public ModelRegistration InheritedBy(params string[] gameIds)
    {
        table.AddInheritors(kind, model, gameIds);
        return this;
    }
}
