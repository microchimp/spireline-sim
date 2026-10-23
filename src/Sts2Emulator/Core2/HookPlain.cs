// The plain-data rule of a hook step (docs/rebuild/TIER1_KERNEL.md section 5): a paused handler's request and its
// resume record are shared by reference when the state holding them is copied (KernelState.CopyTo), so they hold
// handles and amounts, never a live object of the state they came from (the combat, a creature, a card, a list of
// cards): a copy resuming through a shared record would act on its source's objects. HookStep.Paused checks it in a
// debug build; a test checks it in every build.
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static class HookPlain
{
    // The game's state types: live objects even where they are records.
    private static readonly HashSet<Type> Live =
        [typeof(CombatState), typeof(EnemyState), typeof(CardInstance), typeof(Sts2Emulator.Core.Run.RunState),
         typeof(HookRuntime), typeof(ChoiceCtx), typeof(PendingCardSelection)];

    // Collections a record may hold, when their element types are plain.
    private static readonly HashSet<Type> Collections =
        [typeof(List<>), typeof(IList<>), typeof(IReadOnlyList<>), typeof(IEnumerable<>), typeof(IReadOnlyCollection<>),
         typeof(HashSet<>), typeof(Dictionary<,>), typeof(KeyValuePair<,>), typeof(System.Collections.Immutable.ImmutableArray<>)];

    private static readonly ConcurrentDictionary<Type, string?> Verdicts = new();

    /// <summary>Debug builds only: throws when the request or the resume record is not plain data.</summary>
    [Conditional("DEBUG")]
    public static void Check(object? request, HookResume? resume)
    {
        if ((Violation(request) ?? Violation(resume)) is { } why)
            throw new InvalidOperationException($"hook step holds a live object: {why}");
    }

    /// <summary>Why <paramref name="value" /> is not plain data (a path and the offending type), or null when it is plain:
    /// null, a primitive, string, enum, decimal; a struct or a record whose members are plain; an array or collection of
    /// plain elements. A member declared as object or an interface is judged by the value it holds.</summary>
    public static string? Violation(object? value)
    {
        if (value is null)
            return null;
        var t = value.GetType();
        return Type(t) is { } why ? $"{t.Name}: {why}" : Dynamic(value, t.Name, 0);
    }

    // Members of an open type (object, an interface, an abstract record) are judged by their runtime values.
    private static string? Dynamic(object value, string path, int depth)
    {
        if (depth > 8 || value is string || value is Delegate)
            return value is Delegate ? $"{path} is a delegate" : null;
        var t = value.GetType();
        if (t.IsPrimitive || t.IsEnum)
            return null;
        if (value is IEnumerable items and not string)
        {
            int i = 0;
            foreach (var item in items)
            {
                if (item is not null && (Type(item.GetType()) is { } why ? $"{path}[{i}]: {why}" : Dynamic(item, $"{path}[{i}]", depth + 1)) is { } bad)
                    return bad;
                i++;
            }

            return null;
        }

        foreach (var f in Fields(t).Where(f => IsOpen(f.FieldType)))
        {
            if (f.GetValue(value) is { } v
                && (Type(v.GetType()) is { } why ? $"{path}.{f.Name}: {why}" : Dynamic(v, $"{path}.{f.Name}", depth + 1)) is { } bad)
                return bad;
        }

        return null;
    }

    private static bool IsOpen(Type t) => t == typeof(object) || t.IsInterface || t.IsAbstract;

    private static IEnumerable<FieldInfo> Fields(Type t) =>
        t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy)
            .Concat(t.BaseType is { } b && b != typeof(object) && b != typeof(ValueType) ? Fields(b) : []);

    /// <summary>The static verdict on a type: null when its declared members are plain.</summary>
    private static string? Type(Type t) => Verdicts.GetOrAdd(t, x => Judge(x, []));

    private static string? Judge(Type t, HashSet<Type> seen)
    {
        if (t.IsPrimitive || t.IsEnum || t == typeof(string) || t == typeof(decimal) || IsOpen(t))
            return null;
        if (Live.Contains(t) || typeof(Delegate).IsAssignableFrom(t))
            return $"{t.Name} is a live type";
        if (!seen.Add(t))
            return null;
        if (t.IsArray)
            return Judge(t.GetElementType()!, seen);
        if (Nullable.GetUnderlyingType(t) is { } inner)
            return Judge(inner, seen);
        if (t.IsGenericType && Collections.Contains(t.GetGenericTypeDefinition()))
            return t.GetGenericArguments().Select(a => Judge(a, seen)).FirstOrDefault(r => r is not null);
        bool record = !t.IsValueType && t.GetMethod("<Clone>$") is not null;
        if (!t.IsValueType && !record)
            return $"{t.Name} is a reference type that is not a record";
        if (!t.IsValueType && t.Namespace == typeof(CombatState).Namespace)
            return $"{t.Name} is a game state type";
        foreach (var f in Fields(t))
        {
            if (Judge(f.FieldType, seen) is { } why)
                return $"{f.Name}: {why}";
        }

        return null;
    }
}
