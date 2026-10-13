using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Sts2Emulator.Tests.Invariants;

/// <summary>
/// Every field of an object graph, public and private, as "path=value" lines, and a hash of them:
/// the emulator's whole state, not the part a snapshot keeps.
/// </summary>
/// <remarks>
/// Two states are the same state when their lines are equal. That is what a clone, a replay or a
/// relabelled fight has to give back, and it reaches what a snapshot leaves out (enemy AI scratch
/// fields, per-card modifiers, the generators' own lanes). Reference cycles and shared references
/// print as "-> first path", so a clone that stops sharing an object the original shares, or the
/// other way round, is a difference too.
///
/// Only the runtime type's own declared fields are read for types outside the emulator (a
/// <see cref="Random"/> base class keeps a randomly seeded generator of its own that no code reads).
/// No code runs: fields only, never a property getter, so a state read has no effects.
/// <c>skip</c> lists path prefixes to leave out (".Item1.Hand[" hides the hand's cards but not its count).
/// </remarks>
public static class DeepState
{
    private static readonly Dictionary<Type, FieldInfo[]> FieldCache = new();

    /// <summary>The state as sorted "path=value" lines.</summary>
    public static List<string> Lines(object? root, params string[] skip)
    {
        var sink = new ListSink();
        Walk(root, new Walker(sink, skip));
        sink.Lines.Sort(StringComparer.Ordinal);
        return sink.Lines;
    }

    /// <summary>SHA-256 of the state, as hex: cheap to keep for every step of a fight.</summary>
    public static string Digest(object? root, params string[] skip)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Walk(root, new Walker(new HashSink(sha), skip));
        return Convert.ToHexString(sha.GetHashAndReset());
    }

    /// <summary>The first <paramref name="limit"/> lines that differ, as "path: a | b" ("-" for absent).</summary>
    public static List<string> Diff(object? a, object? b, int limit = 6, params string[] skip)
    {
        var left = ToMap(Lines(a, skip));
        var right = ToMap(Lines(b, skip));
        var diffs = new List<string>();
        foreach (string path in left.Keys.Union(right.Keys).OrderBy(p => p, StringComparer.Ordinal))
        {
            left.TryGetValue(path, out string? x);
            right.TryGetValue(path, out string? y);
            if (x != y)
            {
                diffs.Add($"{path}: {x ?? "-"} | {y ?? "-"}");
                if (diffs.Count >= limit)
                {
                    break;
                }
            }
        }

        return diffs;
    }

    private static Dictionary<string, string> ToMap(List<string> lines)
    {
        var map = new Dictionary<string, string>(lines.Count);
        foreach (string line in lines)
        {
            int eq = line.IndexOf('=');
            map[line[..eq]] = line[(eq + 1)..];
        }

        return map;
    }

    private interface ISink
    {
        void Emit(string path, string value);
    }

    private sealed class ListSink : ISink
    {
        public List<string> Lines { get; } = new(1024);

        public void Emit(string path, string value) => Lines.Add(path + "=" + value);
    }

    private sealed class HashSink(IncrementalHash sha) : ISink
    {
        private byte[] _buf = new byte[512];

        public void Emit(string path, string value)
        {
            int need = Encoding.UTF8.GetMaxByteCount(path.Length + value.Length + 2);
            if (_buf.Length < need)
            {
                _buf = new byte[need * 2];
            }

            int n = Encoding.UTF8.GetBytes(path, 0, path.Length, _buf, 0);
            _buf[n++] = (byte)'=';
            n += Encoding.UTF8.GetBytes(value, 0, value.Length, _buf, n);
            _buf[n++] = (byte)'\n';
            sha.AppendData(_buf, 0, n);
        }
    }

    private sealed class Walker(ISink sink, string[] skip)
    {
        public ISink Sink { get; } = sink;

        public StringBuilder Path { get; } = new();

        public Dictionary<object, string> Seen { get; } = new(ReferenceEqualityComparer.Instance);

        /// <summary>Whether the current path is under a skipped prefix (compared without making a string).</summary>
        public bool Skipped()
        {
            foreach (string prefix in skip)
            {
                if (Path.Length >= prefix.Length && Same(prefix))
                {
                    return true;
                }
            }

            return false;

            bool Same(string prefix)
            {
                for (int i = 0; i < prefix.Length; i++)
                {
                    if (Path[i] != prefix[i])
                    {
                        return false;
                    }
                }

                return true;
            }
        }
    }

    private static void Walk(object? root, Walker w) => Visit(root, w);

    private static void Visit(object? o, Walker w)
    {
        if (w.Skipped())
        {
            return;
        }

        if (o is null)
        {
            w.Sink.Emit(w.Path.ToString(), "null");
            return;
        }

        var type = o.GetType();
        if (type.IsPrimitive || o is string || type.IsEnum || o is decimal)
        {
            w.Sink.Emit(w.Path.ToString(), Format(o));
            return;
        }

        if (o is Delegate or Type)
        {
            w.Sink.Emit(w.Path.ToString(), "<" + type.Name + ">");
            return;
        }

        if (!type.IsValueType)
        {
            if (w.Seen.TryGetValue(o, out string? first))
            {
                w.Sink.Emit(w.Path.ToString(), "-> " + first);
                return;
            }

            w.Seen[o] = w.Path.ToString();
        }

        if (o is IDictionary dict)
        {
            w.Sink.Emit(w.Path + ".#", dict.Count.ToString(CultureInfo.InvariantCulture));
            foreach (object k in dict.Keys.Cast<object>().OrderBy(k => Format(k), StringComparer.Ordinal))
            {
                Child(w, "[" + Format(k) + "]", dict[k]);
            }

            return;
        }

        if (o is IEnumerable seq)
        {
            if (o is IList list)
            {
                w.Sink.Emit(w.Path + ".#", list.Count.ToString(CultureInfo.InvariantCulture));
                for (int i = 0; i < list.Count; i++)
                {
                    Child(w, "[" + i.ToString(CultureInfo.InvariantCulture) + "]", list[i]);
                }

                return;
            }

            // A set has no order: its members are read in the order of their own text.
            var members = new List<(string Key, object? Item)>();
            foreach (object? item in seq)
            {
                members.Add((string.Join(";", Lines(item)), item));
            }

            w.Sink.Emit(w.Path + ".#", members.Count.ToString(CultureInfo.InvariantCulture));
            int n = 0;
            foreach (var (_, item) in members.OrderBy(m => m.Key, StringComparer.Ordinal))
            {
                Child(w, "{" + (n++).ToString(CultureInfo.InvariantCulture) + "}", item);
            }

            return;
        }

        foreach (var f in FieldsOf(type))
        {
            string name = f.Name;
            if (name.StartsWith('<') && name.EndsWith(">k__BackingField", StringComparison.Ordinal))
            {
                name = name[1..^">k__BackingField".Length];
            }

            Child(w, "." + name, f.GetValue(o));
        }
    }

    private static void Child(Walker w, string segment, object? value)
    {
        int length = w.Path.Length;
        w.Path.Append(segment);
        Visit(value, w);
        w.Path.Length = length;
    }

    private static FieldInfo[] FieldsOf(Type type)
    {
        lock (FieldCache)
        {
            if (FieldCache.TryGetValue(type, out var cached))
            {
                return cached;
            }

            var fields = new List<FieldInfo>();
            // The emulator's own classes are read up the inheritance chain; framework types only for
            // what they declare themselves, so a Random subclass's base generator is not part of the state.
            for (var t = type; t is not null && t != typeof(object); t = t.BaseType)
            {
                fields.AddRange(
                    t.GetFields(
                        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly
                    )
                );
                if (t.Namespace is null || !t.Namespace.StartsWith("Sts2Emulator", StringComparison.Ordinal))
                {
                    break;
                }

                if (t.BaseType?.Namespace is { } ns && !ns.StartsWith("Sts2Emulator", StringComparison.Ordinal))
                {
                    break;
                }
            }

            cached = [.. fields.Where(f => !typeof(Delegate).IsAssignableFrom(f.FieldType))];
            FieldCache[type] = cached;
            return cached;
        }
    }

    private static string Format(object o) =>
        o switch
        {
            string s => s.Replace("\n", "\\n"),
            double d => d.ToString("R", CultureInfo.InvariantCulture),
            float f => f.ToString("R", CultureInfo.InvariantCulture),
            IFormattable fmt => fmt.ToString(null, CultureInfo.InvariantCulture),
            _ => o.ToString() ?? "",
        };
}
