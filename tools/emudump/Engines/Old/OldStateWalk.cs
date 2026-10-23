// The old engine's side of the full-state hash (IEngine.StateHash, StateFieldHashes). Every field of the emulator's state, found by reflection. One walk serves two gates:
// - tools/emudump.Tests/LifecycleTests.cs enumerates these fields and checks each one is handled by
//   Clone/CopyTo, Reset, the snapshot and Reanchor's write-back (or allow-listed with a reason);
// - the full-state hash (EmuDump fight --state-hash, tools/fidelity/corpus.py identity) is a hash of
//   this canonical text, so it covers every field Clone must copy, hidden ones included, where an
//   EmuDump byte comparison covers only what the dump prints.
// The text is canonical: fields by name, sets and dictionaries sorted, streams as (seed, position),
// enum members by number. It names fields, so a refactor that renames one changes the hash; that is a
// structural change the identity check reports by field (EmuDump fight --state-hash=fields).
using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

static class OldStateWalk
{
    const BindingFlags Inst = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static readonly Assembly Engine = typeof(CombatState).Assembly;
    static readonly Dictionary<Type, FieldInfo[]> FieldCache = new();

    /// <summary>A class, struct or record of the engine whose fields are state (not an enum, not a stream, not the profile).</summary>
    internal static bool IsEngineType(Type t) =>
        t.Assembly == Engine && !t.IsEnum && !IsStream(t) && !IsProfile(t) && !typeof(Delegate).IsAssignableFrom(t);

    /// <summary>
    /// The run's profile (UnlockState): immutable input a run is generated for, one value written
    /// whole (what it lacks) rather than a holder of fields.
    /// </summary>
    internal static bool IsProfile(Type t) => t == typeof(UnlockState);

    internal static bool IsStream(Type t) =>
        t == typeof(CountingRandom) || t == typeof(GameRng) || t == typeof(MegaRandom) || t == typeof(Random);

    /// <summary>Every instance field of <paramref name="t"/> and its base types, by name (auto-property backing fields under the property's name).</summary>
    internal static FieldInfo[] Fields(Type t)
    {
        lock (FieldCache)
        {
            if (FieldCache.TryGetValue(t, out var cached)) return cached;
            var all = new List<FieldInfo>();
            for (var x = t; x != null && x != typeof(object) && x != typeof(ValueType); x = x.BaseType)
            {
                // [IdentityOnly]: the kernel's card uids and their counter, read by no output (CardUid.cs).
                all.AddRange(x.GetFields(Inst | BindingFlags.DeclaredOnly)
                    .Where(f => !typeof(Delegate).IsAssignableFrom(f.FieldType) && !f.IsDefined(typeof(IdentityOnlyAttribute))));
            }

            var fields = all.OrderBy(Name, StringComparer.Ordinal).ToArray();
            FieldCache[t] = fields;
            return fields;
        }
    }

    /// <summary>A field's name as the source spells it (an auto-property's backing field under the property's name).</summary>
    internal static string Name(FieldInfo f)
    {
        var n = f.Name;
        return n.StartsWith('<') && n.Contains(">k__BackingField") ? n[1..n.IndexOf('>')] : n;
    }

    /// <summary>The element type a collection field holds (T of a List, array, set; the value of a dictionary), or null.</summary>
    internal static Type? ElementType(Type t)
    {
        if (t.IsArray) return t.GetElementType();
        if (!t.IsGenericType || t == typeof(string)) return null;
        var def = t.GetGenericTypeDefinition();
        var args = t.GetGenericArguments();
        if (def == typeof(Dictionary<,>) || def == typeof(SortedDictionary<,>)) return args[1];
        if (def == typeof(List<>) || def == typeof(HashSet<>) || def == typeof(Queue<>) || def == typeof(Stack<>)
            || def == typeof(IReadOnlyList<>) || def == typeof(IList<>) || def == typeof(LinkedList<>))
        {
            return args[0];
        }

        return null;
    }

    // ------------------------------------------------------------------------------- canonical text

    public static string Canon(object? v)
    {
        var sb = new StringBuilder(4096);
        Write(sb, v, new HashSet<object>(ReferenceEqualityComparer.Instance), 0);
        return sb.ToString();
    }

    static void Write(StringBuilder sb, object? v, HashSet<object> path, int depth)
    {
        if (v is null)
        {
            sb.Append('~');
            return;
        }

        var t = v.GetType();
        switch (v)
        {
            case string s:
                sb.Append('"').Append(s.Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('"');
                return;
            case bool b:
                sb.Append(b ? 'T' : 'F');
                return;
            case Enum e:
                sb.Append(Convert.ToInt64(e, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture));
                return;
            case float f:
                sb.Append(f.ToString("R", CultureInfo.InvariantCulture));
                return;
            case double d:
                sb.Append(d.ToString("R", CultureInfo.InvariantCulture));
                return;
            case IFormattable n when t.IsPrimitive || v is decimal:
                sb.Append(n.ToString(null, CultureInfo.InvariantCulture));
                return;
            case CountingRandom cr:
                sb.Append("CR(").Append(cr.Seed).Append(',').Append(cr.CallCount).Append(')');
                return;
            case GameRng gr:
                sb.Append("GR(").Append(gr.RawSeed).Append(',').Append(gr.CallCount).Append(')');
                return;
            case Random:
                sb.Append("Random");  // a plain Random has no readable position (only the combat-only env makes one)
                return;
            case UnlockState profile:
                sb.Append("UL(").Append(profile.ToString()).Append(')');
                return;
        }

        if (!t.IsValueType && !path.Add(v))
        {
            sb.Append("^cycle");
            return;
        }

        try
        {
            if (v is IDictionary dict)
            {
                var items = new List<string>();
                foreach (DictionaryEntry kv in dict)
                {
                    var k = new StringBuilder();
                    Write(k, kv.Key, path, depth + 1);
                    k.Append(':');
                    Write(k, kv.Value, path, depth + 1);
                    items.Add(k.ToString());
                }

                items.Sort(StringComparer.Ordinal);
                sb.Append('{').AppendJoin(',', items).Append('}');
            }
            else if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(HashSet<>))
            {
                var items = new List<string>();
                foreach (var x in (IEnumerable)v) items.Add(Canon(x));
                items.Sort(StringComparer.Ordinal);
                sb.Append("s[").AppendJoin(',', items).Append(']');
            }
            else if (v is IEnumerable seq)
            {
                sb.Append('[');
                bool first = true;
                foreach (var x in seq)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    Write(sb, x, path, depth + 1);
                }

                sb.Append(']');
            }
            else if (v is ITuple tuple)
            {
                sb.Append('(');
                for (int i = 0; i < tuple.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    Write(sb, tuple[i], path, depth + 1);
                }

                sb.Append(')');
            }
            else if (IsEngineType(t))
            {
                sb.Append(t.Name).Append('{');
                bool first = true;
                foreach (var f in Fields(t))
                {
                    if (!first) sb.Append(',');
                    first = false;
                    sb.Append(Name(f)).Append('=');
                    Write(sb, f.GetValue(v), path, depth + 1);
                }

                sb.Append('}');
            }
            else
            {
                sb.Append(t.Name).Append(':').Append(Convert.ToString(v, CultureInfo.InvariantCulture));
            }
        }
        finally
        {
            if (!t.IsValueType) path.Remove(v);
        }
    }

    // ------------------------------------------------------------------------------------ hashes

    static string Hex(string text, int chars) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)))[..chars].ToLowerInvariant();

    /// <summary>The full-state hash of one object: SHA-256 of its canonical text, 16 hex digits.</summary>
    public static string Hash(object? state) => Hex(Canon(state), 16);

    /// <summary>One short hash per top-level field, to name what changed when the full hash differs.</summary>
    public static SortedDictionary<string, string> FieldHashes(object state)
    {
        var o = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var f in Fields(state.GetType()))
        {
            o[Name(f)] = Hex(Canon(f.GetValue(state)), 8);
        }

        return o;
    }

    /// <summary>
    /// The run's full state: every RunState field (the active combat's included). The combat's own
    /// streams are fields of the combat; the engine's active-combat stream is RunState.ActiveCombatRng.
    /// </summary>
    public static string RunHash(RunEngine e) => Hash(e.State);
}
