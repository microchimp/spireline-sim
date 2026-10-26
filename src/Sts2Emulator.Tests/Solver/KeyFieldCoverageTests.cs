// The state key's census (docs/rebuild/FIGHT_SOLVER.md section 2, "The checklist"): every field the full-state walk
// reaches from CombatState (the walk of tools/emudump/Engines/Old/OldStateWalk.cs: instance fields of the engine's own
// types and their bases, delegates left out, an auto-property under its property's name; unlike the hash's walk the
// [IdentityOnly] fields are in it too, since CopyTo copies them and the kernel's progress and card_ref live there) is classed in
// tools/solver/key_fields.json, with the type the json says; the walk goes on into the type of every field the key keeps.
// A field added to CombatState, EnemyState, CardInstance, BuffState, RelicInstance, OrbState, PendingCardSelection, the
// stopped step or the kernel's state fails here until it is classed; so does a json row for a field that is gone, an
// exclusion without a reason, and a generated encoder older than its json.
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class KeyFieldCoverageTests
{
    const BindingFlags Inst = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    static readonly Assembly Engine = typeof(CombatState).Assembly;

    internal static string Repo()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            if (File.Exists(Path.Combine(d.FullName, "tools", "solver", "key_fields.json"))) return d.FullName;
        }

        throw new InvalidOperationException("tools/solver/key_fields.json not found above the test assembly");
    }

    internal static JsonObject Json(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo(), "tools", "solver", name)))!.AsObject();

    static bool IsStream(Type t) => t == typeof(CountingRandom) || t == typeof(GameRng) || t == typeof(MegaRandom) || t == typeof(Random);

    static bool IsHolder(Type t) =>
        t.Assembly == Engine && !t.IsEnum && !t.IsArray && !IsStream(t) && t != typeof(UnlockState) && !typeof(Delegate).IsAssignableFrom(t);

    internal static IEnumerable<FieldInfo> Fields(Type t)
    {
        for (var x = t; x != null && x != typeof(object) && x != typeof(ValueType); x = x.BaseType)
        {
            foreach (var f in x.GetFields(Inst))
            {
                if (!typeof(Delegate).IsAssignableFrom(f.FieldType)) yield return f;
            }
        }
    }

    internal static string Name(FieldInfo f) =>
        f.Name.StartsWith('<') && f.Name.Contains(">k__BackingField") ? f.Name[1..f.Name.IndexOf('>')] : f.Name;

    internal static string TypeName(Type t)
    {
        if (Nullable.GetUnderlyingType(t) is { } u) return TypeName(u) + "?";
        if (t.IsArray) return TypeName(t.GetElementType()!) + "[]";
        if (t.IsGenericType) return t.Name[..t.Name.IndexOf('`')] + "<" + string.Join(",", t.GetGenericArguments().Select(TypeName)) + ">";
        return t == typeof(int) ? "int" : t == typeof(bool) ? "bool" : t == typeof(long) ? "long" : t == typeof(uint) ? "uint"
            : t == typeof(string) ? "string" : t == typeof(float) ? "float" : t == typeof(double) ? "double" : t == typeof(byte) ? "byte"
            : t == typeof(ulong) ? "ulong" : t == typeof(short) ? "short" : t.Name;
    }

    /// <summary>The engine types a field's value is or holds (its own type, a collection's element, a nullable's value).</summary>
    static IEnumerable<Type> Held(Type t)
    {
        t = Nullable.GetUnderlyingType(t) ?? t;
        if (IsHolder(t))
        {
            yield return t;
            yield break;
        }

        var args = t.IsArray ? [t.GetElementType()!] : t.IsGenericType ? t.GetGenericArguments() : [];
        foreach (var a in args)
        {
            foreach (var h in Held(a)) yield return h;
        }
    }

    static IEnumerable<Type> Enums(Type t)
    {
        t = Nullable.GetUnderlyingType(t) ?? t;
        if (t.IsEnum) yield return t;
        var args = t.IsArray ? [t.GetElementType()!] : t.IsGenericType ? t.GetGenericArguments() : [];
        foreach (var a in args)
        {
            foreach (var e in Enums(a)) yield return e;
        }
    }

    static readonly string[] Classes = ["plain", "multiset", "regions", "card_ref", "stream", "excluded"];

    /// <summary>Every finding of the census: an unclassed field (with the json row it needs), a stale row, a wrong type.</summary>
    internal static List<string> Census()
    {
        var spec = Json("key_fields.json");
        var types = spec["types"]!.AsObject();
        var enums = spec["enums"]!.AsArray().Select(e => (string)e!).ToHashSet();
        var findings = new List<string>();
        var seen = new HashSet<string>();
        var queue = new Queue<Type>([typeof(CombatState)]);
        while (queue.Count > 0)
        {
            var t = queue.Dequeue();
            if (!seen.Add(t.Name)) continue;
            if (types[t.Name] is not JsonObject tspec)
            {
                findings.Add($"type {t.Name} is reached through a keyed field and has no entry in key_fields.json");
                continue;
            }

            var rows = tspec["fields"]!.AsObject();
            var walked = new HashSet<string>();
            foreach (var f in Fields(t))
            {
                var name = Name(f);
                walked.Add(name);
                var tn = TypeName(f.FieldType);
                if (rows[name] is not JsonObject row)
                {
                    findings.Add($"{t.Name}.{name} is not classed: add \"{name}\": {{\"type\": \"{tn}\", \"class\": ...}}");
                    continue;
                }

                var cls = (string?)row["class"];
                if (!Classes.Contains(cls)) findings.Add($"{t.Name}.{name}: unknown class {cls}");
                if ((string?)row["type"] != tn) findings.Add($"{t.Name}.{name}: the json says type {(string?)row["type"]}, the field is {tn}");
                if (cls is "excluded" or "stream" && string.IsNullOrWhiteSpace((string?)row["reason"]))
                    findings.Add($"{t.Name}.{name}: {cls} with no reason");
                if (cls == "excluded") continue;
                foreach (var e in Enums(f.FieldType).Where(e => !enums.Contains(e.Name)))
                    findings.Add($"{t.Name}.{name}: enum {e.Name} is not in key_fields.json's enums");
                if (cls == "stream") continue;
                foreach (var h in Held(f.FieldType)) queue.Enqueue(h);
            }

            foreach (var (name, _) in rows)
            {
                if (!walked.Contains(name)) findings.Add($"{t.Name}.{name} is in key_fields.json but the type has no such field");
            }
        }

        foreach (var (name, _) in types)
        {
            if (!seen.Contains(name)) findings.Add($"type {name} is in key_fields.json but no keyed field reaches it");
        }

        return findings;
    }

    [Fact]
    public void EveryWalkedFieldIsClassed()
    {
        var findings = Census();
        Assert.True(findings.Count == 0, "key_fields.json census:\n" + string.Join("\n", findings));
    }

    [Fact]
    public void EveryCensusPathIsInAKeptField()
    {
        var spec = Json("key_fields.json");
        var paths = spec["census"]!["paths"]!.AsObject();
        Assert.True(paths.Count >= 25);
        foreach (var (path, target) in paths)
        {
            var parts = ((string)target!).Split('.');
            var row = spec["types"]![parts[0]]?["fields"]?[parts[1]];
            Assert.True(row is not null, $"census {path}: {target} is not a classed field");
            Assert.True((string?)row!["class"] != "excluded", $"census {path}: {target} is excluded from the key");
        }
    }

    [Fact]
    public void TheEncoderIsGeneratedFromTheJsonAsItStands()
    {
        static byte[] Norm(string p) => Encoding.UTF8.GetBytes(File.ReadAllText(p).Replace("\r\n", "\n"));
        var dir = Path.Combine(Repo(), "tools", "solver");
        var digest = Convert.ToHexString(SHA256.HashData([.. Norm(Path.Combine(dir, "key_fields.json")), .. Norm(Path.Combine(dir, "order_readers.json"))]))
            .ToLowerInvariant();
        Assert.True(digest == StateKey.FieldsSha256, "StateKey.g.cs is stale: run python tools/solver/gen_key.py");
    }

    [Fact]
    public void EveryOrderReaderNamesACard()
    {
        Assert.Empty(StateKey.Readers.Unresolved());
    }

    [Fact]
    public void TheDocsExclusionsAreTheJsonsWithTheirReasons()
    {
        // FIGHT_SOLVER.md section 2's table: what leaves the key and why.
        var cs = Json("key_fields.json")["types"]!["CombatState"]!["fields"]!;
        foreach (var n in new[] { "LastCardUid", "LastIntentRoll", "AiStateAssumed" })
            Assert.Equal("excluded", (string?)cs[n]!["class"]);
        foreach (var n in new[] { "ShuffleRng", "TargetRng", "CardSelectionRng", "EnergyCostRng", "CardGenerationRng", "PotionGenerationRng", "OrbGenerationRng", "AiRng", "NicheHpRng" })
            Assert.Equal("stream", (string?)cs[n]!["class"]);
        Assert.Equal("regions", (string?)cs["DrawPile"]!["class"]);
        Assert.Equal("card_ref", (string?)Json("key_fields.json")["types"]!["EnemyState"]!["fields"]!["CurlUpCardUid"]!["class"]);
    }
}
