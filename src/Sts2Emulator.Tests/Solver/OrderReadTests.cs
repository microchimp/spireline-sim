// The pile-order rules of the state key (docs/rebuild/FIGHT_SOLVER.md section 2): tools/solver/scan_order_reads.py finds
// every read of a pile's index or position in the six engine files (tools/solver/order_reads.json) and
// tools/solver/order_readers.json classes each read of a multiset pile. These tests fail on a scan older than the sources,
// on an unclassed or stale read, on a reader outside its category, on a "dead" method that has a caller and on a rolled
// pile hindsight mode does not keep ordered; and they show each order rule is needed: a Dredge position, a hindsight roll
// over the hand, and Whispering Earring, each of which the key merged before the rule.
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class OrderReadTests
{
    static readonly string[] Sources =
    [
        "src/Sts2Emulator/Core/Effects/CardEffects.cs", "src/Sts2Emulator/Core/Effects/RelicEffects.cs",
        "src/Sts2Emulator/Core/Effects/PotionEffects.cs", "src/Sts2Emulator/Core/BuffSystem.cs",
        "src/Sts2Emulator/Core/CombatEngine.cs", "src/Sts2Emulator/Core/EnemyAI.cs",
    ];

    static byte[] Norm(string p)
    {
        var b = File.ReadAllBytes(p);
        var o = new List<byte>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            if (b[i] != (byte)'\r' || i + 1 >= b.Length || b[i + 1] != (byte)'\n') o.Add(b[i]);
        }

        return [.. o];
    }

    [Fact]
    public void TheScanIsAsFreshAsTheEngineSources()
    {
        var repo = KeyFieldCoverageTests.Repo();
        var digest = Convert.ToHexString(SHA256.HashData([.. Sources.SelectMany(s => Norm(Path.Combine(repo, s)))])).ToLowerInvariant();
        Assert.True(digest == (string?)KeyFieldCoverageTests.Json("order_reads.json")["sources_sha256"],
            "order_reads.json is older than the engine sources: run python tools/solver/scan_order_reads.py and class the new reads");
    }

    [Fact]
    public void EveryReadOfAMultisetPileIsClassed()
    {
        var scan = KeyFieldCoverageTests.Json("order_reads.json");
        var readers = KeyFieldCoverageTests.Json("order_readers.json");
        var rules = readers["rules"]!.AsObject();
        var sites = readers["sites"]!.AsObject();
        var cats = readers["readers"]!.AsObject();
        var multiset = scan["multiset_piles"]!.AsArray().Select(x => (string)x!).ToHashSet();
        var keyed = KeyFieldCoverageTests.Json("key_fields.json")["types"]!["CombatState"]!["fields"]!.AsObject()
            .Where(kv => (string?)kv.Value!["class"] == "multiset").Select(kv => kv.Key).ToHashSet();
        Assert.True(keyed.SetEquals(multiset), "the scan's multiset piles are not key_fields.json's");
        var src = KeyFieldCoverageTests.Repo();
        var allSource = string.Join("\n", Directory.GetFiles(Path.Combine(src, "src", "Sts2Emulator"), "*.cs", SearchOption.AllDirectories)
            .Where(p => !p.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !p.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .Select(File.ReadAllText));
        var problems = new List<string>();
        var seen = new HashSet<string>();
        var rolled = new HashSet<string>();
        int n = 0;
        foreach (var s in scan["sites"]!.AsArray())
        {
            string id = (string)s!["id"]!, pile = (string)s["pile"]!, kind = (string)s["kind"]!;
            if (!multiset.Contains(pile)) continue;
            n++;
            seen.Add(id);
            var c = sites[id] ?? rules[kind];
            if (c is null)
            {
                problems.Add($"unclassed read {id} ({kind}, line {(int)s["line"]!})");
                continue;
            }

            string cls = (string)c["class"]!;
            if (cls == "roll" || kind == "roll") rolled.Add(pile);
            if (cls is not ("reader" or "roll" or "flow" or "free" or "dead")) problems.Add($"{id}: unknown class {cls}");
            if (cls != "reader" && string.IsNullOrEmpty((string?)c["why"])) problems.Add($"{id}: class {cls} without a why");
            if (cls == "reader")
            {
                var cat = cats[(string)c["category"]!];
                if (cat is null || (string?)cat["pile"] != pile)
                {
                    problems.Add($"{id}: reader of {pile} in category {(string?)c["category"]}");
                    continue;
                }

                var cards = cat["cards"]!.AsArray().Select(x => (string)x!["name"]!).ToHashSet();
                var relics = cat["relics"]!.AsArray().Select(x => (string)x!["name"]!).ToHashSet();
                foreach (var h in c["holders"]?.AsArray() ?? []) if (!cards.Contains((string)h!)) problems.Add($"{id}: holder {h} not in its category");
                foreach (var h in c["relics"]?.AsArray() ?? []) if (!relics.Contains((string)h!)) problems.Add($"{id}: relic {h} not in its category");
                if ((c["holders"]?.AsArray().Count ?? 0) + (c["relics"]?.AsArray().Count ?? 0) == 0 && cat["always"] is null) problems.Add($"{id}: a reader with no holder");
            }

            if (cls == "dead")
            {
                var method = (string)s["method"]!;
                int refs = Regex.Matches(allSource, $@"\b{Regex.Escape(method)}\b").Count;
                if (refs != 1) problems.Add($"{id}: classed dead but {method} is named {refs} times in src/Sts2Emulator");
            }

            if (cls == "flow")
            {
                foreach (var cat in new[] { "discard_position", "exhaust_position" })
                    if (!(cats[cat]!["implies"]!.AsArray().Any(x => (string?)x == "hand_position"))) problems.Add($"{id}: a flow from the hand, but {cat} does not imply hand_position");
            }
        }

        foreach (var (id, _) in sites)
            if (!seen.Contains(id)) problems.Add($"stale class: {id} (the scan no longer finds it)");
        var hindsight = readers["hindsight_ordered"]!["piles"]!.AsArray().Select(x => (string)x!).ToHashSet();
        foreach (var p in rolled)
            if (!hindsight.Contains(p)) problems.Add($"a roll indexes {p}, but hindsight mode does not keep it ordered");
        Assert.True(hindsight.SetEquals(StateKeyCodec.HindsightOrdered), "StateKey.g.cs's hindsight piles are not the json's");
        Assert.True(n > 100, $"only {n} multiset-pile reads scanned");
        Assert.True(problems.Count == 0, string.Join("\n", problems));
    }

    // ------------------------------------------------------------------------------------------------ each rule is needed

    static CombatState Start(params string[] hand)
    {
        var s = new CombatState();
        var rng = new CountingRandom(77);
        CombatFactory.Reset(s, rng);
        CombatEngine.ResolvePrePlayPhase(s, rng);
        s.Hand.Clear();
        foreach (var h in hand) s.Hand.Add(Card(s, h));
        s.Energy = 9;
        return s;
    }

    static CardInstance Card(CombatState s, string name) => new(Cards.FindId(name) ?? throw new InvalidOperationException(name), false) { Uid = s.NewCardUid() };

    static bool Same(CombatState a, CombatState b, KeyMode mode = KeyMode.Expected) => StateKey.Encode(a, mode).AsSpan().SequenceEqual(StateKey.Encode(b, mode));

    [Fact]
    public void DredgeKeepsTheDiscardPileOrdered()
    {
        // Dredge moves DiscardPile[0] to the hand with no screen: two discard orders are two positions.
        var a = Start("Dredge");
        a.DiscardPile.Clear();
        a.DiscardPile.Add(Card(a, "StrikeIronclad"));
        a.DiscardPile.Add(Card(a, "Bash"));
        var b = a.Clone();
        b.DiscardPile.Reverse();
        Assert.True(StateKey.DiscardOrdered(a) && StateKey.HandOrdered(a), "a discard-position reader orders the discard pile and, by implication, the hand");
        Assert.False(Same(a, b), "Dredge in the combat: two discard orders gave one key");
        // and they play differently: the effect behind the generated-card form of Dredge, Hologram and Reanimate
        var move = typeof(Sts2Emulator.Core.Effects.CardEffects).GetMethod("MoveDiscardCardsToHand", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        move.Invoke(null, [a, 1]);
        move.Invoke(null, [b, 1]);
        Assert.NotEqual(a.Hand.Select(c => c.DefId), b.Hand.Select(c => c.DefId));
        // without a reader the discard order is not in the key
        var c = Start("StrikeIronclad");
        c.DiscardPile.Clear();
        c.DiscardPile.Add(Card(c, "StrikeIronclad"));
        c.DiscardPile.Add(Card(c, "Bash"));
        var d = c.Clone();
        d.DiscardPile.Reverse();
        Assert.True(Same(c, d));
    }

    [Fact]
    public void HindsightKeepsARolledHandOrdered()
    {
        // Cinder exhausts a hand card at a roll's index: with the streams in the key, the roll's answer is fixed, so a
        // permuted hand is another position in hindsight mode (and the same one in expected mode).
        var a = Start("Cinder", "StrikeIronclad", "Bash", "DefendIronclad");
        var b = a.Clone();
        (b.Hand[1], b.Hand[2], b.Hand[3]) = (a.Hand[2], a.Hand[3], a.Hand[1]);
        Assert.True(Same(a, b), "expected mode: a permuted hand with no reader is one position");
        Assert.True(StateKey.HandOrdered(a, KeyMode.Hindsight));
        Assert.False(Same(a, b, KeyMode.Hindsight), "hindsight mode: a permuted hand gave one key though a roll indexes it");
        int target = Enumerable.Range(0, a.Enemies.Count).First(i => EnemyAI.IsTargetable(a.Enemies[i]));
        CombatEngine.Step(a, 0, new CountingRandom(9), target);
        CombatEngine.Step(b, 0, new CountingRandom(9), target);
        Assert.NotEqual(a.ExhaustPile.Select(c => c.DefId), b.ExhaustPile.Select(c => c.DefId));  // the same streams exhaust another card
    }

    [Fact]
    public void WhisperingEarringOrdersTheHandUntilItIsSpent()
    {
        var a = Start("StrikeIronclad", "Bash");
        Assert.False(StateKey.HandOrdered(a));
        a.Relics.Add(new RelicInstance(Relics.FindId("WhisperingEarring")!.Value));
        a.Turn = 0;
        a.WhisperingEarringSpent = false;
        Assert.True(StateKey.HandOrdered(a), "an unspent Whispering Earring on turn zero plays the hand's first playable card");
        a.WhisperingEarringSpent = true;
        Assert.False(StateKey.HandOrdered(a));
        a.WhisperingEarringSpent = false;
        a.Turn = 1;
        Assert.False(StateKey.HandOrdered(a));
    }
}
