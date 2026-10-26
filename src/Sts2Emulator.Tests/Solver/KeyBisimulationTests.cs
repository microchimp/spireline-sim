// The state key is a bisimulation (docs/rebuild/FIGHT_SOLVER.md section 2, KeyBisimulationTests): on positions of fuzzed
// fights, on both engines (old, and core2 with F08 and F10 on), two states the key calls equal must have the same legal
// actions and, for each action and the same chance answers, the same result and successors with equal keys.
//
// The equal-key pairs are made by the changes the key claims not to see: the card uids renumbered (and the card_ref and
// uid counter with them), the hand permuted (where no reader keeps it ordered), the discard and exhaust piles permuted,
// the potion slots permuted, the draw pile's unknown middle permuted, and new streams. "The same chance answers" is the
// same streams on both sides. A permutation changes which card a roll over a pile's index lands on, so for a permuted
// pair a successor that differs after a step that rolled is the chance model's to judge (S2's chance-model gate), not a
// failure; it is counted. A pair whose step rolled nothing must match, and a uid-renumbered pair must match always.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class KeyBisimulationTests(ITestOutputHelper output)
{
    sealed record Position(string Engine, CombatState State, CountingRandom Rng);

    /// <summary>Fuzzed positions: random legal play from a reset combat, a snapshot before every step.</summary>
    static List<Position> Positions(int seeds = 6, int steps = 30)
    {
        var all = new List<Position>();
        foreach (var engine in new[] { "old", "core2" })
        {
            for (int seed = 1; seed <= seeds; seed++)
            {
                var kernel = engine == "core2" ? new Kernel(KernelOptions.FromFamilies(["F08", "F10"]), () => Sts2Emulator.Tests.Core2.ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All) : null;
                var s = new CombatState();
                kernel?.Bind(s);
                var rng = new CountingRandom(seed * 101);
                CombatFactory.Reset(s, rng);
                CombatEngine.ResolvePrePlayPhase(s, rng);
                var pick = new Random(seed * 7919);
                for (int i = 0; i < steps; i++)
                {
                    all.Add(new Position(engine, s.Clone(), rng.Copy()));
                    var moves = Moves(s);
                    if (moves.Count == 0) break;
                    // end turn less often, so turns hold several plays
                    var (a, t) = moves[pick.Next(moves.Count)];
                    if (a == s.Hand.Count && s.PendingSelection is null && moves.Count > 1 && pick.Next(3) > 0) (a, t) = moves[pick.Next(moves.Count)];
                    if (CombatEngine.Step(s, a, rng, t).Terminal) break;
                }
            }
        }

        return all;
    }

    /// <summary>Legal (action, target) pairs: an aimed card at every targetable enemy, a potion at the first.</summary>
    static List<(int Action, int Target)> Moves(CombatState s)
    {
        var targets = Enumerable.Range(0, s.Enemies.Count).Where(i => EnemyAI.IsTargetable(s.Enemies[i])).ToList();
        var moves = new List<(int, int)>();
        foreach (var a in CombatEngine.ValidActions(s))
        {
            if (s.PendingSelection is null && a < s.Hand.Count && Cards.Get(s.Hand[a].DefId).Target == CardTarget.AnyEnemy && targets.Count > 0)
                moves.AddRange(targets.Select(t => (a, t)));
            else if (s.PendingSelection is null && a > s.Hand.Count)
                moves.Add((a, targets.Count > 0 ? targets[0] : -1));
            else
                moves.Add((a, -1));
        }

        return moves;
    }

    static byte[]? Key(CombatState s, KeyMode mode = KeyMode.Expected) => StateKey.TryEncode(s, mode, out var k, out _) ? k : null;

    static bool Same(byte[]? a, byte[]? b) => a is null ? b is null : b is not null && a.AsSpan().SequenceEqual(b);

    // ------------------------------------------------------------------------------------------------ the changes

    static void MapCards(CombatState s, Func<CardInstance, CardInstance> f)
    {
        foreach (var (_, pile) in StateKey.Piles(s))
        {
            for (int i = 0; i < pile.Count; i++) pile[i] = f(pile[i]);
        }

        if (s.HeldPlayedCard is { } h) s.HeldPlayedCard = f(h);
        if (s.LastAttackOrSkillThisTurn is { } a) s.LastAttackOrSkillThisTurn = f(a);
        if (s.LastAttackOrSkillLastTurn is { } b) s.LastAttackOrSkillLastTurn = f(b);
        foreach (var e in s.Enemies)
        {
            if (e.StolenCard is { } c) e.StolenCard = f(c);
        }

        if (s.PendingSelection is { } p)
        {
            for (int i = 0; i < p.AfterSelectionToHand.Count; i++) p.AfterSelectionToHand[i] = f(p.AfterSelectionToHand[i]);
        }
    }

    /// <summary>Every uid moved to a new, shuffled number above the old counter; card_ref and the counter follow.</summary>
    static CombatState RenumberUids(CombatState s0, Random r)
    {
        var s = s0.Clone();
        var uids = StateKey.AllCards(s).Select(c => c.Uid).Where(u => u != 0).Distinct().OrderBy(_ => r.Next()).ToList();
        int offset = Math.Max(s.LastCardUid, uids.DefaultIfEmpty(0).Max()) + 1000;
        var map = uids.Select((u, i) => (u, i)).ToDictionary(x => x.u, x => offset + x.i);
        MapCards(s, c => c.Uid != 0 && map.TryGetValue(c.Uid, out var n) ? c with { Uid = n } : c);
        foreach (var e in s.Enemies)
        {
            if (e.CurlUpCardUid != 0 && map.TryGetValue(e.CurlUpCardUid, out var n)) e.CurlUpCardUid = n;
        }

        s.LastCardUid = offset + uids.Count + 1;
        return s;
    }

    static void Shuffle<T>(IList<T> xs, int from, int count, Random r)
    {
        for (int i = count - 1; i > 0; i--)
        {
            int j = r.Next(i + 1);
            (xs[from + i], xs[from + j]) = (xs[from + j], xs[from + i]);
        }
    }

    static CombatState Permute(CombatState s0, Random r, Func<CombatState, (IList<int>? Ints, List<CardInstance>? Cards, int From, int Count)> pick)
    {
        var s = s0.Clone();
        var (ints, cards, from, count) = pick(s);
        if (ints != null) Shuffle(ints, from, count, r);
        if (cards != null) Shuffle(cards, from, count, r);
        return s;
    }

    static int Middle(CombatState s, out int top)
    {
        top = Math.Clamp(s.KnownTopCount, 0, s.DrawPile.Count);
        int bottom = Math.Clamp(s.KnownBottomCount, 0, s.DrawPile.Count - top);
        return s.DrawPile.Count - top - bottom;
    }

    static void CopyStreams(CombatState from, CombatState to)
    {
        to.ShuffleRng = from.ShuffleRng?.Copy();
        to.TargetRng = from.TargetRng?.Copy();
        to.CardSelectionRng = from.CardSelectionRng?.Copy();
        to.CardGenerationRng = from.CardGenerationRng?.Copy();
        to.PotionGenerationRng = from.PotionGenerationRng?.Copy();
        to.NicheHpRng = from.NicheHpRng?.Copy();
        to.EnergyCostRng = from.EnergyCostRng?.Copy();
        to.OrbGenerationRng = from.OrbGenerationRng?.Copy();
        to.AiRng = (from.AiRng as CountingRandom)?.Copy() ?? from.AiRng;
    }

    static CombatState NewStreams(CombatState s0, int seed)
    {
        var s = s0.Clone();
        var fresh = new CombatState
        {
            ShuffleRng = new CountingRandom(seed + 1), TargetRng = new CountingRandom(seed + 2), CardSelectionRng = new CountingRandom(seed + 3),
            CardGenerationRng = new CountingRandom(seed + 4), PotionGenerationRng = new CountingRandom(seed + 5), NicheHpRng = new CountingRandom(seed + 6),
            EnergyCostRng = new CountingRandom(seed + 7), OrbGenerationRng = new CountingRandom(seed + 8), AiRng = new CountingRandom(seed + 9),
        };
        fresh.ShuffleRng.Next(7);
        CopyStreams(fresh, s);
        return s;
    }

    static long Calls(CombatState s, CountingRandom step) =>
        new[] { s.ShuffleRng, s.TargetRng, s.CardSelectionRng, s.CardGenerationRng, s.PotionGenerationRng, s.NicheHpRng, s.EnergyCostRng, s.OrbGenerationRng, s.AiRng as CountingRandom, step }
            .Sum(x => (long)(x?.CallCount ?? 0));

    // ------------------------------------------------------------------------------------------------ the checks

    sealed class Tally
    {
        public int Pairs, Steps, Rolled, Positions;
        public readonly Dictionary<string, int> PerChange = new();
        public readonly List<string> Failures = [];
    }

    /// <summary>
    /// Steps <paramref name="a" /> and its equal-key partner <paramref name="b" /> by every legal action of a (b's action is
    /// the one on the same card, found by uid where the change kept uids, else by index), with a's streams on both sides.
    /// </summary>
    static void CheckSuccessors(Position p, CombatState a, CombatState b, string change, bool strict, Func<CombatState, CombatState, int, int> mapAction, Tally t)
    {
        var movesA = Moves(a);
        var movesB = Moves(b);
        if (movesA.Count != movesB.Count)
        {
            t.Failures.Add($"{p.Engine} {change}: {movesA.Count} legal moves against {movesB.Count}");
            return;
        }

        foreach (var (act, target) in movesA)
        {
            int actB = mapAction(a, b, act);
            var sa = a.Clone();
            var sb = b.Clone();
            CopyStreams(sa, sb);
            var ra = p.Rng.Copy();
            var rb = p.Rng.Copy();
            long before = Calls(sa, ra);
            var resA = CombatEngine.Step(sa, act, ra, target);
            var resB = CombatEngine.Step(sb, actB, rb, target);
            bool rolled = Calls(sa, ra) != before;
            t.Steps++;
            bool same = resA == resB && Same(Key(sa), Key(sb));
            if (same) continue;
            if (rolled && !strict)
            {
                t.Rolled++;
                continue;
            }

            t.Failures.Add($"{p.Engine} turn {a.Turn} {change}: action {act} target {target} gives different successors (rolled: {rolled})");
        }
    }

    static int SameIndex(CombatState a, CombatState b, int act) => act;

    /// <summary>The hand index in b of the card at a's index (the change kept uids), else the action unchanged.</summary>
    static int ByUid(CombatState a, CombatState b, int act) =>
        a.PendingSelection is null && act < a.Hand.Count ? b.Hand.FindIndex(c => c.Uid == a.Hand[act].Uid) : act;

    /// <summary>The slot in b holding the potion of a's slot (the first such, equal potions being one).</summary>
    static int BySlot(CombatState a, CombatState b, int act)
    {
        if (a.PendingSelection is not null || act <= a.Hand.Count) return act;
        int slot = act - a.Hand.Count - 1;
        return a.Hand.Count + 1 + Array.IndexOf(b.PotionSlots, a.PotionSlots[slot]);
    }

    [Fact]
    public void EqualKeysGiveEqualSuccessorsOnFuzzedPositions()
    {
        var t = new Tally();
        var r = new Random(35);
        foreach (var p in Positions())
        {
            var s = p.State;
            var key = Key(s);
            if (key is null) continue;  // a busy kernel queue: no key, nothing to compare
            t.Positions++;
            var changes = new List<(string Name, CombatState B, bool Strict, Func<CombatState, CombatState, int, int> Map)>();
            if (s.PendingSelection?.Replay is null) changes.Add(("uids renumbered", RenumberUids(s, r), true, SameIndex));
            if (!StateKey.HandOrdered(s) && s.Hand.Count > 1)
                changes.Add(("hand permuted", Permute(s, r, x => (null, x.Hand, 0, x.Hand.Count)), false, ByUid));
            if (s.PendingSelection is null && s.DiscardPile.Count > 1)
                changes.Add(("discard permuted", Permute(s, r, x => (null, x.DiscardPile, 0, x.DiscardPile.Count)), false, SameIndex));
            if (s.PendingSelection is null && s.ExhaustPile.Count > 1)
                changes.Add(("exhaust permuted", Permute(s, r, x => (null, x.ExhaustPile, 0, x.ExhaustPile.Count)), false, SameIndex));
            if (!StateKey.PotionOrdered(s))
                changes.Add(("potions permuted", Permute(s, r, x => (x.PotionSlots, null, 0, Math.Min(x.MaxPotionSlots, x.PotionSlots.Length))), false, BySlot));
            foreach (var (name, b, strict, map) in changes)
            {
                t.Pairs++;
                t.PerChange[name] = t.PerChange.GetValueOrDefault(name) + 1;
                if (!Same(key, Key(b)))
                {
                    t.Failures.Add($"{p.Engine} turn {s.Turn} {name}: the key changed");
                    continue;
                }

                CheckSuccessors(p, s, b, name, strict, map, t);
            }
        }

        output.WriteLine($"positions {t.Positions}, equal-key pairs {t.Pairs} ({string.Join(", ", t.PerChange.Select(kv => $"{kv.Key} {kv.Value}"))}), "
            + $"successor steps {t.Steps}, rolled and left to the chance model {t.Rolled}, failures {t.Failures.Count}");
        Assert.True(t.Positions > 200, $"only {t.Positions} keyed positions");
        Assert.True(t.Steps > 1000, $"only {t.Steps} successor steps");
        Assert.True(t.Failures.Count == 0, string.Join("\n", t.Failures.Take(40)));
    }

    [Fact]
    public void StreamsAndTheUnknownMiddleChangeNoExpectedKeyAndEveryHindsightKey()
    {
        var r = new Random(36);
        int middles = 0, streams = 0;
        foreach (var p in Positions(4, 20))
        {
            var s = p.State;
            var key = Key(s);
            if (key is null) continue;
            var fresh = NewStreams(s, 9000 + streams);
            Assert.True(Same(key, Key(fresh)), "new streams changed an expected-mode key");
            Assert.False(Same(Key(s, KeyMode.Hindsight), Key(fresh, KeyMode.Hindsight)), "new streams left the hindsight key");
            streams++;
            int n = Middle(s, out int top);
            var mid = s.DrawPile.GetRange(top, n);
            if (s.PendingSelection is not null || mid.Distinct().Count() < 2)
                continue;
            CombatState b;
            do b = Permute(s, r, x => (null, x.DrawPile, top, n));
            while (b.DrawPile.Select(c => c.DefId).SequenceEqual(s.DrawPile.Select(c => c.DefId)) && b.DrawPile.Select(c => c.Upgraded).SequenceEqual(s.DrawPile.Select(c => c.Upgraded)) && r.Next(20) > 0);
            Assert.True(Same(key, Key(b)), "permuting the unknown middle changed an expected-mode key");
            middles++;
        }

        output.WriteLine($"stream changes {streams}, middle permutations {middles}");
        Assert.True(streams > 50 && middles > 20);
    }

    [Fact]
    public void AKeptFieldChangesTheKey()
    {
        // Each plain int or bool of CombatState and of an enemy, changed alone, gives another key (the encoder writes it).
        var s = Positions(1, 3)[2].State;
        var key = Key(s)!;
        var rows = KeyFieldCoverageTests.Json("key_fields.json")["types"]!;
        int n = 0;
        foreach (var (owner, get) in new (string, Func<CombatState, object>)[] { ("CombatState", x => x), ("EnemyState", x => x.Enemies[0]) })
        {
            foreach (var f in KeyFieldCoverageTests.Fields(get(s).GetType()))
            {
                var row = rows[owner]!["fields"]![KeyFieldCoverageTests.Name(f)];
                if ((string?)row?["class"] != "plain" || (f.FieldType != typeof(int) && f.FieldType != typeof(bool))) continue;
                var c = s.Clone();
                var o = get(c);
                f.SetValue(o, f.FieldType == typeof(int) ? (int)f.GetValue(o)! + 1 : !(bool)f.GetValue(o)!);
                Assert.False(Same(key, Key(c)), $"{owner}.{KeyFieldCoverageTests.Name(f)} changed and the key did not");
                n++;
            }
        }

        // and each field of a card in the hand
        foreach (var f in KeyFieldCoverageTests.Fields(typeof(CardInstance)).Where(f => f.FieldType == typeof(int) || f.FieldType == typeof(bool)))
        {
            var c = s.Clone();
            object card = c.Hand[0];
            f.SetValue(card, f.FieldType == typeof(int) ? (int)f.GetValue(card)! + 1 : !(bool)f.GetValue(card)!);
            c.Hand[0] = (CardInstance)card;
            Assert.False(Same(key, Key(c)), $"CardInstance.{KeyFieldCoverageTests.Name(f)} changed and the key did not");
            n++;
        }

        Assert.True(n > 100, $"only {n} fields tried");
    }

    [Fact]
    public void ACardRefKeepsItsCardApartAndFollowsItAcrossPiles()
    {
        var s = Positions(1, 1)[0].State;
        var strike = s.Hand[0];
        s.Hand.Add(strike with { Uid = s.LastCardUid + 1 });  // an equal copy with its own uid
        s.LastCardUid += 1;
        var plain = Key(s)!;
        var pointed = s.Clone();
        pointed.Enemies[0].CurlUpCardUid = strike.Uid;
        var pointedOther = s.Clone();
        pointedOther.Enemies[0].CurlUpCardUid = s.LastCardUid;
        Assert.False(Same(plain, Key(pointed)));
        Assert.True(Same(Key(pointed), Key(pointedOther)), "two equal cards in one pile: pointing at either is one position");
        var moved = pointed.Clone();
        var at = moved.Hand.FindIndex(c => c.Uid == strike.Uid);
        moved.DiscardPile.Add(moved.Hand[at]);
        moved.Hand.RemoveAt(at);
        moved.DiscardPile.Add(moved.Hand[0]);
        moved.Hand.RemoveAt(0);
        var movedOther = pointed.Clone();
        movedOther.DiscardPile.Add(movedOther.Hand[0]);
        movedOther.Hand.RemoveAt(0);
        Assert.False(Same(Key(moved), Key(movedOther)) && moved.Hand.Count == movedOther.Hand.Count, "where the pointed card is is part of the key");
    }

    [Fact]
    public void TheKeyChangesNothingInTheState()
    {
        var s = Positions(1, 2)[1].State;
        var before = s.Clone();
        _ = StateKey.Encode(s);
        _ = StateKey.Encode(s, KeyMode.Hindsight);
        _ = StateKey.EnemyClasses(s);
        Assert.Equal(Snapshot(before), Snapshot(s));
        Assert.All(StateKey.EnemyClasses(s).Select((c, i) => (c, i)), x => Assert.Equal(x.i, x.c));  // enemy order is read today
    }

    static string Snapshot(CombatState s) =>
        $"{s.PlayerHp}|{s.Energy}|{string.Join(",", s.Hand.Select(c => $"{c.DefId}:{c.Uid}"))}|{string.Join(",", s.DrawPile.Select(c => c.Uid))}|"
        + $"{string.Join(",", s.Enemies.Select(e => $"{e.Hp}:{e.Block}:{e.MoveIndex}"))}|{s.ShuffleRng?.CallCount}|{(s.AiRng is CountingRandom a ? a.CallCount : -1)}";
}
