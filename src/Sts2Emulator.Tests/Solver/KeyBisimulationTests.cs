// The state key is a bisimulation (docs/rebuild/FIGHT_SOLVER.md section 2, KeyBisimulationTests): on varied positions
// (the default fights, the corpus's decks, relics and potions, and synthetic decks rich in order readers and rolled cards,
// on old and on core2 with F08 and F10, single and multi-enemy, open screens, mid-turn), two states the key calls equal must
// have the same legal actions and, for each action, the same distribution of (result, successor key).
//
// The equal-key pairs are made by the changes the key claims not to see: the card uids renumbered (and the card_ref and
// uid counter with them), the hand, discard and exhaust piles and the potion slots permuted where no reader keeps them
// ordered. A permutation moves which card a roll over the pile lands on, so the comparison is of distributions: the
// step's chance tree is enumerated (every stream cleared, each roll answered over all its outcomes) to the deepest roll
// count both sides fit in a leaf budget, which is the whole tree for most steps. Where a tree is larger, the rolls past
// the cut are answered alike on both sides, which compares unlike things when the next roll shuffles a permuted pile
// (the same fixed answer lands on different cards of a differently ordered pile). So a cut step is compared to its cut
// with the shared tail and, where that differs, up to the first roll past the cut (the state there, the draw pile's
// order forgotten), which both sides reach by the same rolls. No step is excused and none is tolerated. Hindsight mode
// is checked strictly: equal hindsight keys must step to equal hindsight keys under the same streams.
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class KeyBisimulationTests(ITestOutputHelper output)
{
    sealed record Position(string Engine, CombatState State, CountingRandom Rng, string Source);

    /// <summary>Cards and relics that read pile order or roll over a pile, mixed into the synthetic decks.</summary>
    static readonly string[] OrderCards =
    [
        "Dredge", "Hologram", "Reanimate", "AllForOne", "FiendFire", "Stoke", "SecondWind", "Acrobatics", "CalculatedGamble",
        "Eidolon", "Cinder", "Seance", "Transfigure", "Scavenge", "MakeItSo", "SummonForth", "RightHandHand", "Armaments",
        "DualWield", "Havoc", "TrueGrit", "BurningPact", "Headbutt", "Exhume", "Survivor", "Prepared", "StormOfSteel",
        "ShadowStep", "Reboot", "DaggerThrow", "Anticipate", "Compact", "Aggression", "DrainPower", "BulletTime",
    ];

    static readonly string[] OrderRelics = ["WhisperingEarring", "GamblingChip", "MummifiedHand", "Toolbox"];

    static readonly Dictionary<(int, int, bool), List<Position>> Cache = [];

    /// <summary>
    /// Positions of random legal play, a snapshot before every step: the default fights on old and core2 (F08, F10), and
    /// with <paramref name="varied" /> the corpus's decks, relics and potions (tests/fixtures/combat) and synthetic decks
    /// that mix order readers and rolled cards into a corpus deck, with order relics and potions, on both engines.
    /// </summary>
    static List<Position> Positions(int seeds = 6, int steps = 30, bool varied = false)
    {
        lock (Cache)
        {
            if (Cache.TryGetValue((seeds, steps, varied), out var hit)) return hit;
        }

        var all = new List<Position>();
        foreach (var engine in new[] { "old", "core2" })
        {
            for (int seed = 1; seed <= seeds; seed++) Play(all, engine, seed, steps, null, "default");
        }

        if (varied)
        {
            var corpus = CorpusSetups().ToList();
            for (int i = 0; i < corpus.Count; i++) Play(all, i % 2 == 0 ? "old" : "core2", 300 + i, steps, corpus[i], "corpus");
            var r = new Random(4242);
            var order = OrderCards.Select(Cards.FindId).Where(x => x is not null).Select(x => x!.Value).ToArray();
            var relics = OrderRelics.Select(Relics.FindId).Where(x => x is not null).Select(x => x!.Value).ToArray();
            var potions = corpus.SelectMany(c => c.Potions).Concat(Enumerable.Range(1, 200).Where(UsablePotion)).Distinct().ToArray();
            for (int i = 0; i < 48; i++)
            {
                var basis = corpus[i % corpus.Count];
                var deck = basis.Deck.Concat(Enumerable.Range(0, 3 + r.Next(5)).Select(_ => order[r.Next(order.Length)])).ToArray();
                var rel = basis.Relics.Concat(relics.Where(_ => r.Next(3) == 0)).Distinct().ToArray();
                var pots = Enumerable.Range(0, 1 + r.Next(3)).Select(_ => potions.Length > 0 ? potions[r.Next(potions.Length)] : 0).Where(x => x != 0).ToArray();
                Play(all, i % 2 == 0 ? "old" : "core2", 900 + i, steps, (basis.Name, deck, rel, pots), "synthetic");
            }
        }

        lock (Cache) Cache[(seeds, steps, varied)] = all;
        return all;
    }

    static void Play(List<Position> all, string engine, int seed, int steps, (string Name, int[] Deck, int[] Relics, int[] Potions)? setup, string source)
    {
        var kernel = engine == "core2" ? new Kernel(KernelOptions.FromFamilies(["F08", "F10"]), () => Sts2Emulator.Tests.Core2.ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All) : null;
        var s = new CombatState();
        kernel?.Bind(s);
        var rng = new CountingRandom(seed * 101);
        try
        {
            if (setup is { } u) CombatFactory.Reset(s, rng, u.Deck, null, u.Relics, 80, 80, u.Potions);
            else CombatFactory.Reset(s, rng);
            CombatEngine.ResolvePrePlayPhase(s, rng);
        }
        catch (Exception)
        {
            return;  // a setup the engine does not take: no positions from it
        }

        var pick = new Random(seed * 7919);
        for (int i = 0; i < steps; i++)
        {
            all.Add(new Position(engine, s.Clone(), rng.Copy(), source));
            var moves = Moves(s);
            if (moves.Count == 0) break;
            // end turn less often, so turns hold several plays
            var (a, t) = moves[pick.Next(moves.Count)];
            if (a == s.Hand.Count && s.PendingSelection is null && moves.Count > 1 && pick.Next(3) > 0) (a, t) = moves[pick.Next(moves.Count)];
            try
            {
                if (CombatEngine.Step(s, a, rng, t).Terminal) break;
            }
            catch (Exception)
            {
                break;
            }
        }
    }

    /// <summary>The corpus's setups: each combat fixture's deck, relics and potions, by the game's ids.</summary>
    static IEnumerable<(string Name, int[] Deck, int[] Relics, int[] Potions)> CorpusSetups()
    {
        var dir = Path.Combine(KeyFieldCoverageTests.Repo(), "tests", "fixtures", "combat");
        foreach (var f in Directory.GetFiles(dir, "*.json").OrderBy(x => x, StringComparer.Ordinal))
        {
            JsonNode? p;
            try
            {
                p = JsonNode.Parse(File.ReadAllText(f))?["player"];
            }
            catch (Exception)
            {
                continue;
            }

            if (p is null) continue;
            int[] Ids(string key, Func<string, int?> find) =>
                [.. ((p[key] as JsonArray) ?? []).Select(x => x is JsonObject o ? (string?)o["id"] : null).Where(x => x is not null).Select(x => find(Pascal(x!))).Where(x => x is not null).Select(x => x!.Value)];
            var deck = Ids("deck", Cards.FindId);
            if (deck.Length < 5) continue;
            yield return (Path.GetFileName(f), deck, Ids("relics", Relics.FindId), Ids("potions", Potions.FindId));
        }
    }

    static bool UsablePotion(int id)
    {
        try
        {
            return Sts2Emulator.Core.Effects.PotionEffects.CanUseInCombat(id);
        }
        catch (Exception)
        {
            return false;
        }
    }

    static string Pascal(string id) => string.Concat(id.Split('_', StringSplitOptions.RemoveEmptyEntries).Select(w => w[..1] + w[1..].ToLowerInvariant()));

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
        public int Pairs, Steps, Positions, Exact, Truncated, UpToCut, Leaves;
        public readonly Dictionary<string, int> PerChange = new();
        public readonly List<string> Failures = [];
        public readonly List<string> UpToCutSteps = [];
    }

    // ------------------------------------------------------------------------------------------------ chance, exactly

    /// <summary>
    /// The step's chance source while its tree is enumerated: every stream of the state is cleared, so each roll comes
    /// here; the first <c>depth</c> rolls with more than one outcome answer from the prefix (then 0) and are recorded,
    /// each outcome at probability 1/n; later rolls, and Next() and NextDouble(), come from a fixed tail, the same on both
    /// sides of a pair.
    /// </summary>
    sealed class ChanceRandom(int[] prefix, int depth) : Random
    {
        readonly Random _tail = new(1234567);

        /// <summary>The first roll past the depth throws <see cref="CutReached" /> instead of answering from the tail.</summary>
        public bool StopAtCut;
        public readonly List<int> Ranges = [];
        public readonly List<int> Answers = [];
        public double P = 1;
        public bool Truncated;

        public override int Next(int maxValue)
        {
            if (maxValue <= 1) return 0;
            int k = Ranges.Count;
            if (k >= depth)
            {
                Truncated = true;
                if (StopAtCut) throw new CutReached();
                return _tail.Next(maxValue);
            }

            int a = k < prefix.Length ? Math.Min(prefix[k], maxValue - 1) : 0;
            Ranges.Add(maxValue);
            Answers.Add(a);
            P /= maxValue;
            return a;
        }

        public override int Next(int minValue, int maxValue) => minValue + Next(maxValue - minValue);
        public override int Next() => _tail.Next();
        public override double NextDouble() => _tail.NextDouble();
        protected override double Sample() => _tail.NextDouble();
        public override long NextInt64() => _tail.NextInt64();
        public override long NextInt64(long maxValue) => _tail.NextInt64(maxValue);
        public override long NextInt64(long minValue, long maxValue) => _tail.NextInt64(minValue, maxValue);
        public override float NextSingle() => _tail.NextSingle();
        public override void NextBytes(byte[] buffer) => _tail.NextBytes(buffer);
        public override void NextBytes(Span<byte> buffer) => _tail.NextBytes(buffer);
    }

    /// <summary>Thrown by a <see cref="ChanceRandom" /> that stops at its cut: the step is left where it stood.</summary>
    sealed class CutReached : Exception;

    static void ClearStreams(CombatState s)
    {
        s.ShuffleRng = null;
        s.TargetRng = null;
        s.CardSelectionRng = null;
        s.CardGenerationRng = null;
        s.PotionGenerationRng = null;
        s.NicheHpRng = null;
        s.EnergyCostRng = null;
        s.OrbGenerationRng = null;
        s.AiRng = null;
    }

    /// <summary>The successor distribution of one step: (result, successor key) to probability over the chance tree's
    /// first <paramref name="depth" /> branching rolls; null past <paramref name="budget" /> leaves. With
    /// <paramref name="stopAtCut" />, a step that reaches a roll past the depth ends there and its outcome is the state it
    /// stood in (its draw pile's order forgotten: a shuffle may be half done).</summary>
    static Dictionary<string, double>? Distribution(CombatState s0, int act, int target, int depth, int budget, out bool truncated, bool stopAtCut = false)
    {
        var dist = new Dictionary<string, double>();
        truncated = false;
        int[] prefix = [];
        for (int leaves = 1; ; leaves++)
        {
            if (leaves > budget) return null;
            var s = s0.Clone();
            ClearStreams(s);
            var cr = new ChanceRandom(prefix, depth) { StopAtCut = stopAtCut };
            string outcome;
            try
            {
                var res = CombatEngine.Step(s, act, cr, target);
                outcome = $"{res}|{Outcome(s, 3)}";
            }
            catch (CutReached)
            {
                var at = s.Clone();
                at.ForgetDrawOrder();
                outcome = $"cut|{Outcome(at, 3)}";
            }
            catch (Exception e)
            {
                outcome = $"throws {e.GetType().Name}";
            }

            truncated |= cr.Truncated;
            dist[outcome] = dist.GetValueOrDefault(outcome) + cr.P;
            int j = cr.Answers.Count - 1;
            while (j >= 0 && cr.Answers[j] + 1 >= cr.Ranges[j]) j--;
            if (j < 0) return dist;
            prefix = [.. cr.Answers.Take(j), cr.Answers[j] + 1];
        }
    }

    const int Budget = 96;

    /// <summary>The deepest roll count whose outcomes, as the first path through the tree branches, fit the budget.</summary>
    static int Depth(CombatState s0, int act, int target, int budget)
    {
        var s = s0.Clone();
        ClearStreams(s);
        var cr = new ChanceRandom([], 64);
        try
        {
            CombatEngine.Step(s, act, cr, target);
        }
        catch (Exception)
        {
        }

        long leaves = 1;
        int d = 0;
        while (d < cr.Ranges.Count && leaves * cr.Ranges[d] <= budget) leaves *= cr.Ranges[d++];
        return d == cr.Ranges.Count ? 64 : d;
    }

    /// <summary>
    /// A successor's outcome: its key; or, while a screen is open (the key then writes every pile in order, so two
    /// bisimilar successors can differ in it by their piles' order alone), the sorted outcomes of every choice it offers,
    /// a level deeper, to <paramref name="depth" /> screens.
    /// </summary>
    static string Outcome(CombatState s, int depth)
    {
        if (s.PendingSelection is null || depth == 0)
        {
            var k = Key(s);
            return k is null ? "refused" : Convert.ToHexString(SHA256.HashData(k));
        }

        var parts = new List<string>();
        foreach (var (act, target) in Moves(s))
        {
            var c = s.Clone();
            try
            {
                var res = CombatEngine.Step(c, act, new ChanceRandom([], 0), target);
                parts.Add($"{res}|{Outcome(c, depth - 1)}");
            }
            catch (Exception e)
            {
                parts.Add($"throws {e.GetType().Name}");
            }
        }

        parts.Sort(StringComparer.Ordinal);
        return $"screen[{string.Join(",", parts)}]";
    }

    static string Describe(CombatState a, int act) =>
        a.PendingSelection is not null ? $"choice {act}" : act < a.Hand.Count ? Cards.Get(a.Hand[act].DefId).Name
        : act == a.Hand.Count ? "end turn" : $"potion {a.PotionSlots[act - a.Hand.Count - 1]}";

    static bool SameDistribution(Dictionary<string, double> a, Dictionary<string, double> b) =>
        a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out var q) && Math.Abs(q - kv.Value) < 1e-9);

    /// <summary>
    /// Steps <paramref name="a" /> and its equal-key partner <paramref name="b" /> by every legal action of a (b's action is
    /// the one on the same card, found by uid where the change kept uids, else by index) and compares the successor
    /// distributions: the chance tree enumerated to the deepest roll count both sides fit in the leaf budget (whole, for
    /// most steps), the rest of the rolls answered alike; where that differs (a shuffle of a permuted pile past the cut),
    /// the two are compared up to the first roll past the cut instead. A uid-renumbered pair is also stepped on the
    /// state's own streams and must match there. No step is excused.
    /// </summary>
    static void CheckSuccessors(Position p, CombatState a, CombatState b, string change, Func<CombatState, CombatState, int, int> mapAction, Tally t)
    {
        var movesA = Moves(a);
        var movesB = Moves(b);
        if (movesA.Count != movesB.Count)
        {
            t.Failures.Add($"{p.Engine} {p.Source} {change}: {movesA.Count} legal moves against {movesB.Count}");
            return;
        }

        foreach (var (act, target) in movesA)
        {
            int actB = mapAction(a, b, act);
            t.Steps++;
            Dictionary<string, double>? da = null, db = null;
            bool cut = false;
            int used = 0;
            for (int depth = Depth(a, act, target, Budget); depth >= 0; depth--)
            {
                da = Distribution(a, act, target, depth, Budget, out var ca);
                if (da is null) continue;
                db = Distribution(b, actB, target, depth, Budget, out var cb);
                if (db is null) continue;
                cut = ca || cb;
                used = depth;
                break;
            }

            if (cut) t.Truncated++;
            else t.Exact++;
            t.Leaves += da?.Count ?? 0;
            if (da is not null && db is not null && cut && !SameDistribution(da, db))
            {
                // the tree is larger than the budget, and the rolls past the cut, answered alike, land on different cards
                // of the differently ordered pile (a reshuffle, a random pick of its cards): an artefact of the cut, not of
                // the key. Both sides reach the first roll past the cut by the same rolls, so they are compared there.
                da = Distribution(a, act, target, used, Budget, out _, stopAtCut: true);
                db = Distribution(b, actB, target, used, Budget, out _, stopAtCut: true);
                t.UpToCut++;
                t.UpToCutSteps.Add($"{p.Engine} {p.Source} turn {a.Turn} {change}: {Describe(a, act)}");
            }

            if (da is null || db is null || !SameDistribution(da, db))
            {
                t.Failures.Add($"{p.Engine} {p.Source} turn {a.Turn} {change}: action {act} target {target} gives different successor distributions (cut: {cut}; {Describe(a, act)}; {da?.Count} against {db?.Count} outcomes, screen {da?.Keys.Any(x => x.Contains("screen["))})");
                continue;
            }

            if (change != "uids renumbered") continue;
            var sa = a.Clone();
            var sb = b.Clone();
            CopyStreams(sa, sb);
            var resA = CombatEngine.Step(sa, act, p.Rng.Copy(), target);
            var resB = CombatEngine.Step(sb, actB, p.Rng.Copy(), target);
            if (resA != resB || !Same(Key(sa), Key(sb)))
                t.Failures.Add($"{p.Engine} {p.Source} turn {a.Turn} {change}: action {act} differs on the state's own streams");
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
    public void EqualKeysGiveEqualSuccessorDistributionsOnVariedPositions()
    {
        var t = new Tally();
        var r = new Random(35);
        var sources = new Dictionary<string, int>();
        int multiEnemy = 0, screens = 0, midTurn = 0, withRelics = 0, withPotions = 0, ordered = 0;
        foreach (var p in Positions(6, 30, varied: true))
        {
            var s = p.State;
            var key = Key(s);
            if (key is null) continue;  // a busy kernel queue: no key, nothing to compare
            t.Positions++;
            sources[p.Source] = sources.GetValueOrDefault(p.Source) + 1;
            if (s.Enemies.Count > 1) multiEnemy++;
            if (s.PendingSelection is not null) screens++;
            if (s.Energy != s.MaxEnergy) midTurn++;
            if (s.Relics.Count > 1) withRelics++;
            if (s.PotionSlots.Any(x => x != 0)) withPotions++;
            if (StateKey.HandOrdered(s) || StateKey.DiscardOrdered(s)) ordered++;
            var changes = new List<(string Name, CombatState B, Func<CombatState, CombatState, int, int> Map)>();
            if (s.PendingSelection?.Replay is null) changes.Add(("uids renumbered", RenumberUids(s, r), SameIndex));
            if (!StateKey.HandOrdered(s) && s.Hand.Count > 1)
                changes.Add(("hand permuted", Permute(s, r, x => (null, x.Hand, 0, x.Hand.Count)), ByUid));
            if (!StateKey.DiscardOrdered(s) && s.DiscardPile.Count > 1)
                changes.Add(("discard permuted", Permute(s, r, x => (null, x.DiscardPile, 0, x.DiscardPile.Count)), SameIndex));
            if (!StateKey.ExhaustOrdered(s) && s.ExhaustPile.Count > 1)
                changes.Add(("exhaust permuted", Permute(s, r, x => (null, x.ExhaustPile, 0, x.ExhaustPile.Count)), SameIndex));
            if (!StateKey.PotionOrdered(s))
                changes.Add(("potions permuted", Permute(s, r, x => (x.PotionSlots, null, 0, Math.Min(x.MaxPotionSlots, x.PotionSlots.Length))), BySlot));
            // every permutation, and the uid renumbering on one position in three (or where nothing else applies)
            if (changes.Count > 1 && changes[0].Name == "uids renumbered" && r.Next(3) > 0) changes.RemoveAt(0);
            foreach (var (name, b, map) in changes)
            {
                t.Pairs++;
                t.PerChange[name] = t.PerChange.GetValueOrDefault(name) + 1;
                if (!Same(key, Key(b)))
                {
                    t.Failures.Add($"{p.Engine} {p.Source} turn {s.Turn} {name}: the key changed");
                    continue;
                }

                CheckSuccessors(p, s, b, name, map, t);
            }
        }

        output.WriteLine($"positions {t.Positions} ({string.Join(", ", sources.Select(kv => $"{kv.Key} {kv.Value}"))}; multi-enemy {multiEnemy}, open screens {screens}, mid-turn {midTurn}, "
            + $"relics {withRelics}, potions {withPotions}, an order reader {ordered}), equal-key pairs {t.Pairs} ({string.Join(", ", t.PerChange.Select(kv => $"{kv.Key} {kv.Value}"))}), "
            + $"successor steps {t.Steps} (chance enumerated whole {t.Exact}, to a depth {t.Truncated}, of those compared up to the first roll past the cut {t.UpToCut}; "
            + $"outcomes {t.Leaves}), failures {t.Failures.Count}, tolerated 0 ({string.Join("; ", t.UpToCutSteps.Take(6))})");
        foreach (var f in t.Failures.Take(60)) output.WriteLine(f);
        Assert.True(t.Pairs >= 1000, $"only {t.Pairs} equal-key pairs");
        Assert.True(multiEnemy > 50 && screens > 10 && midTurn > 200 && withPotions > 50, "the positions are not varied enough");
        Assert.True(t.Failures.Count == 0, string.Join("\n", t.Failures.Take(40)));
    }

    /// <summary>Swaps two cards of one pile that differ only in their uid (the only permutation a hindsight key allows).</summary>
    static CombatState? SwapEqualCards(CombatState s0, Random r, Func<CombatState, List<CardInstance>> pile)
    {
        var s = s0.Clone();
        var cards = pile(s);
        var pairs = new List<(int, int)>();
        for (int i = 0; i < cards.Count; i++)
        {
            for (int j = i + 1; j < cards.Count; j++)
            {
                if (cards[i] with { Uid = 0 } == cards[j] with { Uid = 0 } && cards[i].Uid != cards[j].Uid) pairs.Add((i, j));
            }
        }

        if (pairs.Count == 0) return null;
        var (x, y) = pairs[r.Next(pairs.Count)];
        (cards[x], cards[y]) = (cards[y], cards[x]);
        return s;
    }

    [Fact]
    public void EqualHindsightKeysGiveEqualSuccessorsOnTheSameStreams()
    {
        // Hindsight mode keys the streams and keeps every rolled pile ordered, so two positions with one hindsight key
        // must step to one hindsight key under the same streams, every step, strictly.
        var t = new Tally();
        var r = new Random(37);
        int refused = 0;
        foreach (var p in Positions(6, 30, varied: true))
        {
            var s = p.State;
            var key = Key(s, KeyMode.Hindsight);
            if (key is null) continue;
            t.Positions++;
            var changes = new List<(string, CombatState?, Func<CombatState, CombatState, int, int>)>
            {
                ("uids renumbered", s.PendingSelection?.Replay is null ? RenumberUids(s, r) : null, SameIndex),
                ("equal hand cards swapped", SwapEqualCards(s, r, x => x.Hand), SameIndex),
                ("equal discard cards swapped", SwapEqualCards(s, r, x => x.DiscardPile), SameIndex),
            };
            var q = Permute(s, r, x => (null, x.Hand, 0, x.Hand.Count));
            if (!q.Hand.Select(c => c with { Uid = 0 }).SequenceEqual(s.Hand.Select(c => c with { Uid = 0 })))
            {
                if (Same(key, Key(q, KeyMode.Hindsight))) t.Failures.Add($"{p.Engine} {p.Source} turn {s.Turn}: a permuted hand kept its hindsight key");
                else refused++;
            }
            foreach (var (name, b, map) in changes)
            {
                if (b is null) continue;
                t.Pairs++;
                t.PerChange[name] = t.PerChange.GetValueOrDefault(name) + 1;
                if (!Same(key, Key(b, KeyMode.Hindsight)))
                {
                    t.Failures.Add($"{p.Engine} {p.Source} turn {s.Turn} {name}: the hindsight key changed");
                    continue;
                }

                foreach (var (act, target) in Moves(s))
                {
                    int actB = map(s, b, act);
                    var sa = s.Clone();
                    var sb = b.Clone();
                    CopyStreams(sa, sb);
                    var resA = CombatEngine.Step(sa, act, p.Rng.Copy(), target);
                    var resB = CombatEngine.Step(sb, actB, p.Rng.Copy(), target);
                    t.Steps++;
                    if (resA != resB || !Same(Key(sa, KeyMode.Hindsight), Key(sb, KeyMode.Hindsight)))
                        t.Failures.Add($"{p.Engine} {p.Source} turn {s.Turn} {name}: action {act} target {target} gives different hindsight successors");
                }
            }
        }

        output.WriteLine($"hindsight: positions {t.Positions}, equal-key pairs {t.Pairs} ({string.Join(", ", t.PerChange.Select(kv => $"{kv.Key} {kv.Value}"))}), "
            + $"successor steps {t.Steps}, permuted hands keyed apart {refused}, failures {t.Failures.Count}");
        Assert.True(t.Pairs >= 1000 && t.Steps > 5000, $"only {t.Pairs} pairs, {t.Steps} steps");
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
