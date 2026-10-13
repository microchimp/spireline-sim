using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

namespace Sts2Emulator.Tests.Invariants;

/// <summary>One broken relation: which, at which step, on which card, and the first differing paths.</summary>
public sealed record Failure(string Relation, int Step, string Card, string Detail);

/// <summary>Counts how often each relation was checked and keeps what it caught.</summary>
public sealed class RelationReport
{
    public Dictionary<string, int> Instances { get; } = new();

    /// <summary>Wall time spent in each relation, in milliseconds.</summary>
    public Dictionary<string, long> Millis { get; } = new();

    public List<Failure> Failures { get; } = [];

    public void Count(string relation) => Instances[relation] = Instances.GetValueOrDefault(relation) + 1;

    public void Fail(string relation, int step, string card, IEnumerable<string> detail) =>
        Failures.Add(new Failure(relation, step, card, string.Join(" ; ", detail.Take(4))));

    public void Fail(string relation, int step, string card, string detail) =>
        Failures.Add(new Failure(relation, step, card, detail));
}

/// <summary>
/// Relations that must hold of the emulator alone: a transformation of a fight that must not change
/// its outcome, or must change it in a known way. Each takes a function that opens the same fresh fight
/// every time, so a relation can build as many copies as it needs.
/// </summary>
/// <remarks>
/// They need no game. A failure is either an emulator bug (state that leaks, a copy that forgot a field,
/// an effect that reads a thing it should not) or a legitimate dependence the relation did not model
/// (a card that looks at the size of the draw pile, a random pick by position); triage tells them apart.
/// </remarks>
public static class Metamorphic
{
    public const int MaxSteps = 120;

    /// <summary>How many differing lines a failure quotes (raised to debug one).</summary>
    public static int DiffLimit { get; set; } = 4;

    public static readonly string[] Names =
        ["invariants", "determinism", "clone", "pad", "twins", "symmetry", "streams", "handorder", "discardorder", "energy", "monotone"];

    private static string Digest(Combat c, params string[] skip) => DeepState.Digest(c.Whole, skip);

    /// <summary>Runs the named relations (all when null), turning an engine exception into a failure.</summary>
    /// <param name="open">Opens the fight; every relation but determinism may be given a copy of one opening.</param>
    /// <param name="fresh">Opens the fight from scratch, for determinism (<paramref name="open"/> when null).</param>
    public static void Run(
        Func<Combat> open, int chooserSeed, RelationReport report, IEnumerable<string>? only = null, Func<Combat>? fresh = null)
    {
        var wanted = only?.ToHashSet() ?? Names.ToHashSet();
        foreach (string name in Names.Where(wanted.Contains))
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                switch (name)
                {
                    case "invariants": Invariants(open, chooserSeed, report); break;
                    case "determinism": Determinism(fresh ?? open, chooserSeed, report); break;
                    case "clone": CloneFidelity(open, chooserSeed, report); break;
                    case "pad": PadBottom(open, chooserSeed, report); break;
                    case "twins": Twins(open, chooserSeed, report); break;
                    case "symmetry": Symmetry(open, chooserSeed, report); break;
                    case "streams": StreamShift(open, chooserSeed, report); break;
                    case "handorder": HandOrder(open, chooserSeed, report); break;
                    case "discardorder": DiscardOrder(open, chooserSeed, report); break;
                    case "energy": EnergySlack(open, chooserSeed, report); break;
                    case "monotone": MoreIsBetter(open, chooserSeed, report); break;
                }
            }
            catch (Exception e)
            {
                // The first frames say where it broke: the engine's, or the relation's own.
                string where = string.Join(
                    " < ",
                    (e.StackTrace ?? "")
                        .Split('\n')
                        .Where(l => l.Contains("Sts2Emulator."))
                        .Take(3)
                        .Select(l => l.Trim().Replace("at Sts2Emulator.", "").Split(" in ")[0])
                );
                report.Fail("exception", -1, name, $"{e.GetType().Name}: {e.Message.Split('\n')[0]} [{where}]");
            }

            report.Millis[name] = report.Millis.GetValueOrDefault(name) + clock.ElapsedMilliseconds;
        }
    }

    /// <summary>The live-state invariants after every step of one fight (see <see cref="StateInvariants"/>).</summary>
    public static void Invariants(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var chooser = new Random(seed);
        var tracker = new StateInvariants.Tracker();
        rep.Count("invariants");
        Note(0, "OPEN", tracker.Step(c.State), rep);
        for (int i = 1; i <= MaxSteps && !c.Over && RandomPlay.Pick(c.State, chooser) is { } m; i++)
        {
            string card = c.Describe(m);
            c.Step(m);
            if (c.Over)
            {
                break;
            }

            Note(i, card, tracker.Step(c.State), rep);
        }
    }

    private static void Note(int step, string card, List<Violation> found, RelationReport rep)
    {
        foreach (var v in found)
        {
            rep.Fail("invariant:" + v.Check, step, card, v.Detail);
        }
    }

    /// <summary>
    /// Two fresh copies of one fight, played in lockstep with one chooser, are the same state after every
    /// step. Interleaving the steps exposes state they would share (a static, a cached list).
    /// </summary>
    public static void Determinism(Func<Combat> open, int seed, RelationReport rep)
    {
        var a = open();
        var b = open();
        var ra = new Random(seed);
        var rb = new Random(seed);
        rep.Count("determinism");
        if (Digest(a) != Digest(b))
        {
            rep.Fail("determinism", 0, "OPEN", DeepState.Diff(a.Whole, b.Whole));
            return;
        }

        for (int i = 1; i <= MaxSteps && !a.Over && !b.Over; i++)
        {
            var ma = RandomPlay.Pick(a.State, ra);
            var mb = RandomPlay.Pick(b.State, rb);
            if (ma != mb)
            {
                rep.Fail("determinism", i, "CHOOSER", $"legal moves differ: {ma} vs {mb}");
                return;
            }

            if (ma is null)
            {
                return;
            }

            string card = a.Describe(ma.Value);
            a.Step(ma.Value);
            b.Step(mb.Value);
            if (Digest(a) != Digest(b))
            {
                rep.Fail("determinism", i, card, DeepState.Diff(a.Whole, b.Whole));
                return;
            }
        }
    }

    /// <summary>
    /// A copy taken mid-fight is the same state, stays the same while the original plays on (it shares
    /// nothing), and then plays the same actions to the same states.
    /// </summary>
    public static void CloneFidelity(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var chooser = new Random(seed);
        RandomPlay.Play(c, chooser, chooser.Next(0, 25));
        if (c.Over)
        {
            return;
        }

        rep.Count("clone");
        var clone = c.Clone();
        var frozen = DeepState.Lines(clone.Whole);
        if (!frozen.SequenceEqual(DeepState.Lines(c.Whole)))
        {
            rep.Fail("clone", 0, "CUT", DeepState.Diff(c.Whole, clone.Whole));
            return;
        }

        var moves = new List<Move>();
        var cards = new List<string>();
        var digests = new List<string>();
        while (moves.Count < 60 && !c.Over && RandomPlay.Pick(c.State, chooser) is { } m)
        {
            cards.Add(c.Describe(m));
            c.Step(m);
            moves.Add(m);
            digests.Add(Digest(c));
        }

        if (!DeepState.Lines(clone.Whole).SequenceEqual(frozen))
        {
            var now = DeepState.Lines(clone.Whole);
            var changed = frozen.Except(now).Concat(now.Except(frozen)).Take(4);
            rep.Fail("clone", moves.Count, "ORIGINAL-RAN", "the copy changed while the original played: " + string.Join(" ; ", changed));
            return;
        }

        for (int i = 0; i < moves.Count; i++)
        {
            clone.Step(moves[i]);
            if (Digest(clone) != digests[i])
            {
                rep.Fail("clone", i + 1, cards[i], "the copy plays differently from the original");
                return;
            }
        }
    }

    /// <summary>
    /// A card added at the bottom of the draw pile, never drawn, changes nothing else: compared with the
    /// pad taken out, the two fights stay the same state until the pile runs down or is reshuffled.
    /// </summary>
    public static void PadBottom(Func<Combat> open, int seed, RelationReport rep)
    {
        var a = open();
        if (a.State.DrawPile.Count == 0)
        {
            return;
        }

        var b = a.Clone();
        // A Status card: no card that picks by type (an Attack or a Skill from the pile) or by tag (Strikes) can take it.
        var pad = new CardInstance(Cards.All.ToArray().First(c => c.Type == CardType.Status).Id, false);
        b.State.DrawPile.Add(pad);
        var chooser = new Random(seed);
        rep.Count("pad");

        // The known-order counters are clamped to the pile's length, so they are not part of this relation.
        // ExecutedThisEnemyTurn is the engine's own log of the card counts it saw, and the pad is one more.
        string[] skip = [".Item1.KnownTopCount", ".Item1.KnownBottomCount", ".Item1.ExecutedThisEnemyTurn"];
        for (int i = 1; i <= MaxSteps && !a.Over && !b.Over; i++)
        {
            var m = RandomPlay.Pick(a.State, chooser);
            if (m is null || !CombatEngine.ValidActions(b.State).SequenceEqual(CombatEngine.ValidActions(a.State)))
            {
                return;
            }

            string card = a.Describe(m.Value);
            int before = a.State.DrawPile.Count;
            a.Step(m.Value);
            b.Step(m.Value);
            if (a.Over || b.Over)
            {
                // The fight ending is an outcome too: both must end together.
                if (a.Over != b.Over)
                {
                    rep.Fail("pad", i, card, "only one of the fights ended");
                }

                return;
            }

            // Out of scope from here: the pile was reshuffled, ran out, or took a card at a random place
            // (whose range depends on the pile's size by design).
            if ((a.State.ShuffleRng?.CallCount ?? 0) != (b.State.ShuffleRng?.CallCount ?? 0)
                || a.State.DrawPile.Count == 0
                || a.State.DrawPile.Count > before)
            {
                return;
            }

            // A card shuffled in at a random place can land below the pad (its range depends on the pile's
            // size by design), and the pad is drawn when the pile runs down: past either, nothing is owed.
            var last = b.State.DrawPile.Count - 1;
            if (last < 0 || b.State.DrawPile[last] != pad)
            {
                return;
            }

            b.State.DrawPile.RemoveAt(last);
            var diff = DeepState.Diff(a.Whole, b.Whole, DiffLimit, skip);
            b.State.DrawPile.Add(pad);
            if (diff.Count > 0)
            {
                rep.Fail("pad", i, card, diff);
                return;
            }
        }
    }

    /// <summary>
    /// Playing either of two identical cards leaves the same state, bar the order of the hand: the game
    /// has no way to tell twins apart.
    /// </summary>
    public static void Twins(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var chooser = new Random(seed);
        RandomPlay.Play(c, chooser, chooser.Next(0, 12));
        if (c.Over || c.State.PendingSelection is not null)
        {
            return;
        }

        var valid = CombatEngine.ValidActions(c.State).ToHashSet();
        var pairs = new List<(int, int)>();
        for (int i = 0; i < c.State.Hand.Count; i++)
        {
            for (int j = i + 1; j < c.State.Hand.Count; j++)
            {
                if (c.State.Hand[i] == c.State.Hand[j] && valid.Contains(i) && valid.Contains(j))
                {
                    pairs.Add((i, j));
                }
            }
        }

        foreach (var (i, j) in pairs.OrderBy(_ => chooser.Next()).Take(2))
        {
            int target = RandomPlay.TargetFor(c.State, i, chooser);
            var x = c.Clone();
            var y = c.Clone();
            string card = c.Describe(new Move(i, target));
            x.Step(new Move(i, target));
            y.Step(new Move(j, target));
            rep.Count("twins");
            var rest = DeepState.Diff(x.Whole, y.Whole, DiffLimit, ".Item1.Hand[");
            if (rest.Count > 0)
            {
                rep.Fail("twins", 1, card, rest);
                continue;
            }

            if (!SortedHand(x).SequenceEqual(SortedHand(y)))
            {
                rep.Fail("twins", 1, card, "the hands hold different cards");
            }
        }
    }

    private static List<string> SortedHand(Combat c) =>
        c.State.Hand.Select(card => string.Join("|", DeepState.Lines(card))).OrderBy(s => s, StringComparer.Ordinal).ToList();

    /// <summary>
    /// Two enemies of one kind made identical: hitting the first or hitting the second gives the same
    /// fight with the two swapped. Judged only where no random pick of a target by position took part.
    /// </summary>
    public static void Symmetry(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var enemies = c.State.Enemies;
        (int, int)? pair = null;
        for (int i = 0; i < enemies.Count && pair is null; i++)
        {
            for (int j = i + 1; j < enemies.Count; j++)
            {
                if (enemies[i].DefId == enemies[j].DefId && enemies[i].Hp > 0 && enemies[j].Hp > 0)
                {
                    pair = (i, j);
                    break;
                }
            }
        }

        if (pair is not var (p, q))
        {
            return;
        }

        enemies[q] = enemies[p].Clone();
        if (!DeepState.Lines(enemies[p]).SequenceEqual(DeepState.Lines(enemies[q])))
        {
            return; // the enemy carries something that tells it apart from a copy of itself
        }

        var chooser = new Random(seed);
        var cards = Enumerable.Range(0, c.State.Hand.Count)
            .Where(h => CombatEngine.ValidActions(c.State).Contains(h)
                && Cards.Get(c.State.Hand[h].DefId).Target == CardTarget.AnyEnemy)
            .OrderBy(_ => chooser.Next())
            .Take(3);
        foreach (int h in cards)
        {
            var x = c.Clone();
            var y = c.Clone();
            string card = x.Describe(new Move(h, -1));
            int rolls = c.State.TargetRng?.CallCount ?? 0;
            x.Step(new Move(h, p));
            y.Step(new Move(h, q));
            if ((x.State.TargetRng?.CallCount ?? 0) != rolls || (y.State.TargetRng?.CallCount ?? 0) != rolls)
            {
                continue; // a random pick by position: the mirror image is not owed
            }

            if (x.State.Enemies.Count != c.State.Enemies.Count || y.State.Enemies.Count != c.State.Enemies.Count)
            {
                continue; // an enemy left the fight, and with it the positions the mirror swaps
            }

            rep.Count("symmetry");
            var mirror = y.Clone();
            (mirror.State.Enemies[p], mirror.State.Enemies[q]) = (mirror.State.Enemies[q], mirror.State.Enemies[p]);
            var d = DeepState.Diff(x.Whole, mirror.Whole, DiffLimit);
            if (d.Count > 0)
            {
                rep.Fail("symmetry", 1, card, d);
                continue;
            }

            // And through the enemies' turn, when none of them rolled for it.
            if (x.Over || y.Over)
            {
                continue;
            }

            int ai = x.State.AiRng is CountingRandom r ? r.CallCount : 0;
            x.Step(new Move(x.State.Hand.Count, -1));
            mirror.Step(new Move(mirror.State.Hand.Count, -1));
            int aiAfter = x.State.AiRng is CountingRandom r2 ? r2.CallCount : 0;
            if (x.Over || mirror.Over || aiAfter != ai || (x.State.TargetRng?.CallCount ?? 0) != rolls
                || x.State.Enemies.Count != mirror.State.Enemies.Count)
            {
                continue;
            }

            rep.Count("symmetry");
            d = DeepState.Diff(x.Whole, mirror.Whole, DiffLimit);
            if (d.Count > 0)
            {
                rep.Fail("symmetry", 2, "END_TURN after " + card, d);
            }
        }
    }

    private static int Rolls(Combat c) => StateInvariants.Streams(c.State).Sum(s => s.Rng?.CallCount ?? 0);

    /// <summary>A shuffled copy of the list, drawn from <paramref name="r"/>.</summary>
    private static List<T> Shuffled<T>(IEnumerable<T> items, Random r) => items.OrderBy(_ => r.Next()).ToList();

    /// <summary>
    /// The order of the cards in the hand is the player's view, not the engine's: playing a card from a hand
    /// put in another order, then ending the turn, leaves the same fight bar the order of the piles it
    /// went to. Judged only where nothing rolled (a random pick by position is owed no such thing).
    /// </summary>
    public static void HandOrder(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var chooser = new Random(seed);
        RandomPlay.Play(c, chooser, chooser.Next(0, 10));
        if (c.Over || c.State.PendingSelection is not null || c.State.Hand.Count < 3)
        {
            return;
        }

        var playable = CombatEngine.ValidActions(c.State).Where(a => a < c.State.Hand.Count).ToArray();
        if (playable.Length == 0)
        {
            return;
        }

        int i = playable[chooser.Next(playable.Length)];
        int target = RandomPlay.TargetFor(c.State, i, chooser);
        string card = c.Describe(new Move(i, target));
        var order = Shuffled(Enumerable.Range(0, c.State.Hand.Count), chooser);
        var x = c.Clone();
        var y = c.Clone();
        y.State.Hand = order.Select(k => c.State.Hand[k]).ToList();
        int j = order.IndexOf(i);
        int before = Rolls(x);
        x.Step(new Move(i, target));
        y.Step(new Move(j, target));
        if (Rolls(x) != before || x.State.PendingSelection is not null || x.Over || y.Over)
        {
            return; // a roll or a screen over the hand's positions: not owed
        }

        rep.Count("handorder");
        string[] skip = [".Item1.Hand[", ".Item1.DiscardPile[", ".Item1.ExhaustPile["];
        var d = DeepState.Diff(x.Whole, y.Whole, DiffLimit, skip);
        if (d.Count > 0 || !SamePiles(x, y))
        {
            rep.Fail("handorder", 1, card, d.Count > 0 ? d : ["a pile holds different cards"]);
            return;
        }

        if (HasHiddenTwins(x.State) || PicksByPosition(x.State))
        {
            return; // cards alike in name and upgrade but not in every field keep their relative order through a shuffle
        }

        x.Step(new Move(x.State.Hand.Count, -1));
        y.Step(new Move(y.State.Hand.Count, -1));
        if (x.Over || y.Over || Rolls(x) != Rolls(y))
        {
            return;
        }

        d = DeepState.Diff(x.Whole, y.Whole, DiffLimit, skip);
        if (d.Count > 0 || !SamePiles(x, y))
        {
            rep.Fail(
                "handorder", 2, "END_TURN after " + card,
                (d.Count > 0 ? d : ["a pile holds different cards"]).Append($"x {Piles(x)} | y {Piles(y)}"));
        }
    }

    /// <summary>
    /// Powers that pick a card at random BY POSITION in a pile, so the order of that pile is their input by
    /// design: StampedePower rolls an index into the attacks of the hand, AggressionPower takes the first
    /// of an UnstableShuffle (no sort first) of the discard pile.
    /// </summary>
    private static bool PicksByPosition(CombatState s) =>
        BuffSystem.Get(s.PlayerBuffs, BuffId.Stampede) > 0 || BuffSystem.Get(s.PlayerBuffs, BuffId.Aggression) > 0;

    /// <summary>
    /// Whether two cards of one name and upgrade differ in anything else (a bump, an enchantment, a growth).
    /// A shuffle sorts by name and upgrade only, so such cards keep the relative order the piles held them in.
    /// </summary>
    private static bool HasHiddenTwins(CombatState s) =>
        StateInvariants.Piles(s).SelectMany(p => p.Pile).GroupBy(k => (k.DefId, k.Upgraded)).Any(g => g.Distinct().Count() > 1);

    /// <summary>The piles in a line, by def id: hand | draw (first eight) | discard.</summary>
    private static string Piles(Combat c) =>
        $"hand {string.Join(",", c.State.Hand.Select(k => k.DefId))} draw {string.Join(",", c.State.DrawPile.Take(8).Select(k => k.DefId))}"
        + $" discard {string.Join(",", c.State.DiscardPile.Select(k => k.DefId))} shuffled {c.State.ShuffleRng?.CallCount}"
        + $" buffs {string.Join(",", c.State.PlayerBuffs.Select(b => $"{b.Id}{b.Magnitude}"))}"
        + $" relics {string.Join(",", c.State.Relics.Select(r => r.DefId))}";

    private static bool SamePiles(Combat x, Combat y)
    {
        List<string> Sorted(List<CardInstance> pile) =>
            pile.Select(card => string.Join("|", DeepState.Lines(card))).OrderBy(t => t, StringComparer.Ordinal).ToList();

        return Sorted(x.State.Hand).SequenceEqual(Sorted(y.State.Hand))
            && Sorted(x.State.DiscardPile).SequenceEqual(Sorted(y.State.DiscardPile))
            && Sorted(x.State.ExhaustPile).SequenceEqual(Sorted(y.State.ExhaustPile));
    }

    /// <summary>
    /// The discard pile's order does not reach the future: the reshuffle sorts the pile before it shuffles
    /// (the game's StableShuffle), so a discard put in another order, with each card kind keeping its own
    /// order, is the same fight, draw pile included, once the pile is shuffled back.
    /// </summary>
    public static void DiscardOrder(Func<Combat> open, int seed, RelationReport rep)
    {
        var c = open();
        var chooser = new Random(seed);
        for (int n = 0; n < 60 && !c.Over && c.State.DiscardPile.Count < 4 && RandomPlay.Pick(c.State, chooser) is { } m; n++)
        {
            c.Step(m);
        }

        if (c.Over || c.State.DiscardPile.Count < 4 || c.State.PendingSelection is not null || HasHiddenTwins(c.State))
        {
            return;
        }

        var kinds = c.State.DiscardPile.GroupBy(k => k.DefId).Select(g => g.ToList()).ToList();
        var x = c.Clone();
        var y = c.Clone();
        y.State.DiscardPile = Shuffled(kinds, chooser).SelectMany(g => g).ToList();
        rep.Count("discardorder");
        string[] skip = [".Item1.DiscardPile["];
        for (int i = 1; i <= 80 && !x.Over && !y.Over && RandomPlay.Pick(x.State, chooser) is { } move; i++)
        {
            // An open screen over the discard pile names its cards by position, and AggressionPower picks
            // with UnstableShuffle: both read the pile's order by design.
            if (x.State.PendingSelection is not null || BuffSystem.Get(x.State.PlayerBuffs, BuffId.Aggression) > 0)
            {
                return;
            }

            string card = x.Describe(move);
            x.Step(move);
            y.Step(move);
            if (HasHiddenTwins(x.State))
            {
                return;
            }

            var d = DeepState.Diff(x.Whole, y.Whole, DiffLimit, skip);
            if (d.Count > 0)
            {
                rep.Fail("discardorder", i, card, d.Append($"x {Piles(x)} | y {Piles(y)}"));
                return;
            }

            if (!SamePiles(x, y))
            {
                rep.Fail("discardorder", i, card, "a pile holds different cards");
                return;
            }
        }
    }

    /// <summary>
    /// Energy to spare changes nothing but the energy: two more points at the start of the turn, the same
    /// plays, the same fight with two more left over. Stops at an X-cost card, which spends what is there.
    /// </summary>
    public static void EnergySlack(Func<Combat> open, int seed, RelationReport rep)
    {
        var x = open();
        var y = x.Clone();
        const int Extra = 2;
        y.State.Energy += Extra;
        var chooser = new Random(seed);
        rep.Count("energy");
        for (int i = 1; i <= 40 && !x.Over && !y.Over; i++)
        {
            var move = RandomPlay.Pick(x.State, chooser);
            if (move is null || move.Value.Action >= x.State.Hand.Count)
            {
                return; // the end of the turn hands energy to relics that keep it: out of scope
            }

            var def = Cards.Get(x.State.Hand[move.Value.Action].DefId);
            if (def.HasEnergyCostX || def.StarCost > 0)
            {
                return;
            }

            x.Step(move.Value);
            y.Step(move.Value);
            if (x.Over || y.Over)
            {
                return;
            }

            // EnergySpentThisTurn and its kin do not depend on what was left; the energy itself does.
            var d = DeepState.Diff(x.Whole, y.Whole, DiffLimit, ".Item1.Energy");
            if (d.Count > 0 || y.State.Energy != x.State.Energy + Extra)
            {
                rep.Fail("energy", i, def.Entry, d.Count > 0 ? d : [$"energy {x.State.Energy} vs {y.State.Energy}"]);
                return;
            }
        }
    }

    /// <summary>
    /// More is better, in the direction the game says: block spares hit points (by at most the block
    /// added), strength never leaves an enemy with more. The same plays through the first turn.
    /// </summary>
    public static void MoreIsBetter(Func<Combat> open, int seed, RelationReport rep)
    {
        const int ExtraBlock = 9;
        const int ExtraStrength = 3;
        var low = open();
        var chooser = new Random(seed);

        // Block: put on just before the enemies' turn, so no card of the player's can read it.
        var a = low.Clone();
        var local = new Random(chooser.Next());
        while (!a.Over && RandomPlay.Pick(a.State, local) is { } m
               && !(m.Action == a.State.Hand.Count && a.State.PendingSelection is null))
        {
            a.Step(m);
        }

        if (!a.Over && a.State.PendingSelection is null)
        {
            var b = a.Clone();
            b.State.PlayerBlock += ExtraBlock;
            int hp = a.State.PlayerHp;
            a.Step(new Move(a.State.Hand.Count, -1));
            b.Step(new Move(b.State.Hand.Count, -1));
            if (!a.Over && !b.Over)
            {
                rep.Count("monotone");
                int saved = b.State.PlayerHp - a.State.PlayerHp;
                if (saved < 0 || saved > ExtraBlock)
                {
                    rep.Fail("monotone", 0, "BLOCK", $"{ExtraBlock} more block before the enemies' turn changed hit points by {saved} (from {hp})");
                }
            }
        }

        // Strength: no enemy ends with more health.
        var c = low.Clone();
        var d = low.Clone();
        BuffSystem.Apply(d.State.PlayerBuffs, BuffId.Strength, ExtraStrength);
        PlayThroughFirstTurn(c, d, chooser, stopAtEndTurn: true);
        if (!c.Over && !d.Over && c.State.Enemies.Count == d.State.Enemies.Count)
        {
            rep.Count("monotone");
            for (int e = 0; e < c.State.Enemies.Count; e++)
            {
                if (d.State.Enemies[e].Hp > c.State.Enemies[e].Hp)
                {
                    rep.Fail("monotone", 0, "STRENGTH", $"enemy {e}: {c.State.Enemies[e].Hp} hp, with {ExtraStrength} more strength {d.State.Enemies[e].Hp}");
                }
            }
        }
    }

    /// <summary>Both fights make the same plays (chosen on <paramref name="x"/>) up to and through the end of the turn.</summary>
    private static void PlayThroughFirstTurn(Combat x, Combat y, Random chooser, bool stopAtEndTurn)
    {
        int turn = x.State.Turn;
        var local = new Random(chooser.Next());
        for (int i = 0; i < 40 && !x.Over && !y.Over; i++)
        {
            var move = RandomPlay.Pick(x.State, local);
            if (move is null)
            {
                return;
            }

            bool endTurn = move.Value.Action == x.State.Hand.Count && x.State.PendingSelection is null;
            if (stopAtEndTurn && endTurn)
            {
                return;
            }

            if (!CombatEngine.ValidActions(y.State).Contains(move.Value.Action))
            {
                return;
            }

            x.Step(move.Value);
            y.Step(move.Value);
            if (endTurn || x.State.Turn != turn)
            {
                return;
            }
        }
    }

    /// <summary>
    /// A stream the fight never draws from can be moved ahead without changing anything else: a state
    /// that depends on it anyway reads a stream it should not, or reads one without counting.
    /// </summary>
    public static void StreamShift(Func<Combat> open, int seed, RelationReport rep)
    {
        var baseline = open();
        var start = StateInvariants.Streams(baseline.State).Where(s => s.Rng is not null).ToDictionary(s => s.Name, s => s.Rng!.CallCount);
        var moves = RandomPlay.Play(baseline, new Random(seed), MaxSteps);
        var untouched = StateInvariants.Streams(baseline.State)
            .Where(s => s.Rng is not null && s.Rng.CallCount == start[s.Name])
            .Select(s => s.Name)
            .ToList();
        var fields = new Dictionary<string, string>
        {
            ["Shuffle"] = "ShuffleRng", ["CombatTargets"] = "TargetRng", ["CombatCardSelection"] = "CardSelectionRng",
            ["CombatCardGeneration"] = "CardGenerationRng", ["CombatPotionGeneration"] = "PotionGenerationRng",
            ["CombatEnergyCosts"] = "EnergyCostRng", ["CombatOrbGeneration"] = "OrbGenerationRng",
            ["Niche"] = "NicheHpRng", ["MonsterAi"] = "AiRng",
        };
        foreach (string name in untouched)
        {
            string field = fields[name];
            // The step stream is the Niche stream's twin in a run; leave it out when that one moves.
            string[] skip = name == "Niche" ? [".Item1." + field, ".Item2"] : [".Item1." + field];
            rep.Count("streams");

            // Cheap first: the end of the fight. Only a fight that differs there is replayed step by step.
            if (EndStates(open, name, moves, skip, perStep: false) is null)
            {
                continue;
            }

            var found = EndStates(open, name, moves, skip, perStep: true);
            if (found is { } f)
            {
                rep.Fail("streams", f.Step, f.Card, $"moving {name} changed the fight: " + f.Detail);
            }
        }
    }

    /// <summary>Plays <paramref name="moves"/> on a fresh fight and on one whose stream was moved; the first difference.</summary>
    private static (int Step, string Card, string Detail)? EndStates(
        Func<Combat> open, string stream, List<Move> moves, string[] skip, bool perStep)
    {
        var a = open();
        var b = open();
        foreach (var (n, rng) in StateInvariants.Streams(b.State))
        {
            if (n == stream)
            {
                for (int k = 0; k < 3; k++)
                {
                    rng!.Next();
                }
            }
        }

        for (int i = 0; i < moves.Count; i++)
        {
            string card = a.Describe(moves[i]);
            a.Step(moves[i]);
            b.Step(moves[i]);
            bool last = i == moves.Count - 1 || a.Over || b.Over;
            if ((perStep || last) && Digest(a, skip) != Digest(b, skip))
            {
                return (i + 1, card, string.Join(" ; ", DeepState.Diff(a.Whole, b.Whole, DiffLimit, skip)));
            }

            if (a.Over || b.Over)
            {
                break;
            }
        }

        return null;
    }
}
