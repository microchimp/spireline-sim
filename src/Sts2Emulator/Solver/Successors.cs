namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;

/// <summary>A position the solver will not value, with the reason (FIGHT_SOLVER.md section 7: refused, never solved wrong).</summary>
public sealed class SolveRefused(string reason) : Exception($"solve refused: {reason}")
{
    public string Reason { get; } = reason;
}

/// <summary>
/// One outcome of a move: the stepped copy, the step's stream after it, the step's result and the outcome's probability.
/// A <see cref="WideRoll" /> outcome is a roll wider than the width cap met on the way: no step result, its probability
/// valued as a horizon leaf at the move's position (L0, bound L0's maximum, never exact), never refused.
/// </summary>
public sealed record Outcome(CombatState State, CountingRandom Rng, StepResult Result, Rational P, bool WideRoll = false);

/// <summary>A roll wider than the width cap: the run stops and its branch is a horizon leaf (optimisation review change 9).</summary>
internal sealed class WideRollLeaf(ChanceCall call) : Exception("wide roll")
{
    public ChanceCall Call { get; } = call;
}

/// <summary>Counters shared by a solve: engine steps, decision nodes, table hits, outcomes merged.</summary>
public sealed class SolveCounters
{
    public long Steps;
    public long Nodes;
    public long TableHits;
    public long Merged;
    public long Leaves;
    public long HorizonLeaves;
    public long Deduped;

    /// <summary>Decision nodes on an open screen.</summary>
    public long Screens;

    /// <summary>Moves expanded, by <see cref="MoveKind" />.</summary>
    public readonly long[] MovesByKind = new long[4];

    /// <summary>Engine runs (step calls) of the expansions, by <see cref="MoveKind" />.</summary>
    public readonly long[] RunsByKind = new long[4];

    /// <summary>Chance levels summed over runs (the answered rolls and the drawn positions a run starts with), and the deepest.</summary>
    public long ChanceLevels;

    public long ChanceLevelMax;

    /// <summary>Outcomes (chance leaves) of the end-turn expansions, before merging.</summary>
    public long EndTurnLeaves;

    /// <summary>Table hits on an exact entry, hits on a cut entry at the same turns left, and lookups that missed.</summary>
    public long TableHitsExact;

    public long TableHitsSameTurns;

    public long TableMisses;

    /// <summary>Branches cut at a roll wider than the width cap (valued as horizon leaves).</summary>
    public long WideRollLeaves;

    /// <summary>Macro draw nodes (a run of two or more plain draws taken as one node over the sub-multisets drawn).</summary>
    public long MacroNodes;

    /// <summary>Sequential draw levels the macro nodes stood in for: the cards each took, less one, summed.</summary>
    public long MacroDrawsAvoided;

    /// <summary>Draw nodes with the macro draw on that took one card (the front alone was drawn, or not by a plain draw).</summary>
    public long DrawNodesSingle;

    /// <summary>Draw nodes taken one card at a time because the macro draw was off, by reason (hand-ordered, observer:Name).</summary>
    public readonly Dictionary<string, long> MacroOff = [];

    /// <summary>Why the macro draw was off at the solve's root (<c>on</c> when it was on; <c>pruning-off</c> without the bit).</summary>
    public string MacroRoot = "";

    /// <summary>The tight bound at the root: <c>on</c>, <c>off:kind:Name</c> (the gain holder that turned it off) or <c>pruning-off</c>.</summary>
    public string BoundRoot = "";

    /// <summary>Max nodes below the root where the tight bound was on (no gain reachable).</summary>
    public long BoundOn;

    /// <summary>Max nodes below the root where it was off, by the holder that turned it off.</summary>
    public readonly Dictionary<string, long> BoundOff = [];

    /// <summary>Max nodes closed by the bound (an exact move reached it; the rest not expanded).</summary>
    public long BoundClosed;

    /// <summary>Moves cut by the bound: before their step (a potion's use) and after it (a single outcome's bound).</summary>
    public long BoundCutsPreStep;

    public long BoundCutsPostStep;

    /// <summary>Table entries stored as bounds (a cut move's child), read back as bounds, and value lookups that met a bound entry (never read as a value).</summary>
    public long TableBoundsStored;

    public long TableBoundsReused;

    public long TableBoundsRefusedAsValue;

    /// <summary>Refusals by reason class (the reason up to its first colon).</summary>
    public readonly Dictionary<string, long> Refusals = [];

    /// <summary>The deepening's horizons in order: where each ended and at what cumulative cost.</summary>
    public readonly List<HorizonMark> Horizons = [];

    /// <summary>The phase timers' exclusive time per <see cref="Phase" /> in Stopwatch ticks, when they were on.</summary>
    public long[]? PhaseTicks;

    /// <summary>The merge audit's end-turn expansions and their distinct keys (full, all candidates masked, all but block).</summary>
    public long AuditExpansions;

    public readonly HashSet<UInt128> AuditFull = [];

    public readonly HashSet<UInt128> AuditMasked = [];

    public readonly HashSet<UInt128> AuditMaskedKeepBlock = [];

    public static MoveKind KindOf(CombatState s, Move m) =>
        s.PendingSelection is not null ? MoveKind.Screen : m.Action == s.Hand.Count ? MoveKind.EndTurn : m.Action > s.Hand.Count ? MoveKind.Potion : MoveKind.Play;

    /// <summary>Called at every engine step and every node: throws <see cref="BudgetExceeded" /> past the budget.</summary>
    public Action? Check;
}

/// <summary>
/// The draw model in the engine's own pile (FIGHT_SOLVER.md section 4): the draw pile's unknown middle is a
/// multiset, and its order is chosen when a draw happens. The engine draws off the top of one ordered list, so
/// each step runs on a copy whose middle is laid out by the solver: the cards already drawn in this branch first
/// (in the order drawn), then the rest in a canonical order. A step that takes a card of the rest has drawn
/// from the unknown middle: it is run again once per value the next position can hold, at count / size
/// (or once per card for the reference). A shuffle of the discard into the draw pile inside a step is laid out
/// the same way by answering its Fisher-Yates calls. A step that takes no card of the rest is independent of the
/// rest's order, which the key does not hold.
/// </summary>
public static class DrawSteer
{
    public static readonly int ShuffleStream = Array.IndexOf(StateCloning.StreamNames, nameof(CombatState.ShuffleRng));

    /// <summary>A card's value as the key writes it: its card_ref mark and its encoding (equal values are interchangeable copies).</summary>
    public static byte[] CardValue(CombatState s, CardInstance c)
    {
        int mark = 0;
        if (c.Uid != 0)
        {
            for (int i = 0; i < s.Enemies.Count; i++)
            {
                if (s.Enemies[i].CurlUpCardUid == c.Uid)
                {
                    mark = i + 1;
                    break;
                }
            }
        }

        var w = new KeyWriter(KeyMode.Expected);
        w.P(mark);
        StateKeyCodec.Enc_CardInstance(w, c);
        return w.ToArray();
    }

    /// <summary>The cards of <paramref name="cards" /> in canonical order: by value, then by uid.</summary>
    internal static List<(CardInstance Card, byte[] Value)> Canonical(CombatState s, IEnumerable<CardInstance> cards)
    {
        var list = cards.Select(c => (Card: c, Value: CardValue(s, c))).ToList();
        list.Sort(static (a, b) =>
        {
            int c = a.Value.AsSpan().SequenceCompareTo(b.Value);
            return c != 0 ? c : a.Card.Uid.CompareTo(b.Card.Uid);
        });
        return list;
    }

    /// <summary>
    /// Lays <paramref name="cards" /> out: the cards whose uids <paramref name="first" /> names, in that order, then the
    /// rest canonically. Returns the order, the rest's uids and the rest's branches (one per distinct value with its
    /// count, or one per card when <paramref name="byCard" />).
    /// </summary>
    internal static (List<CardInstance> Order, HashSet<int> RestUids, List<(int Uid, int Count)> Branches) Layout(
        CombatState s, IEnumerable<CardInstance> cards, IReadOnlyList<int> first, bool byCard)
    {
        var canon = Canonical(s, cards);
        var seen = new HashSet<int>();
        foreach (var (c, _) in canon)
        {
            if (c.Uid == 0 || !seen.Add(c.Uid))
            {
                throw new SolveRefused("card-uid: a draw-pile card without a distinct uid");
            }
        }

        var order = new List<CardInstance>(canon.Count);
        foreach (int uid in first)
        {
            int at = canon.FindIndex(x => x.Card.Uid == uid);
            if (at < 0)
            {
                throw new SolveRefused("draw-order: a drawn card is not where the draw model put it");
            }

            order.Add(canon[at].Card);
            canon.RemoveAt(at);
        }

        var rest = new HashSet<int>();
        var branches = new List<(int Uid, int Count)>();
        byte[]? last = null;
        foreach (var (c, v) in canon)
        {
            order.Add(c);
            rest.Add(c.Uid);
            if (!byCard && last is not null && last.AsSpan().SequenceEqual(v))
            {
                branches[^1] = (branches[^1].Uid, branches[^1].Count + 1);
            }
            else
            {
                branches.Add((c.Uid, 1));
            }

            last = v;
        }

        return (order, rest, branches);
    }

    /// <summary>
    /// A card of the draw pile that ties another one on <c>ShuffleSortKey</c> but not on value, when at least two cards
    /// of <paramref name="rest" /> (the canonically laid-out part) are still in the pile; null when there is none (see
    /// <c>SolverChance.CheckTie</c>).
    /// </summary>
    internal static CardInstance? LayoutTie(CombatState s, HashSet<int> rest)
    {
        var pile = s.DrawPile;
        int inRest = 0;
        foreach (var c in pile)
        {
            inRest += rest.Contains(c.Uid) ? 1 : 0;
        }

        if (inRest < 2)
        {
            return null;
        }

        var groups = new Dictionary<long, CardInstance>();
        HashSet<long>? tied = null;
        foreach (var c in pile)
        {
            long k = Sts2Emulator.Core.Effects.CardEffects.ShuffleSortKey(c);
            if (!groups.TryAdd(k, c))
            {
                (tied ??= []).Add(k);
            }
        }

        if (tied is null)
        {
            return null;
        }

        foreach (var c in pile)
        {
            long k = Sts2Emulator.Core.Effects.CardEffects.ShuffleSortKey(c);
            if (tied.Contains(k) && !CardValue(s, groups[k]).AsSpan().SequenceEqual(CardValue(s, c)))
            {
                return c;
            }
        }

        return null;
    }
}

/// <summary>
/// The step took a card of the rest of epoch <see cref="Epoch" />: the next position branches, or, when <see cref="Take" />
/// is two or more, the next <see cref="Take" /> positions as one node over the sub-multisets taken (the macro draw).
/// </summary>
internal sealed class DrawPending(int epoch, List<(int Uid, int Count)> branches, int size, int take = 1, int[]? rest = null, int call = 0) : Exception("draw pending")
{
    public int Epoch { get; } = epoch;

    public List<(int Uid, int Count)> Branches { get; } = branches;

    public int Size { get; } = size;

    /// <summary>The rest's cards one draw call's loop took in a row from the front, before the check: 1 for the front alone.</summary>
    public int Take { get; } = take;

    /// <summary>The rest's uids in layout order (<see cref="Branches" />' counts split it into its values).</summary>
    public int[] Rest { get; } = rest ?? [];

    /// <summary>The chance calls the run had seen at the check (<see cref="MacroBlock.Call" />).</summary>
    public int Call { get; } = call;
}

/// <summary>The chance source of one run of a step: the rolls answered from a list, the draw pile's layout steered.</summary>
internal sealed class SolverChance(CombatState state, int[] answers, List<List<int>> drawn, bool byCard, int widthCap, bool macro = false, IReadOnlyList<MacroBlock>? blocks = null)
    : IChanceSource
{
    int _next;
    int _steerNext;
    List<CardInstance>? _target;

    /// <summary>The chance calls seen (each runs the rest check first): a take's place between two of them.</summary>
    int _calls;

    /// <summary>This run's takes off the draw pile: the card, its draw call's number when the draw loop took it (else 0), and the calls seen before it.</summary>
    readonly List<(int Uid, long Action, int Call)> _takes = [];

    readonly IReadOnlyList<MacroBlock> _blocks = blocks ?? [];

    int _verified;

    public int Epoch;
    public HashSet<int> Rest = [];
    public List<(int Uid, int Count)> Branches = [];

    /// <summary>The rest's uids in layout order, front first.</summary>
    public List<int> RestOrder = [];

    /// <summary>Why the macro draw was off at the last pending that could have taken it (an observer met in the step).</summary>
    public string? MacroOff;

    long _drawCalls;

    /// <summary>The draw watch (<c>CombatEngine.DrawWatch</c>): numbers each draw call on this run's pile from 1 (0 for another pile's).</summary>
    public long OnDraw(CombatState s) => ReferenceEquals(s, state) ? ++_drawCalls : 0;

    /// <summary>The take watch (<c>CombatEngine.TakeWatch</c>): records each take off this run's pile.</summary>
    public void OnTake(CombatState s, int index)
    {
        if (ReferenceEquals(s, state) && index >= 0 && index < s.DrawPile.Count)
        {
            _takes.Add((s.DrawPile[index].Uid, Sts2Emulator.Core.Effects.CardEffects.PlainTake, _calls));
        }
    }

    /// <summary>The rest's first card in the layout: the next card a draw off the top of the unknown middle takes.</summary>
    public int Front;

    public int? Answer(in ChanceCall call)
    {
        int prof = SolverProf.Enter(Phase.Chance);
        try
        {
            return AnswerCore(call);
        }
        finally
        {
            SolverProf.Exit(prof);
        }
    }

    int? AnswerCore(in ChanceCall call)
    {
        if (CombatEngine.Running is { Record: { } record } run && run.Used < record.Screens.Count)
        {
            // A screen's answer replays its step from the start: the way to the screen is the past, and its rolls are
            // given again as they fell (recorded answers come first; a step stopped unhooked draws on its own streams).
            return null;
        }

        CheckRest();
        _calls++;
        CheckTie(call);
        if (call.Kind == ChanceKind.RawDouble)
        {
            throw new ChanceRefused("unlisted-roll", call);
        }

        if (call.Stream == DrawSteer.ShuffleStream && call.Kind == ChanceKind.Next && call.Min == 0)
        {
            if (_steerNext > 0 && call.Max == _steerNext)
            {
                return Steer(call);
            }

            if (_steerNext == 0 && call.Max >= 2 && call.Max == state.DrawPile.Count && FromDiscardShuffle())
            {
                Epoch++;
                while (drawn.Count <= Epoch)
                {
                    drawn.Add([]);
                }

                var (order, rest, branches) = DrawSteer.Layout(state, state.DrawPile, drawn[Epoch], byCard);
                (_target, Rest, Branches) = (order, rest, branches);
                RestOrder = [.. order.Skip(drawn[Epoch].Count).Select(c => c.Uid)];
                Front = rest.Count > 0 ? order[drawn[Epoch].Count].Uid : 0;
                _steerNext = call.Max;
                return Steer(call);
            }
        }

        if (_next < answers.Length)
        {
            return answers[_next++];
        }

        if (call.Width > widthCap)
        {
            throw new WideRollLeaf(call);
        }

        throw new ChancePending(call);
    }

    /// <summary>One Fisher-Yates call (<c>ShufflePile</c>, from the end): the index that puts the laid-out card at position Max - 1.</summary>
    int Steer(in ChanceCall call)
    {
        int i = call.Max - 1;
        int uid = _target![i].Uid;
        var pile = state.DrawPile;
        int j = -1;
        for (int k = 0; k <= i; k++)
        {
            if (pile[k].Uid == uid)
            {
                j = k;
                break;
            }
        }

        if (j < 0)
        {
            throw new SolveRefused("draw-order: the shuffle's pile is not the one laid out");
        }

        _steerNext = call.Max - 1 >= 2 ? call.Max - 1 : 0;
        return j;
    }

    /// <summary>Whether the call comes from inside the discard reshuffle (the engine's explicit site, not a stack walk).</summary>
    static bool FromDiscardShuffle() => Sts2Emulator.Core.Effects.CardEffects.DiscardShuffleDepth > 0;

    /// <summary>
    /// A uniform pick over the draw pile (Uproar, Catastrophe, a draw-pile sample screen) goes through
    /// <c>CardEffects.StableShuffle</c>, which sorts by <c>ShuffleSortKey</c> (id, upgraded) before its Fisher-Yates. The
    /// sort puts cards of different sort keys in an order their values fix, but which of two tied cards comes first is
    /// the pile's order (the runtime's sort is not stable at three cards or past sixteen, so any tie in the pile can
    /// move with the rest's layout). When two cards of the pile tie there and differ in their key value (a grown
    /// Rampage beside a fresh one), the value picked depends on the rest's order, which the canonical layout fixes and
    /// the front's branch does not average: Uproar over the two Rampages came out 1/4 and 3/4, not 1/2 each. The order
    /// left behind is not uniform either, so the multiset model cannot carry the pick: the whole solve is refused.
    /// Without such a tie the picked value is the same in every layout, so the canonical one is exact.
    /// </summary>
    void CheckTie(in ChanceCall call)
    {
        // FromStableShuffle first: a thread-static read, false on every roll outside a sorted pick, before the layout's scan.
        if (!FromStableShuffle() || call.Kind != ChanceKind.Next || call.Min != 0 || call.Max < 2 || DrawSteer.LayoutTie(state, Rest) is not { } tie)
        {
            return;
        }

        throw new SolveRefused($"draw-tie: {Cards.Get(Math.Abs(tie.DefId)).Name} ties another draw-pile card of another value in a sorted pick");
    }

    /// <summary>Whether the call comes from inside a sorted pick (the engine's explicit site, not a stack walk).</summary>
    static bool FromStableShuffle() => Sts2Emulator.Core.Effects.CardEffects.StableShuffleDepth > 0;

    /// <summary>
    /// Throws <see cref="DrawPending" /> when the rest's front card has left the draw pile: the next position of the
    /// unknown middle was taken, so its value branches at count / size. That is exact for a draw off the top, and for a
    /// sorted pick that landed there when the picked value does not depend on the rest's layout (<see cref="CheckTie" />
    /// refuses the case where it does): every layout then picks that value, and the branches only split the rest's
    /// order, uniformly. A card of the rest taken from further down by such a pick, or by a screen's choice, leaves the
    /// rest's order unread, so it does not branch.
    /// </summary>
    public void CheckRest(bool end = false)
    {
        VerifyBlocks(end);
        if (Front == 0)
        {
            return;
        }

        foreach (var c in state.DrawPile)
        {
            if (c.Uid == Front)
            {
                return;
            }
        }

        int take = macro ? PlainRun() : 1;
        if (take >= 2 && DrawObservers.Off(state) is { } off)
        {
            MacroOff = off;
            take = 1;
        }

        throw new DrawPending(Epoch, Branches, Rest.Count, take, [.. RestOrder], _calls);
    }

    /// <summary>
    /// How many of the rest's cards, from the front in layout order, one draw call's loop took in a row since the last
    /// check (1 when the front was not such a take). A front taken right after a macro block by the same loop between the
    /// same checks means the replay drew more than the block's run did: the count drawn depended on the cards. A card of
    /// the rest taken earlier in the run outside a draw loop (a <c>rest-take</c> of draw_observers.json: Thieving Hopper's
    /// steal, a pick by place, a top-card play) read the layout, and a block laid out at the front would change what it
    /// took: the draw is sequential then (<c>rest-taken</c>).
    /// </summary>
    int PlainRun()
    {
        int i = _takes.FindIndex(x => x.Uid == Front && x.Call == _calls);
        if (i < 0 || _takes[i].Action == 0)
        {
            return 1;
        }

        var (_, action, call) = _takes[i];
        if (i > 0 && _takes[i - 1].Action == action && _takes[i - 1].Call == call && EndsBlock(_takes[i - 1].Uid, call))
        {
            throw new SolveRefused("macro-draw-mismatch: a replay drew past a macro draw's cards in the same run of draws");
        }

        int j = 1;
        while (j < RestOrder.Count && i + j < _takes.Count && _takes[i + j].Uid == RestOrder[j] && _takes[i + j].Action == action && _takes[i + j].Call == call)
        {
            j++;
        }

        for (int t = 0; j >= 2 && t < i; t++)
        {
            if (_takes[t].Action == 0 && RestOrder.Contains(_takes[t].Uid))
            {
                MacroOff = "rest-taken";
                return 1;
            }
        }

        return j;
    }

    bool EndsBlock(int uid, int call)
    {
        foreach (var b in _blocks)
        {
            if (b.Epoch == Epoch && b.Call == call && b.Uids[^1] == uid)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Each macro block is taken again as its run took it: at the check it was found at, every card of it taken, in its
    /// order, by one draw call's loop, between the same two checks. Otherwise the replay drew a different count, which
    /// the hypergeometric node cannot carry: refused (an observer the registry does not know).
    /// </summary>
    void VerifyBlocks(bool end)
    {
        while (_verified < _blocks.Count && _blocks[_verified].Call <= _calls)
        {
            var b = _blocks[_verified];
            int i = _takes.FindIndex(x => x.Uid == b.Uids[0] && x.Call == b.Call);
            bool ok = i >= 0 && _takes[i].Action != 0 && i + b.Uids.Length <= _takes.Count;
            for (int r = 1; ok && r < b.Uids.Length; r++)
            {
                ok = _takes[i + r].Uid == b.Uids[r] && _takes[i + r].Action == _takes[i].Action && _takes[i + r].Call == b.Call;
            }

            if (!ok)
            {
                throw new SolveRefused($"macro-draw-mismatch: a replay did not draw a macro draw's cards as its run did (block [{string.Join(",", b.Uids)}] at check {b.Call}; takes [{string.Join(" ", _takes.Take(16).Select(x => $"{x.Uid}/{x.Action}/{x.Call}"))}])");
            }

            _verified++;
        }

        if (end && _verified < _blocks.Count)
        {
            throw new SolveRefused("macro-draw-mismatch: a replay ended before a macro draw's cards were drawn");
        }
    }
}

/// <summary>
/// The outcomes of one move at the rules' odds (FIGHT_SOLVER.md sections 3 and 4): the step is run on copies, each
/// pending roll and each draw from the unknown middle branching, until every run is independent of what is still
/// unrolled. The position given is never changed.
/// </summary>
public static class Successors
{
    /// <summary>Cards whose rules look at the draw pile's order in a way the steered layout cannot answer.</summary>
    static readonly HashSet<int> OrderPeekers =
        [.. new[] { "IAmInvincible", "PerfectFit" }.Select(Cards.FindId).Where(i => i is not null).Select(i => i!.Value)];

    public static StepResult Step(CombatState s, CountingRandom rng, Move m) =>
        s.Kernel is { AnyOn: true } k ? KernelGate.Step(k.Shared, s, rng, m.Action, m.Target) : CombatEngine.Step(s, m.Action, rng, m.Target);

    /// <summary>Every outcome of the move at once (the reference, the tests): <see cref="Stream" /> collected.</summary>
    public static List<Outcome> Expand(CombatState state, CountingRandom rng, Move move, bool byCard, int widthCap, SolveCounters counters, bool macro = false) =>
        [.. Stream(state, rng, move, byCard, widthCap, counters, macro)];

    /// <summary>
    /// The outcomes of the move one at a time, each produced on demand: a caller that solves an outcome's children before
    /// asking for the next keeps one outcome's copy alive per expansion, not all of them (the expansion itself holds only
    /// its pending answer lists). <paramref name="macro" />: a run of plain draws from the unknown middle is one node over
    /// the sub-multisets drawn (FIGHT_SOLVER.md section 17), unless the hand is kept in order or a draw observer is present.
    /// </summary>
    public static IEnumerable<Outcome> Stream(CombatState state, CountingRandom rng, Move move, bool byCard, int widthCap, SolveCounters counters, bool macro = false)
    {
        var kind = SolveCounters.KindOf(state, move);
        counters.MovesByKind[(int)kind]++;
        int prof = SolverProf.Enter(Phase.Chance);
        try
        {
            foreach (var c in StateKey.AllCards(state))
            {
                if (OrderPeekers.Contains(Math.Abs(c.DefId)))
                {
                    throw new SolveRefused($"draw-peek: {Cards.Get(Math.Abs(c.DefId)).Name} reads the draw pile's order");
                }
            }

        }
        finally
        {
            SolverProf.Exit(prof);
        }

        bool useMacro = macro && !byCard;
        // An open screen's answer replays its step from the start, and the pile after the screen is the screen's own copy:
        // the takes there need not follow the layout, so the macro draw is off (the replay's takes refused it: #424 of the harness).
        var gate = new ParentGate(state, state.PendingSelection is not null ? "screen-answer" : null);
        var stack = new Stack<(int[] Answers, List<List<int>> Drawn, List<MacroBlock> Blocks, Rational P)>();
        stack.Push(([], [[]], [], Rational.One));
        while (stack.Count > 0)
        {
            Outcome? produced;
            prof = SolverProf.Enter(Phase.Chance);
            try
            {
                produced = Run(state, rng, move, byCard, widthCap, counters, kind, stack, useMacro, gate);
            }
            finally
            {
                SolverProf.Exit(prof);
            }

            if (produced is not null)
            {
                if (kind == MoveKind.EndTurn)
                {
                    counters.EndTurnLeaves++;
                }

                yield return produced;
            }
        }
    }

    /// <summary>Why the macro draw is off at the expansion's position (<see cref="DrawObservers.Off" />), asked once, at its first run of plain draws.</summary>
    sealed class ParentGate(CombatState state, string? preset)
    {
        /// <summary>A run of this expansion reached a screen: the rest run with the engine's pre-step record on.</summary>
        public bool Record;

        bool _done = preset is not null;
        string? _reason = preset;

        public string? Reason
        {
            get
            {
                if (!_done)
                {
                    (_done, _reason) = (true, DrawObservers.Off(state));
                }

                return _reason;
            }
        }
    }

    /// <summary>One run of the step off the stack: its outcome, or null when it stopped at a roll or a draw (its branches pushed).</summary>
    static Outcome? Run(
        CombatState state, CountingRandom rng, Move move, bool byCard, int widthCap, SolveCounters counters, MoveKind kind,
        Stack<(int[] Answers, List<List<int>> Drawn, List<MacroBlock> Blocks, Rational P)> stack, bool macro, ParentGate gate)
    {
        {
            var (answers, drawn, blocks, p) = stack.Pop();
            counters.Steps++;
            counters.RunsByKind[(int)kind]++;
            long levels = answers.Length;
            foreach (var d in drawn)
            {
                levels += d.Count;
            }

            counters.ChanceLevels += levels;
            counters.ChanceLevelMax = Math.Max(counters.ChanceLevelMax, levels);
            counters.Check?.Invoke();
            var copy = state.Clone();
            var r = rng.Copy();
            int kt = Math.Clamp(copy.KnownTopCount, 0, copy.DrawPile.Count);
            int kb = Math.Clamp(copy.KnownBottomCount, 0, copy.DrawPile.Count - kt);
            int mid = copy.DrawPile.Count - kt - kb;
            // A screen choosing from the draw pile keeps its chosen card where the screen shows it (its candidate is an
            // index): the choice is the player's, not a draw, and the rest is laid out around it.
            int keep = copy.PendingSelection is { } open && open.Kind.ChoosesFromDrawPile() && move.Action < open.Candidates.Count
                && open.Candidates[move.Action] is int at && at >= kt && at < kt + mid ? at - kt : -1;
            var middle = copy.DrawPile.GetRange(kt, mid);
            CardInstance? kept = keep >= 0 ? middle[keep] : null;
            if (keep >= 0)
            {
                middle.RemoveAt(keep);
            }

            var (order, rest, branches) = DrawSteer.Layout(copy, middle, drawn[0], byCard);
            List<int> restOrder = [.. order.Skip(drawn[0].Count).Select(c => c.Uid)];
            int front = rest.Count > 0 ? order[drawn[0].Count].Uid : 0;
            if (kept is { } k)
            {
                order.Insert(keep, k);
            }

            for (int i = 0; i < mid; i++)
            {
                copy.DrawPile[kt + i] = order[i];
            }

            var known = new HashSet<int>();
            for (int i = 0; i < copy.DrawPile.Count; i++)
            {
                if (i < kt || i >= kt + mid)
                {
                    known.Add(copy.DrawPile[i].Uid);
                }
            }

            var source = new SolverChance(copy, answers, [.. drawn.Select(d => d.ToList())], byCard, widthCap, macro, blocks)
            {
                Rest = rest, Branches = branches, Front = front, RestOrder = restOrder,
            };
            try
            {
                ChanceHook.Attach(copy, r, source);
                StepResult result;
                int stepProf = SolverProf.Enter(Phase.Step);
                CombatEngine.TakeWatch = source.OnTake;
                CombatEngine.DrawWatch = source.OnDraw;
                CombatEngine.LazyRecord = !gate.Record;
                try
                {
                    result = Step(copy, r, move);
                    source.CheckRest(end: true);
                }
                finally
                {
                    CombatEngine.LazyRecord = false;
                    CombatEngine.TakeWatch = null;
                    CombatEngine.DrawWatch = null;
                    SolverProf.Exit(stepProf);
                    ChanceHook.Attach(copy, r, null);
                }

                CheckLayout(copy, source, known);
                return new Outcome(copy, r, result, p);
            }
            catch (ChancePending pending)
            {
                var odds = Odds(pending.Call);
                for (int i = odds.Count - 1; i >= 0; i--)
                {
                    stack.Push(([.. answers, odds[i].Value], drawn, blocks, p * odds[i].P));
                }
            }
            catch (DrawPending d)
            {
                if (d.Take >= 2 && gate.Reason is null)
                {
                    counters.MacroNodes++;
                    counters.MacroDrawsAvoided += d.Take - 1;
                    var sizes = d.Branches.Select(b => b.Count).ToArray();
                    var parts = MacroDraw.Parts(sizes, d.Take);
                    for (int x = parts.Count - 1; x >= 0; x--)
                    {
                        var (counts, q) = parts[x];
                        var uids = new List<int>(d.Take);
                        int off0 = 0;
                        for (int g = 0; g < sizes.Length; g++)
                        {
                            for (int c = 0; c < counts[g]; c++)
                            {
                                uids.Add(d.Rest[off0 + c]);
                            }

                            off0 += sizes[g];
                        }

                        var next = drawn.Select(y => y.ToList()).ToList();
                        while (next.Count <= d.Epoch)
                        {
                            next.Add([]);
                        }

                        next[d.Epoch].AddRange(uids);
                        stack.Push((answers, next, [.. blocks, new MacroBlock(d.Epoch, [.. uids], d.Call)], p * q));
                    }

                    return null;
                }

                string? why = source.MacroOff ?? (d.Take >= 2 ? gate.Reason : null);
                if (why is not null)
                {
                    counters.MacroOff[why] = counters.MacroOff.GetValueOrDefault(why) + 1;
                }
                else if (macro)
                {
                    counters.DrawNodesSingle++;
                }

                for (int b = d.Branches.Count - 1; b >= 0; b--)
                {
                    var next = drawn.Select(x => x.ToList()).ToList();
                    while (next.Count <= d.Epoch)
                    {
                        next.Add([]);
                    }

                    next[d.Epoch].Add(d.Branches[b].Uid);
                    stack.Push((answers, next, blocks, p * new Rational(d.Branches[b].Count, d.Size)));
                }
            }
            catch (RecordNeeded)
            {
                // The step reached a screen without its pre-step record: this copy is dropped and the run made again with it.
                gate.Record = true;
                stack.Push((answers, drawn, blocks, p));
            }
            catch (WideRollLeaf)
            {
                // A roll wider than the cap: this branch (probability p) is a horizon leaf at the move's position.
                counters.WideRollLeaves++;
                return new Outcome(state, rng, default!, p, WideRoll: true);
            }
            catch (ChanceRefused e)
            {
                throw new SolveRefused($"{e.Reason}: {e.Call}");
            }
            catch (StateKeyRefusedException e)
            {
                throw new SolveRefused(e.Reason);
            }
            catch (SolveRefused)
            {
                throw;
            }
            catch (BudgetExceeded)
            {
                throw;
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                throw new SolveRefused($"engine-error: {e.GetType().Name}: {e.Message}");
            }
        }

        return null;
    }

    /// <summary>
    /// A run that leaves a card the player knew the place of in the unknown middle, with no shuffle in the step, is
    /// refused: a random placement into a known run, after which the player knows more than a multiset (a hidden branch).
    /// </summary>
    static void CheckLayout(CombatState copy, SolverChance source, HashSet<int> known)
    {
        int kt = Math.Clamp(copy.KnownTopCount, 0, copy.DrawPile.Count);
        int kb = Math.Clamp(copy.KnownBottomCount, 0, copy.DrawPile.Count - kt);
        for (int i = kt; i < copy.DrawPile.Count - kb; i++)
        {
            if (source.Epoch == 0 && known.Contains(copy.DrawPile[i].Uid))
            {
                throw new SolveRefused("hidden-branch: a card of a known run is in the unknown middle");
            }
        }
    }

    /// <summary>A pending call's outcomes as exact rationals: 1 / width, or weight / sum over the positive weights.</summary>
    public static List<(int Value, Rational P)> Odds(in ChanceCall call)
    {
        var list = new List<(int, Rational)>();
        switch (call.Kind)
        {
            case ChanceKind.Next:
                if ((long)call.Max <= call.Min)
                {
                    throw new ChanceRefused("empty-range", call);
                }

                var u = new Rational(1, (long)call.Max - call.Min);
                for (long v = call.Min; v < call.Max; v++)
                {
                    list.Add(((int)v, u));
                }

                break;
            case ChanceKind.Branch:
                var w = call.Weights!.Select(x => x > 0 ? Rational.FromDouble(x) : Rational.Zero).ToArray();
                var sum = w.Aggregate(Rational.Zero, (a, b) => a + b);
                if (sum == Rational.Zero)
                {
                    list.Add((0, Rational.One));
                    break;
                }

                for (int i = 0; i < w.Length; i++)
                {
                    if (w[i] > Rational.Zero)
                    {
                        list.Add((i, w[i] / sum));
                    }
                }

                break;
            default:
                throw new ChanceRefused("unlisted-roll", call);
        }

        return list;
    }
}
