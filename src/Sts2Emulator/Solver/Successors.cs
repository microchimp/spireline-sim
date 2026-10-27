namespace Sts2Emulator.Solver;

using System.Diagnostics;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;

/// <summary>A position the solver will not value, with the reason (FIGHT_SOLVER.md section 7: refused, never solved wrong).</summary>
public sealed class SolveRefused(string reason) : Exception($"solve refused: {reason}")
{
    public string Reason { get; } = reason;
}

/// <summary>One outcome of a move: the stepped copy, the step's stream after it, the step's result and the outcome's probability.</summary>
public sealed record Outcome(CombatState State, CountingRandom Rng, StepResult Result, Rational P);

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
}

/// <summary>The step took a card of the rest of epoch <see cref="Epoch" />: the next position branches.</summary>
internal sealed class DrawPending(int epoch, List<(int Uid, int Count)> branches, int size) : Exception("draw pending")
{
    public int Epoch { get; } = epoch;

    public List<(int Uid, int Count)> Branches { get; } = branches;

    public int Size { get; } = size;
}

/// <summary>The chance source of one run of a step: the rolls answered from a list, the draw pile's layout steered.</summary>
internal sealed class SolverChance(CombatState state, int[] answers, List<List<int>> drawn, bool byCard, int widthCap) : IChanceSource
{
    int _next;
    int _steerNext;
    List<CardInstance>? _target;

    public int Epoch;
    public HashSet<int> Rest = [];
    public List<(int Uid, int Count)> Branches = [];

    /// <summary>The rest's first card in the layout: the next card a draw off the top of the unknown middle takes.</summary>
    public int Front;

    public int? Answer(in ChanceCall call)
    {
        if (CombatEngine.Running is { Record: { } record } run && run.Used < record.Screens.Count)
        {
            // A screen's answer replays its step from the start: the way to the screen is the past, and its rolls are
            // given again as they fell (recorded answers come first; a step stopped unhooked draws on its own streams).
            return null;
        }

        CheckRest();
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
            throw new ChanceRefused("chance-width", call);
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

    static bool FromDiscardShuffle()
    {
        foreach (var f in new StackTrace(2, false).GetFrames())
        {
            var m = f.GetMethod();
            if (m?.DeclaringType == typeof(Sts2Emulator.Core.Effects.CardEffects) && m.Name == nameof(Sts2Emulator.Core.Effects.CardEffects.ShuffleDiscardIntoDraw))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Throws <see cref="DrawPending" /> when the rest's front card has left the draw pile: the next position of the
    /// unknown middle was taken, so its value branches. Conditioning on that position is exact whatever took it (a draw
    /// off the top, or a uniform pick that landed there); a card of the rest taken from further down by a uniform pick
    /// or a screen's choice leaves the rest's order unread, so it does not branch.
    /// </summary>
    public void CheckRest()
    {
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

        throw new DrawPending(Epoch, Branches, Rest.Count);
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

    public static List<Outcome> Expand(CombatState state, CountingRandom rng, Move move, bool byCard, int widthCap, SolveCounters counters)
    {
        foreach (var c in StateKey.AllCards(state))
        {
            if (OrderPeekers.Contains(Math.Abs(c.DefId)))
            {
                throw new SolveRefused($"draw-peek: {Cards.Get(Math.Abs(c.DefId)).Name} reads the draw pile's order");
            }
        }

        var outcomes = new List<Outcome>();
        var stack = new Stack<(int[] Answers, List<List<int>> Drawn, Rational P)>();
        stack.Push(([], [[]], Rational.One));
        while (stack.Count > 0)
        {
            var (answers, drawn, p) = stack.Pop();
            counters.Steps++;
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

            var source = new SolverChance(copy, answers, [.. drawn.Select(d => d.ToList())], byCard, widthCap) { Rest = rest, Branches = branches, Front = front };
            try
            {
                ChanceHook.Attach(copy, r, source);
                StepResult result;
                try
                {
                    result = Step(copy, r, move);
                    source.CheckRest();
                }
                finally
                {
                    ChanceHook.Attach(copy, r, null);
                }

                CheckLayout(copy, source, known);
                outcomes.Add(new Outcome(copy, r, result, p));
            }
            catch (ChancePending pending)
            {
                var odds = Odds(pending.Call);
                for (int i = odds.Count - 1; i >= 0; i--)
                {
                    stack.Push(([.. answers, odds[i].Value], drawn, p * odds[i].P));
                }
            }
            catch (DrawPending d)
            {
                for (int b = d.Branches.Count - 1; b >= 0; b--)
                {
                    var next = drawn.Select(x => x.ToList()).ToList();
                    while (next.Count <= d.Epoch)
                    {
                        next.Add([]);
                    }

                    next[d.Epoch].Add(d.Branches[b].Uid);
                    stack.Push((answers, next, p * new Rational(d.Branches[b].Count, d.Size)));
                }
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

        return outcomes;
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
