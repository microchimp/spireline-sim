namespace Sts2Emulator.Solver;

using System.Diagnostics;
using Sts2Emulator.Core;

/// <summary>
/// One subset's entry of the potion value curve (FIGHT_SOLVER.md section 8). <see cref="Subset" /> is a bit set over
/// <see cref="PotionCurve.HeldSlots" /> (bit j: the j-th held potion usable). <see cref="ExpectedL0" /> is the solve's
/// value with only those potions usable, L0 counting every potion still held, those outside the subset included;
/// <see cref="WinChance" /> the chance of a won fight under the solve's policy (the first move of greatest value),
/// <see cref="ExpectedWinHp" /> the expected HP at the end of a won fight (NaN when no win), and <see cref="Held" /> for
/// each potion of the subset the chance it is still held at the end (by potion: the k-th copy of an id is held when
/// at least k are). A horizon leaf counts as no win and holds what it holds at the cut, so with
/// <see cref="ExactMass" /> below one the win chance and the held chances are those of the cut tree. The win chance, the
/// win HP and the held chances follow the first move of greatest value, so among moves of equal value they depend on the
/// move order (the tie-break); the value does not. Every entry of one curve is solved at the same <see cref="Horizon" />.
/// A refused entry's numbers are zero here beside its status and cross the boundary as NaN (null, None).
/// </summary>
public sealed record CurveEntry(
    int Subset,
    SolveStatus Status,
    string? Reason,
    int Horizon,
    Rational ExpectedL0,
    bool Exact,
    Rational ExactMass,
    Rational WinChance,
    double ExpectedWinHp,
    IReadOnlyDictionary<int, Rational> Held);

/// <summary>
/// The potion value curve: the root solved once per subset S of the potions held, with only the potions in S usable
/// (<see cref="SolveOptions.AllowedPotions" />; a slot empty at the root stays usable, so a potion gained in the fight is
/// usable in every subset). The potions outside S stay in the copy (the always-valid form of section 8: the subsets share
/// no table; removing them where no rule reads potions held is a later saving). Each subset is the expected solve's
/// expectimax with its iterative deepening and budget, carrying the policy's statistics beside the value; its value
/// equals <see cref="Expectimax.SolveAt" />'s under the same mask at the curve's horizon (<c>PotionCurveTests</c>).
/// One horizon per curve: every subset is solved at horizon 0, 1, ... in turn, and the curve is the deepest horizon that
/// every subset finished within its budget (each subset's budget counts across its horizons, as an expected solve's does),
/// so V(S + p) - V(S) always compares two trees cut alike, and an entry never falls below a subset of it (section 8's
/// monotonicity holds between entries of one curve). When no horizon was finished by every subset, the curve is cut: every
/// entry is refused with a <c>budget:</c> reason.
/// </summary>
public static class PotionCurve
{
    /// <summary>The slots holding a potion at the root, in slot order (at most the belt).</summary>
    public static List<int> HeldSlots(CombatState s)
    {
        var held = new List<int>();
        int slots = Math.Min(s.PotionSlots.Length, s.MaxPotionSlots);
        for (int i = 0; i < slots; i++)
        {
            if (s.PotionSlots[i] != 0)
            {
                held.Add(i);
            }
        }

        return held;
    }

    /// <summary>The potion mask of subset <paramref name="subset" />: every slot but the held ones outside it.</summary>
    public static int MaskOf(IReadOnlyList<int> held, int subset)
    {
        int mask = -1;
        for (int j = 0; j < held.Count; j++)
        {
            if ((subset & (1 << j)) == 0)
            {
                mask &= ~(1 << held[j]);
            }
        }

        return mask;
    }

    /// <summary>Every subset in bit order, the empty one (the base) first; at most <paramref name="maxSubsets" />; one horizon for all.</summary>
    public static List<CurveEntry> Solve(CombatState state, CountingRandom rng, SolveOptions options, int maxSubsets = 64)
    {
        var held = HeldSlots(state);
        int n = Math.Min(1 << held.Count, Math.Max(0, maxSubsets));
        var solvers = new CurveSolver[n];
        var refused = new string?[n];
        for (int i = 0; i < n; i++)
        {
            solvers[i] = new CurveSolver(options with { AllowedPotions = MaskOf(held, i) }, state, held);
        }

        Stats?[]? last = null;
        int lastHorizon = -1;
        string reason = "horizon";
        for (int h = 0; h <= options.MaxHorizon && n > 0; h++)
        {
            if (options.UseEstimate && h > 0 && SizeEstimate.Log10(state, h) > Math.Log10(options.NodeBudget) + options.EstimateSlack)
            {
                reason = $"estimate: horizon {h} is about 10^{SizeEstimate.Log10(state, h):0.0} states";
                break;
            }

            var at = new Stats?[n];
            bool cut = false;
            for (int i = 0; i < n && !cut; i++)
            {
                if (refused[i] is not null)
                {
                    continue;
                }

                try
                {
                    at[i] = solvers[i].At(state, rng, h);
                }
                catch (BudgetExceeded e)
                {
                    reason = $"subset {i} at horizon {h}: {e.Message}";
                    cut = true;
                }
                catch (SolveRefused e)
                {
                    refused[i] = e.Reason;
                }
            }

            if (cut)
            {
                break;
            }

            (last, lastHorizon) = (at, h);
            if (at.All(x => x is not { } st || (st.V.Exact && st.V.Mass == Rational.One)))
            {
                break;
            }
        }

        var entries = new List<CurveEntry>(n);
        for (int i = 0; i < n; i++)
        {
            if (refused[i] is { } why)
            {
                entries.Add(solvers[i].Entry(i, SolveStatus.Refused, why, -1, null));
            }
            else if (last?[i] is not { } st)
            {
                entries.Add(solvers[i].Entry(i, SolveStatus.Refused, $"budget: no horizon finished by every subset ({reason})", -1, null));
            }
            else
            {
                bool exact = st.V.Exact && st.V.Mass == Rational.One;
                entries.Add(solvers[i].Entry(i, exact ? SolveStatus.Exact : SolveStatus.Budget, exact ? null : reason, lastHorizon, st));
            }
        }

        return entries;
    }

    /// <summary>A node's value with the policy's statistics: P(win), the sum of P x HP over wins, P(held) per held potion.</summary>
    internal readonly record struct Stats(NodeValue V, Rational Win, Rational WinHp, Rational[] Held);

    sealed class CurveSolver(SolveOptions options, CombatState root, List<int> held)
    {
        readonly int[] _ids = [.. held.Select(i => root.PotionSlots[i])];
        readonly Dictionary<(UInt128, int), Stats> _memo = [];
        readonly HashSet<UInt128> _path = [];
        readonly SolveCounters _c = new();

        bool On(Pruning p) => (options.Prune & p) != 0;

        readonly Stopwatch _sw = new();

        /// <summary>
        /// The subset's tree at horizon <paramref name="horizon" />. The budget counts across the horizons this subset has run
        /// (the time budget its own solving time only); throws <see cref="BudgetExceeded" /> or <see cref="SolveRefused" />.
        /// </summary>
        public Stats At(CombatState state, CountingRandom rng, int horizon)
        {
            _c.Check ??= () =>
            {
                if (_c.Nodes > options.NodeBudget)
                {
                    throw new BudgetExceeded($"{options.NodeBudget} nodes");
                }

                if (_c.Steps > options.StepBudget)
                {
                    throw new BudgetExceeded($"{options.StepBudget} steps");
                }

                if (options.TimeBudgetMs > 0 && _sw.ElapsedMilliseconds > options.TimeBudgetMs)
                {
                    throw new BudgetExceeded($"{options.TimeBudgetMs} ms");
                }
            };
            _sw.Start();
            try
            {
                _memo.Clear();
                _path.Clear();
                return Node(state, rng, horizon, root: true);
            }
            finally
            {
                _sw.Stop();
            }
        }

        public CurveEntry Entry(int subset, SolveStatus status, string? reason, int horizon, Stats? stats)
        {
            var heldChance = new Dictionary<int, Rational>();
            for (int j = 0; j < held.Count; j++)
            {
                if ((subset & (1 << j)) != 0)
                {
                    heldChance[held[j]] = stats?.Held[j] ?? Rational.Zero;
                }
            }

            if (stats is not { } s)
            {
                return new CurveEntry(subset, status, reason, horizon, Rational.Zero, false, Rational.Zero, Rational.Zero, double.NaN, heldChance);
            }

            double winHp = s.Win == Rational.Zero ? double.NaN : (s.WinHp / s.Win).ToDouble();
            return new CurveEntry(subset, status, reason, horizon, s.V.Value, status == SolveStatus.Exact, s.V.Mass, s.Win, winHp, heldChance);
        }

        /// <summary>P(held) per held potion as the leaf stands: the k-th copy of an id is held when at least k are.</summary>
        Rational[] HeldAt(CombatState s)
        {
            var counts = new Dictionary<int, int>();
            int slots = Math.Min(s.PotionSlots.Length, s.MaxPotionSlots);
            for (int i = 0; i < slots; i++)
            {
                if (s.PotionSlots[i] != 0)
                {
                    counts[s.PotionSlots[i]] = counts.GetValueOrDefault(s.PotionSlots[i]) + 1;
                }
            }

            var seen = new Dictionary<int, int>();
            var h = new Rational[_ids.Length];
            for (int j = 0; j < _ids.Length; j++)
            {
                int k = seen[_ids[j]] = seen.GetValueOrDefault(_ids[j]) + 1;
                h[j] = counts.GetValueOrDefault(_ids[j]) >= k ? Rational.One : Rational.Zero;
            }

            return h;
        }

        Stats Leaf(CombatState s) => new(NodeValue.Horizon(s), Rational.Zero, Rational.Zero, HeldAt(s));

        Stats End(CombatState s, bool won) =>
            new(NodeValue.End(s, won), won ? Rational.One : Rational.Zero, won ? new Rational(s.PlayerHp, 1) : Rational.Zero, HeldAt(s));

        static UInt128 Key(CombatState s)
        {
            try
            {
                return StateKey.Hash(s, KeyMode.Expected);
            }
            catch (StateKeyRefusedException e)
            {
                throw new SolveRefused(e.Reason);
            }
        }

        Stats Node(CombatState s, CountingRandom rng, int remaining, bool root = false)
        {
            if (!root)
            {
                _c.Nodes++;
                _c.Check?.Invoke();
            }

            var key = Key(s);
            if (On(Pruning.Transposition) && _memo.TryGetValue((key, remaining), out var hit))
            {
                _c.TableHits++;
                return hit;
            }

            if (!_path.Add(key))
            {
                throw new SolveRefused("cycle: a position reached again below itself");
            }

            var moves = Actions.Legal(s, options.Prune, options.AllowedPotions);
            var stats = new List<Stats>(moves.Count);
            foreach (var m in moves)
            {
                stats.Add(MoveStats(s, rng, m, remaining));
            }

            _path.Remove(key);
            var result = Max(stats, s);
            if (On(Pruning.Transposition))
            {
                _memo[(key, remaining)] = result;
            }

            return result;
        }

        /// <summary>The max node: <see cref="NodeValue.Max" />'s value, the statistics of the move it takes.</summary>
        Stats Max(List<Stats> moves, CombatState s)
        {
            if (moves.Count == 0)
            {
                return Leaf(s);
            }

            var v = NodeValue.Max(moves.Select(x => x.V), s);
            int at = v.Exact ? moves.FindIndex(x => x.V.Exact && x.V.Value == v.Value) : moves.FindIndex(x => x.V.Value == v.Value);
            var pick = moves[Math.Max(0, at)];
            return pick with { V = v };
        }

        Stats MoveStats(CombatState s, CountingRandom rng, Move m, int remaining)
        {
            if (Actions.EndsTurn(s, m) && remaining <= 0)
            {
                _c.HorizonLeaves++;
                return Leaf(s);
            }

            var outcomes = Successors.Expand(s, rng, m, byCard: !On(Pruning.DrawByValue), options.ChanceWidthCap, _c);
            var acc = new Acc(_ids.Length);
            if (On(Pruning.Transposition))
            {
                var groups = new Dictionary<UInt128, (Outcome First, Rational P)>();
                var order = new List<UInt128>();
                foreach (var o in outcomes)
                {
                    if (o.Result.Terminal)
                    {
                        _c.Leaves++;
                        acc.Add(o.P, End(o.State, o.Result.PlayerWon));
                        continue;
                    }

                    var k = Key(o.State);
                    if (groups.TryGetValue(k, out var g))
                    {
                        groups[k] = (g.First, g.P + o.P);
                    }
                    else
                    {
                        groups[k] = (o, o.P);
                        order.Add(k);
                    }
                }

                foreach (var k in order)
                {
                    var (o, p) = groups[k];
                    acc.Add(p, Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn)));
                }
            }
            else
            {
                foreach (var o in outcomes)
                {
                    acc.Add(o.P, o.Result.Terminal ? End(o.State, o.Result.PlayerWon) : Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn)));
                }
            }

            return acc.Done();
        }

        /// <summary>A chance node's sums, as <c>Expectimax.MoveValueOf</c> sums its values.</summary>
        sealed class Acc(int n)
        {
            Rational _value = Rational.Zero, _mass = Rational.Zero, _upper = Rational.Zero, _win = Rational.Zero, _winHp = Rational.Zero;
            readonly Rational[] _held = [.. Enumerable.Repeat(Rational.Zero, n)];
            bool _exact = true;

            public void Add(Rational p, Stats s)
            {
                _value += p * s.V.Value;
                _mass += p * s.V.Mass;
                _upper += p * s.V.Upper;
                _exact &= s.V.Exact;
                _win += p * s.Win;
                _winHp += p * s.WinHp;
                for (int j = 0; j < _held.Length; j++)
                {
                    _held[j] += p * s.Held[j];
                }
            }

            public Stats Done() => new(new NodeValue(_value, _exact, _mass, _exact ? _value : _upper), _win, _winHp, _held);
        }
    }
}
