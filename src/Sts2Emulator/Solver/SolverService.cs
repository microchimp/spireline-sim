namespace Sts2Emulator.Solver;

using System.Text.Json.Nodes;
using Sts2Emulator.Core;

/// <summary>
/// The one code path behind the solver's front ends (FIGHT_SOLVER.md section 1, piece S4): the native exports
/// (<c>Interop/SolverExports.cs</c>) and EmuDump's <c>solve</c> and <c>hindsight</c> commands (<c>tools/emudump/Engines/Old/Solve.cs</c>)
/// both call these, so they answer alike on any position (<c>NativeSolverTests</c>, EmuDump's <c>SolveModeTests</c>).
/// The expected solve and the hindsight bound are separate calls with separate result types, and the hindsight call
/// refuses without its flag, without the record's own streams, or with a stream unset (section 4a). A refused result
/// carries no number across the boundary: NaN natively, null in EmuDump's JSON, None in Python (the C# Rational stays
/// beside the status).
/// </summary>
public static class SolverService
{
    /// <summary>The solver's version, recorded with every result for provenance.</summary>
    public const string Version = "s4-1";

    /// <summary>Stride of the native q buffer: one entry for an untargeted move, then one per enemy slot.</summary>
    public const int QStride = 1 + Interop.NativeExports.MAX_ENEMIES;

    /// <summary>Why a hindsight search on a determinized or invented world is refused.</summary>
    public const string NoTrueStreams =
        "hindsight-streams: the handle does not carry the record's own streams (a resampled fork or a loaded document), so its draws and rolls are not the record's";

    public static string EngineOf(CombatState s) => s.Kernel is { AnyOn: true } ? "core2" : "old";

    /// <summary>The expected solve (iterative deepening within the budget) on the given table, or the solver's own.</summary>
    public static ExpectedResult SolveExpected(CombatState state, CountingRandom rng, SolveOptions options, TranspositionTable? table = null) =>
        new Expectimax(options, table).SolveExpected(ExpectedView(state), rng);

    /// <summary>The potion value curve of the position (the expected solve per potion subset).</summary>
    public static List<CurveEntry> SolvePotionCurve(CombatState state, CountingRandom rng, SolveOptions options, int maxSubsets = 64) =>
        PotionCurve.Solve(ExpectedView(state), rng, options, maxSubsets);

    /// <summary>
    /// The position the expected solve reads: the position itself, or, when a stream is unset, a copy with each unset
    /// stream given a stream of its own. A native combat handle leaves some unset (Sts2_Create's fight draws its shuffles
    /// from the step's stream, a reset leaves the energy-cost and orb streams to it), and the chance hook refuses an unset
    /// stream, since the engine's fallbacks would roll unseen. In expected mode every stream is chance and none is in the
    /// key, so a stream of its own changes no value: it only lets each call reach the hook on its own stream, as the game
    /// draws it (the draw model steers the shuffle on the shuffle stream). The hindsight bound never does this: it refuses
    /// a handle with an unset stream (<see cref="UnsetStreams" />).
    /// </summary>
    public static CombatState ExpectedView(CombatState state)
    {
        bool unset = false;
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            unset |= StateCloning.Stream(state, i) is null;
        }

        if (!unset)
        {
            return state;
        }

        var view = state.Clone();
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(view, i) is null)
            {
                StateCloning.SetStream(view, i, new CountingRandom(7919 + i));
            }
        }

        return view;
    }

    /// <summary>Why a hindsight search on a handle with an unset stream is refused.</summary>
    public const string UnsetStreams =
        "hindsight-unset-streams: a stream of the handle is unset, so the engine's fallback rolls it on the step's stream, which the line itself advances: the draws after a reshuffle would depend on the line, which is neither the game's world nor every stream fixed";

    /// <summary>The streams of <paramref name="state" /> that are unset, by StateCloning's stream index.</summary>
    public static List<int> UnsetStreamsOf(CombatState state) =>
        [.. Enumerable.Range(0, StateCloning.StreamCount).Where(i => StateCloning.Stream(state, i) is null)];

    /// <summary>
    /// The hindsight bound: refused without the flag, on a handle whose streams are not the record's, and on one with an
    /// unset stream (section 9 fixes every stream; an unset one falls back to the step's stream).
    /// </summary>
    public static HindsightResult SolveHindsight(CombatState state, CountingRandom rng, HindsightOptions options, bool trueStreams)
    {
        string? refusal = !options.AllowHindsight ? Hindsight.NotAllowed
            : !trueStreams ? NoTrueStreams
            : UnsetStreamsOf(state) is { Count: > 0 } unset ? $"{UnsetStreams} (unset: {string.Join(", ", unset.Select(i => StateCloning.StreamNames[i]))})"
            : null;
        return refusal is null
            ? new Hindsight(options).Solve(state, rng)
            : new HindsightResult(HindsightOutcome.Refused, refusal, WinLine.Unknown, Utility.Loss, false, [], 0, 0, 0);
    }

    /// <summary>
    /// The value of every legal move of the root (each hand index, each target), as the native q buffer holds it: a move
    /// the solve merged into an equal one (an identical card, an interchangeable enemy, an identical potion) takes that
    /// move's value; a move with none is NaN. Keyed (action, target), target -1 for none.
    /// </summary>
    public static Dictionary<Move, double> QValues(CombatState s, ExpectedResult r)
    {
        var q = new Dictionary<Move, double>();
        if (r.Status == SolveStatus.Refused)
        {
            return q;
        }

        var byMove = r.Moves.ToDictionary(m => m.Move, m => m.Value);
        int[] classes = s.PendingSelection is null && s.Enemies.Count > 0 ? StateKey.EnemyClasses(s) : [];
        int hand = s.Hand.Count;
        foreach (var m in Actions.Legal(s, Pruning.None))
        {
            if (byMove.TryGetValue(m, out var v))
            {
                q[m] = v.ToDouble();
                continue;
            }

            Rational? found = null;
            foreach (var kept in r.Moves)
            {
                if (s.PendingSelection is not null || !SameTarget(kept.Move.Target, m.Target, classes) || !SameThing(s, kept.Move.Action, m.Action, hand))
                {
                    continue;
                }

                found = kept.Value;
                break;
            }

            q[m] = found is { } f ? f.ToDouble() : double.NaN;
        }

        return q;
    }

    static bool SameTarget(int a, int b, int[] classes) =>
        a == b || (a >= 0 && b >= 0 && a < classes.Length && b < classes.Length && classes[a] == classes[b]);

    static bool SameThing(CombatState s, int a, int b, int hand)
    {
        if (a < hand && b < hand)
        {
            return DrawSteer.CardValue(s, s.Hand[a]).AsSpan().SequenceEqual(DrawSteer.CardValue(s, s.Hand[b]));
        }

        if (a > hand && b > hand)
        {
            return a == b || s.PotionSlots[a - hand - 1] == s.PotionSlots[b - hand - 1];
        }

        return a == b;
    }

    /// <summary>The first move of greatest value (the move the expected-best policy plays), or null.</summary>
    public static Move? BestMove(ExpectedResult r) => r.Moves.Count == 0 ? null : r.Best.First();

    /// <summary>
    /// Plays the expected-best policy from the position on its own streams (a copy: the position is not changed): at each
    /// decision the expected solve's first best move, stepped with the real draws and rolls. Returns the fight's terminal
    /// utility, or null when a solve was refused or the fight did not end within <paramref name="maxSteps" />.
    /// </summary>
    public static (Rational Outcome, int Decisions, bool AllExact)? PlayExpectedPolicy(CombatState state, CountingRandom rng, SolveOptions options, int maxSteps = 400)
    {
        var solver = new Expectimax(options);
        var s = state.Clone();
        var r = rng.Copy();
        bool allExact = true;
        for (int i = 0; i < maxSteps; i++)
        {
            var res = solver.SolveExpected(ExpectedView(s), r);
            if (res.Status == SolveStatus.Refused || BestMove(res) is not { } m)
            {
                return null;
            }

            allExact &= res.Exact;
            var step = Successors.Step(s, r, m);
            if (step.Terminal)
            {
                return (Utility.Terminal(s, step.PlayerWon), i + 1, allExact);
            }
        }

        return null;
    }

    static string StatusName(SolveStatus s) => s switch
    {
        SolveStatus.Exact => "exact",
        SolveStatus.Budget => "budget",
        _ => "refused",
    };

    static JsonObject MoveJson(Move m) => new() { ["action"] = m.Action, ["target"] = m.Target };

    /// <summary>EmuDump's <c>expected</c> object: the result's every field, values as exact rationals and as doubles.</summary>
    public static JsonObject ExpectedJson(CombatState s, ExpectedResult r, int allowedPotions = -1)
    {
        bool refused = r.Status == SolveStatus.Refused;
        var moves = new JsonArray();
        foreach (var m in r.Moves)
        {
            moves.Add((JsonNode)new JsonObject
            {
                ["action"] = m.Move.Action, ["target"] = m.Move.Target, ["value"] = m.Value.ToString(), ["value_f"] = m.Value.ToDouble(), ["exact"] = m.Exact,
            });
        }

        var q = new JsonArray();
        foreach (var (m, v) in QValues(s, r).OrderBy(x => x.Key.Action).ThenBy(x => x.Key.Target))
        {
            q.Add((JsonNode)new JsonArray(m.Action, m.Target, double.IsNaN(v) ? null : JsonValue.Create(v)));
        }

        return new JsonObject
        {
            ["status"] = StatusName(r.Status),
            ["reason"] = r.Reason,
            ["horizon"] = r.Horizon,
            // A refused solve has no value: null, never a 0 that reads as a certain loss.
            ["value"] = refused ? null : r.Value.ToString(),
            ["value_f"] = refused ? null : r.Value.ToDouble(),
            ["exact"] = r.Exact,
            ["exact_mass"] = refused ? null : r.ExactMass.ToString(),
            ["exact_mass_f"] = refused ? null : r.ExactMass.ToDouble(),
            ["allowed_potions"] = allowedPotions,
            ["best"] = BestMove(r) is { } b ? MoveJson(b) : null,
            ["moves"] = moves,
            ["q"] = q,
        };
    }

    /// <summary>EmuDump's <c>stats</c> object (the native <c>Sts2_SolverStats</c>): every counter of the baseline round.</summary>
    public static JsonObject StatsJson(ExpectedResult r) => SolverStatsJson.Of(r);

    public static JsonObject CountersJson(SolveCounters c, double seconds) => new()
    {
        ["nodes"] = c.Nodes,
        ["steps"] = c.Steps,
        ["table_hits"] = c.TableHits,
        ["merged"] = c.Merged,
        ["leaves"] = c.Leaves,
        ["horizon_leaves"] = c.HorizonLeaves,
        ["screens"] = c.Screens,
        ["seconds"] = seconds,
    };

    /// <summary>EmuDump's <c>hindsight</c> object. Its keys share none with the expected object's.</summary>
    public static JsonObject HindsightJson(HindsightResult h)
    {
        var line = new JsonArray();
        foreach (var m in h.Line)
        {
            line.Add((JsonNode)MoveJson(m));
        }

        return new JsonObject
        {
            ["outcome"] = h.Outcome switch { HindsightOutcome.Searched => "searched", HindsightOutcome.Cut => "cut", _ => "refused" },
            ["refusal"] = h.Refusal,
            ["win_line_exists"] = h.WinLineExists switch { WinLine.Yes => "yes", WinLine.No => "no", _ => "unknown" },
            // The bound only when every line was searched: a cut search's best line is not the bound, a refused one has none.
            ["best_utility"] = h.Bound?.ToString(),
            ["best_utility_f"] = h.Bound is { } b ? b.ToDouble() : null,
            ["complete"] = h.Complete,
            ["line"] = line,
        };
    }

    public static JsonArray CurveJson(IEnumerable<CurveEntry> curve, CombatState root)
    {
        var held = PotionCurve.HeldSlots(root);
        var arr = new JsonArray();
        foreach (var e in curve)
        {
            bool refused = e.Status == SolveStatus.Refused;
            var potions = new JsonArray();
            for (int j = 0; j < held.Count; j++)
            {
                if ((e.Subset & (1 << j)) != 0)
                {
                    potions.Add((JsonNode)new JsonObject { ["slot"] = held[j], ["potion"] = root.PotionSlots[held[j]], ["held"] = refused ? null : e.Held[held[j]].ToDouble() });
                }
            }

            arr.Add((JsonNode)new JsonObject
            {
                ["subset"] = e.Subset,
                ["status"] = StatusName(e.Status),
                ["reason"] = e.Reason,
                ["horizon"] = e.Horizon,
                ["allowed_potions"] = PotionCurve.MaskOf(held, e.Subset),
                ["expected_l0"] = refused ? null : e.ExpectedL0.ToString(),
                ["expected_l0_f"] = refused ? null : e.ExpectedL0.ToDouble(),
                ["exact"] = e.Exact,
                ["exact_mass_f"] = refused ? null : e.ExactMass.ToDouble(),
                ["win_chance"] = refused ? null : e.WinChance.ToDouble(),
                ["expected_win_hp"] = double.IsNaN(e.ExpectedWinHp) ? null : JsonValue.Create(e.ExpectedWinHp),
                ["potions"] = potions,
            });
        }

        return arr;
    }
}
