using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Solver;

namespace Sts2Emulator.Interop;

// The fight solver's native exports (docs/rebuild/FIGHT_SOLVER.md section 1, piece S4; native API version 39). The two
// outputs are kept apart (section 4a): Sts2_SolveExpected fills an ExpectedResultNative, the only output a label, a regret
// or a grade may read; Sts2_SolveHindsight fills a HindsightResultNative, a separate type that shares no field with it, and
// refuses unless the solver was made with allowHindsight and the handle carries the record's own streams. Both run
// SolverService, the code path EmuDump's solve and hindsight commands print. The leaf protocol for the network at the
// horizon (Sts2Solver_PendingLeaves / Sts2Solver_SetLeafValues) comes with S5: until then a horizon leaf is valued by L0.
// sts2_gym/native.py mirrors each struct field for field; tests/python/test_solver_binding.py reads this file to hold the two
// alike. A refused result carries no number: its value fields are NaN, never a 0 that reads as a certain loss. Solver handles
// are claimed and freed atomically, but one solver handle serves one thread at a time (its tables are not shared-safe), and
// Sts2_GetLastError is one buffer for the process.

/// <summary>
/// A solve's options. A budget of 0 or less takes the default; <c>Prune</c> is the <see cref="Pruning" /> bits.
/// <c>ForbiddenPotions</c> is a slot mask (bit i: the potion in slot i may not be used), so a zeroed struct leaves every
/// potion usable; the mask the solve used comes back as the result's <c>AllowedPotions</c> (bit i: slot i usable, -1 all).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SolveOptionsNative
{
    public long NodeBudget;
    public long StepBudget;
    public int TimeBudgetMs;
    public int MaxHorizon;
    public int Prune;
    public int ChanceWidthCap;
    public int ForbiddenPotions;
    public int Utility;
    public int AllowHindsight;
    public int UseEstimate;
}

/// <summary>
/// The expected solve's result. Values are doubles of the exact rationals (NaN when refused); the q buffer holds every root
/// move's. <c>AllowedPotions</c> is the potion slot mask the solve used (bit i: slot i usable; -1 every slot).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ExpectedResultNative
{
    public int Status;
    public int Horizon;
    public int Exact;
    public int MoveCount;
    public int BestAction;
    public int BestTarget;
    public int QStride;
    public int UtilityKind;
    public double Value;
    public double ExactMass;
    public double Seconds;
    public long Nodes;
    public long Steps;
    public long TableHits;
    public long Leaves;
    public long HorizonLeaves;
    public int AllowedPotions;
}

/// <summary>A hindsight search's options. A budget of 0 or less takes the default.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct HindsightOptionsNative
{
    public long SearchNodeBudget;
    public long SearchStepBudget;
    public int SearchTimeMs;
    public int MaxTurns;
}

/// <summary>
/// The hindsight bound. Shares no field with <see cref="ExpectedResultNative" />. <c>BestUtility</c> is the bound only when
/// <c>Outcome</c> is 0 (searched): NaN when the search was cut (its best line is a lower bound on the bound) or refused.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct HindsightResultNative
{
    public const int LineCap = 64;

    public int Outcome;
    public int WinLineExists;
    public int Complete;
    public int LineLength;
    public double BestUtility;
    public double Elapsed;
    public long NodesSearched;
    public long EngineSteps;

    /// <summary>The best line's first <see cref="LineCap" /> moves as (action, target) pairs.</summary>
    public fixed int Line[128];
}

/// <summary>
/// One potion subset's entry of the curve (section 8). <c>Subset</c> is a bit set over the held potions in slot order (bit j:
/// the j-th held potion usable); <c>AllowedPotions</c> is the same subset as a slot mask (bit i: slot i usable), the
/// convention of <see cref="ExpectedResultNative.AllowedPotions" />. <c>Held</c> is per slot, NaN for a slot outside the
/// subset. Every entry of a curve has the same <c>CurveHorizon</c>. A refused entry's values and chances are NaN.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CurveEntryNative
{
    public int Subset;
    public int CurveStatus;
    public int CurveExact;
    public int CurveHorizon;
    public double ExpectedL0;
    public double WinChance;
    public double ExpectedWinHp;
    public double CurveExactMass;
    public fixed double Held[8];
    public int AllowedPotions;
}

public static class SolverExports
{
    /// <summary>A solver handle: the options it was made with, its tables for the fight (one per potion mask) and the hindsight flag.</summary>
    internal sealed class NativeSolver(SolveOptions options, bool allowHindsight)
    {
        public readonly SolveOptions Options = options;
        public readonly bool AllowHindsight = allowHindsight;
        public readonly Dictionary<int, TranspositionTable> Tables = [];

        /// <summary>The last expected solve's stats object (<see cref="SolverStatsJson" />), for <c>Sts2_SolverStats</c>.</summary>
        public System.Text.Json.Nodes.JsonObject? LastStats;

        public TranspositionTable TableFor(int mask)
        {
            if (!Tables.TryGetValue(mask, out var t))
            {
                Tables[mask] = t = new TranspositionTable();
            }

            return t;
        }
    }

    static readonly NativeSolver?[] _solvers = new NativeSolver?[256];

    internal const string OnlyL0 = "utility: only L0 (0) is built; SearchHp and L0Priced come with the oracle (S5)";

    /// <summary>The options a native struct gives over <paramref name="basis" /> (null: the basis).</summary>
    internal static unsafe SolveOptions Options(SolveOptionsNative* o, SolveOptions basis)
    {
        if (o is null)
        {
            return basis;
        }

        var d = new SolveOptions();
        return basis with
        {
            NodeBudget = o->NodeBudget > 0 ? o->NodeBudget : d.NodeBudget,
            StepBudget = o->StepBudget > 0 ? o->StepBudget : d.StepBudget,
            TimeBudgetMs = Math.Max(0, o->TimeBudgetMs),
            MaxHorizon = o->MaxHorizon >= 0 ? o->MaxHorizon : d.MaxHorizon,
            Prune = (Pruning)o->Prune & Pruning.All,
            ChanceWidthCap = o->ChanceWidthCap > 0 ? o->ChanceWidthCap : d.ChanceWidthCap,
            AllowedPotions = ~o->ForbiddenPotions,
            UseEstimate = o->UseEstimate != 0,
        };
    }

    internal static NativeSolver? SolverOf(int solver) => solver >= 0 && solver < _solvers.Length ? Volatile.Read(ref _solvers[solver]) : null;

    static int Fail(string why, int code)
    {
        NativeExports.Refuse(why);
        return code;
    }

    /// <summary>A solver for the fight of <paramref name="combatHandle" />. <c>allowHindsight</c> is read here and only here.</summary>
    /// <returns>
    /// The solver handle; -1 for an unknown combat, -2 for a full pool (as <c>Sts2Run_ForkCombat</c>), -3 for refused options
    /// (<c>Sts2_GetLastError</c>). A slot is claimed atomically, so two threads never get the same handle.
    /// </returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2Solver_Create")]
    public static unsafe int Sts2Solver_Create(int combatHandle, SolveOptionsNative* options)
    {
        try
        {
            if (!NativeExports.TryGetCombat(combatHandle, out _, out _, out _))
            {
                return Fail("unknown combat handle", -1);
            }

            if (options is not null && options->Utility != 0)
            {
                return Fail(OnlyL0, -3);
            }

            var solver = new NativeSolver(Options(options, new SolveOptions()), options is not null && options->AllowHindsight != 0);
            for (int i = 0; i < _solvers.Length; i++)
            {
                if (Interlocked.CompareExchange(ref _solvers[i], solver, null) is null)
                {
                    return i;
                }
            }

            return Fail("the solver pool is full", -2);
        }
        catch (Exception e)
        {
            return Fail($"internal error: {e.GetType().Name}: {e.Message}", -1);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Solver_Destroy")]
    public static void Sts2Solver_Destroy(int solver)
    {
        if (solver >= 0 && solver < _solvers.Length)
        {
            Interlocked.Exchange(ref _solvers[solver], null);
        }
    }

    /// <summary>Forgets the solver's tables (a new fight on the same solver): 0, or -1 for an unknown solver.</summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2Solver_Reset")]
    public static int Sts2Solver_Reset(int solver)
    {
        if (SolverOf(solver) is not { } s)
        {
            return Fail("unknown solver handle", -1);
        }

        s.Tables.Clear();
        return 0;
    }

    /// <summary>
    /// The expected value of the combat's position (section 1): status 0 exact, 1 budget reached (the last completed
    /// horizon), -1 error, -3 refused (<c>Sts2_GetLastError</c>). <paramref name="qBuf" /> receives
    /// <paramref name="maxActions" /> x <c>QStride</c> floats: entry action x QStride + 0 for a move without a target,
    /// + 1 + t for target t; NaN where there is no such move.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolveExpected")]
    public static unsafe int Sts2_SolveExpected(int solver, int combatHandle, SolveOptionsNative* options, ExpectedResultNative* result, float* qBuf, int maxActions)
    {
        try
        {
            if (SolverOf(solver) is not { } sv || !NativeExports.TryGetCombat(combatHandle, out var state, out var rng, out _))
            {
                return Fail("unknown solver or combat handle", -1);
            }

            if (options is not null && options->Utility != 0)
            {
                return Fail(OnlyL0, -3);
            }

            var opts = Options(options, sv.Options);
            var r = SolverService.SolveExpected(state, rng, opts, sv.TableFor(opts.AllowedPotions));
            sv.LastStats = SolverStatsJson.Of(r);
            WriteExpected(state, r, result, qBuf, maxActions, opts.AllowedPotions);
            if (r.Status == SolveStatus.Refused)
            {
                NativeExports.Refuse(r.Reason ?? "refused");
            }

            return (int)r.Status;
        }
        catch (Exception e)
        {
            return Fail($"internal error: {e.GetType().Name}: {e.Message}", -1);
        }
    }

    internal static unsafe void WriteExpected(CombatState state, ExpectedResult r, ExpectedResultNative* result, float* qBuf, int maxActions, int allowedPotions = -1)
    {
        if (result is not null)
        {
            var best = SolverService.BestMove(r);
            bool refused = r.Status == SolveStatus.Refused;
            *result = new ExpectedResultNative
            {
                Status = (int)r.Status,
                Horizon = r.Horizon,
                Exact = r.Exact ? 1 : 0,
                MoveCount = r.Moves.Count,
                BestAction = best?.Action ?? -1,
                BestTarget = best?.Target ?? -1,
                QStride = SolverService.QStride,
                UtilityKind = 0,
                Value = refused ? double.NaN : r.Value.ToDouble(),
                ExactMass = refused ? double.NaN : r.ExactMass.ToDouble(),
                Seconds = r.Seconds,
                Nodes = r.Counters.Nodes,
                Steps = r.Counters.Steps,
                TableHits = r.Counters.TableHits,
                Leaves = r.Counters.Leaves,
                HorizonLeaves = r.Counters.HorizonLeaves,
                AllowedPotions = allowedPotions,
            };
        }

        if (qBuf is null || maxActions <= 0)
        {
            return;
        }

        int n = maxActions * SolverService.QStride;
        for (int i = 0; i < n; i++)
        {
            qBuf[i] = float.NaN;
        }

        foreach (var (m, v) in SolverService.QValues(state, r))
        {
            int col = m.Target < 0 ? 0 : 1 + m.Target;
            if (m.Action >= 0 && m.Action < maxActions && col < SolverService.QStride)
            {
                qBuf[m.Action * SolverService.QStride + col] = (float)v;
            }
        }
    }

    /// <summary>
    /// The hindsight bound (sections 4a and 9), for lost-fight attribution only: 0 searched, 1 cut (budget or turn cap),
    /// -1 error, -3 refused: the solver was not made with allowHindsight, or the handle does not carry the record's own
    /// streams (a resampled fork, a loaded document), or a stream of the handle is unset (<c>hindsight-unset-streams</c>), or
    /// the position was refused. The flag set at Sts2Solver_Create is the gate.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolveHindsight")]
    public static unsafe int Sts2_SolveHindsight(int solver, int combatHandle, HindsightOptionsNative* options, HindsightResultNative* result)
    {
        try
        {
            if (SolverOf(solver) is not { } sv || !NativeExports.TryGetCombat(combatHandle, out var state, out var rng, out bool trueStreams))
            {
                return Fail("unknown solver or combat handle", -1);
            }

            var d = new HindsightOptions();
            var opts = new HindsightOptions
            {
                AllowHindsight = sv.AllowHindsight,
                SearchNodeBudget = options is not null && options->SearchNodeBudget > 0 ? options->SearchNodeBudget : d.SearchNodeBudget,
                SearchStepBudget = options is not null && options->SearchStepBudget > 0 ? options->SearchStepBudget : d.SearchStepBudget,
                SearchTimeMs = options is not null ? Math.Max(0, options->SearchTimeMs) : 0,
                MaxTurns = options is not null && options->MaxTurns > 0 ? options->MaxTurns : d.MaxTurns,
            };
            var h = SolverService.SolveHindsight(state, rng, opts, trueStreams);
            WriteHindsight(h, result);
            if (h.Outcome == HindsightOutcome.Refused)
            {
                NativeExports.Refuse(h.Refusal ?? "refused");
            }

            return (int)h.Outcome;
        }
        catch (Exception e)
        {
            return Fail($"internal error: {e.GetType().Name}: {e.Message}", -1);
        }
    }

    internal static unsafe void WriteHindsight(HindsightResult h, HindsightResultNative* result)
    {
        if (result is null)
        {
            return;
        }

        *result = new HindsightResultNative
        {
            Outcome = (int)h.Outcome,
            WinLineExists = (int)h.WinLineExists,
            Complete = h.Complete ? 1 : 0,
            LineLength = h.Line.Count,
            BestUtility = h.BoundDouble,
            Elapsed = h.Elapsed,
            NodesSearched = h.NodesSearched,
            EngineSteps = h.EngineSteps,
        };
        int n = Math.Min(h.Line.Count, HindsightResultNative.LineCap);
        for (int i = 0; i < n; i++)
        {
            result->Line[2 * i] = h.Line[i].Action;
            result->Line[2 * i + 1] = h.Line[i].Target;
        }
    }

    /// <summary>
    /// The potion value curve (section 8): one entry per subset of the potions held, the empty one first, in bit order over
    /// the held slots. Returns the number of subsets (only the first <paramref name="maxSubsets" /> are solved and written),
    /// -1 for an error, -3 for refused options. Every entry has the curve's one horizon; a refused entry's numbers are NaN and
    /// the first refused entry's reason is in <c>Sts2_GetLastError</c>.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolvePotionCurve")]
    public static unsafe int Sts2_SolvePotionCurve(int solver, int combatHandle, SolveOptionsNative* options, CurveEntryNative* buf, int maxSubsets)
    {
        try
        {
            if (SolverOf(solver) is not { } sv || !NativeExports.TryGetCombat(combatHandle, out var state, out var rng, out _))
            {
                return Fail("unknown solver or combat handle", -1);
            }

            if (options is not null && options->Utility != 0)
            {
                return Fail(OnlyL0, -3);
            }

            var held = PotionCurve.HeldSlots(state);
            var curve = SolverService.SolvePotionCurve(state, rng, Options(options, sv.Options), buf is null ? 0 : maxSubsets);
            for (int i = 0; i < curve.Count; i++)
            {
                WriteCurveEntry(curve[i], held, buf + i);
            }

            if (curve.FirstOrDefault(e => e.Status == SolveStatus.Refused) is { } first)
            {
                NativeExports.Refuse($"subset {first.Subset}: {first.Reason}");
            }

            return 1 << held.Count;
        }
        catch (Exception e)
        {
            return Fail($"internal error: {e.GetType().Name}: {e.Message}", -1);
        }
    }

    internal static unsafe void WriteCurveEntry(CurveEntry e, List<int> held, CurveEntryNative* o)
    {
        bool refused = e.Status == SolveStatus.Refused;
        *o = new CurveEntryNative
        {
            Subset = e.Subset,
            CurveStatus = (int)e.Status,
            CurveExact = e.Exact ? 1 : 0,
            CurveHorizon = e.Horizon,
            ExpectedL0 = refused ? double.NaN : e.ExpectedL0.ToDouble(),
            WinChance = refused ? double.NaN : e.WinChance.ToDouble(),
            ExpectedWinHp = refused ? double.NaN : e.ExpectedWinHp,
            CurveExactMass = refused ? double.NaN : e.ExactMass.ToDouble(),
            AllowedPotions = PotionCurve.MaskOf(held, e.Subset),
        };
        for (int i = 0; i < 8; i++)
        {
            o->Held[i] = !refused && e.Held.TryGetValue(i, out var p) ? p.ToDouble() : double.NaN;
        }
    }

    /// <summary>
    /// The last expected solve's stats on <paramref name="solver" /> as UTF-8 JSON (size-and-retry): the counters, the deepening's
    /// horizons, and the phase split and merge audit when their switches were on. Returns the length, or -1 for an unknown
    /// solver and -2 for one that has not solved yet.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolverStats")]
    public static unsafe int Sts2_SolverStats(int solver, byte* buf, int cap)
    {
        if (SolverOf(solver) is not { } s)
        {
            return Fail("unknown solver handle", -1);
        }

        if (s.LastStats is not { } stats)
        {
            return Fail("the solver has not solved yet", -2);
        }

        byte[] json = Encoding.UTF8.GetBytes(stats.ToJsonString());
        if (buf is not null && cap >= json.Length)
        {
            json.CopyTo(new Span<byte>(buf, cap));
        }

        return json.Length;
    }

    /// <summary>
    /// The solver's debug switches for the whole process (<see cref="SolverDebug.Flags" />: bit 0 the phase timers, bit 1 the
    /// merge audit); a negative value only reads them. Returns the flags before the call. Off by default; neither changes a value.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolverDebug")]
    public static int Sts2_SolverDebug(int flags)
    {
        int before = SolverDebug.Flags;
        if (flags >= 0)
        {
            SolverDebug.Flags = flags;
        }

        return before;
    }

    /// <summary>The solver's version, its pruning names with their bits and its utility names, as UTF-8 JSON; size-and-retry.</summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_SolverInfo")]
    public static unsafe int Sts2_SolverInfo(byte* buf, int cap)
    {
        byte[] json = Encoding.UTF8.GetBytes(Info().ToJsonString());
        if (buf is not null && cap >= json.Length)
        {
            json.CopyTo(new Span<byte>(buf, cap));
        }

        return json.Length;
    }

    internal static JsonObject Info()
    {
        var prunings = new JsonObject();
        // Named by hand, not through the enum's metadata, which a trimmed build need not keep.
        foreach (var (name, bit) in new[]
        {
            ("Transposition", Pruning.Transposition), ("MultisetHand", Pruning.MultisetHand), ("InterchangeableEnemies", Pruning.InterchangeableEnemies),
            ("SubtreeReuse", Pruning.SubtreeReuse), ("DrawByValue", Pruning.DrawByValue), ("MacroDraw", Pruning.MacroDraw),
        })
        {
            prunings[name] = (int)bit;
        }

        return new JsonObject
        {
            ["solver_version"] = SolverService.Version,
            ["native_api_version"] = NativeExports.NATIVE_API_VERSION,
            ["prunings"] = prunings,
            ["prune_all"] = (int)Pruning.All,
            ["utilities"] = new JsonObject { ["L0"] = 0 },
            ["q_stride"] = SolverService.QStride,
            ["hindsight_line_cap"] = HindsightResultNative.LineCap,
        };
    }
}
