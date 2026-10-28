using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;

namespace Sts2Emulator.Tests.Solver;

/// <summary>The solver's native exports called as a C caller calls them (through their unmanaged entry points).</summary>
internal static unsafe class NativeSolverCalls
{
    public const int MaxActions = 64;

    public static SolveOptionsNative Options(SolveOptions o, bool allowHindsight = false) => new()
    {
        NodeBudget = o.NodeBudget,
        StepBudget = o.StepBudget,
        TimeBudgetMs = o.TimeBudgetMs,
        MaxHorizon = o.MaxHorizon,
        Prune = (int)o.Prune,
        ChanceWidthCap = o.ChanceWidthCap,
        AllowedPotions = o.AllowedPotions,
        Utility = 0,
        AllowHindsight = allowHindsight ? 1 : 0,
        UseEstimate = o.UseEstimate ? 1 : 0,
    };

    public static int Adopt(CombatState s, CountingRandom rng, bool trueStreams = true) => NativeExports.Adopt(s, rng, trueStreams);

    public static int Create(int combat, SolveOptionsNative? options)
    {
        delegate* unmanaged<int, SolveOptionsNative*, int> f = &SolverExports.Sts2Solver_Create;
        if (options is { } o)
        {
            return f(combat, &o);
        }

        return f(combat, null);
    }

    public static void Destroy(int solver)
    {
        delegate* unmanaged<int, void> f = &SolverExports.Sts2Solver_Destroy;
        f(solver);
    }

    public static void DestroyCombat(int handle)
    {
        delegate* unmanaged<int, void> f = &NativeExports.Sts2_Destroy;
        f(handle);
    }

    public static int Clone(int handle, bool resample, int seed)
    {
        delegate* unmanaged<int, int, int, int*, int> f = &NativeExports.Sts2_Clone;
        var obs = new int[NativeExports.OBS_SIZE];
        fixed (int* p = obs)
        {
            return f(handle, resample ? 1 : 0, seed, p);
        }
    }

    public static (int Status, ExpectedResultNative Result, float[] Q) Expected(int solver, int combat, SolveOptionsNative? options)
    {
        delegate* unmanaged<int, int, SolveOptionsNative*, ExpectedResultNative*, float*, int, int> f = &SolverExports.Sts2_SolveExpected;
        var q = new float[MaxActions * SolverService.QStride];
        ExpectedResultNative r;
        int status;
        fixed (float* qp = q)
        {
            if (options is { } o)
            {
                status = f(solver, combat, &o, &r, qp, MaxActions);
            }
            else
            {
                status = f(solver, combat, null, &r, qp, MaxActions);
            }
        }

        return (status, r, q);
    }

    public static (int Status, HindsightResultNative Result) Hindsight(int solver, int combat, long nodes = 0)
    {
        delegate* unmanaged<int, int, HindsightOptionsNative*, HindsightResultNative*, int> f = &SolverExports.Sts2_SolveHindsight;
        var o = new HindsightOptionsNative { SearchNodeBudget = nodes };
        HindsightResultNative r;
        int status = f(solver, combat, &o, &r);
        return (status, r);
    }

    public static (int Count, CurveEntryNative[] Entries) Curve(int solver, int combat, SolveOptionsNative? options, int max = 16)
    {
        delegate* unmanaged<int, int, SolveOptionsNative*, CurveEntryNative*, int, int> f = &SolverExports.Sts2_SolvePotionCurve;
        var buf = new CurveEntryNative[max];
        int n;
        fixed (CurveEntryNative* b = buf)
        {
            if (options is { } o)
            {
                n = f(solver, combat, &o, b, max);
            }
            else
            {
                n = f(solver, combat, null, b, max);
            }
        }

        return (n, buf);
    }

    public static string LastError()
    {
        delegate* unmanaged<byte*, int, int> f = &NativeExports.GetLastError;
        int n = f(null, 0);
        var buf = new byte[Math.Max(1, n)];
        fixed (byte* p = buf)
        {
            n = f(p, buf.Length);
        }

        return Encoding.UTF8.GetString(buf, 0, n);
    }

    public static string Info()
    {
        delegate* unmanaged<byte*, int, int> f = &SolverExports.Sts2_SolverInfo;
        int n = f(null, 0);
        var buf = new byte[n];
        fixed (byte* p = buf)
        {
            f(p, n);
        }

        return Encoding.UTF8.GetString(buf);
    }
}
