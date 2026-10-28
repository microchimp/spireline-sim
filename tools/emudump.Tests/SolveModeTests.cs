// EmuDump's `solve` and `hindsight` commands (tools/emudump/Solve.cs, piece S4) against the native exports on the same
// fights: a native handle made by Sts2_Create(seed) and stepped through Sts2_Step with the command's seeded policy reaches
// the command's position, and Sts2_SolveExpected and Sts2_SolveHindsight answer as the command prints. The command prints the
// expected result or the hindsight bound, never both, and its hindsight refuses without --allow-hindsight.
using System.Text.Json.Nodes;
using Sts2Emulator.Interop;
using Xunit;

namespace EmuDump.Tests;

public unsafe class SolveModeTests
{
    static JsonObject Command(params string[] args)
    {
        var w = new StringWriter();
        Assert.Equal(0, SolveMode.Run(args[0], args[1..], w));
        return JsonNode.Parse(w.ToString().Trim())!.AsObject();
    }

    static int DefaultAscension()
    {
        delegate* unmanaged<int> f = &NativeExports.Sts2_DefaultAscension;
        return f();
    }

    /// <summary>The native handle at the command's position: Sts2_Create(seed), then <paramref name="at" /> moves of the command's policy.</summary>
    static int NativeAt(int seed, int at)
    {
        delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
        delegate* unmanaged<int, int*, int, void> valid = &NativeExports.Sts2_ValidActions;
        delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
        int h = create(seed);
        var policy = new Random(seed);
        var mask = new int[128];
        var obs = new int[NativeExports.OBS_SIZE];
        float reward;
        for (int i = 0; i < at; i++)
        {
            fixed (int* m = mask)
            {
                valid(h, m, mask.Length);
            }

            int[] ids = [.. Enumerable.Range(0, mask.Length).Where(a => mask[a] != 0)];
            fixed (int* o = obs)
            {
                Assert.Equal(0, step(h, ids[policy.Next(ids.Length)], o, &reward));
            }
        }

        return h;
    }

    [Theory]
    [InlineData(3, 0)]
    [InlineData(4, 6)]
    [InlineData(9, 11)]
    public void SolveEqualsTheNativeExportOnTheSameFight(int seed, int at)
    {
        EmuDump.Engines.EngineFactory.Selected = "old";
        int asc = DefaultAscension();
        var doc = Command("solve", $"{asc}", $"{seed}", "--at", $"{at}", "--budget-nodes", "20000", "--horizon", "1");
        Assert.Null(doc["hindsight"]);
        Assert.Null(doc["error"]);
        var e = doc["expected"]!.AsObject();

        int h = NativeAt(seed, at);
        var o = new SolveOptionsNative { NodeBudget = 20_000, StepBudget = 2_000_000, MaxHorizon = 1, Prune = 31, ChanceWidthCap = 32, AllowedPotions = -1, UseEstimate = 1 };
        delegate* unmanaged<int, SolveOptionsNative*, int> mk = &SolverExports.Sts2Solver_Create;
        delegate* unmanaged<int, int, SolveOptionsNative*, ExpectedResultNative*, float*, int, int> solve = &SolverExports.Sts2_SolveExpected;
        int solver = mk(h, &o);
        Assert.True(solver >= 0);
        ExpectedResultNative r;
        var q = new float[64 * 7];
        int status;
        fixed (float* qp = q)
        {
            status = solve(solver, h, &o, &r, qp, 64);
        }

        Assert.Equal((string)e["status"]!, status switch { 0 => "exact", 1 => "budget", _ => "refused" });
        if (status >= 0)
        {
            Assert.Equal((double)e["value_f"]!, r.Value);
            Assert.Equal((int)e["horizon"]!, r.Horizon);
            Assert.Equal(e["moves"]!.AsArray().Count, r.MoveCount);
            Assert.Equal((int)e["best"]!["action"]!, r.BestAction);
            foreach (var row in e["q"]!.AsArray())
            {
                int a = (int)row![0]!, t = (int)row[1]!;
                float got = q[a * 7 + (t < 0 ? 0 : 1 + t)];
                if (row[2] is { } v)
                {
                    Assert.Equal((float)(double)v, got);
                }
                else
                {
                    Assert.True(float.IsNaN(got));
                }
            }
        }

        // The hindsight bound of the same fight: refused without the flag, the native bound with it.
        var off = Command("hindsight", $"{asc}", $"{seed}", "--at", $"{at}");
        Assert.Null(off["expected"]);
        Assert.Equal("refused", (string)off["hindsight"]!["outcome"]!);
        var on = Command("hindsight", $"{asc}", $"{seed}", "--at", $"{at}", "--allow-hindsight", "--budget-nodes", "3000", "--max-turns", "1");
        var hb = on["hindsight"]!.AsObject();
        o.AllowHindsight = 1;
        int flagged = mk(h, &o);
        delegate* unmanaged<int, int, HindsightOptionsNative*, HindsightResultNative*, int> bound = &SolverExports.Sts2_SolveHindsight;
        var ho = new HindsightOptionsNative { SearchNodeBudget = 3000, SearchStepBudget = 2_000_000, MaxTurns = 1 };
        HindsightResultNative hr;
        int hs = bound(flagged, h, &ho, &hr);
        Assert.Equal((string)hb["outcome"]!, hs switch { 0 => "searched", 1 => "cut", _ => "refused" });
        Assert.Equal((double)hb["best_utility_f"]!, hr.BestUtility);
        Assert.Equal(hb["line"]!.AsArray().Count, hr.LineLength);

        delegate* unmanaged<int, void> free = &SolverExports.Sts2Solver_Destroy;
        delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
        free(solver);
        free(flagged);
        destroy(h);
    }

    [Fact]
    public void TheCurveIsAnExpectedOutputAndBadArgumentsExitTwo()
    {
        EmuDump.Engines.EngineFactory.Selected = "old";
        var doc = Command("solve", $"{DefaultAscension()}", "5", "--potions", "FirePotion,BlockPotion", "--hp", "9", "--curve", "--budget-nodes", "5000", "--horizon", "0");
        Assert.Equal(4, doc["curve"]!.AsArray().Count);
        Assert.Null(doc["hindsight"]);
        Assert.Equal(2, SolveMode.Run("hindsight", ["10", "5", "--curve"], new StringWriter()));
        Assert.Equal(2, SolveMode.Run("solve", ["10"], new StringWriter()));
        Assert.Equal(2, SolveMode.Run("solve", ["10", "5", "--prune", "Nonsense"], new StringWriter()));
    }
}
