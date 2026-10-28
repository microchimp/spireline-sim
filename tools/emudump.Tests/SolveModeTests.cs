// EmuDump's `solve` and `hindsight` commands (tools/emudump/Solve.cs, piece S4) against the native exports on the same
// fights: a native handle made by Sts2_Create(seed) and stepped through Sts2_Step with the command's seeded policy reaches
// the command's position, and Sts2_SolveExpected and Sts2_SolveHindsight answer as the command prints. The command prints the
// expected result or the hindsight bound, never both, and its hindsight refuses without --allow-hindsight and on a fight
// with an unset stream (the seed form: hindsight-unset-streams, as the native export says on the same handle); on a run's
// fight (--run: every stream set, the fight's own) it is the native bound on the same fork.
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Run;
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
        return Walk(create(seed), seed, at);
    }

    /// <summary><paramref name="at" /> moves of the command's policy on a native handle.</summary>
    static int Walk(int h, int seed, int at)
    {
        delegate* unmanaged<int, int*, int, void> valid = &NativeExports.Sts2_ValidActions;
        delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
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
        var o = new SolveOptionsNative { NodeBudget = 20_000, StepBudget = 2_000_000, MaxHorizon = 1, Prune = 31, ChanceWidthCap = 32, ForbiddenPotions = 0, UseEstimate = 1 };
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

        // The hindsight bound of the same fight: refused without the flag, and with it refused as unset streams (Sts2_Create's
        // fight shuffles on the step's stream), with no number, by the command and by the native export alike.
        var off = Command("hindsight", $"{asc}", $"{seed}", "--at", $"{at}");
        Assert.Null(off["expected"]);
        Assert.Equal("refused", (string)off["hindsight"]!["outcome"]!);
        var on = Command("hindsight", $"{asc}", $"{seed}", "--at", $"{at}", "--allow-hindsight", "--budget-nodes", "3000", "--max-turns", "1");
        var hb = on["hindsight"]!.AsObject();
        Assert.Equal("refused", (string)hb["outcome"]!);
        Assert.StartsWith("hindsight-unset-streams", (string)hb["refusal"]!);
        Assert.Null(hb["best_utility_f"]);
        o.AllowHindsight = 1;
        int flagged = mk(h, &o);
        var (hs, hr) = Hindsight(flagged, h, 3000);
        Assert.Equal(-3, hs);
        Assert.Equal((string)hb["refusal"]!, LastError());
        Assert.True(double.IsNaN(hr.BestUtility));

        delegate* unmanaged<int, void> free = &SolverExports.Sts2Solver_Destroy;
        delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
        free(solver);
        free(flagged);
        destroy(h);
    }

    static (int Status, HindsightResultNative Result) Hindsight(int solver, int h, long nodes)
    {
        delegate* unmanaged<int, int, HindsightOptionsNative*, HindsightResultNative*, int> bound = &SolverExports.Sts2_SolveHindsight;
        var ho = new HindsightOptionsNative { SearchNodeBudget = nodes, SearchStepBudget = 2_000_000, MaxTurns = 1 };
        HindsightResultNative hr;
        int hs = bound(solver, h, &ho, &hr);
        return (hs, hr);
    }

    static string LastError()
    {
        delegate* unmanaged<byte*, int, int> f = &NativeExports.GetLastError;
        var buf = new byte[4096];
        fixed (byte* p = buf)
        {
            return System.Text.Encoding.UTF8.GetString(buf, 0, f(p, buf.Length));
        }
    }

    [Theory]
    [InlineData(4, 0)]
    [InlineData(6, 3)]
    public void HindsightOnARunsFightIsTheNativeBoundOnTheSameFork(int seed, int at)
    {
        EmuDump.Engines.EngineFactory.Selected = "old";
        int asc = DefaultAscension();
        var doc = Command("hindsight", $"{asc}", $"{seed}", "--run", "--hp", "64", "--at", $"{at}", "--allow-hindsight", "--budget-nodes", "3000", "--max-turns", "1");
        Assert.Null(doc["error"]);
        var hb = doc["hindsight"]!.AsObject();
        Assert.NotEqual("refused", (string)hb["outcome"]!);

        // The native side: Sts2Run_Reset and Sts2Run_StartCombat as the run form makes its fight, forked as it stands.
        int run = RunNativeExports.Sts2Run_Create();
        int[] deck = [.. RunConstants.StarterDeckIds];
        var robs = new int[RunConstants.RunObsSize];
        byte[] seedText = System.Text.Encoding.UTF8.GetBytes($"{seed}");
        fixed (byte* sp = seedText)
        fixed (int* d = deck)
        fixed (int* ro = robs)
        {
            Assert.Equal(0, RunNativeExports.Sts2Run_Reset(run, sp, seedText.Length, asc, ro));
            Assert.Equal(0, RunNativeExports.Sts2Run_StartCombat(run, d, deck.Length, RunConstants.SlitheringStranglerEncounterId, null, 0, 64,
                RunConstants.StartingMaxHp, null, 0, 0, 99, ro));
        }

        delegate* unmanaged<int, int, int, int*, int> fork = &NativeExports.Sts2Run_ForkCombat;
        var obs = new int[NativeExports.OBS_SIZE];
        int h;
        fixed (int* ob = obs)
        {
            h = fork(run, 0, 0, ob);
        }

        Assert.True(h >= 0);
        h = Walk(h, seed, at);
        var o = new SolveOptionsNative { AllowHindsight = 1 };
        delegate* unmanaged<int, SolveOptionsNative*, int> mk = &SolverExports.Sts2Solver_Create;
        int flagged = mk(h, &o);
        var (hs, hr) = Hindsight(flagged, h, 3000);
        Assert.Equal((string)hb["outcome"]!, hs switch { 0 => "searched", 1 => "cut", _ => "refused" });
        if (hb["best_utility_f"] is { } bu)
        {
            Assert.Equal((double)bu, hr.BestUtility);
        }
        else
        {
            Assert.True(double.IsNaN(hr.BestUtility));
        }

        Assert.Equal(hb["line"]!.AsArray().Count, hr.LineLength);
        delegate* unmanaged<int, void> free = &SolverExports.Sts2Solver_Destroy;
        delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
        free(flagged);
        destroy(h);
        RunNativeExports.Sts2Run_Destroy(run);
    }

    [Fact]
    public void TheCurveIsAnExpectedOutputAndBadArgumentsExitTwo()
    {
        EmuDump.Engines.EngineFactory.Selected = "old";
        var doc = Command("solve", $"{DefaultAscension()}", "5", "--potions", "FirePotion,BlockPotion", "--hp", "9", "--curve", "--budget-nodes", "5000", "--horizon", "0");
        Assert.Equal(4, doc["curve"]!.AsArray().Count);
        Assert.Null(doc["hindsight"]);
        Assert.Equal(2, SolveMode.Run("hindsight", ["10", "5", "--curve"], new StringWriter()));
        Assert.Equal(2, SolveMode.Run("hindsight", ["--snapshot", "x.json", "--run"], new StringWriter()));
        Assert.Equal(2, SolveMode.Run("solve", ["10"], new StringWriter()));
        Assert.Equal(2, SolveMode.Run("solve", ["10", "5", "--prune", "Nonsense"], new StringWriter()));
    }
}
