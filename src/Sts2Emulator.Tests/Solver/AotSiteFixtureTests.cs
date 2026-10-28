// The documents the solver's AOT-against-JIT gate solves (tests/python/test_solver_aot_sites.py): the published native
// library (NativeAOT) through the Python binding against `dotnet EmuDump.dll solve --snapshot` (the JIT), on the same
// `sts2-emu-snapshot/1` documents. tests/fixtures/solver/aot_sites.jsonl holds the reshuffle and draw-tie positions of
// SolverSiteTests (the two the stack-walk defect broke under NativeAOT) and a sample of the equal-value harness's positions
// (every 40th recipe), so that the gate compares the native library with EmuDump on harness positions in every run of
// tools/build_and_test.sh, from documents in the repo. This test keeps the file current (STS2_WRITE_SOLVER_FIXTURES=1
// rewrites it) and holds the two named documents to the positions' own behaviour under the JIT.
using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public class AotSiteFixtureTests(ITestOutputHelper output)
{
    public const int LoadSeed = 5;

    static string FixturePath => Path.Combine(KeyFieldCoverageTests.Repo(), "tests", "fixtures", "solver", "aot_sites.jsonl");

    static JsonObject Line(string name, string engine, int horizon, CombatState s) => new()
    {
        ["name"] = name,
        ["engine"] = engine,
        ["horizon"] = horizon,
        ["doc"] = JsonNode.Parse(Encoding.UTF8.GetString(CombatSnapshot.Write(s))),
    };

    static string Fixture()
    {
        var (r, _) = SolverSiteTests.Reshuffle();
        var (t, _) = SolverSiteTests.DrawTie();
        var lines = new List<string> { Line("reshuffle", "old", 1, r).ToJsonString(), Line("draw-tie", "old", 0, t).ToJsonString() };
        var corpus = SolverPositions.Corpus();
        foreach (var recipe in SolverPositions.Recipes().Where(x => x.Id % 40 == 0))
        {
            if (SolverPositions.Build(recipe, corpus) is not { } pos)
            {
                continue;
            }

            try
            {
                var line = Line($"harness-{recipe.Id}", recipe.Engine, recipe.Horizon, pos.State);
                line["id"] = recipe.Id;
                lines.Add(line.ToJsonString());
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                // a position the document cannot hold: not in the sample
            }
        }

        return string.Join("\n", lines) + "\n";
    }

    [Fact]
    public void TheCommittedAotFixtureIsCurrentAndBehavesUnderTheJit()
    {
        string fresh = Fixture();
        if (Environment.GetEnvironmentVariable("STS2_WRITE_SOLVER_FIXTURES") == "1")
        {
            File.WriteAllText(FixturePath, fresh);
        }

        Assert.Equal(fresh, File.ReadAllText(FixturePath).Replace("\r\n", "\n"));
        var all = fresh.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.True(all.Length >= 10, $"only {all.Length} documents");
        foreach (var line in all.Take(2))
        {
            var o = JsonNode.Parse(line)!.AsObject();
            var (s, rng) = NativeExports.LoadFight(Encoding.UTF8.GetBytes(o["doc"]!.ToJsonString()), LoadSeed, null);
            var r = SolverService.SolveExpected(s, rng, new SolveOptions { MaxHorizon = (int)o["horizon"]!, UseEstimate = false });
            output.WriteLine($"{o["name"]}: {r.Status} {r.ValueDouble:0.000000} {r.Reason} nodes {r.Counters.Nodes} steps {r.Counters.Steps}");
            if ((string)o["name"]! == "draw-tie")
            {
                Assert.Equal(SolveStatus.Refused, r.Status);
                Assert.StartsWith("draw-tie", r.Reason);
            }
            else
            {
                Assert.NotEqual(SolveStatus.Refused, r.Status);
            }
        }
    }
}
