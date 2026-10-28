// The documents the solver's AOT-against-JIT gate solves (tests/python/test_solver_aot_sites.py): the published native
// library (NativeAOT) through the Python binding against `dotnet EmuDump.dll solve --snapshot` (the JIT), on the same
// `sts2-emu-snapshot/1` documents. tests/fixtures/solver/aot_sites.jsonl holds the reshuffle and draw-tie positions of
// SolverSiteTests (the two the stack-walk defect broke under NativeAOT); this test keeps it current
// (STS2_WRITE_SOLVER_FIXTURES=1 rewrites it) and holds the loaded documents to the positions' own behaviour under the JIT.
// With STS2_SOLVER_DOCS set to a file path, the harness set's documents are written there for the same gate (not committed).
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
        return string.Join("\n", Line("reshuffle", "old", 1, r).ToJsonString(), Line("draw-tie", "old", 0, t).ToJsonString()) + "\n";
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
        foreach (var line in fresh.Split('\n', StringSplitOptions.RemoveEmptyEntries))
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

    [Fact]
    public void ExportTheHarnessDocumentsWhenAsked()
    {
        if (Environment.GetEnvironmentVariable("STS2_SOLVER_DOCS") is not { Length: > 0 } path)
        {
            return;
        }

        var corpus = SolverPositions.Corpus();
        var lines = new List<string>();
        int skipped = 0;
        foreach (var recipe in SolverPositions.Recipes())
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
                skipped++;
            }
        }

        File.WriteAllText(path, string.Join("\n", lines) + "\n");
        output.WriteLine($"wrote {lines.Count} harness documents, {skipped} not writable");
    }
}
