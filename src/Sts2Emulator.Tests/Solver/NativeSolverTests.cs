// The native exports against EmuDump's code path (piece S4, docs/rebuild/FIGHT_SOLVER.md section 1): on the equal-value
// harness's positions (tests/fixtures/solver/positions.jsonl), Sts2_SolveExpected called through its unmanaged entry point
// gives what EmuDump's `solve` prints for the same position (SolverService.SolveExpected and SolverService.ExpectedJson,
// which tools/emudump/Engines/Old/Solve.cs writes and nothing else), field for field and move for move, the q buffer included. EmuDump's
// own command is held to the native handle on its seed fights in tools/emudump.Tests/SolveModeTests.cs. Both run in this
// JIT host: the gate that sets the published NativeAOT library against `dotnet EmuDump.dll` is
// tests/python/test_solver_aot_sites.py (the harness documents come from AotSiteFixtureTests).
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;
using Xunit;
using Xunit.Abstractions;

namespace Sts2Emulator.Tests.Solver;

public unsafe class NativeSolverTests(ITestOutputHelper output)
{
    [Fact]
    public void TheApiVersionSaysTheSolverExportsAreThere()
    {
        Assert.Equal(39, NativeExports.NATIVE_API_VERSION);
        var info = JsonNode.Parse(NativeSolverCalls.Info())!.AsObject();
        Assert.Equal(39, (int)info["native_api_version"]!);
        Assert.Equal(SolverService.Version, (string)info["solver_version"]!);
        Assert.Equal((int)Pruning.All, (int)info["prune_all"]!);
        Assert.Equal(5, info["prunings"]!.AsObject().Count);
        Assert.Equal(SolverService.QStride, (int)info["q_stride"]!);
    }

    internal static SolveOptions HarnessOptions(int horizon) =>
        new() { MaxHorizon = horizon, NodeBudget = 20_000, StepBudget = 200_000, UseEstimate = false };

    [Fact]
    public void TheExportsWriteWhatEmuDumpPrintsOnTheHarnessSet()
    {
        var corpus = SolverPositions.Corpus();
        int compared = 0, exact = 0, refused = 0, budget = 0, qEntries = 0;
        var diffs = new List<string>();
        foreach (var recipe in SolverPositions.Recipes())
        {
            if (SolverPositions.Build(recipe, corpus) is not { } pos)
            {
                continue;
            }

            var opts = HarnessOptions(recipe.Horizon);
            // EmuDump's side: the service and the JSON it prints.
            var lib = SolverService.SolveExpected(pos.State, pos.Rng, opts);
            var json = SolverService.ExpectedJson(pos.State, lib);
            // The native side, on a handle holding the same position.
            int handle = NativeSolverCalls.Adopt(pos.State, pos.Rng);
            Assert.True(handle >= 0);
            int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(opts));
            Assert.True(solver >= 0, NativeSolverCalls.LastError());
            var (status, r, q) = NativeSolverCalls.Expected(solver, handle, NativeSolverCalls.Options(opts));
            string error = status == -3 ? NativeSolverCalls.LastError() : "";
            NativeSolverCalls.Destroy(solver);
            NativeSolverCalls.DestroyCombat(handle);

            compared++;
            exact += lib.Status == SolveStatus.Exact ? 1 : 0;
            refused += lib.Status == SolveStatus.Refused ? 1 : 0;
            budget += lib.Status == SolveStatus.Budget ? 1 : 0;
            string tag = $"#{recipe.Id}";
            if (status != (int)lib.Status || r.Status != status)
            {
                diffs.Add($"{tag}: status native {status} vs emudump {json["status"]}");
                continue;
            }

            if (lib.Status == SolveStatus.Refused)
            {
                if (error != lib.Reason)
                {
                    diffs.Add($"{tag}: refusal native '{error}' vs emudump '{lib.Reason}'");
                }

                continue;
            }

            if (r.Value != (double)json["value_f"]! || r.ExactMass != (double)json["exact_mass_f"]! || (r.Exact == 1) != (bool)json["exact"]!
                || r.Horizon != (int)json["horizon"]! || r.MoveCount != json["moves"]!.AsArray().Count)
            {
                diffs.Add($"{tag}: native value {r.Value} mass {r.ExactMass} exact {r.Exact} horizon {r.Horizon} vs emudump {json.ToJsonString()}");
            }

            var best = json["best"]!;
            if (r.BestAction != (int)best["action"]! || r.BestTarget != (int)best["target"]!)
            {
                diffs.Add($"{tag}: best native {r.BestAction}@{r.BestTarget} vs emudump {best.ToJsonString()}");
            }

            // The q buffer: every legal move EmuDump lists, at its (action, target) entry, and NaN everywhere else.
            var expectedQ = new Dictionary<int, float>();
            foreach (var row in json["q"]!.AsArray())
            {
                int a = (int)row![0]!, t = (int)row[1]!;
                if (row[2] is { } v && a < NativeSolverCalls.MaxActions)
                {
                    expectedQ[a * SolverService.QStride + (t < 0 ? 0 : 1 + t)] = (float)(double)v;
                }
            }

            for (int i = 0; i < q.Length; i++)
            {
                bool want = expectedQ.TryGetValue(i, out float v);
                if (want ? q[i] != v : !float.IsNaN(q[i]))
                {
                    diffs.Add($"{tag}: q[{i}] native {q[i]} vs emudump {(want ? v : float.NaN)}");
                }
            }

            qEntries += expectedQ.Count;
        }

        output.WriteLine($"native vs emudump: {compared} positions ({exact} exact, {budget} budget, {refused} refused), {qEntries} q entries, {diffs.Count} differences");
        Assert.True(compared >= 300, $"only {compared} harness positions built");
        Assert.True(exact >= 50, $"only {exact} exact solves");
        Assert.True(compared - refused >= 250, $"only {compared - refused} solved: equal refusals would carry the comparison");
        Assert.Empty(diffs);
    }

    [Fact]
    public void AMergedMoveTakesTheValueOfTheMoveItWasMergedInto()
    {
        // Ten Strikes: the hand holds five equal cards, the solve keeps one per target; the q buffer gives all five.
        var (s, rng) = SolverPositions.Fight("old", 21, [.. Enumerable.Repeat(GeneratedData.Cards.FindId("StrikeIronclad")!.Value, 10)], [], []);
        foreach (var e in s.Enemies.Where(e => e.Hp > 0))
        {
            e.Hp = 7;
        }

        var r = SolverService.SolveExpected(s, rng, new SolveOptions { MaxHorizon = 0 });
        var q = SolverService.QValues(s, r);
        var strikes = q.Where(kv => kv.Key.Action < s.Hand.Count).ToList();
        Assert.True(r.Moves.Count(m => m.Move.Action < s.Hand.Count) < strikes.Count);
        Assert.All(strikes, kv => Assert.False(double.IsNaN(kv.Value), $"{kv.Key} has no value"));
        Assert.Single(strikes.Select(kv => kv.Value).Distinct());
    }

    [Fact]
    public unsafe void AHandleWithUnsetStreamsIsSolvedOnStreamsOfItsOwn()
    {
        // Sts2_Create's fight draws its shuffles from the step's stream (no ShuffleRng); the chance hook refuses an unset
        // stream, so the expected solve reads a copy with streams of its own (SolverService.ExpectedView).
        delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
        int handle = create(11);
        var state = NativeExports.StateOf(handle)!;
        Assert.Contains(Enumerable.Range(0, StateCloning.StreamCount), i => StateCloning.Stream(state, i) is null);
        int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(new SolveOptions { NodeBudget = 20_000, MaxHorizon = 0 }));
        var (status, _, _) = NativeSolverCalls.Expected(solver, handle, null);
        Assert.True(status is 0 or 1, $"status {status}: {NativeSolverCalls.LastError()}");
        var view = SolverService.ExpectedView(state);
        Assert.NotSame(state, view);
        Assert.Contains(Enumerable.Range(0, StateCloning.StreamCount), i => StateCloning.Stream(state, i) is null);  // the handle is untouched
        Assert.All(Enumerable.Range(0, StateCloning.StreamCount), i => Assert.NotNull(StateCloning.Stream(view, i)));
        NativeSolverCalls.Destroy(solver);
        NativeSolverCalls.DestroyCombat(handle);
    }

    [Fact]
    public void TheFilledStreamsOfTheExpectedViewChangeNoValue()
    {
        // The slots Sts2_Create leaves unset, nulled on harness positions (whose streams are all set): the expected solve
        // reads the view (each unset slot a stream of its own) and must give the value and the root moves of the original.
        delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
        int probe = create(11);
        var unset = SolverService.UnsetStreamsOf(NativeExports.StateOf(probe)!);
        NativeSolverCalls.DestroyCombat(probe);
        Assert.NotEmpty(unset);
        var corpus = SolverPositions.Corpus();
        int compared = 0, solved = 0;
        var diffs = new List<string>();
        foreach (var recipe in SolverPositions.Recipes().Where(r => r.Id % 7 == 0))
        {
            if (compared >= 50 || SolverPositions.Build(recipe, corpus) is not { } pos)
            {
                continue;
            }

            var opts = HarnessOptions(recipe.Horizon);
            var nulled = pos.State.Clone();
            foreach (int i in unset)
            {
                StateCloning.SetStream(nulled, i, null);
            }

            Assert.NotSame(nulled, SolverService.ExpectedView(nulled));
            var a = SolverService.SolveExpected(pos.State, pos.Rng, opts);
            var b = SolverService.SolveExpected(nulled, pos.Rng, opts);
            compared++;
            solved += a.Status == SolveStatus.Refused ? 0 : 1;
            if (a.Status != b.Status || a.Value != b.Value || a.Exact != b.Exact || a.ExactMass != b.ExactMass || a.Reason != b.Reason
                || !a.Moves.SequenceEqual(b.Moves))
            {
                diffs.Add($"#{recipe.Id}: {a.Status} {a.Value} vs {b.Status} {b.Value}");
            }
        }

        output.WriteLine($"expected view: {compared} positions with slots {string.Join(",", unset)} nulled, {solved} solved, {diffs.Count} differences");
        Assert.True(compared >= 50 && solved >= 30, $"{compared} compared, {solved} solved");
        Assert.Empty(diffs);
    }

    [Fact]
    public void ARefusedSolveCarriesNoValue()
    {
        // The draw-tie position is refused: NaN natively and null in EmuDump's object, never the 0 of a certain loss.
        var (s, rng) = SolverSiteTests.DrawTie();
        int handle = NativeSolverCalls.Adopt(s, rng);
        int solver = NativeSolverCalls.Create(handle, NativeSolverCalls.Options(new SolveOptions { MaxHorizon = 0 }));
        var (status, r, q) = NativeSolverCalls.Expected(solver, handle, null);
        string reason = NativeSolverCalls.LastError();
        NativeSolverCalls.Destroy(solver);
        NativeSolverCalls.DestroyCombat(handle);
        Assert.Equal(-3, status);
        Assert.StartsWith("draw-tie", reason);
        Assert.True(double.IsNaN(r.Value) && double.IsNaN(r.ExactMass), $"value {r.Value}, mass {r.ExactMass}");
        Assert.All(q, v => Assert.True(float.IsNaN(v)));
        var lib = SolverService.SolveExpected(s, rng, new SolveOptions { MaxHorizon = 0 });
        var json = SolverService.ExpectedJson(s, lib);
        Assert.Equal("refused", (string)json["status"]!);
        Assert.Null(json["value"]);
        Assert.Null(json["value_f"]);
        Assert.Null(json["exact_mass_f"]);
    }

    [Fact]
    public void BadOptionsAndHandlesAreRefusedNotThrown()
    {
        var (s, rng) = SolverPositions.Fight("old", 22, TestDeck.StarterDeckIds.ToArray(), [], []);
        int handle = NativeSolverCalls.Adopt(s, rng);
        var bad = NativeSolverCalls.Options(new SolveOptions()) with { Utility = 2 };
        Assert.Equal(-3, NativeSolverCalls.Create(handle, bad));
        Assert.StartsWith("utility:", NativeSolverCalls.LastError());
        Assert.Equal(-1, NativeSolverCalls.Create(-5, null));
        int solver = NativeSolverCalls.Create(handle, null);
        Assert.True(solver >= 0);
        Assert.Equal(-1, NativeSolverCalls.Expected(solver, 255, null).Status);
        Assert.Equal(-1, NativeSolverCalls.Expected(-1, handle, null).Status);
        // A full pool is -2, as Sts2Run_ForkCombat's; every handle claimed is distinct.
        var claimed = new List<int> { solver };
        int next;
        while ((next = NativeSolverCalls.Create(handle, null)) >= 0)
        {
            claimed.Add(next);
        }

        Assert.Equal(-2, next);
        Assert.Equal(claimed.Count, claimed.Distinct().Count());
        foreach (int c in claimed.Skip(1))
        {
            NativeSolverCalls.Destroy(c);
        }

        NativeSolverCalls.Destroy(solver);
        NativeSolverCalls.DestroyCombat(handle);
    }
}
