// Nodeless host (SPEED_PLAN "Fallback"): the same game process, the same RunManager, CombatManager
// and rules code, but no NRun scene. NGame stays (its tree drives frames, timers and the action
// executor); the main menu is swapped for an empty Control; a run starts the way NGame.StartRun
// starts it, minus NRun.Create and the asset loads. Every node singleton that hangs off NRun
// (NCombatRoom, NOverlayStack, NMapScreen, NPlayerHand, ...) is then null: the ~300 null-safe (?.)
// node calls in the rules do nothing, and the few UI helpers that would dereference null are
// skipped by NodelessPatches.cs.
//
// Job flag "nodeless": true selects it; without it nothing here is installed. Per fight:
//   - a spec whose deck_add holds a fallback model goes straight to the full host;
//   - otherwise it plays nodeless; the first fallback model created (AbstractModel.MutableClone,
//     any card, monster, power, relic...) aborts the fight at the driver's next frame check, and
//     the fight is played again from the start on the full host (NGame.StartRun, NRun);
//   - any game error (Log.Error) or exception thrown (first chance, cancellations aside) while a
//     nodeless fight runs fails that fight ("error"), unless "nodeless_on_error": "full" asks for
//     the full host instead.
// Diagnostics go under "cost" (never compared): host, fallback reason, errors, health counters.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

/// <summary>Thrown by the driver (never inside game code) to abandon a nodeless attempt.</summary>
public sealed class NodelessFallbackException : Exception
{
    public NodelessFallbackException(string reason) : base("nodeless fallback: " + reason) { }
}

public static class Nodeless
{
    /// <summary>The job asked for the nodeless host.</summary>
    public static bool Enabled { get; private set; }

    /// <summary>The fight being played right now runs nodeless (every patch checks this).</summary>
    public static bool Active { get; private set; }

    /// <summary>Where a nodeless fight is, for the watchdog's message ("" when the job is not nodeless).</summary>
    internal static string Where => Enabled ? $" (nodeless: {_phase}, step {Driver.CurrentStep}, active {Active})" : "";

    private static string _phase = "idle";

    /// <summary>First fallback-listed model seen in this attempt, or null.</summary>
    internal static string? FallbackHit;

    private static bool _errorsToFull;
    private static readonly object _lock = new();
    private static readonly List<string> _errors = new();
    private static int _cancellations;
    private static Control? _emptyScene;

    /// <summary>Per-process tallies, reported with every fight.</summary>
    private static int _nodelessFights, _fullFights, _fallbacks, _failed;

    public static void Configure(JsonObject job)
    {
        if ((bool?)job["nodeless"] != true || Enabled)
        {
            return;
        }

        Enabled = true;
        _errorsToFull = (string?)job["nodeless_on_error"] == "full";
        NodelessPatches.Apply();
        Log.LogCallback += OnLog;
        AppDomain.CurrentDomain.FirstChanceException += OnFirstChance;
        GD.Print($"[referee] nodeless host on ({NodelessPatches.Count} patches, {NodelessPatches.FallbackCount} fallback models)");
    }

    /// <summary>
    /// Before the first fight: let the main menu's own 2 s fade-in (NMainMenu, started before the driver
    /// sets Instant mode) run out. Its last step writes a shader parameter on the transition material,
    /// which the first run's RoomFadeOut has replaced and freed by then, so it throws
    /// ObjectDisposedException a moment into the first fight, on both hosts. Waiting keeps that
    /// menu-only error out of the nodeless error check.
    /// </summary>
    internal static async Task Prepare()
    {
        if (!Enabled)
        {
            return;
        }

        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        int fps = Engine.MaxFps;
        Engine.MaxFps = 30;
        while (clock.Elapsed.TotalSeconds < 2.5)
        {
            Driver.Heartbeat();
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        Engine.MaxFps = fps;
    }

    private static void OnLog(LogLevel level, string text, int _)
    {
        if (Active && level >= LogLevel.Error)
        {
            Record("log: " + text);
        }
    }

    private static void OnFirstChance(object? sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
    {
        if (!Active || e.Exception is NodelessFallbackException)
        {
            return;
        }

        if (e.Exception is OperationCanceledException)
        {
            lock (_lock) _cancellations++;
            return;
        }

        if (NodelessPatches.IsExpectedGameOverException(e.Exception))
        {
            return;  // the one null dereference the game-over guard turns back into the full host's flow
        }

        Record("exception: " + e.Exception.GetType().Name + ": " + e.Exception.Message + " at "
            + new System.Diagnostics.StackTrace(1, false).ToString().Split('\n').Take(12).Aggregate("", (a, l) => a + l.Trim() + " | "));
    }

    private static void Record(string what)
    {
        lock (_lock)
        {
            if (FallbackHit != null)
            {
                return;  // the attempt is being thrown away; what a fallback model does nodeless is expected
            }

            if (_errors.Count < 20)
            {
                _errors.Add(what.Length > 4000 ? what[..4000] : what);
            }
            else if (_errors.Count == 20)
            {
                _errors.Add("...");
            }
        }

        GD.Print("[referee] nodeless error: " + (what.Length > 600 ? what[..600] : what));
    }

    /// <summary>Called by the MutableClone postfix: a model whose own code needs nodes entered this fight.</summary>
    internal static void MarkFallback(string reason)
    {
        if (Active && FallbackHit == null)
        {
            FallbackHit = reason;
            GD.Print("[referee] nodeless: fallback model " + reason + "; replaying on the full host");
        }
    }

    /// <summary>The driver's frame loops call this: a fallback model seen means stop this attempt now.</summary>
    internal static void ThrowIfFallback()
    {
        if (Active && FallbackHit != null)
        {
            throw new NodelessFallbackException(FallbackHit);
        }
    }

    /// <summary>
    /// NGame.StartRun without the scene: same calls in the same order, minus the asset loads
    /// (PreloadManager) and RootSceneContainer.SetCurrentScene(NRun.Create(state)).
    /// </summary>
    internal static async Task StartRun(RunState state)
    {
        UseEmptyScene();
        using (new NetLoadingHandle(RunManager.Instance.NetService))
        {
            await RunManager.Instance.FinalizeStartingRelics();
            RunManager.Instance.Launch();
            await RunManager.Instance.EnterAct(0, doTransition: false);
        }
    }

    /// <summary>The root scene is an empty Control: frees the main menu, or the last full-host NRun.</summary>
    private static void UseEmptyScene()
    {
        var root = NGame.Instance!.RootSceneContainer;
        if (_emptyScene != null && GodotObject.IsInstanceValid(_emptyScene) && root.CurrentScene == _emptyScene)
        {
            return;
        }

        _emptyScene = new Control { Name = "SpirelineNodeless" };
        root.SetCurrentScene(_emptyScene);
    }

    /// <summary>
    /// One combat spec: nodeless if it can be, else (or after a fallback model showed up) the full host.
    /// <paramref name="play" /> is Driver.PlayCombat; <paramref name="reset" /> ends a run as PlaySpec does.
    /// The patches stay on (Active) from the start of a nodeless fight through its reset and until a
    /// full-host fight starts, so game coroutines still running between fights see the same host.
    /// Errors are collected over the same span: anything logged between fights counts for the next one.
    /// </summary>
    internal static async Task<JsonObject> Play(JsonObject spec, Func<JsonObject, Task<JsonObject>> play, Action reset)
    {
        string? reason = NodelessPatches.FallbackInSpec(spec);
        JsonObject? attempt = null;
        if (reason == null)
        {
            FallbackHit = null;
            Active = true;
            int gameOvers = NodelessPatches.GameOvers;
            _phase = "fight";
            attempt = await play(spec);
            _phase = "reset";
            var clock = System.Diagnostics.Stopwatch.StartNew();
            reset();  // RunManager.CleanUp with the patches on; PlaySpec's own reset then finds no run
            long resetMs = clock.ElapsedMilliseconds;
            _phase = "after reset";
            gameOvers = NodelessPatches.GameOvers - gameOvers;
            string[] errors;
            int cancellations;
            lock (_lock)
            {
                errors = _errors.ToArray();
                cancellations = _cancellations;
                _errors.Clear();
                _cancellations = 0;
            }

            if (FallbackHit == null)
            {
                if (errors.Length == 0 || !_errorsToFull)
                {
                    _nodelessFights++;
                    var cost = Cost(attempt);
                    cost["host"] = "nodeless";
                    cost["reset_ms"] = resetMs;
                    cost["cancellations"] = cancellations;
                    cost["game_overs"] = gameOvers;
                    if (errors.Length > 0)
                    {
                        _failed++;
                        cost["nodeless_errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());
                        string first = "nodeless: " + errors.Length + " game error(s); first: " + errors[0];
                        attempt["error"] = attempt["error"] is { } prior ? first + " || " + prior : first;
                    }

                    _phase = "returned";
                    return attempt;
                }

                reason = "errors: " + errors[0];
            }
            else
            {
                reason = FallbackHit;
                _fallbacks++;
            }

            if (errors.Length > 0 && attempt["cost"] is JsonObject ac)
            {
                ac["nodeless_errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());  // reported below
            }
        }

        Active = false;
        FallbackHit = null;
        _fullFights++;
        _phase = "full-host fight";
        var result = await play(spec);
        _phase = "full-host done";
        var c = Cost(result);
        c["host"] = "full";
        c["fallback"] = reason;
        if (attempt != null && attempt["cost"] is JsonObject wasted)
        {
            c["nodeless_attempt"] = wasted.DeepClone();
        }

        return result;
    }

    private static JsonObject Cost(JsonObject result)
    {
        if (result["cost"] is not JsonObject cost)
        {
            cost = new JsonObject();
            result["cost"] = cost;
        }

        return cost;
    }

    /// <summary>Leak and growth counters, taken after the run is reset (stability proof).</summary>
    internal static JsonObject Health()
    {
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        return new JsonObject
        {
            ["orphan_nodes"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectOrphanNodeCount),
            ["nodes"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectNodeCount),
            ["objects"] = (long)Performance.GetMonitor(Performance.Monitor.ObjectCount),
            ["godot_static_mb"] = Math.Round(Performance.GetMonitor(Performance.Monitor.MemoryStatic) / 1048576.0, 1),
            ["managed_mb"] = Math.Round(GC.GetTotalMemory(false) / 1048576.0, 1),
            ["private_mb"] = Math.Round(proc.PrivateMemorySize64 / 1048576.0, 1),
            ["working_set_mb"] = Math.Round(proc.WorkingSet64 / 1048576.0, 1),
            ["fights_nodeless"] = _nodelessFights, ["fights_full"] = _fullFights,
            ["fallbacks"] = _fallbacks, ["failed"] = _failed,
        };
    }
}
