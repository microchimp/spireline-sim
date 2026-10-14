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
//     any card, monster, power, relic...) or seen in a recorded state aborts the fight at the
//     driver's next frame check, and the fight is played again from the start on the full host
//     (NGame.StartRun, NRun);
//   - any game error (Log.Error) or exception thrown (first chance) while a nodeless fight runs
//     fails that fight ("error"), unless "nodeless_on_error": "full" asks for the full host instead;
//   - a cancellation exception thrown during a nodeless fight fails it too, unless its type and
//     site are on the proven-harmless list (HarmlessCancels) or the same cancellation (type, site,
//     token owner) happens when the spec is replayed on the full host right away (cost.cancel_check);
//   - "nodeless_audit": f also plays a stable fraction f of nodeless fights on the full host and
//     records in cost.audit whether both results agree;
//   - "nodeless_full": "pool" (set by serve.Pool) leaves the fallback and audit replays to the
//     dispatcher, which plays them ("host": "full") on its full-host lane and merges the lines;
//   - a fight broken off mid-turn can leave the turn flags set (Driver.StaleState): nothing more is
//     played in this process, and the server retires after the result (Server.cs).
// A full-host fight (fallback, audit, cancel check, or an "error_log" job) with a game error during
// the fight gets "error" too ("full host: ..."), so a UI abort is never output as a clean result.
// Diagnostics go under "cost" (never compared): host, fallback reason, errors, cancellations, audit.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
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
    internal static string Where => Enabled ? $" (nodeless: {_fight} {_phase}, step {Driver.CurrentStep}, active {Active})" : "";

    /// <summary>"fight" (BeginFight to the end of play), "reset", "teardown", then "between" until the next fight.</summary>
    private static string _phase = "idle";

    /// <summary>First fallback-listed model seen in this attempt, or null.</summary>
    internal static string? FallbackHit;

    private static bool _errorsToFull;

    /// <summary>
    /// Job "nodeless_full": "pool" (serve.Pool sets it when it routes full-host work): a fallback or an
    /// audit is not replayed in this process; the result line carries "_full" ({"why": "fallback" |
    /// "audit", "reason"}) and the dispatcher plays the spec again with "host": "full" on a server of its
    /// full-host lane, then merges the two lines into what Play would have returned here. Cancel checks
    /// stay in this process (they compare cancellation sites, which only this process records).
    /// </summary>
    private static bool _fullInPool;

    private static readonly object _lock = new();
    private static Control? _emptyScene;

    /// <summary>Per-process tallies, reported with every fight.</summary>
    private static int _nodelessFights, _fullFights, _fallbacks, _failed, _cancelChecks, _cancelFails;

    /// <summary>
    /// Job "error_log": true (implied by "nodeless"): game errors, first-chance exceptions and
    /// cancellations of each full-host fight go to its cost, so a comparison can tell a full-host UI
    /// failure (such as "Hand size 11") from a nodeless divergence. Off, the full host runs as before.
    /// </summary>
    public static bool ErrorLog { get; private set; }

    /// <summary>
    /// Recording span: null before the first fight, else the host of the fight that began last. Errors
    /// and cancellations are collected from BeginFight on; those after a fight's harvest (between two
    /// fights) are reported with the next fight.
    /// </summary>
    private static string? _span;

    private static readonly List<string> _errors = new();

    /// <summary>Nodeless frame pacing (job "frame_sleep_usec"); full-host fights keep the default.</summary>
    private static int? _sleepNodeless;

    private static int _sleepDefault;

    /// <summary>Frames waited after a fight's reset, so its own teardown is recorded with it.</summary>
    private const int TeardownFrames = 2;

    public static void Configure(JsonObject job)
    {
        bool nodeless = (bool?)job["nodeless"] == true;

        // Job "frame_sleep_usec": N sets OS.LowProcessorUsageModeSleepUsec. Headless, Godot cannot
        // draw, so OS::add_frame_delay paces every frame to at least that long (6900 us by default).
        // A full-host frame mostly costs more CPU than that; an empty nodeless frame costs ~1 ms and
        // then sleeps the rest. Frame pacing only: no rule reads wall time. Unset, nothing changes.
        // In a nodeless job it paces nodeless fights only: the full host's settle check depends on
        // the default pacing (0 there changes 67 of 300 full-host results), so fallback replays,
        // audits and cancel checks in the same process run at the default.
        if (job["frame_sleep_usec"] is { } sleep)
        {
            if (nodeless)
            {
                _sleepDefault = OS.LowProcessorUsageModeSleepUsec;
                _sleepNodeless = (int)sleep;
                GD.Print($"[referee] frame sleep {_sleepNodeless} us in nodeless fights, {_sleepDefault} us on the full host");
            }
            else
            {
                OS.LowProcessorUsageModeSleepUsec = (int)sleep;
                GD.Print($"[referee] frame sleep {OS.LowProcessorUsageModeSleepUsec} us");
            }
        }

        if ((bool?)job["error_log"] == true && !ErrorLog && !nodeless)
        {
            ErrorLog = true;
            Log.LogCallback += OnLog;
            AppDomain.CurrentDomain.FirstChanceException += OnFirstChance;
            GD.Print("[referee] full-host error log on");
            return;
        }

        if (!nodeless || Enabled)
        {
            return;
        }

        Enabled = true;
        ErrorLog = true;
        _errorsToFull = (string?)job["nodeless_on_error"] == "full";
        _fullInPool = (string?)job["nodeless_full"] == "pool";
        _audit = (double?)job["nodeless_audit"] ?? 0;
        NodelessPatches.Apply();
        Log.LogCallback += OnLog;
        AppDomain.CurrentDomain.FirstChanceException += OnFirstChance;
        GD.Print($"[referee] nodeless host on ({NodelessPatches.Count} patches, {NodelessPatches.FactoryCount} test-mode-null factories, {NodelessPatches.FallbackCount} fallback models, audit {_audit})");
    }

    /// <summary>
    /// Before the first fight: let the main menu's own 2 s fade-in (NMainMenu, started before the driver
    /// sets Instant mode) run out. Its last step writes a shader parameter on the transition material,
    /// which the first run's RoomFadeOut has replaced and freed by then, so it throws
    /// ObjectDisposedException a moment into the first fight, on both hosts. Waiting keeps that
    /// menu-only error out of the error check (nodeless, and error_log jobs on the full host, where
    /// it would mark the second fight "full host: ..."). The main menu stays on the full host.
    /// </summary>
    internal static async Task Prepare()
    {
        if (!Enabled && !ErrorLog)
        {
            return;
        }

        var tree = (SceneTree)Engine.GetMainLoop();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        int fps = Engine.MaxFps;
        // The fade counts process deltas (NTransition.FadeIn adds GetProcessDeltaTime() each frame until it
        // reaches its 2 s), which Engine.TimeScale scales: at 100x it runs out within a few frames instead of
        // 2.5 s of wall time per boot. Only the menu's own visuals run meanwhile, no run exists yet, and the
        // scale is 1 again before Prepare returns.
        double scale = Engine.TimeScale;
        Engine.TimeScale = 100;
        Engine.MaxFps = 0;
        double menuTime = 0;
        int menuFrames = 0;
        try
        {
            while ((menuTime < 2.5 || menuFrames < 10) && clock.Elapsed.TotalSeconds < 2.5)
            {
                Driver.Heartbeat();
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
                menuTime += tree.Root.GetProcessDeltaTime();
                menuFrames++;
            }
        }
        finally
        {
            Engine.TimeScale = scale;
        }

        GD.Print($"[referee] menu fade-in run out: {menuTime:F1} s of game time in {menuFrames} frames, {clock.ElapsedMilliseconds} ms");

        if (!Enabled)
        {
            Engine.MaxFps = fps;
            return;
        }

        // The main menu goes now, and what an empty nodeless frame costs is logged once (process CPU
        // over 200 uncapped frames), to tell engine overhead from rules cost in the fight numbers.
        UseEmptyScene();
        Engine.MaxFps = 0;
        await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        proc.Refresh();
        double cpu0 = proc.TotalProcessorTime.TotalMilliseconds;
        var wall = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 200; i++)
        {
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        proc.Refresh();
        GD.Print($"[referee] nodeless: empty frame {(proc.TotalProcessorTime.TotalMilliseconds - cpu0) / 200:F2} ms CPU, {wall.Elapsed.TotalMilliseconds / 200:F2} ms wall; nodes {Performance.GetMonitor(Performance.Monitor.ObjectNodeCount)}");
        Engine.MaxFps = fps;
    }

    private static void OnLog(LogLevel level, string text, int _)
    {
        if (_span != null && level >= LogLevel.Error)
        {
            Record("log: " + text);
        }
    }

    private static void OnFirstChance(object? sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
    {
        if (_span == null || e.Exception is NodelessFallbackException)
        {
            return;
        }

        if (e.Exception is OperationCanceledException oce)
        {
            RecordCancel(oce);
            return;
        }

        if (Active && NodelessPatches.IsExpectedGameOverException(e.Exception))
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
            if (Active && FallbackHit != null)
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

        GD.Print((_span == "full" ? "[referee] full-host error: " : "[referee] nodeless error: ") + (what.Length > 600 ? what[..600] : what));
    }

    // --- Cancellations ---------------------------------------------------------------------
    // A cancellation exception is not a game error (the game throws them on purpose), but a rules
    // chain cut short by one would go unnoticed: the game logs neither a cancelled action nor a
    // cancelled fire-and-forget task. So each one is recorded once (an async exception is raised
    // again by every frame that awaits it; those repeats are the same object) with where it was
    // first thrown, what cancelled it and when, and Play decides whether the fight may stand.

    /// <summary>One cancellation: exception type, throw site (game frames), token owner, phase, step, fight.</summary>
    internal sealed record Cancel(string Type, string Site, string Token, string Phase, int Step, string Fight)
    {
        /// <summary>What has to match between hosts: the type, the site and what cancelled it.</summary>
        public string Key => Type + " at " + Site + " [" + Token + "]";

        /// <summary>The game methods the exception went up through, one per await that rethrew it (diagnostic).</summary>
        public List<string> Awaiters { get; } = new();

        public override string ToString() => Key + (Awaiters.Count > 0 ? " awaited by " + string.Join(" < ", Awaiters) : "")
            + " (" + Phase + ", step " + Step + (Fight.Length > 0 ? ", " + Fight : "") + ")";
    }

    /// <summary>
    /// Cancellations proven harmless from the decompile, by type and throw site (Cancel.Key). Empty:
    /// every cancellation in a nodeless fight is checked against the full host.
    /// </summary>
    internal static readonly HashSet<string> HarmlessCancels = new();

    private static readonly ConditionalWeakTable<Exception, Cancel> _seenCancels = new();
    private static readonly List<Cancel> _cancels = new();

    /// <summary>
    /// CombatManager._combatCts (cancelled by Reset and by the next AfterCombatRoomLoaded) and
    /// ActionExecutor._actionCancelToken: the two rules-side token sources a fight can see.
    /// </summary>
    private static readonly FieldInfo? _combatCts = typeof(CombatManager).GetField("_combatCts", BindingFlags.NonPublic | BindingFlags.Instance);
    private static readonly FieldInfo? _actionCts = typeof(MegaCrit.Sts2.Core.GameActions.ActionExecutor).GetField("_actionCancelToken", BindingFlags.NonPublic | BindingFlags.Instance);

    /// <summary>Each finished fight's combat token source, with the fight's label, to tell an old combat's token.</summary>
    private static readonly List<(WeakReference<CancellationTokenSource> Cts, string Fight)> _combats = new();

    /// <summary>The fight being played (or last played): "nodeless 12", "full 13".</summary>
    private static string _fight = "none";

    private static int _fightNo;

    private static void BeginFight(string host)
    {
        _fight = host + " " + ++_fightNo;
        _span = host;
        _phase = "fight";
        Active = host == "nodeless";
        if (_sleepNodeless is int s)
        {
            OS.LowProcessorUsageModeSleepUsec = Active ? s : _sleepDefault;
        }
    }

    /// <summary>
    /// After play: the run's own reset, then (full host) the NRun scene goes, then a few frames, all
    /// still in this fight's span, so what its teardown throws is recorded with it and not with the
    /// next fight. The scene would go anyway when the next run starts (NGame.StartRun replaces it).
    /// </summary>
    private static async Task EndFight(Action reset)
    {
        if (_combatCts?.GetValue(CombatManager.Instance) is CancellationTokenSource cts)
        {
            lock (_lock)
            {
                _combats.RemoveAll(c => !c.Cts.TryGetTarget(out var t) || t == cts);
                _combats.Add((new WeakReference<CancellationTokenSource>(cts), _fight));
            }
        }

        _phase = "reset";
        reset();
        _phase = "teardown";
        UseEmptyScene();
        var tree = (SceneTree)Engine.GetMainLoop();
        for (int i = 0; i < TeardownFrames; i++)
        {
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }
    }

    /// <summary>Takes what was recorded since the last harvest.</summary>
    private static (string[] Errors, Cancel[] Cancels) Harvest()
    {
        lock (_lock)
        {
            var r = (_errors.ToArray(), _cancels.ToArray());
            _errors.Clear();
            _cancels.Clear();
            return r;
        }
    }

    /// <summary>Who owns the token: a finished fight's combat, this fight's combat, the action executor, other, none.</summary>
    private static string TokenOwner(CancellationToken token, out string fight)
    {
        fight = "";
        if (!token.CanBeCanceled)
        {
            return "no token";
        }

        lock (_lock)
        {
            foreach (var (weak, label) in _combats)
            {
                if (weak.TryGetTarget(out var cts) && cts.Token == token)
                {
                    fight = "token of " + label;
                    return "an earlier fight's combat";
                }
            }
        }

        if (_combatCts?.GetValue(CombatManager.Instance) is CancellationTokenSource now && now.Token == token)
        {
            return "this fight's combat";
        }

        if (_actionCts?.GetValue(RunManager.Instance.ActionExecutor) is CancellationTokenSource act && act.Token == token)
        {
            return "action executor";
        }

        return "other token";
    }

    /// <summary>
    /// The throw site: the first game (or Godot) frames of the stack, runtime and handler frames left out,
    /// async state machines named after their method ("CombatManager.StartTurn").
    /// </summary>
    internal static string Site(System.Diagnostics.StackTrace trace, int frames = 4)
    {
        var names = new List<string>();
        foreach (var f in trace.GetFrames())
        {
            var m = f.GetMethod();
            var t = m?.DeclaringType;
            string ns = t?.Namespace ?? t?.DeclaringType?.Namespace ?? "";
            if (m == null || t == null || ns.StartsWith("System", StringComparison.Ordinal) || t == typeof(Nodeless))
            {
                continue;
            }

            names.Add(MethodName(m));
            if (names.Count == frames)
            {
                break;
            }
        }

        return names.Count == 0 ? "runtime" : string.Join(" < ", names);
    }

    private static string MethodName(MethodBase m)
    {
        var t = m.DeclaringType!;
        var outer = t;
        while (outer.Name.StartsWith('<') && outer.DeclaringType != null)
        {
            outer = outer.DeclaringType;  // compiler-generated nested types: state machines, closures
        }

        string name = t.Name.StartsWith('<') && t.Name.IndexOf('>') > 1 && m.Name == "MoveNext"
            ? t.Name[1..t.Name.IndexOf('>')]  // <StartTurn>d__78.MoveNext
            : m.Name.StartsWith('<') && m.Name.IndexOf('>') > 1
                ? m.Name[1..m.Name.IndexOf('>')] + " (lambda)"
                : m.Name;
        return outer.Name + "." + name;
    }

    private static void RecordCancel(OperationCanceledException oce)
    {
        if (_seenCancels.TryGetValue(oce, out var seen))
        {
            // raised again by an await: the method it now leaves is one the cancellation cuts short
            string by = Site(new System.Diagnostics.StackTrace(2, false), 1);
            lock (_lock)
            {
                if (seen.Awaiters.Count < 6 && (seen.Awaiters.Count == 0 || seen.Awaiters[^1] != by))
                {
                    seen.Awaiters.Add(by);
                }
            }

            return;
        }

        string owner = TokenOwner(oce.CancellationToken, out string fight);
        var c = new Cancel(oce.GetType().Name, Site(new System.Diagnostics.StackTrace(2, false)), owner, _phase, Driver.CurrentStep, fight);
        lock (_lock)
        {
            _seenCancels.AddOrUpdate(oce, c);
            if (_cancels.Count < 500)
            {
                _cancels.Add(c);
            }
        }
    }

    /// <summary>Distinct cancellations with their counts, for cost.cancel_sites.</summary>
    private static JsonArray CancelSites(IEnumerable<Cancel> cancels) =>
        new(cancels.GroupBy(c => c.ToString()).Select(g => (JsonNode)(g.Count() + " x " + g.Key)).ToArray());

    /// <summary>Cancellations that count against a nodeless fight: thrown while it played, not proven harmless.</summary>
    private static List<Cancel> Unexplained(IEnumerable<Cancel> cancels) =>
        cancels.Where(c => c.Phase == "fight" && !HarmlessCancels.Contains(c.Key)).ToList();

    // --- Full host ---------------------------------------------------------------------------

    /// <summary>
    /// A full-host fight with its game errors in cost.errors and its cancellations in cost.cancel_sites
    /// (error_log jobs, fallback replays, audits, cancel checks). A game error while the fight played
    /// sets "error" ("full host: ..."); errors of its teardown go to cost.teardown_errors only.
    /// </summary>
    internal static async Task<JsonObject> PlayFull(JsonObject spec, Func<JsonObject, Task<JsonObject>> play, Action reset) =>
        (await PlayFullCore(spec, play, reset)).Result;

    private static async Task<(JsonObject Result, Cancel[] Cancels)> PlayFullCore(JsonObject spec, Func<JsonObject, Task<JsonObject>> play, Action reset)
    {
        var (between, betweenCancels) = Harvest();  // what the last fight's coroutines threw since its harvest
        BeginFight("full");
        FallbackHit = null;
        _fullFights++;
        var result = await play(spec);
        var (errors, cancels) = Harvest();
        await EndFight(reset);
        var (teardown, teardownCancels) = Harvest();
        _phase = "between";
        var cost = Cost(result);
        cost["host"] = "full";
        cost["frame_sleep_usec"] = OS.LowProcessorUsageModeSleepUsec;
        cost["errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());
        if (teardown.Length > 0 || between.Length > 0)
        {
            cost["teardown_errors"] = new JsonArray(between.Concat(teardown).Select(e => (JsonNode)e).ToArray());
        }

        var all = betweenCancels.Concat(cancels).Concat(teardownCancels).ToArray();
        cost["cancellations"] = all.Length;
        if (all.Length > 0)
        {
            cost["cancel_sites"] = CancelSites(all);
        }

        if (errors.Length > 0 && result["error"] == null)
        {
            result["error"] = "full host: " + errors.Length + " game error(s); first: " + errors[0];
        }

        return (result, cancels);
    }

    /// <summary>Called when a model whose own code needs nodes entered this fight.</summary>
    internal static void MarkFallback(string reason)
    {
        if (Active && FallbackHit == null)
        {
            FallbackHit = reason;
            GD.Print("[referee] nodeless: fallback model " + reason + (_fullInPool ? "; the dispatcher replays it on the full host" : "; replaying on the full host"));
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
    /// The patches stay on (Active) from the start of a nodeless fight through its reset and until the
    /// next fight starts, so game coroutines still running between fights see the same host.
    /// Errors are collected over the same span: anything logged between fights counts for the next one.
    /// </summary>
    internal static async Task<JsonObject> Play(JsonObject spec, Func<JsonObject, Task<JsonObject>> play, Action reset)
    {
        if ((string?)spec["host"] == "full")
        {
            return await PlayFull(spec, play, reset);  // the dispatcher's full-host replay of a fallback or an audit
        }

        string? reason = NodelessPatches.FallbackInSpec(spec);
        JsonObject? attempt = null;
        if (reason != null && _fullInPool)
        {
            _fallbacks++;
            return ToPool(new JsonObject { ["seed"] = (string?)spec["seed"] }, "fallback", reason);
        }

        if (reason == null)
        {
            var (between, betweenCancels) = Harvest();  // what the last fight's coroutines threw since its harvest
            BeginFight("nodeless");
            FallbackHit = null;
            int gameOvers = NodelessPatches.GameOvers;
            NodelessPatches.OrphansByStep.Clear();
            long orphans0 = (long)Performance.GetMonitor(Performance.Monitor.ObjectOrphanNodeCount);
            attempt = await play(spec);
            var (errors, cancels) = Harvest();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            await EndFight(reset);  // RunManager.CleanUp with the patches on; PlaySpec's own reset then finds no run
            long resetMs = clock.ElapsedMilliseconds;
            var (teardown, teardownCancels) = Harvest();
            _phase = "between";
            errors = between.Concat(errors).Concat(teardown).ToArray();  // nodeless, an error anywhere counts
            long orphans1 = (long)Performance.GetMonitor(Performance.Monitor.ObjectOrphanNodeCount);
            gameOvers = NodelessPatches.GameOvers - gameOvers;
            var allCancels = betweenCancels.Concat(cancels).Concat(teardownCancels).ToArray();
            // An attempt broken off mid-turn (a fallback model seen during an end of turn) can leave the turn
            // flags set (Driver.StaleState): no full-host fight can settle in this process any more, so
            // nothing is replayed here; the server retires after this result (Server.cs).
            string? stale = Driver.StaleState();
            if (stale != null)
            {
                Cost(attempt)["stale"] = stale;
            }

            if (FallbackHit == null)
            {
                if (errors.Length == 0 || !_errorsToFull)
                {
                    _nodelessFights++;
                    var cost = Cost(attempt);
                    cost["host"] = "nodeless";
                    cost["frame_sleep_usec"] = OS.LowProcessorUsageModeSleepUsec;
                    cost["reset_ms"] = resetMs;  // the reset and the teardown frames
                    cost["cancellations"] = allCancels.Length;
                    if (allCancels.Length > 0)
                    {
                        cost["cancel_sites"] = CancelSites(allCancels);
                    }

                    cost["game_overs"] = gameOvers;
                    if (orphans1 != orphans0)
                    {
                        // where in the fight orphan nodes appeared: count before, after each step, after the reset
                        cost["orphans_by_step"] = new JsonArray(new[] { orphans0 }.Concat(NodelessPatches.OrphansByStep).Append(orphans1)
                            .Select(n => (JsonNode)n).ToArray());
                    }

                    if (errors.Length > 0)
                    {
                        _failed++;
                        cost["nodeless_errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());
                        string first = "nodeless: " + errors.Length + " game error(s); first: " + errors[0];
                        attempt["error"] = attempt["error"] is { } prior ? first + " || " + prior : first;
                    }
                    else if (Unexplained(cancels) is { Count: > 0 } unexplained)
                    {
                        if (stale != null)
                        {
                            _failed++;
                            string first = "nodeless: cancellation not checked on the full host (game state stale: " + stale + "): " + unexplained[0].Key;
                            attempt["error"] = attempt["error"] is { } prior ? first + " || " + prior : first;
                            return attempt;
                        }

                        var full = await CancelCheck(spec, attempt, unexplained, play, reset);
                        if (full != null)
                        {
                            return full;  // nodeless_on_error "full": the full host's result stands instead
                        }
                    }

                    if (IsAudited(spec))
                    {
                        if (_fullInPool)
                        {
                            return ToPool(attempt, "audit", null);
                        }

                        if (stale == null)
                        {
                            await Audit(spec, attempt, play, reset);
                        }
                    }

                    return attempt;
                }

                reason = "errors: " + errors[0];
            }
            else
            {
                reason = FallbackHit;
                _fallbacks++;
            }

            if (attempt["cost"] is JsonObject ac)
            {
                if (errors.Length > 0)
                {
                    ac["nodeless_errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());  // reported below
                }

                if (allCancels.Length > 0)
                {
                    ac["cancel_sites"] = CancelSites(allCancels);
                }
            }

            if (_fullInPool)
            {
                return ToPool(attempt, "fallback", reason);
            }

            if (stale != null)
            {
                _failed++;
                attempt["error"] = $"nodeless: fallback ({reason}) not replayed: game state stale after the attempt ({stale})";
                return attempt;
            }
        }

        var result = await PlayFull(spec, play, reset);
        var c = Cost(result);
        c["fallback"] = reason;
        if (attempt != null && attempt["cost"] is JsonObject wasted)
        {
            c["nodeless_attempt"] = wasted.DeepClone();
        }

        return result;
    }

    /// <summary>
    /// Job "nodeless_full": "pool": the full-host replay is the dispatcher's (serve.Pool). "fallback": the
    /// line is the wasted attempt (its cost becomes the replay's cost.nodeless_attempt); "audit": the line is
    /// the nodeless result, which stands, and the replay fills its cost.audit.
    /// </summary>
    private static JsonObject ToPool(JsonObject line, string why, string? reason)
    {
        var full = new JsonObject { ["why"] = why };
        if (reason != null)
        {
            full["reason"] = reason;
        }

        line["_full"] = full;
        return line;
    }

    /// <summary>
    /// A nodeless fight threw cancellations that are not proven harmless: the spec is played again on
    /// the full host at once. The nodeless result stands only if every one of them (type, site, token
    /// owner) also happened there; else the fight fails ("nodeless: cancellation ..."), or with
    /// "nodeless_on_error": "full" the full host's result is returned instead (the return value).
    /// </summary>
    private static async Task<JsonObject?> CancelCheck(JsonObject spec, JsonObject nodeless, List<Cancel> unexplained,
        Func<JsonObject, Task<JsonObject>> play, Action reset)
    {
        _cancelChecks++;
        var (full, fullCancels) = await PlayFullCore(spec, play, reset);
        var seen = fullCancels.Select(c => c.Key).ToHashSet();
        var missing = unexplained.Select(c => c.Key).Distinct().Where(k => !seen.Contains(k)).ToList();
        string? diff = FirstDiff(nodeless, full);
        Cost(nodeless)["cancel_check"] = new JsonObject
        {
            ["same_on_full"] = missing.Count == 0,
            ["missing_on_full"] = new JsonArray(missing.Select(k => (JsonNode)k).ToArray()),
            ["full_cancel_sites"] = (full["cost"] as JsonObject)?["cancel_sites"]?.DeepClone(),
            ["result_same"] = diff == null, ["first_diff"] = diff, ["full_error"] = full["error"]?.DeepClone(),
        };
        if (missing.Count == 0)
        {
            return null;
        }

        _cancelFails++;
        GD.Print($"[referee] nodeless: {(string?)spec["seed"]} cancellation not seen on the full host: {missing[0]}");
        if (_errorsToFull)
        {
            var c = Cost(full);
            c["fallback"] = "cancellation: " + missing[0];
            c["nodeless_attempt"] = Cost(nodeless).DeepClone();
            return full;
        }

        _failed++;
        string first = "nodeless: cancellation not seen on the full host: " + missing[0];
        nodeless["error"] = nodeless["error"] is { } prior ? first + " || " + prior : first;
        return null;
    }

    /// <summary>Job "nodeless_audit": fraction (0..1) of nodeless fights also played on the full host.</summary>
    private static double _audit;

    private static int _audits, _auditDiffs;

    /// <summary>A stable pick per spec (seed, chooser seed, deck), so a rerun audits the same fights.</summary>
    private static bool IsAudited(JsonObject spec)
    {
        if (_audit <= 0)
        {
            return false;
        }

        string key = (string?)spec["seed"] + "/" + (int?)spec["chooser_seed"] + "/" + spec["deck_add"]?.ToJsonString();
        uint h = 2166136261;
        foreach (char ch in key)
        {
            h = (h ^ ch) * 16777619;
        }

        return h % 10000 < _audit * 10000;
    }

    /// <summary>The first result key (timing and cost keys aside) where two results differ, or null.</summary>
    private static string? FirstDiff(JsonObject a, JsonObject b)
    {
        foreach (var key in a.Select(kv => kv.Key).Union(b.Select(kv => kv.Key)))
        {
            if (key is "cost" or "cost_setup" or "phase_ms" or "timing_ms")
            {
                continue;
            }

            if (a[key]?.ToJsonString() != b[key]?.ToJsonString())
            {
                return key;
            }
        }

        return null;
    }

    /// <summary>
    /// The audit: the same spec again on the full host; both results compared without timing and cost
    /// keys. The nodeless result is kept; cost.audit says whether they agree (and the first key that
    /// does not). Catches what nodeless cannot show, such as full-game UI errors.
    /// </summary>
    private static async Task Audit(JsonObject spec, JsonObject nodeless, Func<JsonObject, Task<JsonObject>> play, Action reset)
    {
        var full = await PlayFull(spec, play, reset);
        _audits++;
        string? diff = FirstDiff(nodeless, full);
        if (diff != null)
        {
            _auditDiffs++;
            GD.Print($"[referee] nodeless audit: {(string?)spec["seed"]} differs from the full host at '{diff}'");
        }

        Cost(nodeless)["audit"] = new JsonObject
        {
            ["same"] = diff == null, ["first_diff"] = diff, ["full_error"] = full["error"]?.DeepClone(),
            ["full_cpu_ms"] = (full["cost"] as JsonObject)?["cpu_ms"]?.DeepClone(),
            ["full_errors"] = (full["cost"] as JsonObject)?["errors"]?.DeepClone(),
        };
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
            ["fallbacks"] = _fallbacks, ["failed"] = _failed, ["audits"] = _audits, ["audit_diffs"] = _auditDiffs,
            ["cancel_checks"] = _cancelChecks, ["cancel_fails"] = _cancelFails,
        };
    }
}
