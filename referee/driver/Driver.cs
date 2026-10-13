// The referee driver. Loaded as a mod by the user's own copy of the game, run headless, it starts
// seeded runs through the game's own run manager and writes what the game produced as JSON.
// It contains no game code: only calls into the game's public API.
//
// It does nothing unless SPIRELINE_REFEREE_JOB names a job file, so an installed copy is inert in
// normal play. Job: {"out": "<path>", "runs": [{"seed": "ABC", "character": "IRONCLAD", "ascension": 10}]}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;

namespace Spireline.Referee;

[ModInitializer(nameof(Init))]
public static partial class Driver
{
    private const string JobEnv = "SPIRELINE_REFEREE_JOB";

    public static void Init()
    {
        string? job = System.Environment.GetEnvironmentVariable(JobEnv);
        if (string.IsNullOrEmpty(job))
        {
            return;
        }

        // A fault before the job's own try (e.g. a type initializer) would otherwise hang silently.
        _ = RunJob(job).ContinueWith(t =>
        {
            GD.Print($"[referee] job failed outside its handler: {t.Exception}");
            System.Environment.Exit(4);
        }, TaskContinuationOptions.OnlyOnFaulted);
    }

    private static long _lastProgress = System.Environment.TickCount64;
    private static long _frameTicks;
    internal static long _lastDriverFrame;

    /// <summary>Marks progress for the watchdog: called on every action, not just every fight.</summary>
    internal static void Heartbeat() => Interlocked.Exchange(ref _lastProgress, System.Environment.TickCount64);
    private static readonly object _saveLock = new();

    private static void Save(string path, JsonArray results)
    {
        lock (_saveLock)
        {
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, new JsonObject { ["results"] = results.DeepClone() }.ToJsonString());
            File.Move(tmp, path, overwrite: true);
        }
    }

    private static async Task RunJob(string jobPath)
    {
        var job = JsonNode.Parse(File.ReadAllText(jobPath))!.AsObject();
        string outPath = (string)job["out"]!;
        var results = new JsonArray();
        // Watchdog on a pool thread: if no fight finishes for stall_s seconds, the main loop is
        // stuck (a load or a screen nothing answers); keep what is saved and end the process.
        int stallMs = ((int?)job["stall_s"] ?? 20) * 1000;
        _ = Task.Run(async () =>
        {
            while (true)
            {
                await Task.Delay(5000);
                if (System.Environment.TickCount64 - Interlocked.Read(ref _lastProgress) > stallMs)
                {
                    GD.Print($"[referee] watchdog: no progress for {stallMs / 1000} s, exiting with {results.Count} results saved; "
                        + $"frames ticked {Interlocked.Read(ref _frameTicks)} (last driver frame {Interlocked.Read(ref _lastDriverFrame)}), "
                        + $"step {CurrentStep}, fast tweens {Interlocked.Read(ref Patches.FastTweens)}, in fast tween {Patches.InFastTween}");
                    System.Environment.Exit(3);
                }
            }
        });
        int exitCode = 0;
        try
        {
            // Wait until the game has finished its own startup and shows the main menu: content,
            // localisation and the profile are loaded by then, exactly as for a player.
            var tree = (SceneTree)Engine.GetMainLoop();
            // For the watchdog: are frames still ticking (an await never resolves) or is the main thread stuck?
            tree.ProcessFrame += () => Interlocked.Increment(ref _frameTicks);
            while (NGame.Instance?.MainMenu == null)
            {
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            }

            // Player-facing speed settings only: the game's own "Instant" fast mode skips animation
            // waits, and the frame cap only paces rendering, which is off anyway.
            SaveManager.Instance.PrefsSave.FastMode = MegaCrit.Sts2.Core.Settings.FastModeType.Instant;
            IdleFrames = (double?)job["idle_frames"] ?? IdleFrames;
            SettleMode = (string?)job["settle"] ?? SettleMode;
            SettleAudit = (double?)job["settle_audit"] ?? SettleAudit;
            TraceFrames = (bool?)job["trace_frames"] ?? false;
            StepCosts = (bool?)job["step_costs"] ?? false;
            IdleProbe = (int?)job["idle_probe"] ?? 0;
            if (job["sleep_usec"] is { } sleep)
            {
                // Headless, the main loop pads every frame to this many microseconds (Godot's
                // OS::add_frame_delay applies the low-processor sleep when no window can draw).
                OS.LowProcessorUsageModeSleepUsec = (int)sleep;
            }

            GD.Print($"[referee] low_processor_mode={OS.LowProcessorUsageMode} sleep_usec={OS.LowProcessorUsageModeSleepUsec} max_fps={Engine.MaxFps}");
            if (TraceFrames)
            {
                Patches.ApplyTrace();
            }

            if ((bool?)job["fast_tweens"] == true)
            {
                Patches.ApplyFastTweens();
            }
            // Tutorials off, as a player can set it: first-time popups (FTUE) on a fresh profile
            // open modals that nothing headless answers.
            SaveManager.Instance.Progress.EnableFtues = false;
            // The game's own preloader switch: background asset sessions only warm a cache,
            // and back-to-back runs (worse with many copies) hang waiting on one. Off, assets
            // load on demand.
            if ((bool?)job["fast_patches"] == true)
            {
                Patches.Apply();
            }

            if ((bool?)job["no_preload"] == true)
            {
                MegaCrit.Sts2.Core.Assets.PreloadManager.Enabled = false;
            }
            Engine.MaxFps = 0;

            Heartbeat();
            foreach (var spec in job["runs"]!.AsArray())
            {
                var o = spec!.AsObject();
                var clock = System.Diagnostics.Stopwatch.StartNew();
                var result = (string?)o["kind"] switch
                {
                    "combat" => await PlayCombat(o),
                    "export" => Export(),
                    _ => await StartRun(o),
                };
                long playMs = clock.ElapsedMilliseconds;
                if ((bool?)job["fast_reset"] == true)
                {
                    // Skip the menu: end the run the way ReturnToMainMenu does at its core
                    // (RunManager.CleanUp) without reloading menu assets and scenes.
                    RunManager.Instance.CleanUp();
                }
                else
                {
                    await NGame.Instance!.ReturnToMainMenu();  // the game's own way out of a run
                }

                result["timing_ms"] = new JsonObject { ["play"] = playMs, ["reset"] = clock.ElapsedMilliseconds - playMs };
                results.Add(result);
                Save(outPath, results);  // after every fight: a hang or kill keeps what is done
                Interlocked.Exchange(ref _lastProgress, System.Environment.TickCount64);
            }
        }
        catch (Exception ex)
        {
            results.Add(new JsonObject { ["error"] = ex.ToString() });
            exitCode = 1;
        }

        Save(outPath, results);
        ((SceneTree)Engine.GetMainLoop()).Quit(exitCode);
    }

    /// <summary>
    /// <c>NGame.StartNewSingleplayerRun</c>, with one difference: the profile is the fully
    /// unlocked, all-discovered one the emulator models, not the empty scratch profile.
    /// </summary>
    private static async Task<JsonObject> StartRun(JsonObject spec) => (await BeginRun(spec)).dump;

    private static async Task<(RunState state, JsonObject dump)> BeginRun(JsonObject spec)
    {
        string seed = SeedHelper.CanonicalizeSeed((string)spec["seed"]!);
        string characterId = (string?)spec["character"] ?? "IRONCLAD";
        int ascension = (int?)spec["ascension"] ?? 10;

        UnlockState unlock = UnlockState.all;
        foreach (ActModel act in ModelDb.ActsByIndex.SelectMany(a => a))
        {
            SaveManager.Instance.Progress.MarkActAsSeen(act.Id);
        }

        // StartRunLobby.BeginRunLocally's act selection, single player.
        var actRng = new Rng((uint)StringHelper.GetDeterministicHashCode(seed), "act_selection");
        List<ActModel> acts = ActModel.GetRandomList(actRng, unlock, isMultiplayer: false).ToList();
        CharacterModel character = ModelDb.AllCharacters.First(c => c.Id.Entry == characterId);

        RunState state = RunState.CreateForNewRun(
            new[] { Player.CreateForNewRun(character, unlock, 1uL) },
            acts.Select(a => a.ToMutable()).ToList(),
            Array.Empty<ModifierModel>(),
            GameMode.Standard,
            ascension,
            seed);
        RunManager.Instance.SetUpNewSingleplayer(state, shouldSave: false);
        if (Patches.Applied)
        {
            // SetUpNewSingleplayer turns the combat replay recorder on (RunManager.cs ~:416);
            // nothing reads its output here.
            RunManager.Instance.CombatReplayWriter.IsEnabled = false;
        }
        MethodInfo startRun = typeof(NGame).GetMethod("StartRun", BindingFlags.NonPublic | BindingFlags.Instance)!;
        await (Task)startRun.Invoke(NGame.Instance, new object[] { state })!;

        return (state, new JsonObject
        {
            ["seed"] = seed,
            ["character"] = characterId,
            ["ascension"] = ascension,
            ["acts"] = new JsonArray(state.Acts.Select(a => (JsonNode)a.Id.Entry).ToArray()),
            ["rng"] = Counters(state.Rng),
            ["player_rng"] = Counters(state.Players[0].PlayerRng),
            ["map"] = Map(state),
            ["save"] = JsonNode.Parse(SaveManager.ToJson(RunManager.Instance.ToSave(null))),
        });
    }

    /// <summary>Every public <see cref="Rng" /> property of a stream set: name, seed, counter.</summary>
    private static JsonObject Counters(object set)
    {
        var o = new JsonObject();
        foreach (PropertyInfo p in set.GetType().GetProperties().Where(p => p.PropertyType == typeof(Rng)))
        {
            var rng = (Rng)p.GetValue(set)!;
            o[p.Name] = new JsonObject { ["seed"] = rng.Seed, ["counter"] = rng.Counter };
        }

        return o;
    }

    private static JsonArray Map(RunState state)
    {
        var points = new JsonArray();
        // GetAllMapPoints is the grid; the start (Ancient) and boss rooms sit outside it.
        var all = state.Map.GetAllMapPoints()
            .Append(state.Map.StartingMapPoint)
            .Append(state.Map.BossMapPoint)
            .Concat(state.Map.SecondBossMapPoint is { } second ? new[] { second } : Array.Empty<MapPoint>())
            .Distinct();
        foreach (var p in all.OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
        {
            points.Add(new JsonObject
            {
                ["col"] = p.coord.col,
                ["row"] = p.coord.row,
                ["type"] = p.PointType.ToString(),
                ["children"] = new JsonArray(p.Children.OrderBy(c => c.coord.row).ThenBy(c => c.coord.col)
                    .Select(c => (JsonNode)new JsonArray(c.coord.col, c.coord.row)).ToArray()),
            });
        }

        return points;
    }
}
