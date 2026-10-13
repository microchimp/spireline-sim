// The referee driver. Loaded as a mod by the user's own copy of the game, run headless, it starts
// seeded runs through the game's own run manager and writes what the game produced as JSON.
// It contains no game code: only calls into the game's public API.
//
// It does nothing unless SPIRELINE_REFEREE_JOB names a job file, so an installed copy is inert in
// normal play. Job: {"out": "<path>", "runs": [{"seed": "ABC", "character": "IRONCLAD", "ascension": 10}]}
// Server mode (Server.cs): {"out": "<path>.jsonl", "inbox": "<dir>"} plays fight-spec files from the
// inbox one at a time, appending one JSON line per fight to "out", until a STOP file appears.

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

        // The game's preloader switch, thrown before NGame starts: with it off the menu's background
        // "Common" asset session (686 assets, ~2-4 s of CPU, and the source of a rare native hang
        // after "texture_2d_initialize: Parameter t is null" when several copies boot together) is
        // never queued, and every asset loads on demand on the main thread instead.
        try
        {
            if ((bool?)JsonNode.Parse(File.ReadAllText(job))?["no_preload"] == true)
            {
                MegaCrit.Sts2.Core.Assets.PreloadManager.Enabled = false;
            }
        }
        catch (Exception ex) when (ex is IOException or JsonException)
        {
            // RunJob reports an unreadable job.
        }

        _ = RunJob(job);
    }

    private static long _lastProgress = System.Environment.TickCount64;

    /// <summary>Marks progress for the watchdog: called on every action, not just every fight.</summary>
    internal static void Heartbeat()
    {
        long now = System.Environment.TickCount64;
        Interlocked.Exchange(ref _lastProgress, now);
        WriteStatus(now);
    }

    /// <summary>Fights finished in this process, for the watchdog's message.</summary>
    private static int _done;
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

    /// <summary>
    /// One fight spec, start to finish: play it (<c>kind</c>), then leave the run. Shared by the
    /// "runs" list and the server inbox, so both take exactly the same code path.
    /// </summary>
    private static async Task<JsonObject> PlaySpec(JsonObject o, bool fastReset)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var result = (string?)o["kind"] switch
        {
            "combat" => await PlayCombat(o),
            "export" => Export(),
            _ => await StartRun(o),
        };
        long playMs = clock.ElapsedMilliseconds;
        if (fastReset)
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
        return result;
    }

    /// <summary>
    /// <c>NGame.LaunchMainMenu</c> queues a background asset session for the common assets right after
    /// the menu shows (NGame.cs:975-996, LoadDeferredStartupAssetsAsync); <c>NAssetLoader</c> polls it
    /// every frame and stops processing when its queue is empty. With the intro logo the 9 s of logo
    /// animation hid that load; without it (skip_intro_logo) the first run's own asset loads race the
    /// session, and under load the game hangs on its first Neow screen. So wait for the loader to drain.
    /// </summary>
    private static async Task WaitForStartupPreload(SceneTree tree)
    {
        for (int i = 0; i < 10; i++)  // the deferred task queues its session a frame or two after the menu
        {
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        var loader = MegaCrit.Sts2.Core.Nodes.NAssetLoader.Instance;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (loader.IsProcessing() && clock.Elapsed.TotalSeconds < 90)
        {
            Heartbeat();
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }

        GD.Print($"[referee] startup preload drained after {clock.ElapsedMilliseconds} ms");
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
                    GD.Print($"[referee] watchdog: no progress for {stallMs / 1000} s, exiting after {_done} fights");
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
            while (NGame.Instance?.MainMenu == null)
            {
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            }

            // Player-facing speed settings only: the game's own "Instant" fast mode skips animation
            // waits, and the frame cap only paces rendering, which is off anyway.
            SaveManager.Instance.PrefsSave.FastMode = MegaCrit.Sts2.Core.Settings.FastModeType.Instant;
            IdleFrames = (double?)job["idle_frames"] ?? IdleFrames;
            IdleTimeoutS = (double?)job["idle_timeout_s"] ?? IdleTimeoutS;
            // Tutorials off, as a player can set it: first-time popups (FTUE) on a fresh profile
            // open modals that nothing headless answers.
            SaveManager.Instance.Progress.EnableFtues = false;
            if ((bool?)job["fast_patches"] == true)
            {
                Patches.Apply();
            }

            Lean.Configure(job, tree);  // frame-cost profiling (Lean.cs)

            // The game's own preloader switch: background asset sessions only warm a cache,
            // and back-to-back runs (worse with many copies) hang waiting on one. Off, assets
            // load on demand.
            if ((bool?)job["no_preload"] == true)
            {
                MegaCrit.Sts2.Core.Assets.PreloadManager.Enabled = false;
            }
            Engine.MaxFps = 0;
            await WaitForStartupPreload(tree);

            Heartbeat();
            bool fastReset = (bool?)job["fast_reset"] == true;
            if (job["inbox"] != null)
            {
                exitCode = await ServeInbox(job, outPath, fastReset);
                ((SceneTree)Engine.GetMainLoop()).Quit(exitCode);
                return;
            }

            foreach (var spec in job["runs"]!.AsArray())
            {
                results.Add(await PlaySpec(spec!.AsObject(), fastReset));
                _done++;
                Save(outPath, results);  // after every fight: a hang or kill keeps what is done
                Interlocked.Exchange(ref _lastProgress, System.Environment.TickCount64);
            }
        }
        catch (Exception ex)
        {
            var err = new JsonObject { ["error"] = ex.ToString() };
            exitCode = 1;
            if (job["inbox"] != null)
            {
                AppendLine(outPath, err);  // server mode: the out file is a JSONL, never rewritten
                ((SceneTree)Engine.GetMainLoop()).Quit(exitCode);
                return;
            }

            results.Add(err);
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
