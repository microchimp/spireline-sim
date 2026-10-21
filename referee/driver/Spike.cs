// Speed spike (branch spike/fast-forward, referee/design/real-game-speed.md): opt-in job keys that
// measure where a headless fight's time goes, try engine-level fast-forward settings and time the
// game's own save and state-capture paths for branching. Every key is off by default; with none set
// nothing here runs except a null check per call.
//
//   "spike_prof": true       cost.spike = driver time per phase of the step loop (choose, act,
//                            settle, snapshot), in ms and main-thread megacycles
//   "time_scale": X          Engine.TimeScale = X (real-time timers and tweens run X times faster)
//   "physics_ticks": N       Engine.PhysicsTicksPerSecond = N
//   "max_physics_steps": N   Engine.MaxPhysicsStepsPerFrame = N
//   "branch_probe": K        after step K's snapshot, time RunManager.ToSave, its JSON round trip,
//                            NetFullCombatState.FromRun and its packet serialisation (diagnostics only)

using System;
using System.Diagnostics;
using System.Text.Json.Nodes;
using Godot;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Spireline.Referee;

internal static class Spike
{
    internal const int Choose = 0, Act = 1, Settle = 2, Snapshot = 3, Slots = 4;
    private static readonly string[] Names = { "choose", "act", "settle", "snapshot" };

    internal static bool Prof;
    internal static int BranchProbe;
    private static readonly double[] _ms = new double[Slots];
    private static readonly double[] _mcyc = new double[Slots];
    private static ulong _c0;

    internal static void Configure(JsonNode job)
    {
        Prof = (bool?)job["spike_prof"] == true;
        BranchProbe = (int?)job["branch_probe"] ?? 0;
        if (job["time_scale"] is { } ts) Engine.TimeScale = (double)ts;
        if (job["physics_ticks"] is { } pt) Engine.PhysicsTicksPerSecond = (int)pt;
        if (job["max_physics_steps"] is { } mp) Engine.MaxPhysicsStepsPerFrame = (int)mp;
        if (Prof || BranchProbe > 0 || job["time_scale"] != null || job["physics_ticks"] != null)
        {
            GD.Print($"[referee] spike prof={Prof} branch_probe={BranchProbe} time_scale={Engine.TimeScale} "
                     + $"physics_ticks={Engine.PhysicsTicksPerSecond} max_physics_steps={Engine.MaxPhysicsStepsPerFrame}");
        }
    }

    internal static void BeginFight()
    {
        if (!Prof) return;
        Array.Clear(_ms);
        Array.Clear(_mcyc);
    }

    /// <summary>A timestamp for <see cref="Add"/>; 0 when profiling is off.</summary>
    internal static long Now()
    {
        if (!Prof) return 0;
        _c0 = Lean.ThreadCycles();
        return Stopwatch.GetTimestamp();
    }

    internal static void Add(int slot, long t0)
    {
        if (!Prof) return;
        _ms[slot] += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        _mcyc[slot] += (Lean.ThreadCycles() - _c0) / 1e6;
    }

    internal static void EndFight(JsonObject cost)
    {
        if (!Prof) return;
        var o = new JsonObject();
        for (int i = 0; i < Slots; i++)
        {
            o[Names[i] + "_ms"] = Math.Round(_ms[i], 2);
            o[Names[i] + "_mcyc"] = Math.Round(_mcyc[i], 1);
        }

        cost["spike"] = o;
    }

    /// <summary>Branching cost: the game's run save (no combat state in it) and its full combat state capture.</summary>
    internal static void Probe(RunState state, int step, JsonObject dump)
    {
        if (BranchProbe <= 0 || step != BranchProbe) return;
        const int n = 20;
        var o = new JsonObject { ["step"] = step, ["n"] = n };
        try
        {
            var sw = Stopwatch.StartNew();
            SerializableRun? save = null;
            for (int i = 0; i < n; i++) save = RunManager.Instance.ToSave(null);
            o["to_save_ms"] = Math.Round(sw.Elapsed.TotalMilliseconds / n, 3);
            sw.Restart();
            string json = "";
            for (int i = 0; i < n; i++) json = JsonSerializationUtility.ToJson(save!);
            o["save_to_json_ms"] = Math.Round(sw.Elapsed.TotalMilliseconds / n, 3);
            o["save_json_bytes"] = json.Length;
            sw.Restart();
            bool ok = false;
            for (int i = 0; i < n; i++) ok = JsonSerializationUtility.FromJson<SerializableRun>(json).Success;
            o["save_from_json_ms"] = Math.Round(sw.Elapsed.TotalMilliseconds / n, 3);
            o["save_from_json_ok"] = ok;
            o["save_has_combat"] = save!.PreFinishedRoom != null;
        }
        catch (Exception ex)
        {
            o["save_error"] = ex.GetType().Name + ": " + ex.Message;
        }

        try
        {
            var sw = Stopwatch.StartNew();
            NetFullCombatState? net = null;
            for (int i = 0; i < n; i++) net = NetFullCombatState.FromRun(state, null);
            o["net_from_run_ms"] = Math.Round(sw.Elapsed.TotalMilliseconds / n, 3);
            sw.Restart();
            var w = new PacketWriter();
            for (int i = 0; i < n; i++) { w.Reset(); w.Write(net!); }
            o["net_write_ms"] = Math.Round(sw.Elapsed.TotalMilliseconds / n, 3);
            o["net_bytes"] = w.BytePosition;
        }
        catch (Exception ex)
        {
            o["net_error"] = ex.GetType().Name + ": " + ex.Message;
        }

        dump["branch_probe"] = o;
    }
}
