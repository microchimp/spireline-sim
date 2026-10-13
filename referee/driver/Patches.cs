// Speed patches (SPEED_PLAN.md step 2), applied only when a job sets "fast_patches": true so the
// old and new paths can be compared on the same seeds. Each removes presentation or cache work
// only; none touches a game rule.

using System;
using System.Reflection;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Nodes;

namespace Spireline.Referee;

public static class Patches
{
    private static bool _applied;

    public static bool Applied => _applied;

    public static void Apply()
    {
        if (_applied) return;
        _applied = true;
        var h = new Harmony("spireline.referee.speed");
        // NTransition.RoomFadeIn builds and awaits a 0.8 s tween even in Instant mode
        // (NTransition.cs ~:276-314). End it at once in the state the Instant branch sets first.
        h.Patch(AccessTools.Method(typeof(NTransition), nameof(NTransition.RoomFadeIn)),
            prefix: new HarmonyMethod(typeof(Patches), nameof(RoomFadeInPrefix)));
        // Load{Run,Act,Room}Assets purge the cache and force GC.Collect every run and room
        // (PreloadManager.cs ~:79-146), even with PreloadManager.Enabled off. Assets load on demand.
        foreach (var name in new[] { "LoadRunAssets", "LoadActAssets", "LoadRoomAssets" })
        {
            h.Patch(AccessTools.Method(typeof(PreloadManager), name),
                prefix: new HarmonyMethod(typeof(Patches), nameof(SkipTaskPrefix)));
        }
    }

    private static bool _fastTweensApplied;

    /// <summary>Tweens completed so far by <see cref="FastTweenPrefix" /> (diagnostics).</summary>
    internal static long FastTweens;

    /// <summary>True while a tween is being run to its end (diagnostics for the watchdog).</summary>
    internal static volatile bool InFastTween;

    /// <summary>
    /// SPEED_PLAN.md step 4 (job "fast_tweens"): an awaited tween is run to its end at once, the way
    /// the game's own <c>TweenHelper.FastForwardToCompletion</c> does (TweenHelper.cs ~:9-12, used
    /// e.g. at CardPileCmd.cs ~:739), so its callbacks run in order and <c>finished</c> fires before
    /// the awaiter looks. Game code awaits card-pile tweens even in Instant mode (CardPileCmd.cs
    /// ~:176, ~:623-626, ~:679): 1 frame per card moved, ~0.3 s per exhaust (~:171, ~:577).
    /// AwaitFinishedInternal (TweenHelper.cs ~:45-66) then sees a finished tween and returns a
    /// completed task, so the awaiting code continues in the same frame.
    /// </summary>
    public static void ApplyFastTweens()
    {
        if (_fastTweensApplied) return;
        _fastTweensApplied = true;
        var h = new Harmony("spireline.referee.tweens");
        h.Patch(AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.GodotExtensions.TweenHelper), "AwaitFinishedInternal"),
            prefix: new HarmonyMethod(typeof(Patches), nameof(FastTweenPrefix)) { priority = Priority.Last });
    }

    private static void FastTweenPrefix(Tween tween, Node owner)
    {
        // Left to the original: invalid or stopped tweens (it returns at once), infinite loops (a
        // huge step would spin), and owners outside the tree (it resolves false via TreeExiting).
        if (!GodotObject.IsInstanceValid(tween) || !tween.IsValid() || !tween.IsRunning() || tween.GetLoopsLeft() < 0
            || !GodotObject.IsInstanceValid(owner) || !owner.IsInsideTree())
        {
            return;
        }

        InFastTween = true;
        try
        {
            tween.CustomStep(999999999.0);
        }
        finally
        {
            InFastTween = false;
        }

        System.Threading.Interlocked.Increment(ref FastTweens);
    }

    private static bool _traceApplied;

    /// <summary>Diagnostics only (job "trace_frames"): log every awaited tween and its caller into the step trace.</summary>
    public static void ApplyTrace()
    {
        if (_traceApplied) return;
        _traceApplied = true;
        var h = new Harmony("spireline.referee.trace");
        h.Patch(AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.GodotExtensions.TweenHelper), "AwaitFinishedInternal"),
            prefix: new HarmonyMethod(typeof(Patches), nameof(TraceTweenPrefix)));
    }

    private static void TraceTweenPrefix(Tween tween)
    {
        var st = new System.Diagnostics.StackTrace(2, false);
        var names = new System.Collections.Generic.List<string>();
        for (int i = 0; i < st.FrameCount && names.Count < 3; i++)
        {
            var m = st.GetFrame(i)?.GetMethod();
            string? n = m?.DeclaringType?.FullName;
            if (n == null || n.StartsWith("System.")) continue;
            names.Add(n.Replace("MegaCrit.Sts2.Core.", "") + "." + m!.Name);
        }

        Driver.TraceNote($"T {Engine.GetProcessFrames()} running={tween.IsRunning()} {string.Join(" < ", names)}");
    }

    private static bool RoomFadeInPrefix(NTransition __instance, ref Task __result)
    {
        if (AccessTools.Field(typeof(NTransition), "_simpleTransition")?.GetValue(__instance) is Control simple)
        {
            var m = simple.Modulate;
            m.A = 0f;
            simple.Modulate = m;
        }

        __instance.MouseFilter = Control.MouseFilterEnum.Ignore;
        AccessTools.Property(typeof(NTransition), nameof(NTransition.InTransition))?.SetValue(__instance, false);
        __result = Task.CompletedTask;
        return false;
    }

    private static bool SkipTaskPrefix(ref Task __result)
    {
        __result = Task.CompletedTask;
        return false;
    }
}
