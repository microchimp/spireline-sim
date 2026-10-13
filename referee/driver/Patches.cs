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
