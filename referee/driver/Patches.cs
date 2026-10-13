// Speed patches (SPEED_PLAN.md step 2), applied only when a job sets "fast_patches": true so the
// old and new paths can be compared on the same seeds. Each removes presentation or cache work
// only; none touches a game rule.
//
// Two more sets cut what each frame and each process costs (job keys, each on its own so it can be
// measured alone): "silent" (no sound is ever started) and "still_visuals" (visual-only nodes do no
// per-frame work). Both rest on reads of the decompiled game, cited at each patch.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.Vfx;

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

    /// <summary>Prefix for a method whose whole effect is skipped (a non-void one returns its default).</summary>
    private static bool SkipPrefix() => false;

    private static void SkipAll(Harmony h, Type type, params string[] names)
    {
        foreach (MethodInfo m in AccessTools.GetDeclaredMethods(type).Where(m => names.Contains(m.Name)))
        {
            h.Patch(m, prefix: new HarmonyMethod(typeof(Patches), nameof(SkipPrefix)));
        }
    }

    private static bool _silent;

    /// <summary>
    /// Job key "silent": the referee never starts a sound. Every sound the game makes goes through three
    /// nodes, all fire-and-forget calls whose results nothing reads: NAudioManager and
    /// NRunMusicController forward to their GDScript "Proxy" node and so to FMOD (NAudioManager.cs
    /// PlayLoop..StopMusic, NRunMusicController.cs UpdateMusic..StopMusic, which also loads the act's
    /// music bank), and NDebugAudioManager plays placeholder mp3s on Godot AudioStreamPlayers
    /// (NDebugAudioManager.Play: a card_deal.mp3 per card drawn, CardPileCmd). With volume 0 the
    /// mixers still decode, mix and run their threads; with nothing started they idle. Left alone:
    /// the volume setters (startup) and ResolveMusic (a fresh "bg_music" Rng, no run stream).
    /// NRunMusicController's merchant/elite/campfire triggers also throw when called in the wrong
    /// room, a programmer check the game never trips in a correct run.
    /// </summary>
    public static void Silence()
    {
        if (_silent) return;
        _silent = true;
        var h = new Harmony("spireline.referee.silent");
        SkipAll(h, typeof(NAudioManager), "PlayLoop", "StopLoop", "SetParam", "StopAllLoops", "PlayOneShot",
            "PlayMusic", "UpdateMusicParameter", "StopMusic");
        SkipAll(h, typeof(NRunMusicController), "UpdateMusic", "PlayCustomMusic", "UpdateCustomTrack",
            "StopCustomMusic", "UpdateAmbience", "UpdateTrack", "UpdateMusicParameter", "ToggleMerchantTrack",
            "TriggerEliteSecondPhase", "TriggerCampfireGoingOut", "StopMusic");
        SkipAll(h, typeof(NDebugAudioManager), "Play", "Stop", "StopAll");
    }

    private static bool _still;

    /// <summary>
    /// Job key "still_visuals": nodes that only animate pixels do no per-frame work. Headless, nothing
    /// is drawn, yet every frame these still simulate and redraw (measured: about 2 ms of a 3.5 ms
    /// quiet combat frame). Three mechanisms, each only where no game rule reads the result:
    /// <list type="bullet">
    /// <item>Native visual nodes without a script (GPU/CPU particles, AnimatedSprite2D, Spine
    /// skeletons) get ProcessMode Disabled as they enter the tree. Checked in the decompile: game
    /// code awaits particle "Finished" only inside VFX classes (NHitSparkVfx, NGroundFireVfx, ...,
    /// fire-and-forget), never AnimatedSprite2D or AudioStreamPlayer signals; Spine "animation_*"
    /// signals feed CreatureAnimator's visual state, NVfxSpine's self-free and two VFX
    /// (MegaSprite.cs, CreatureAnimator.cs); every game tween is created by a C# node on itself
    /// (CreateTween receivers), so none is bound to a node disabled here. The one rule read of a
    /// Spine track, NCreature.AnimDie's remaining time, only sizes a Cmd.Wait that Instant mode skips.
    /// Spine's random track offsets use Rng.Chaotic (CreatureAnimator.cs), never a run stream.</item>
    /// <item>Rich-text effects (every RichTextEffect subclass in the game, e.g. RichTextSine,
    /// RichTextGold) skip _ProcessCustomFX: a C# call per glyph per frame that only moves or tints
    /// glyphs.</item>
    /// <item>NCardTrail._Process (a card's Line2D trail, NCardTrail.cs) is skipped: it only adds and
    /// ages trail points.</item>
    /// </list>
    /// </summary>
    public static void StillVisuals(SceneTree tree)
    {
        if (_still) return;
        _still = true;
        var h = new Harmony("spireline.referee.still");
        foreach (Type t in typeof(NGame).Assembly.GetTypes().Where(t => t.IsSubclassOf(typeof(RichTextEffect))))
        {
            if (AccessTools.DeclaredMethod(t, "_ProcessCustomFX") is { IsAbstract: false } fx)
            {
                h.Patch(fx, prefix: new HarmonyMethod(typeof(Patches), nameof(SkipFxPrefix)));
            }
        }

        h.Patch(AccessTools.DeclaredMethod(typeof(NCardTrail), nameof(NCardTrail._Process)),
            prefix: new HarmonyMethod(typeof(Patches), nameof(SkipPrefix)));
        tree.NodeAdded += StillNode;
    }

    private static bool SkipFxPrefix(ref bool __result)
    {
        __result = true;
        return false;
    }

    private static void StillNode(Node n)
    {
        // Exact wrapper types only: a game script deriving from these is not touched.
        Type t = n.GetType();
        bool still = t == typeof(GpuParticles2D) || t == typeof(CpuParticles2D) || t == typeof(AnimatedSprite2D)
            || (t == typeof(Node2D) && n.IsClass("SpineSprite"));
        if (still && n.GetScript().VariantType == Variant.Type.Nil)
        {
            n.ProcessMode = Node.ProcessModeEnum.Disabled;
        }
    }
}
