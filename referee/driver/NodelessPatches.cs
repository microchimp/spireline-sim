// Harmony patches for the nodeless host (Nodeless.cs). Installed only when a job sets
// "nodeless": true, and every one of them is a no-op unless Nodeless.Active (the fight being played
// runs nodeless), so full-host fights in the same process run the game untouched.
//
// Rule: the rules stay the game's own compiled code. A prefix may only skip work whose whole
// purpose is UI or visual (scene instancing, node lookups, tweens, VFX, screens) and that no rule
// reads; each one names the method and why. Models whose own rules code dereferences a node
// unguarded are not patched: fights that create them go to the full host (FallbackNames).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace Spireline.Referee;

public static class NodelessPatches
{
    private static bool _applied;

    /// <summary>Explicit patches (each listed in Apply).</summary>
    internal static int Count { get; private set; }

    /// <summary>Node factories nulled by the test-mode rule (see PatchTestModeFactories).</summary>
    internal static int FactoryCount { get; private set; }

    internal static int FallbackCount => _fallback.Count;

    /// <summary>
    /// Models whose own rules code needs a node: an unguarded node dereference (it would throw
    /// nodeless and cut the rules chain short) or a rule that waits on a VFX node. A fight that
    /// creates one (AbstractModel.MutableClone) is replayed from the start on the full host.
    /// </summary>
    private static readonly string[] FallbackNames =
    {
        // Cards
        "BouncingFlask",       // OnPlay: NCombatRoom.Instance.GetCreatureNode(owner) under TestMode.IsOff
        "HandOfGreed",         // OnPlay: NCombatRoom.Instance.GetCreatureNode(target) under TestMode.IsOff
        "Nightmare",           // OnPlay: NGame.Instance.CurrentRunNode.GlobalUi under TestMode.IsOff
        "Begone", "Charge", "Compact", "Guards", "PrimalForce", "Seance",  // CardCmd.Transform of a hand card: NCombatRoom.Instance.Ui
        // Monsters
        "BowlbugSilk",         // SpitMove: NCombatRoom.Instance.GetCreatureNode(target).GlobalPosition
        "Guardbot", "Noisebot", "Stabbot", "Zapbot",  // AfterAddedToRoom: NCombatRoom.Instance.GetCreatureNode under TestMode.IsOff
        "ToughEgg",            // AfterAddedToRoom: same, when hatching
        "SoulNexus",           // Died handler: NCombatRoom.Instance.GetCreatureNode(...)
        "KinFollower", "MagiKnight",  // attack moves: target.GetCreatureNode().GlobalPosition under TestMode.IsOff
        "Crusher", "Rocket",   // Background: NCombatRoom.Instance?.Background ?? ... ?? throw
        "Osty",                // OstyCmd.Summon: NCombatRoom.Instance?.GetCreatureNode(osty).OstyScaleToSize under TestMode.IsOff
        // Powers
        "EntropyPower",        // AfterPlayerTurnStart: CardCmd.Transform of a hand card
        "InfestedPower",       // AfterDeath: NRunMusicController.Instance.TriggerEliteSecondPhase()
        "RollingBoulderPower", // its damage is dealt from NRollingBoulderVfx's HitCreature signal
        "SandpitPower",        // AfterApplied/AfterRemoved: NCombatRoom.Instance.GetCreatureNode, tweens
        // Relics
        "LordsParasol",        // merchant: NMerchantRoom.Instance (never in a fight, listed for completeness)
        // Events
        "Trial",               // NEventRoom.Instance (never in a fight, listed for completeness)
    };

    private static readonly HashSet<Type> _fallback = new();

    public static void Apply()
    {
        if (_applied) return;
        _applied = true;
        var h = new Harmony("spireline.referee.nodeless");

        // --- Node factories (the game's own test-mode rule) --------------------------------
        // Every static node factory in the game's Nodes namespaces that starts with
        // `if (TestMode.IsOn) return null;` (NCard.Create, NDamageNumVfx.Create, NEventRoom.Create,
        // NCardRewardSelectionScreen.ShowScreen, NThoughtBubbleVfx.Create, ...) returns null, as it does
        // under the game's tests. All of them instance a scene for display; rules code that calls one
        // null-checks the result, because the game's tests run those rules with null. Without this,
        // nodeless every such node would be an orphan (never added to a tree) or crash on a lookup.
        FactoryCount = PatchTestModeFactories(h);

        // --- Assets -------------------------------------------------------------------------
        // PreloadManager.Load{Run,Act,Room}Assets warm the asset cache for the scenes a run shows
        // (and purge it, and force a GC); nodeless no scene is shown. Skipped as Patches.cs does with
        // "fast_patches", here for nodeless fights only and whatever that job flag says.
        // LoadActAssets would also throw without LoadRunAssets, which NGame.StartRun calls and the
        // nodeless start does not (AssetSets.RunSet is null).
        foreach (var name in new[] { "LoadRunAssets", "LoadActAssets", "LoadRoomAssets" })
        {
            Prefix(h, typeof(MegaCrit.Sts2.Core.Assets.PreloadManager), name, nameof(SkipTask));
        }

        // --- Run and room flow -------------------------------------------------------------
        // RunManager.ClearScreens: NOverlayStack.Clear, NCapstoneContainer.Close, NMapScreen.Close,
        // unguarded, under TestMode.IsOff. Screens only; no run state.
        Prefix(h, typeof(RunManager), "ClearScreens", nameof(SkipVoid));
        // MapRoom.EnterInternal: under TestMode.IsOff creates NMapRoom and hands it to
        // NRun.Instance.SetCurrentRoom (unguarded); in test mode it returns at once, as here. The
        // isRestoringRoomStackBase throw is kept.
        Prefix(h, typeof(MapRoom), nameof(MapRoom.EnterInternal), nameof(MapRoomEnterPrefix));
        // NRewardsScreen.ShowScreen: instances the rewards screen and pushes it on NOverlayStack
        // (unguarded; no test-mode check, RewardsSet.Offer skips the call in test mode). Offer ignores
        // the returned screen; the driver takes rewards through RewardsSetSynchronizer on both hosts.
        Prefix(h, Ui("Screens.NRewardsScreen"), "ShowScreen", nameof(ReturnNull));

        // --- Cards and VFX -----------------------------------------------------------------
        // CardPileCmd.CreateCardNodeAndUpdateVisuals: NCard.Create, then NCombatRoom.Instance.Ui and
        // the pile position (unguarded). Its two callers (Add, AddDuringManualCardPlay) call it only
        // under TestMode.IsOff and treat a null node as "no visuals": no tween, no reparenting.
        Prefix(h, typeof(CardPileCmd), "CreateCardNodeAndUpdateVisuals", nameof(ReturnNull));
        // CardCmd.PreviewInternal: the card preview that pops up mid-screen and flies to its pile;
        // reads NCombatRoom.Instance.Ui containers (unguarded). Returns null in test mode and when
        // there is no container; no caller awaits the preview.
        Prefix(h, typeof(CardCmd), "PreviewInternal", nameof(ReturnNull));
        // ForgeCmd.PreviewSovereignBlade: smith VFX over the hand and top bar (NCombatRoom.Instance.Ui,
        // NRun.Instance.GlobalUi, unguarded); returns at once in test mode.
        Prefix(h, typeof(ForgeCmd), "PreviewSovereignBlade", nameof(SkipVoid));
        // NGame.ScreenShake / ScreenRumble / ScreenShakeTrauma: move the shake target (the combat room's
        // scene container on the full host) for a moment; the angle comes from Rng.Chaotic. Nodeless
        // the target is gone once a full-host fight's NCombatRoom has left the tree (it clears it), and
        // every shake then logs "Missing screenShake target!".
        foreach (var name in new[] { "ScreenShake", "ScreenRumble", "ScreenShakeTrauma" })
        {
            Prefix(h, typeof(MegaCrit.Sts2.Core.Nodes.NGame), name, nameof(SkipVoid));
        }

        // VfxCmd.PlayVfx: instances a VFX scene and adds it to the given container if there is one
        // (returns at once in test mode). Nodeless the container is null and the instance stayed an
        // orphan node: 12 per Stomp, Thunderclap or Pact's End (VfxCmd.PlayOnSide for AOE hits).
        Prefix(h, typeof(VfxCmd), nameof(VfxCmd.PlayVfx), nameof(SkipVoid));
        // NAudioManager play/stop/param calls (each returns at once in test mode): sound only, at
        // volume 0 here. The volume setters are left alone.
        var audio = Ui("Audio.NAudioManager");
        foreach (var m in audio.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (m.ReturnType == typeof(void) && !m.Name.EndsWith("Vol", StringComparison.Ordinal)
                && (m.Name.StartsWith("Play", StringComparison.Ordinal) || m.Name.StartsWith("Stop", StringComparison.Ordinal)
                    || m.Name is "SetParam" or "UpdateMusicParameter"))
            {
                h.Patch(m, prefix: new HarmonyMethod(typeof(NodelessPatches), nameof(SkipVoid)));
                Count++;
            }
        }

        // CreatureCmd.TriggerAnim: creature node lookup, character sfx, animation trigger and a wait
        // (none in Instant mode). With no node it only logs an error and returns; skipped instead.
        Prefix(h, typeof(CreatureCmd), nameof(CreatureCmd.TriggerAnim), nameof(SkipTask));

        // --- Bundle screen (Scroll Boxes) --------------------------------------------------------
        // CardSelectCmd.FromChooseABundleScreen shows NChooseABundleSelectionScreen (no selector hook)
        // and awaits its CardsSelected(), which awaits the screen's completion source and then calls
        // NOverlayStack.Instance.Remove(this) (unguarded). The screen is recorded so the driver answers
        // it exactly as on the full host (Driver.AnswerOpenScreens, bundle 0); CardsSelected returns
        // the same result without the overlay removal, and the never-shown node is freed.
        var bundle = Ui("Screens.CardSelection.NChooseABundleSelectionScreen");
        h.Patch(AccessTools.Method(bundle, "ShowScreen"), postfix: new HarmonyMethod(typeof(NodelessPatches), nameof(BundleShowPostfix)));
        Count++;
        Prefix(h, bundle, "CardsSelected", nameof(BundleSelectedPrefix));
        _bundleSource = AccessTools.Field(bundle, "_completionSource");

        // --- Game over -----------------------------------------------------------------------
        // CreatureCmd.Kill, when every player is dead: after CombatManager.LoseCombat it runs, under
        // TestMode.IsOff, NRun.Instance.RunMusicController.StopMusic(), audio calls,
        // RunManager.OnEnded(false) and NRun.Instance.ShowGameOverScreen(...). The first one throws
        // nodeless (NRun is null), which would cut the rest of the turn and the loss short. The
        // postfix wraps Kill's task: that one exception, and only it, is replaced by what the block
        // does besides music and the screen, the game's own RunManager.OnEnded(false); then Kill
        // returns to its caller as on the full host. "Only it" is checked at the throw: Kill's
        // MoveNext, IL offset inside that statement (IsExpectedGameOverException); any other null
        // dereference in Kill is a nodeless error.
        h.Patch(AccessTools.Method(typeof(CreatureCmd), nameof(CreatureCmd.Kill), new[] { typeof(IReadOnlyCollection<Creature>), typeof(bool) }),
            postfix: new HarmonyMethod(typeof(NodelessPatches), nameof(KillPostfix)));
        Count++;
        FindStopMusic();

        // --- Combat won ------------------------------------------------------------------------
        // CombatManager.EndCombatInternal raises CombatWon (after AfterCombatEnd, AfterCombatVictory and the save,
        // before CombatEnded). On the full host NCombatUi.OnCombatWon answers it, and that answer holds rules: for an
        // encounter that gives rewards it calls the room's own CombatRoom.OfferRoomEndRewards (RewardsSet
        // .GenerateWithoutOffering: each reward's Populate, Hook.ModifyRewards, Hook.AfterModifyingRewards; it draws
        // from the RNG, e.g. Wing Charm's Niche draw, and adds Amethyst Aubergine's gold reward, which flashes), else
        // RunManager.ProceedFromTerminalRewardsScreen. Nodeless there is no NCombatUi, so no room-end reward was ever
        // generated. The subscription makes the same calls at the same moment through the game's own methods.
        CombatManager.Instance.CombatWon += OnCombatWonNodeless;

        // --- Timing ------------------------------------------------------------------------
        // CombatStateTracker.CallCombatStateChangedDeferred waits one process frame on NRun before it
        // recalculates card values (enchantments such as Adroit) and raises CombatStateChanged. With
        // no NRun it would run at once, every time instead of once per frame. The wrapper waits the
        // same frame on the scene tree, then runs the game's own method.
        Prefix(h, typeof(CombatStateTracker), "CallCombatStateChangedDeferred", nameof(DeferredPrefix));
        _callDeferred = AccessTools.Method(typeof(CombatStateTracker), "CallCombatStateChangedDeferred");

        // --- Fallback detection (read-only) ----------------------------------------------------
        foreach (var t in typeof(AbstractModel).Assembly.GetTypes())
        {
            if (FallbackNames.Contains(t.Name) && typeof(AbstractModel).IsAssignableFrom(t))
            {
                _fallback.Add(t);
            }
        }

        if (_fallback.Count != FallbackNames.Length)
        {
            throw new InvalidOperationException("nodeless: fallback list names a type the game does not have: "
                + string.Join(",", FallbackNames.Except(_fallback.Select(t => t.Name))));
        }

        h.Patch(AccessTools.Method(typeof(AbstractModel), nameof(AbstractModel.MutableClone)),
            postfix: new HarmonyMethod(typeof(NodelessPatches), nameof(MutableClonePostfix)));
        Count++;

        // Diagnostics (driver code, read-only): the orphan-node count after each recorded step.
        // The state snapshot (RunState, Player), not Export.cs's Snapshot(CardModel): by name alone the lookup is ambiguous
        // and every nodeless server died at boot (AmbiguousMatchException) once that overload existed.
        h.Patch(AccessTools.Method(typeof(Driver), "Snapshot", new[] { typeof(RunState), typeof(Player) }), postfix: new HarmonyMethod(typeof(NodelessPatches), nameof(SnapshotPostfix)));
    }

    /// <summary>Room-end reward generations started by OnCombatWonNodeless in this process (diagnostics).</summary>
    internal static int RoomEndRewards { get; private set; }

    /// <summary>NCombatUi.OnCombatWon's rules, nodeless only (the full host has the node, which does it).</summary>
    private static void OnCombatWonNodeless(CombatRoom room)
    {
        if (!Nodeless.Active) return;
        RoomEndRewards++;
        TaskHelper.RunSafely(room.Encounter.ShouldGiveRewards ? OfferRoomEndRewardsNodeless(room) : ProceedWithoutRewardsNodeless());
    }

    /// <summary>
    /// NCombatUi.ShowRewards without its display: the same scaled wait (none in Instant mode, which the driver
    /// sets; its extra time for death animations still playing is 0 with no creature nodes), then the room's own
    /// OfferRoomEndRewards. Run inside the CombatWon event, as on the full host, so in Instant mode the rewards
    /// are generated before CombatEnded and the fight's last recorded step holds them on both hosts.
    /// </summary>
    private static async Task OfferRoomEndRewardsNodeless(CombatRoom room)
    {
        if (room.RoomType == RoomType.Boss)
        {
            await Cmd.CustomScaledWait(0f, 1f);
        }
        else
        {
            await Cmd.CustomScaledWait(0.5f, 1f);
        }

        await room.OfferRoomEndRewards();
    }

    /// <summary>NCombatUi.ProceedWithoutRewards: a one-second wait (none in Instant mode), then the screen's Proceed.</summary>
    private static async Task ProceedWithoutRewardsNodeless()
    {
        await Cmd.Wait(1f);
        await RunManager.Instance.ProceedFromTerminalRewardsScreen();
    }

    /// <summary>Orphan nodes after each step of the current fight (cost.orphans_by_step).</summary>
    internal static readonly List<long> OrphansByStep = new();

    private static void SnapshotPostfix(Player player)
    {
        if (Nodeless.Active)
        {
            OrphansByStep.Add((long)Performance.GetMonitor(Performance.Monitor.ObjectOrphanNodeCount));
            if (FallbackInState(player) is { } hit)
            {
                Nodeless.MarkFallback(hit + " (seen in the recorded state)");
            }
        }
    }

    /// <summary>
    /// The second fallback check, independent of the MutableClone postfix (a caller the JIT inlined
    /// before patching would bypass it): any fallback model among the combat's creatures, their
    /// powers, the player's cards in every pile and the deck, or the player's relics.
    /// </summary>
    internal static string? FallbackInState(Player player)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var models = new List<AbstractModel>();
        foreach (var c in combat?.Creatures ?? Enumerable.Empty<Creature>())
        {
            if (c.Monster != null) models.Add(c.Monster);
            models.AddRange(c.Powers);
        }

        models.AddRange(player.PlayerCombatState?.AllCards ?? Enumerable.Empty<CardModel>());
        models.AddRange(player.Deck.Cards);
        models.AddRange(player.Relics);
        return models.FirstOrDefault(m => _fallback.Contains(m.GetType()))?.GetType().Name;
    }

    private static Type Ui(string name) =>
        AccessTools.TypeByName("MegaCrit.Sts2.Core.Nodes." + name) ?? throw new InvalidOperationException("nodeless: no type " + name);

    private static void Prefix(Harmony h, Type type, string method, string prefix)
    {
        var target = AccessTools.Method(type, method) ?? throw new InvalidOperationException($"nodeless: no method {type.Name}.{method}");
        h.Patch(target, prefix: new HarmonyMethod(typeof(NodelessPatches), prefix));
        Count++;
    }

    /// <summary>
    /// Finds every static method in the game's Nodes namespaces whose IL begins
    /// `call TestMode.get_IsOn; brfalse; ldnull; ret` and returns a reference type, and nulls it.
    /// The names go to the game log once.
    /// </summary>
    private static int PatchTestModeFactories(Harmony h)
    {
        var isOn = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOn));
        var prefix = new HarmonyMethod(typeof(NodelessPatches), nameof(ReturnNull));
        var names = new List<string>();
        foreach (var t in typeof(AbstractModel).Assembly.GetTypes())
        {
            string ns = t.Namespace ?? "";
            if (!ns.StartsWith("MegaCrit.Sts2.Core.Nodes", StringComparison.OrdinalIgnoreCase)
                && !ns.StartsWith("MegaCrit.sts2.Core.Nodes", StringComparison.Ordinal))
            {
                continue;
            }

            foreach (var m in t.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (m.ReturnType.IsValueType || m.ReturnType == typeof(void) || m.IsGenericMethodDefinition
                    || m.GetMethodBody()?.GetILAsByteArray() is not { } raw || !TestModeNullPrologue(raw))
                {
                    continue;  // the raw prologue first: decoding every method's IL cost most of a nodeless boot's patch time
                }

                List<CodeInstruction> il;
                try
                {
                    il = PatchProcessor.GetOriginalInstructions(m);
                }
                catch (Exception)
                {
                    continue;
                }

                if (il.Count >= 4 && il[0].opcode == OpCodes.Call && Equals(il[0].operand, isOn)
                    && (il[1].opcode == OpCodes.Brfalse || il[1].opcode == OpCodes.Brfalse_S)
                    && il[2].opcode == OpCodes.Ldnull && il[3].opcode == OpCodes.Ret)
                {
                    h.Patch(m, prefix: prefix);
                    names.Add(t.Name + "." + m.Name);
                }
            }
        }

        GD.Print($"[referee] nodeless: {names.Count} test-mode-null node factories: " + string.Join(", ", names.Distinct()));
        return names.Count;
    }

    /// <summary>
    /// Whether raw IL can start `call &lt;token&gt;; brfalse(.s) &lt;target&gt;; ldnull; ret` (0x28 + 4 bytes, then 0x2C + 1 byte
    /// or 0x39 + 4 bytes, 0x14, 0x2A). A filter only: the decoded instructions and the call's target are checked after it.
    /// </summary>
    private static bool TestModeNullPrologue(byte[] il) =>
        il.Length >= 9 && il[0] == 0x28
        && ((il[5] == 0x2C && il[7] == 0x14 && il[8] == 0x2A)
            || (il.Length >= 12 && il[5] == 0x39 && il[10] == 0x14 && il[11] == 0x2A));

    private static bool SkipVoid() => !Nodeless.Active;

    private static bool SkipTask(ref Task __result)
    {
        if (!Nodeless.Active) return true;
        __result = Task.CompletedTask;
        return false;
    }

    private static bool ReturnNull(ref object? __result)
    {
        if (!Nodeless.Active) return true;
        __result = null;
        return false;
    }

    private static bool MapRoomEnterPrefix(bool isRestoringRoomStackBase, ref Task __result)
    {
        if (!Nodeless.Active || isRestoringRoomStackBase) return true;
        __result = Task.CompletedTask;
        return false;
    }

    // --- Bundle screen ---------------------------------------------------------------------

    /// <summary>Bundle screens shown nodeless and not yet answered (never in a scene tree).</summary>
    internal static readonly List<Node> Screens = new();

    private static FieldInfo? _bundleSource;

    private static void BundleShowPostfix(Node __result)
    {
        if (Nodeless.Active && __result != null)
        {
            Screens.Add(__result);
        }
    }

    private static bool BundleSelectedPrefix(Node __instance, ref Task<IEnumerable<IReadOnlyList<CardModel>>> __result)
    {
        if (!Nodeless.Active) return true;
        __result = BundleSelected(__instance);
        return false;
    }

    private static async Task<IEnumerable<IReadOnlyList<CardModel>>> BundleSelected(Node screen)
    {
        var source = (TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>>)_bundleSource!.GetValue(screen)!;
        try
        {
            return await source.Task;
        }
        finally
        {
            Screens.Remove(screen);
            screen.QueueFree();
        }
    }

    // --- Game over -------------------------------------------------------------------------

    private static void KillPostfix(ref Task __result)
    {
        if (Nodeless.Active)
        {
            __result = GuardGameOver(__result);
        }
    }

    internal static int GameOvers;

    private static async Task GuardGameOver(Task kill)
    {
        try
        {
            await kill;
        }
        catch (NullReferenceException e) when (Nodeless.Active && _gameOverNre.TryGetValue(e, out _))
        {
            GameOvers++;
            RunManager.Instance.OnEnded(isVictory: false);
        }
    }

    private static bool AllPlayersDead() =>
        RunManager.Instance.DebugOnlyGetState() is { } run && run.Players.All(p => p.Creature.IsDead);

    /// <summary>The null dereferences matched at their throw (IsExpectedGameOverException); only these are caught.</summary>
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Exception, object> _gameOverNre = new();

    private static readonly object _marker = new();

    /// <summary>Kill(IReadOnlyCollection&lt;Creature&gt;, bool)'s state machine MoveNext, where the game-over block lives.</summary>
    private static MethodBase? _killMoveNext;

    /// <summary>
    /// IL offsets in _killMoveNext of the one statement NRun.Instance.RunMusicController.StopMusic():
    /// from the call to NRun.get_Instance to the call to StopMusic. Found by reading the method's IL
    /// (FindStopMusic), so a game update that moves or drops the statement turns the guard off and the
    /// dereference shows as a nodeless error.
    /// </summary>
    private static (int From, int To)? _stopMusicIl;

    private static bool _loggedGameOverIl;

    /// <summary>
    /// For the first-chance check, and the only thing the guard catches: a null dereference thrown in
    /// Kill's MoveNext at the StopMusic statement, while NRun.Instance is null and every player is dead.
    /// The first game frame on the stack is the throwing one; its IL offset must lie in the statement
    /// (an inlined callee reports its call site's offset, which is outside it). The exception is
    /// remembered, so its rethrow by GuardGameOver's await and the guard's catch know it.
    /// </summary>
    internal static bool IsExpectedGameOverException(Exception e)
    {
        if (e is not NullReferenceException)
        {
            return false;
        }

        if (_gameOverNre.TryGetValue(e, out _))
        {
            return true;
        }

        if (_stopMusicIl is not { } il || _killMoveNext == null || MegaCrit.Sts2.Core.Nodes.NRun.Instance != null || !AllPlayersDead())
        {
            return false;
        }

        foreach (var f in new System.Diagnostics.StackTrace(1, false).GetFrames())
        {
            var m = f.GetMethod();
            string ns = m?.DeclaringType?.Namespace ?? "";
            if (m == null || ns.StartsWith("System", StringComparison.Ordinal) || m.DeclaringType == typeof(Nodeless)
                || m.DeclaringType == typeof(NodelessPatches))
            {
                continue;  // runtime frames and the first-chance handler itself
            }

            int at = f.GetILOffset();
            bool match = m.MetadataToken == _killMoveNext.MetadataToken && m.Module == _killMoveNext.Module
                && at >= il.From && at <= il.To;
            if (!_loggedGameOverIl)
            {
                _loggedGameOverIl = true;
                GD.Print($"[referee] nodeless: game-over dereference in {m.DeclaringType?.Name}.{m.Name} at IL {at}, StopMusic statement IL {il.From}..{il.To}: {(match ? "guarded" : "not the statement")}");
            }

            if (match)
            {
                _gameOverNre.Add(e, _marker);
            }

            return match;
        }

        return false;
    }

    /// <summary>Finds Kill's MoveNext and the IL range of its NRun.Instance.RunMusicController.StopMusic() statement.</summary>
    private static void FindStopMusic()
    {
        var kill = AccessTools.Method(typeof(CreatureCmd), nameof(CreatureCmd.Kill), new[] { typeof(IReadOnlyCollection<Creature>), typeof(bool) });
        var sm = kill.GetCustomAttribute<System.Runtime.CompilerServices.AsyncStateMachineAttribute>()?.StateMachineType;
        var moveNext = sm?.GetMethod("MoveNext", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        var nrun = Ui("NRun");
        var getInstance = AccessTools.PropertyGetter(nrun, "Instance");
        var getMusic = AccessTools.PropertyGetter(nrun, "RunMusicController");
        var stop = AccessTools.Method(Ui("Audio.NRunMusicController"), "StopMusic");
        if (moveNext == null || getInstance == null || getMusic == null || stop == null)
        {
            GD.Print("[referee] nodeless: game-over guard off (Kill's MoveNext or the StopMusic statement not found)");
            return;
        }

        var calls = IlCalls(moveNext);
        for (int i = 0; i + 2 < calls.Count; i++)
        {
            if (calls[i].Method == getInstance && calls[i + 1].Method == getMusic && calls[i + 2].Method == stop)
            {
                _killMoveNext = moveNext;
                _stopMusicIl = (calls[i].Offset, calls[i + 2].Offset);
                GD.Print($"[referee] nodeless: game-over guard on Kill IL {calls[i].Offset}..{calls[i + 2].Offset}");
                return;
            }
        }

        GD.Print("[referee] nodeless: game-over guard off (no NRun.Instance.RunMusicController.StopMusic() in Kill)");
    }

    /// <summary>Every call/callvirt in a method body with its IL offset and resolved target, in order.</summary>
    internal static List<(int Offset, MethodBase? Method)> IlCalls(MethodBase method)
    {
        var il = method.GetMethodBody()?.GetILAsByteArray() ?? Array.Empty<byte>();
        var one = new OpCode[0x100];
        var two = new OpCode[0x100];
        foreach (var f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            var op = (OpCode)f.GetValue(null)!;
            ushort v = (ushort)op.Value;
            if (op.Size == 1) one[v & 0xff] = op;
            else two[v & 0xff] = op;
        }

        var calls = new List<(int, MethodBase?)>();
        var typeArgs = method.DeclaringType?.IsGenericType == true ? method.DeclaringType.GetGenericArguments() : null;
        int pos = 0;
        while (pos < il.Length)
        {
            int start = pos;
            OpCode op = il[pos] == 0xFE ? two[il[++pos]] : one[il[pos]];
            pos++;
            int size = op.OperandType switch
            {
                OperandType.InlineNone => 0,
                OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
                OperandType.InlineVar => 2,
                OperandType.InlineI8 or OperandType.InlineR => 8,
                OperandType.InlineSwitch => 4 + 4 * BitConverter.ToInt32(il, pos),
                _ => 4,
            };
            if (op == OpCodes.Call || op == OpCodes.Callvirt)
            {
                MethodBase? target = null;
                try
                {
                    target = method.Module.ResolveMethod(BitConverter.ToInt32(il, pos), typeArgs, null);
                }
                catch (ArgumentException)
                {
                    // a token this module cannot resolve without generic context: not one we look for
                }

                calls.Add((start, target));
            }

            pos += size;
        }

        return calls;
    }

    // --- Deferred CombatStateChanged -------------------------------------------------------

    private static MethodInfo? _callDeferred;
    private static bool _inDeferred;

    private static bool DeferredPrefix(CombatStateTracker __instance, ref Task __result)
    {
        if (!Nodeless.Active || _inDeferred) return true;
        __result = Deferred(__instance);
        return false;
    }

    private static async Task Deferred(CombatStateTracker tracker)
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        _inDeferred = true;
        Task inner;
        try
        {
            inner = (Task)_callDeferred!.Invoke(tracker, null)!;  // no NRun: runs to the end at once
        }
        finally
        {
            _inDeferred = false;
        }

        await inner;
    }

    // --- Fallback detection ----------------------------------------------------------------

    private static void MutableClonePostfix(AbstractModel __result)
    {
        if (Nodeless.Active && _fallback.Contains(__result.GetType()))
        {
            Nodeless.MarkFallback(__result.GetType().Name);
        }
    }

    /// <summary>A spec whose added deck cards include a fallback model, or null.</summary>
    internal static string? FallbackInSpec(JsonObject spec)
    {
        foreach (var id in spec["deck_add"]?.AsArray() ?? new JsonArray())
        {
            var card = ModelDb.AllCards.FirstOrDefault(c => c.Id.Entry == (string?)id);
            if (card != null && _fallback.Contains(card.GetType()))
            {
                return "deck_add " + card.GetType().Name;
            }
        }

        return null;
    }
}
