// Takes every reward on the game's rewards screens, as a player clicking each one would, through
// the game's own RewardsSetSynchronizer. Card rewards resolve through FirstChoiceSelector.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;

    /// <summary>
    /// A card reward shows NCardRewardSelectionScreen and awaits its completion source; a click on
    /// a card sets it to that card's index. Set it to 0, the first card, as FirstChoiceSelector does.
    /// </summary>
    private static void AnswerCardRewardScreen(Reward reward, JsonArray log)
    {
        if (reward is not CardReward) return;
        var screen = reward.GetType().GetField("_currentlyShownScreen", Private)?.GetValue(reward);
        if (screen == null) return;
        var source = screen.GetType().GetField("_completionSource", Private)?.GetValue(screen) as TaskCompletionSource<int?>;
        if (source == null || source.Task.IsCompleted) return;
        var cards = (IEnumerable)reward.GetType().GetField("_cards", Private)!.GetValue(reward)!;
        log.Add(new JsonObject
        {
            ["kind"] = "card_reward",
            ["options"] = new JsonArray(cards.Cast<MegaCrit.Sts2.Core.Entities.Cards.CardCreationResult>()
                .Select(c => (JsonNode)c.Card.Id.Entry).ToArray()),
            ["picked"] = 0,
        });
        source.SetResult(0);
    }

    /// <summary>
    /// Screens that await a click outside the selector hook, answered as the first option:
    /// NChooseABundleSelectionScreen (Scroll Boxes) completes with its first bundle, as clicking
    /// it and Confirm would. Called every frame while the driver waits.
    /// </summary>
    internal static void AnswerOpenScreens()
    {
        var stack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
        // Nodeless there is no overlay stack: the screens shown are recorded instead (NodelessPatches).
        var open = stack?.GetChildren().ToList() ?? NodelessPatches.Screens.ToList();
        foreach (var node in open)
        {
            if (node.GetType().Name != "NChooseABundleSelectionScreen") continue;
            var src = node.GetType().GetField("_completionSource", Private)?.GetValue(node)
                as TaskCompletionSource<IEnumerable<IReadOnlyList<MegaCrit.Sts2.Core.Models.CardModel>>>;
            var bundles = node.GetType().GetField("_bundles", Private)?.GetValue(node)
                as IReadOnlyList<IReadOnlyList<MegaCrit.Sts2.Core.Models.CardModel>>;
            if (src == null || bundles == null || bundles.Count == 0 || src.Task.IsCompleted) continue;
            int bi = Env.Decide("bundle", bundles.Count,
                () => new JsonArray(bundles.Select(b => (JsonNode)new JsonArray(b.Select(c => (JsonNode)c.Id.Entry).ToArray())).ToArray()), () => 0);
            ScreenLog?.Add(new JsonObject
            {
                ["kind"] = "bundle", ["step"] = CurrentStep, ["picked"] = bi,
                ["options"] = new JsonArray(bundles.Select(b => (JsonNode)new JsonArray(b.Select(c => (JsonNode)c.Id.Entry).ToArray())).ToArray()),
            });
            src.SetResult(new[] { bundles[bi] });
        }
    }

    /// <summary>Where AnswerOpenScreens logs, set per fight.</summary>
    internal static JsonArray? ScreenLog;

    private static async Task ClaimRewards(JsonArray log)
    {
        for (int guard = 0; guard < 20 && TopRewardsSet() is { } top; guard++)
        {
            await ClaimSet(top, log, depth: 0);
        }
    }

    /// <summary>
    /// The local player's stack of open rewards screens, bottom first
    /// (RewardsSetSynchronizer._rewardStates[slot].rewardsStack; single player, so the only slot).
    /// RewardsSetSynchronizer.BeginRewardsSet pushes a set, CompleteRewardsSet removes it.
    /// </summary>
    private static IList RewardsStack()
    {
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var states = (IList)sync.GetType().GetField("_rewardStates", Private)!.GetValue(sync)!;
        return states.Count == 0 ? Array.Empty<object>() : (IList)states[0]!.GetType().GetField("rewardsStack")!.GetValue(states[0])!;
    }

    /// <summary>The rewards screen on top of the local player's stack, or null.</summary>
    private static object? TopRewardsSet()
    {
        IList stack = RewardsStack();
        return stack.Count > 0 ? stack[stack.Count - 1] : null;
    }

    /// <summary>
    /// The screen opened from inside <paramref name="top" />: the top of the stack when it sits above
    /// <paramref name="top" />. Null when nothing is above it, and also when <paramref name="top" /> is
    /// no longer on the stack: its last reward completed it (CompleteRewardsSet pops it before that
    /// reward's SelectLocalReward task finishes), so what is on top now is its parent, not a child.
    /// SelectLocalReward always selects into the top of the stack, so only the real top may be claimed.
    /// </summary>
    private static object? NestedAbove(object top)
    {
        IList stack = RewardsStack();
        int at = stack.IndexOf(top);
        return at >= 0 && at < stack.Count - 1 ? stack[stack.Count - 1] : null;
    }

    /// <summary>
    /// Takes every reward on one screen. A reward can open a screen of its own while it is
    /// being taken (Neow's Bones offers two relics from inside its relic reward), so while a
    /// reward is pending, a new screen on top of the stack is claimed first, recursively.
    /// </summary>
    private static async Task ClaimSet(object top, JsonArray log, int depth)
    {
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var set = (RewardsSet)top.GetType().GetField("set")!.GetValue(top)!;
        log.Add(new JsonObject
        {
            ["kind"] = "rewards", ["depth"] = depth,
            ["rewards"] = new JsonArray(set.Rewards.Select(r => (JsonNode)r.GetType().Name).ToArray()),
        });
        var tree = (Godot.SceneTree)Godot.Engine.GetMainLoop();
        foreach (var reward in set.Rewards.Where(r => !r.SuccessfullySelected).ToList())
        {
            if (reward.SuccessfullySelected) continue;
            if (!RewardsStack().Contains(top)) break;  // the set is complete: nothing left to select into it
            Task task = sync.SelectLocalReward(reward);
            var budget = new Budget(60);
            while (!task.IsCompleted)
            {
                if (budget.Spent)
                    throw new TimeoutException("SelectLocalReward " + reward.GetType().Name + " did not finish within 60 s");
                AnswerCardRewardScreen(reward, log);
                // A reward that opens a screen outside the selector hook while it is taken (Neow's Bones' relic
                // reward handing over Scroll Boxes, whose choose-a-bundle screen opens at once) waits on it here.
                AnswerOpenScreens();
                if (NestedAbove(top) is { } nested && depth < 5)
                {
                    await ClaimSet(nested, log, depth + 1);
                    continue;
                }

                await tree.ToSignal(tree, Godot.SceneTree.SignalName.ProcessFrame);
            }

            await task;
            await Idle();
        }

        if (ReferenceEquals(TopRewardsSet(), top))
        {
            sync.SkipLocalRewardsSet();  // leave the screen, as Proceed does
            await Idle();
        }
    }
}
