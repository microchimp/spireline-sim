// Potions outside a fight (agent channel only): use (AnyTime potions) and discard, offered with the screens of the map, rewards,
// shops, events and rest sites (Env.DecideWithPotions). The answer is carried out here, through the same actions the potion popup's
// buttons enqueue, and the same decision is asked again with the new belt.

using System;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

public static partial class Driver
{
    /// <summary>Env.Decide with the outside-fight potion actions: an answer past the screen's n is a potion action, done, then asked again.</summary>
    internal static async Task<int> DecideP(string kind, int n, Func<JsonArray> actions, Func<int> fallback, Func<JsonNode?>? ctx = null)
    {
        for (int guard = 0; guard < 40; guard++)
        {
            int a = Env.DecideWithPotions(kind, n, actions, fallback, ctx);
            if (a < n) return a;
            var (act, slot, potion) = Env.PotionActs[a - n];
            await OutsidePotion(act, slot, potion);
        }

        return fallback();  // an agent that only cycles its belt: the seeded pick ends it
    }

    private static async Task OutsidePotion(string act, int slot, PotionModel potion)
    {
        var player = Env.Run!.Players[0];
        if (act == "use") potion.EnqueueManualUse(null);
        else RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(player, (uint)slot, false));
        await Idle();
    }
}
