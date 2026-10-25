// The pick loop of Env.DecideMany, free of the game and of the agent channel so a test can drive it with a fake agent.
//
// A card screen with several picks is asked one pick at a time, each from what is left of the offer. The game's own
// screens hand the selection back in the order the player clicked: NSimpleCardSelectScreen (the run-layer grids through
// CardSelectCmd.FromSimpleGridForRewards, and the combat grids through FromSimpleGrid / FromCombatPile) completes with its
// selection set filled in click order, and NPlayerHand (CardSelectCmd.FromHand) with its click-ordered list. Nothing in
// CardSelectCmd re-sorts the answer. The emulator takes each pick as it is made. So the driver hands the cards back in
// pick order at every site; sorting them into grid order, as this loop once did outside full select, made the game add
// the two cards of a Room Full of Cheese Gorge to the deck in grid order against the emulator's pick order (F11 mechanism A).

using System;
using System.Collections.Generic;
using System.Linq;

namespace Spireline.Referee;

internal static class PickLoop
{
    /// <summary>
    /// The picks of one screen, as indices into the offer, in the order the agent made them.
    /// `ask` gets what is left of the offer (indices), whether the last action is "none"/"done", and the pick number (from 0),
    /// and returns an index into `left`, or `left.Count` for "none"/"done".
    /// </summary>
    internal static List<int> Run(int offerCount, int cap, bool full, int minSelect, bool allowNone,
        Func<IReadOnlyList<int>, bool, int, int> ask)
    {
        var left = Enumerable.Range(0, offerCount).ToList();
        var picked = new List<int>();
        for (int k = 0; k < cap && left.Count > 0; k++)
        {
            var now = left.ToList();
            bool none = full ? k >= minSelect : allowNone && k == 0;  // "done" (full) or "none" (the screen allows choosing 0 cards) is the last action
            int a = ask(now, none, k);
            if (a >= now.Count) break;
            picked.Add(now[a]);
            left.Remove(now[a]);
        }

        return picked;
    }
}
