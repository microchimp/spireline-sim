// Which targets an action is offered at, as a rule over two numbers, so it is tested without the game
// (referee/driver.Tests, ActionTargetsTests).
//
// The fight loop of RunPlay.cs offers each legal play and each usable potion at every target it can take. An action that
// aims at an enemy needs a living one: the game refuses a potion handed no target (UsePotionAction cancels it, the potion
// stays in its slot and nothing else changes) and a card handed none (CardModel.TryManualPlay returns false). When no
// creature is alive there are no such actions to offer. The Test Subject has no living creature between its forms (it falls
// to nothing, stands up on its own next move); the loop offered every enemy-target potion once, at target -1, the game
// ignored the pick, the stand-in answered the same screen again, and the run ended on the 400-action cap as
// `fight_unfinished` (14 of 96 Test Subject arrivals of the 2026-11-19 real block).
//
// A target of -1 means "no target to choose": an action that aims at nothing (Self, the player, all enemies, random).

using System.Collections.Generic;
using System.Linq;

namespace Spireline.Referee;

internal static class ActionTargets
{
    private static readonly IReadOnlyList<int> NoChoice = new[] { -1 };

    /// <summary>
    /// The targets an action is offered at. An action that aims at an enemy: one per living enemy (indices into the list of
    /// living enemies), and none at all when no enemy lives. Any other action: the single target -1.
    /// </summary>
    internal static IReadOnlyList<int> For(bool aimsAtEnemy, int livingEnemies) =>
        aimsAtEnemy ? Enumerable.Range(0, livingEnemies).ToList() : NoChoice;
}
