// How a fight picks its actions, and which relics fired. Spec "chooser": "random" (the default: a seeded
// random legal action, for coverage fights) or "heuristic" (realistic fights: a deterministic, plain
// player's policy, so a fight lasts as long as a real one rather than ending in a random-play death).
// Spec "relic_flashes": true records every relic flash (RelicModel.Flashed, the game's own "this relic
// did something" signal) as [step, relic] in the result's "relic_flashes": the evidence that a relic's
// effect fired (coverage, regression triggers). Both only read the state; neither changes a rule.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace Spireline.Referee;

public static partial class Driver
{
    private static (int round, CardModel? card, int energy, int handCount) _lastHeuristic;
    private static readonly HashSet<CardModel> _refusedThisRound = new();
    private static int _refusedRound = -1;

    /// <summary>
    /// The heuristic chooser, one action from the legal options (as Combat.cs lists them: (-1, -1) ends the
    /// turn, (-2 - slot, target) uses a potion, (hand, target) plays a card):
    ///   1. at or below half HP, drink the first usable potion (aimed at the lowest-HP enemy);
    ///   2. else play a card, never ending the turn while one is playable: a card that gains block when an
    ///      enemy intends to attack, else an attack or a power, else any other card; an attack is aimed at the
    ///      living enemy with the lowest HP (the first such enemy on a tie);
    ///   3. else end the turn.
    /// Ties within a rank go to the seeded rng, so a fight is still one seed's. A play the game refused (the
    /// same card offered again in the same state) is not offered again that round.
    /// </summary>
    internal static (int hand, int target) ChooseHeuristic(List<(int hand, int target)> options, IReadOnlyList<CardModel> hand,
        List<Creature> enemies, CombatState combat, Player player, Random rng)
    {
        int round = combat.RoundNumber;
        int energy = player.PlayerCombatState?.Energy ?? 0;
        if (_refusedRound != round)
        {
            _refusedThisRound.Clear();
            _refusedRound = round;
        }

        if (_lastHeuristic.card != null && _lastHeuristic.round == round && _lastHeuristic.energy == energy
            && _lastHeuristic.handCount == hand.Count && hand.Contains(_lastHeuristic.card))
        {
            _refusedThisRound.Add(_lastHeuristic.card);  // nothing changed since it was chosen: the play was refused
        }

        int LowestHp(IEnumerable<int> targets) => targets.OrderBy(t => t < 0 || t >= enemies.Count ? int.MaxValue : enemies[t].CurrentHp)
            .ThenBy(t => t).First();

        (int, int) Pick(List<(int hand, int target)> group)
        {
            var byFirst = group.GroupBy(o => o.hand).ToList();
            var chosen = byFirst[rng.Next(byFirst.Count)];
            return (chosen.Key, LowestHp(chosen.Select(o => o.target)));
        }

        _lastHeuristic = (round, null, energy, hand.Count);
        var potions = options.Where(o => o.hand <= -2).ToList();
        if (potions.Count > 0 && player.Creature.CurrentHp * 2 <= player.Creature.MaxHp)
        {
            int first = potions.Max(o => o.hand);  // -2 - slot: the highest is the lowest slot
            return (first, LowestHp(potions.Where(o => o.hand == first).Select(o => o.target)));
        }

        var cards = options.Where(o => o.hand >= 0 && o.hand < hand.Count && !_refusedThisRound.Contains(hand[o.hand])).ToList();
        if (cards.Count == 0) return (-1, -1);
        bool attacked = enemies.Any(e => e.IsAlive && (e.Monster?.NextMove?.Intents?.Any(i => i is AttackIntent) ?? false));
        int Rank((int hand, int target) o)
        {
            var c = hand[o.hand];
            if (attacked && c.GainsBlock) return 3;
            return c.Type is CardType.Attack or CardType.Power ? 2 : 1;
        }

        int best = cards.Max(Rank);
        var (h, t) = Pick(cards.Where(o => Rank(o) == best).ToList());
        _lastHeuristic = (round, hand[h], energy, hand.Count);
        return (h, t);
    }

    private static readonly List<(RelicModel relic, Action<RelicModel, IEnumerable<Creature>> handler)> _flashHandlers = new();

    /// <summary>Starts recording the flashes of the relics the player holds into <paramref name="log" /> ([step, relic]).</summary>
    internal static void RecordRelicFlashes(Player player, JsonArray log)
    {
        StopRelicFlashes();
        foreach (var relic in player.Relics)
        {
            Action<RelicModel, IEnumerable<Creature>> h = (r, _) => log.Add(new JsonArray(CurrentStep, r.Id.Entry));
            relic.Flashed += h;
            _flashHandlers.Add((relic, h));
        }
    }

    internal static void StopRelicFlashes()
    {
        foreach (var (relic, h) in _flashHandlers) relic.Flashed -= h;
        _flashHandlers.Clear();
    }

    /// <summary>Resets the heuristic's per-fight memory.</summary>
    internal static void ResetChooser()
    {
        _lastHeuristic = default;
        _refusedThisRound.Clear();
        _refusedRound = -1;
    }
}
