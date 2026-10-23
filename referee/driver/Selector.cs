// Answers the game's card-choice screens without UI, through the hook its own autoplayer and test
// mode use (CardSelectCmd.UseSelector). Deterministic: always the first cards offered, and every
// choice is logged so the emulator can be given the same one.
//
// When it answers. With a selector installed, CardSelectCmd asks it and goes on with the answer at
// once, without signalling a player choice. A player using the UI gets the other branch:
// SignalPlayerChoiceBegun, then the screen, then SignalPlayerChoiceEnded. In a card play or a potion
// (GameActionPlayerChoiceContext) the two give the same order, since the paused action is the front
// of the only queue. In the player's turn-start setup they do not: SetupPlayerTurn runs under a
// HookPlayerChoiceContext, and CombatManager.StartTurn awaits it only until it completes OR pauses
// (WaitForPauseOrCompletionWithoutAssigningTask). A pause lets StartTurn run on through
// Hook.AfterSideTurnStart, the orbs' AfterTurnStart and the win check; the screen's hook action is
// enqueued when RunAutoPrePlayPhase assigns the context its task, and the screen shows once that
// action starts executing in the play phase. So a player picks after all of it. Answered at once,
// the pick came first and recordings disagreed with real play (owner decision, 2026-10-14).
//
// So a screen raised in the turn-start setup (a HookPlayerChoiceContext with no source model, for a
// player in PlayerTurnPhase.Start) is answered the way the UI answers it: SignalPlayerChoiceBegun,
// which returns once nothing but the open screen is left (the hook action is executing, every
// action queued before it has run, StartTurn has moved on), then the pick, from the pile as it
// stands then (the UI's hand and pile screens show the live pile, not the list counted before the
// pause), then SignalPlayerChoiceEnded. Job key "pick_timing": "ui" (default) or "instant" (the
// old answer at once, for comparison).
//
// Edge answers. Spec key "choice_edges" (absent: off, and every output is as it always was) makes the
// answers probe the counts and the eligibility of a screen rather than take its first card: "none" (the
// fewest the screen allows), "one", "max" (the most it takes), "last" (one card, the last eligible) and
// "unusual" (one card, the least ordinary: enchanted, upgraded, a curse or status, a rare); "cycle" walks
// through the five, one per screen, starting where the spec's chooser_seed says. The emulator is told what
// the game was given (the log's "picked" indices, and "min"/"max" for the decline), so a difference in how
// many a screen takes, or which cards it offers, shows as a divergence. Each such entry says which edge it
// is ("edge"). Cards are told apart for the emulator by id only, so a copy that is not the first of its id
// in the offer is picked only when nothing else is left to pick.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace Spireline.Referee;

public sealed class FirstChoiceSelector : ICardSelector
{
    /// <summary>"ui" (default): turn-start screens are answered where the UI would show them; "instant": at once.</summary>
    internal static string PickTiming = "ui";

    /// <summary>Set inside a whole run with an agent channel or a trace (Env.cs): which of the list to pick, ascending.</summary>
    internal static Func<List<CardModel>, int, bool, List<int>>? PickHook;  // (options, n, the screen allows choosing 0)

    /// <summary>As PickHook, for a card reward offered inside a fight.</summary>
    internal static Func<IReadOnlyList<CardCreationResult>, int>? CardRewardHook;

    public JsonArray Log { get; } = new();

    public FirstChoiceSelector() => CaptureChoiceContexts();

    private static readonly string[] EdgeCycle = { "none", "one", "max", "last", "unusual" };
    private string? _edges;
    private int _edgeSeed;
    private int _edgeScreens;

    /// <summary>The spec's "choice_edges" (null: off), with the seed that sets where "cycle" starts.</summary>
    public void ConfigureEdges(string? mode, int seed)
    {
        _edges = string.IsNullOrEmpty(mode) ? null : mode;
        _edgeSeed = Math.Abs(seed % EdgeCycle.Length);
        _edgeScreens = 0;
    }

    /// <summary>The edge for the next screen, or null when edge answers are off.</summary>
    private string? NextEdge()
    {
        if (_edges == null) return null;
        string edge = _edges == "cycle" || _edges == "true" ? EdgeCycle[(_edgeSeed + _edgeScreens) % EdgeCycle.Length] : _edges;
        _edgeScreens++;
        return Array.IndexOf(EdgeCycle, edge) >= 0 ? edge : "one";
    }

    /// <summary>How unusual a card is to pick: enchanted first, then a curse or status, a rare, an upgraded one.</summary>
    private static int Unusual(CardModel c)
    {
        int s = c.Enchantment != null ? 8 : 0;
        s += c.Type is CardType.Curse or CardType.Status ? 5 : c.Type == CardType.Power ? 2 : 0;
        s += c.Rarity switch { CardRarity.Rare or CardRarity.Ancient => 3, CardRarity.Uncommon => 1, CardRarity.Basic => -1, _ => 0 };
        return s + (c.IsUpgraded ? 2 : 0);
    }

    /// <summary>The indices an edge answer takes, ascending: how many (by the screen's min and max) and which.</summary>
    internal static List<int> EdgeIndices(string edge, List<CardModel> list, int minSelect, int maxSelect)
    {
        int cap = Math.Min(list.Count, Math.Max(maxSelect, 0));
        int lo = Math.Min(Math.Max(minSelect, 0), cap);
        int n = edge switch { "none" => lo, "max" => cap, _ => Math.Min(Math.Max(1, lo), cap) };
        IEnumerable<int> order = edge switch
        {
            "last" => Enumerable.Range(0, list.Count).Reverse(),
            "unusual" => Enumerable.Range(0, list.Count).OrderByDescending(i => Unusual(list[i])).ThenByDescending(i => i),
            _ => Enumerable.Range(0, list.Count),
        };
        var ordered = order.ToList();
        bool First(int i) => list.FindIndex(c => c.Id.Entry == list[i].Id.Entry) == i;
        return ordered.Where(First).Concat(ordered.Where(i => !First(i))).Take(n).OrderBy(i => i).ToList();
    }

    /// <summary>The CardSelectCmd call now asking the selector: its context, the UI's choice options, and how to read the live pile.</summary>
    private sealed record Asking(PlayerChoiceContext Context, PlayerChoiceOptions Options, Func<List<CardModel>>? Live);

    [ThreadStatic]
    private static Asking? _asking;

    private static bool _captured;

    /// <summary>
    /// Harmony prefixes on the CardSelectCmd entry points that take a PlayerChoiceContext: each notes
    /// its call for the selector it is about to ask (the selector's interface carries no context).
    /// </summary>
    private static void CaptureChoiceContexts()
    {
        if (_captured)
        {
            return;
        }

        _captured = true;
        var h = new Harmony("spireline.referee.choice-context");
        foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static))
        {
            var ps = m.GetParameters();
            if (!m.Name.StartsWith("From", StringComparison.Ordinal) || ps.Length == 0 || ps[0].ParameterType != typeof(PlayerChoiceContext))
            {
                continue;
            }

            string? prefix = m.Name switch
            {
                nameof(CardSelectCmd.FromHand) => nameof(HandPrefix),
                nameof(CardSelectCmd.FromHandForDiscard) => nameof(HandPrefix),
                nameof(CardSelectCmd.FromHandForUpgrade) => nameof(UpgradePrefix),
                nameof(CardSelectCmd.FromCombatPile) when ps.Length == 5 => nameof(PilePrefix),
                nameof(CardSelectCmd.FromCombatPile) => null,  // the 4-argument overload calls the 5-argument one
                _ => nameof(OtherPrefix),
            };
            if (prefix != null)
            {
                h.Patch(m, prefix: new HarmonyMethod(typeof(FirstChoiceSelector), prefix));
            }
        }
    }

    // FromHand(context, player, prefs, filter, source): the UI selects in the live hand with the filter.
    private static void HandPrefix(PlayerChoiceContext context, Player player, Func<CardModel, bool>? filter) =>
        _asking = new(context, PlayerChoiceOptions.CancelPlayCardActions,
            () => PileType.Hand.GetPile(player).Cards.Where(filter ?? (_ => true)).ToList());

    // FromHandForUpgrade(context, player, source): the upgradable cards in the live hand.
    private static void UpgradePrefix(PlayerChoiceContext context, Player player) =>
        _asking = new(context, PlayerChoiceOptions.CancelPlayCardActions,
            () => PileType.Hand.GetPile(player).Cards.Where(c => c.IsUpgradable).ToList());

    // FromCombatPile(context, pile, player, prefs, filter): the UI's grid is built over the live pile.
    private static void PilePrefix(PlayerChoiceContext context, CardPile pile, Func<CardModel, bool> filter) =>
        _asking = new(context, PlayerChoiceOptions.None, () =>
        {
            var list = pile.Cards.Where(filter).ToList();
            // The selector's order for a draw pile, as CardSelectCmd hands it to a selector.
            return pile.Type == PileType.Draw ? list.OrderBy(c => c.Rarity).ThenBy(c => c.Id).ToList() : list;
        });

    // FromChooseACardScreen, FromSimpleGrid, ...: the screen shows the list the command was given.
    private static void OtherPrefix(PlayerChoiceContext context) => _asking = new(context, PlayerChoiceOptions.None, null);

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var asking = _asking;
        _asking = null;
        if (PickTiming != "instant" && asking?.Context is HookPlayerChoiceContext { Source: null, Owner: { } owner } hook
            && CombatManager.Instance.IsInProgress && owner.PlayerCombatState?.Phase == PlayerTurnPhase.Start)
        {
            return AnswerOnTheScreen(hook, asking, options.ToList(), minSelect, maxSelect);
        }

        return Task.FromResult<IEnumerable<CardModel>>(Pick(options.ToList(), minSelect, maxSelect, waited: false));
    }

    /// <summary>A turn-start screen, answered as the UI answers it (see the file's head).</summary>
    private async Task<IEnumerable<CardModel>> AnswerOnTheScreen(HookPlayerChoiceContext hook, Asking asking,
        List<CardModel> counted, int minSelect, int maxSelect)
    {
        await hook.SignalPlayerChoiceBegun(asking.Options);
        var picked = Pick(asking.Live?.Invoke() ?? counted, minSelect, maxSelect, waited: true);
        await hook.SignalPlayerChoiceEnded();
        return picked;
    }

    private List<CardModel> Pick(List<CardModel> list, int minSelect, int maxSelect, bool waited)
    {
        string? edge = NextEdge();
        int n = Math.Min(list.Count, Math.Max(minSelect, Math.Min(1, maxSelect)));
        var at = edge == null ? PickHook?.Invoke(list, n, Env.Extras && minSelect <= 0 && n > 0) ?? Enumerable.Range(0, n).ToList() : EdgeIndices(edge, list, minSelect, maxSelect);
        var picked = at.Select(i => list[i]).ToList();
        var entry = new JsonObject
        {
            ["kind"] = "cards",
            ["step"] = Driver.CurrentStep,
            ["options"] = new JsonArray(list.Select(c => (JsonNode)c.Id.Entry).ToArray()),
            ["min"] = minSelect, ["max"] = maxSelect,
            ["picked"] = new JsonArray(at.Select(i => (JsonNode)i).ToArray()),
        };
        if (edge != null)
        {
            entry["edge"] = edge;
        }

        if (waited)
        {
            // Answered on the screen, after the rest of the turn start (pick_timing "ui").
            entry["waited"] = true;
        }

        Log.Add(entry);
        return picked;
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        int pick = 0;
        // Nodeless there is no card reward screen, so this answers what Driver.AnswerCardRewardScreen
        // answers on the full host; log it the way that does (card 0, no step), so both hosts agree.
        Log.Add(Nodeless.Active
            ? new JsonObject
            {
                ["kind"] = "card_reward",
                ["options"] = new JsonArray(options.Select(o => (JsonNode)o.Card.Id.Entry).ToArray()),
                ["picked"] = 0,
            }
            : new JsonObject
            {
                ["kind"] = "card_reward",
                ["step"] = Driver.CurrentStep,
                ["options"] = new JsonArray(options.Select(o => (JsonNode)o.Card.Id.Entry).ToArray()),
                ["picked"] = options.Count > 0 ? (pick = CardRewardHook?.Invoke(options) ?? 0) : -1,
            });
        return options.Count == 0 ? default : new CardRewardSelection { card = options[pick].Card, alternative = null };
    }
}
