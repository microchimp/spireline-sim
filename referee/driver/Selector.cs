// Answers the game's card-choice screens without UI, through the hook its own autoplayer and test
// mode use (CardSelectCmd.UseSelector). Deterministic: always the first cards offered, and every
// choice is logged so the emulator can be given the same one.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;

namespace Spireline.Referee;

public sealed class FirstChoiceSelector : ICardSelector
{
    public JsonArray Log { get; } = new();

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var list = options.ToList();
        int n = Math.Min(list.Count, Math.Max(minSelect, Math.Min(1, maxSelect)));
        var picked = list.Take(n).ToList();
        Log.Add(new JsonObject
        {
            ["kind"] = "cards",
            ["step"] = Driver.CurrentStep,
            ["options"] = new JsonArray(list.Select(c => (JsonNode)c.Id.Entry).ToArray()),
            ["min"] = minSelect, ["max"] = maxSelect,
            ["picked"] = new JsonArray(Enumerable.Range(0, n).Select(i => (JsonNode)i).ToArray()),
        });
        return Task.FromResult<IEnumerable<CardModel>>(picked);
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
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
                ["picked"] = options.Count > 0 ? 0 : -1,
            });
        return options.Count == 0 ? default : new CardRewardSelection { card = options[0].Card, alternative = null };
    }
}
