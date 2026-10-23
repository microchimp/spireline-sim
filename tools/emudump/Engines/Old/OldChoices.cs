using System.Text.Json.Nodes;
using EmuDump.Engine;
using Sts2Emulator.Core;

namespace EmuDump.Engines.Old;

/// <summary>
/// The old engine's card-selection screens as <see cref="OpenScreen"/>s, and the thin forwarders the
/// callers not yet moved onto <see cref="IEngine"/> use: they hold a bare combat and a step function.
/// </summary>
static class OldChoices
{
    /// <summary>The pile a non-generated selection's candidates index.</summary>
    static List<CardInstance> Pile(CombatState c, PendingCardSelection sel) =>
        sel.Kind.ChoosesFromDrawPile() ? c.DrawPile
        : sel.Kind.ToString().StartsWith("DiscardTo") ? c.DiscardPile
        : c.Hand;

    /// <summary>Card def id of each candidate: generated options, else the pile whose cards match.</summary>
    internal static List<int> CandidateIds(CombatState c, PendingCardSelection sel)
    {
        if (sel.GeneratedCandidates.Count > 0) return sel.Candidates.Select(i => sel.GeneratedCandidates[i]).ToList();
        var pile = Pile(c, sel);
        return sel.Candidates.Select(i => i < pile.Count ? pile[i].DefId : -1).ToList();
    }

    /// <summary>The combat's open card selection as a screen; null when none is open.</summary>
    internal static OpenScreen? ScreenOf(CombatState? c)
    {
        if (c?.PendingSelection is not { } sel) return null;
        bool generated = sel.GeneratedCandidates.Count > 0;
        var ids = CandidateIds(c, sel);
        var pile = generated ? null : Pile(c, sel);
        var cards = new List<Candidate>(ids.Count);
        for (int k = 0; k < ids.Count; k++)
        {
            int i = sel.Candidates[k];
            bool upgraded = generated ? sel.GeneratedUpgraded : i < pile!.Count && pile[i].Upgraded;
            cards.Add(new Candidate(OldIds.SnakeByCardId(ids[k]), upgraded));
        }

        return new OpenScreen(sel.Kind.ToString(), cards, null, sel.Skippable, OldIds.SnakeByCardId(sel.SourceCardDefId),
            Native: new JsonArray(ids.Select(i => (JsonNode)i).ToArray()));
    }

    /// <summary>A bare combat and a step function as a screen source (Reanchor and RunReplay hold no <see cref="IEngine"/>).</summary>
    sealed class CombatScreens(Func<CombatState?> active, Action<int> step) : IScreenSource
    {
        public OpenScreen? Screen => ScreenOf(active());

        public void Pick(int candidate) => step(candidate);

        // One past the last candidate (PendingCardSelection.Skippable).
        public void Decline() => step(active()!.PendingSelection!.Candidates.Count);
    }

    /// <summary><see cref="Fight.ApplyChoice(IScreenSource, JsonObject)"/> over a bare combat.</summary>
    internal static JsonObject Apply(Func<CombatState?> active, Action<int> step, JsonObject choice) =>
        Fight.ApplyChoice(new CombatScreens(active, step), choice);

    /// <summary><see cref="Fight.UntaggedChoices.AnswerOpen(IScreenSource, JsonArray)"/> over a bare combat.</summary>
    internal static List<JsonObject> AnswerOpen(this Fight.UntaggedChoices untagged, Func<CombatState?> active, Action<int> step, JsonArray log) =>
        untagged.AnswerOpen(new CombatScreens(active, step), log);
}
