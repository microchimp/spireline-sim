// Replays a fight the referee recorded (referee/driver/Combat.cs) in an engine and dumps the
// state after each action in the same shape. Usage: EmuDump fight <referee out.json> > emu.json
// The engine is reached only through IEngine (Engine/IEngine.cs): this file names no engine type.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines;

static class Fight
{
    /// <summary>Called with each step as it is recorded (EmuDump fight --netstate: IEngine.GameState).</summary>
    internal static Action<IEngine, JsonObject>? OnStep;
    /// <summary>Called with the engine after the opening (0) and after each replayed step; the live invariant checker sets it.</summary>
    internal static Action<IEngine, int>? AfterStep;

    public static JsonObject Replay(JsonObject game)
    {
        var outp = new JsonObject();
        var steps = new JsonArray();
        outp["steps"] = steps;
        try
        {
            IEngine e;
            using (Prof.Scope("enter_room"))
            {
                e = EngineFactory.Create();
                e.StartFight(new FightSpec(game), outp);
            }

            steps.Add(new JsonObject { ["action"] = null, ["state"] = e.DumpStep() });
            OnStep?.Invoke(e, steps[^1]!.AsObject());
            AfterStep?.Invoke(e, 0);
            var gameChoices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
            var untagged = UntaggedChoices.Of(game);
            int stepNo = 0;
            foreach (var step in game["steps"]!.AsArray().Skip(1))
            {
                stepNo++;  // the referee tags a choice with the index of the step its action produced
                var a = step!["action"]!.AsObject();
                if (!e.CombatOpen) break;
                var engineSpan = Prof.Scope("step");
                if ((string)a["kind"]! == "end_turn")
                {
                    e.EndTurn();
                }
                else if ((string)a["kind"]! == "potion")
                {
                    // used=false: the game cancelled the use (the target was no longer valid).
                    if ((bool?)a["used"] != false)
                    {
                        e.UsePotion((int)a["slot"]!, (int)a["target_slot"]!);
                    }
                }
                else if ((bool?)a["played"] != false)
                {
                    // played=false: the game refused the play (CanPlay held, the target did not), so
                    // nothing happened there and nothing is replayed here.
                    e.PlayCard((int)a["hand_index"]!, (int)a["target_slot"]!);
                }

                // The choices the game made for this action, applied as the same cards, then the
                // screen closed the same way. FirstChoiceSelector logs every screen with the step
                // whose action led to it: a card's own, and a hook's in the middle of the action
                // (Stratagem's shuffle inside a draw, Entropy and Tyranny at the turn start an
                // end_turn runs into), which the emulator stops on until it is answered.
                var chosen = new JsonArray();
                foreach (var choice in gameChoices.Where(c => (int?)c!["step"] == stepNo && (string?)c["kind"] == "cards"))
                {
                    chosen.Add(ApplyChoice(e, choice!.AsObject()));
                }

                untagged?.AnswerOpen(e, chosen);

                if (e.Screen is { } extra)
                {
                    chosen.Add(new JsonObject { ["unexpected_selection"] = extra.Kind });
                }

                engineSpan.Dispose();
                steps.Add(new JsonObject { ["action"] = a.DeepClone(), ["choices"] = chosen, ["state"] = e.DumpStep() });
                OnStep?.Invoke(e, steps[^1]!.AsObject());
                AfterStep?.Invoke(e, stepNo);
            }
        }
        catch (Exception ex)
        {
            outp["error"] = ex.ToString();
        }

        return outp;
    }

    /// <summary>
    /// One game choice: pick each card the game picked (matched by model id among the open
    /// screen's candidates), then decline the rest if the game took fewer than the screen allows.
    /// The decline goes to that screen only: once its picks are spent the next screen of the same
    /// action may already be open.
    /// </summary>
    internal static JsonObject ApplyChoice(IScreenSource src, JsonObject choice)
    {
        var options = choice["options"]!.AsArray().Select(x => (string)x!).ToList();
        var picked = choice["picked"]!.AsArray().Select(x => options[(int)x!]).ToList();
        var log = new JsonObject { ["game_options"] = choice["options"]!.DeepClone(), ["game_picked"] = choice["picked"]!.DeepClone() };
        var first = src.Screen;
        foreach (var want in picked)
        {
            if (src.Screen is not { } sel) { log["error"] = "no selection open for a pick"; return log; }
            int idx = sel.Candidates.ToList().FindIndex(c => c.ModelId == want);
            if (idx < 0)
            {
                log["error"] = $"{want} not among emulator candidates";
                log["emu_candidates"] = sel.Native?.DeepClone() ?? new JsonArray(sel.Candidates.Select(c => (JsonNode)c.ModelId).ToArray());
                return log;
            }

            src.Pick(idx);
        }

        if (picked.Count < ((int?)choice["max"] ?? int.MaxValue)
            && first != null && src.Screen is { Skippable: true } rest && rest.SourceModelId == first.SourceModelId)
        {
            src.Decline();
        }

        return log;
    }

    /// <summary>
    /// A format-1 recording's card choices (corpus.recording_format): the referee logged each screen's
    /// options and picks in order, without the step whose action led to it. A screen left open after
    /// an action takes the first unused choice whose options are its candidates, so a card's screen
    /// gets the answer the game gave it rather than standing unanswered. Null for a recording whose
    /// choices carry their step (format 2 on): those are applied by step.
    /// </summary>
    internal sealed class UntaggedChoices
    {
        readonly List<JsonObject> _left;

        UntaggedChoices(List<JsonObject> left) => _left = left;

        internal static UntaggedChoices? Of(JsonObject game)
        {
            var all = (game["choices"]?.AsArray() ?? new JsonArray()).Select(c => c!.AsObject()).ToList();
            if (all.Any(c => c.ContainsKey("step"))) return null;
            var cards = all.Where(c => (string?)c["kind"] == "cards" && c["options"] is JsonArray && c["picked"] is JsonArray).ToList();
            return cards.Count == 0 ? null : new UntaggedChoices(cards);
        }

        /// <summary>Answers every screen open now that an unused choice fits; returns the choices used, in order.</summary>
        internal List<JsonObject> AnswerOpen(IScreenSource src, JsonArray log)
        {
            var used = new List<JsonObject>();
            for (int guard = 0; guard < 16 && src.Screen is { } sel; guard++)
            {
                var have = sel.Candidates.Select(c => c.ModelId).Order(StringComparer.Ordinal).ToList();
                int at = _left.FindIndex(ch => OptionIds(ch).SequenceEqual(have));
                if (at < 0) break;
                var choice = _left[at];
                _left.RemoveAt(at);
                used.Add(choice);
                log.Add(ApplyChoice(src, choice));
            }

            return used;
        }

        static List<string> OptionIds(JsonObject choice) =>
            choice["options"]!.AsArray().Select(o => (string)o!).Order(StringComparer.Ordinal).ToList();
    }
}
