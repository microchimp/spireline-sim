// Replays a fight the referee recorded (referee/driver/Combat.cs) in the emulator and dumps the
// state after each action in the same shape. Usage: EmuDump fight <referee out.json> > emu.json
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class Fight
{
    /// <summary>
    /// The run up to the first combat, as the referee drove it: deck additions, the Neow pick,
    /// card choices, the walk into the first room. Fills <c>seed</c> and <c>neow_options</c> of
    /// <paramref name="outp"/> and returns the engine with the fight open.
    /// </summary>
    internal static RunEngine EnterRoom(JsonObject game, JsonObject outp)
    {
        var e = new RunEngine();
        int asc = (int)game["ascension"]!;
        e.Reset((string)game["seed"]!, asc);
        var s = e.State;
        outp["seed"] = s.StringSeed;
        outp["neow_options"] = new JsonArray(s.NeowOptions.Select(i => (JsonNode)i).ToArray());
        // The referee's deck additions, appended the same way (before Neow).
        foreach (var name in game["deck_add"]?.AsArray() ?? new JsonArray())
        {
            RunNonCombatEffects.AddCardToDeck(s, new CardInstance(CardIdBySnake((string)name!), Upgraded: false));
        }

        e.Step((int)game["neow_pick"]!, -1, out _, out _, out _);
        int guard = 0;
        while (s.Phase != RunPhase.Map && s.Phase != RunPhase.Combat && guard++ < 10)
        {
            e.Step(0, -1, out _, out _, out _);  // first offered card, as the referee's selector
            if (s.Phase == RunPhase.BundleSelect)
            {
                e.Step(RunConstants.BundleConfirmAction, -1, out _, out _, out _);  // bundle 0 picked; confirm
            }
        }

        var room = game["room"]!.AsArray();
        (int Col, int Row) want = ((int)room[0]!, (int)room[1]!);
        int idx = Array.FindIndex(s.MapOptionCoords, c => c == want);
        if (idx < 0) throw new InvalidOperationException($"room {want} not offered; phase {s.Phase}");
        e.Step(idx, -1, out _, out _, out _);
        return e;
    }

    public static JsonObject Replay(JsonObject game)
    {
        var outp = new JsonObject();
        var steps = new JsonArray();
        outp["steps"] = steps;
        try
        {
            var e = EnterRoom(game, outp);
            var s = e.State;
            steps.Add(new JsonObject { ["action"] = null, ["state"] = Dump(e) });
            var gameChoices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
            int stepNo = 0;
            foreach (var step in game["steps"]!.AsArray().Skip(1))
            {
                stepNo++;  // the referee tags a choice with the index of the step its action produced
                var a = step!["action"]!.AsObject();
                if (s.ActiveCombat == null) break;
                if ((string)a["kind"]! == "end_turn")
                {
                    e.Step(s.ActiveCombat.Hand.Count, -1, out _, out _, out _);
                }
                else if ((bool?)a["played"] != false)
                {
                    // played=false: the game refused the play (CanPlay held, the target did not), so
                    // nothing happened there and nothing is replayed here.
                    e.Step((int)a["hand_index"]!, (int)a["target_slot"]!, out _, out _, out _);
                }

                // A choice the card opened (Splash, Armaments, ...): the first option, as the
                // referee's FirstChoiceSelector answers it in the game.
                // The choices the game made for this action (FirstChoiceSelector logs them with
                // their step), applied as the same cards, then the screen closed the same way.
                var chosen = new JsonArray();
                foreach (var choice in gameChoices.Where(c => (int?)c!["step"] == stepNo && (string?)c["kind"] == "cards"))
                {
                    chosen.Add(ApplyChoice(() => s.ActiveCombat, i => e.Step(i, -1, out _, out _, out _), choice!.AsObject()));
                }

                if (s.ActiveCombat?.PendingSelection is { } extra)
                {
                    chosen.Add(new JsonObject { ["unexpected_selection"] = extra.Kind.ToString() });
                }

                steps.Add(new JsonObject { ["action"] = a.DeepClone(), ["choices"] = chosen, ["state"] = Dump(e) });
            }
        }
        catch (Exception ex)
        {
            outp["error"] = ex.ToString();
        }

        return outp;
    }

    /// <summary>
    /// One game choice: pick each card the game picked (matched by card id among the open
    /// selection's candidates, in whichever pile they live), then decline the rest if the
    /// game took fewer than the screen allows.
    /// </summary>
    internal static JsonObject ApplyChoice(Func<CombatState?> active, Action<int> step, JsonObject choice)
    {
        var options = choice["options"]!.AsArray().Select(x => (string)x!).ToList();
        var picked = choice["picked"]!.AsArray().Select(x => options[(int)x!]).ToList();
        var log = new JsonObject { ["game_options"] = choice["options"]!.DeepClone(), ["game_picked"] = choice["picked"]!.DeepClone() };
        foreach (var want in picked)
        {
            if (active()?.PendingSelection is not { } sel) { log["error"] = "no selection open for a pick"; return log; }
            var ids = CandidateIds(active()!, sel);
            int idx = ids.FindIndex(id => id == CardIdBySnake(want));
            if (idx < 0) { log["error"] = $"{want} not among emulator candidates"; log["emu_candidates"] = new JsonArray(ids.Select(i => (JsonNode)i).ToArray()); return log; }
            step(idx);
        }

        if (active()?.PendingSelection is { Skippable: true } rest)
        {
            step(rest.Candidates.Count);  // decline: one past the last candidate
        }

        return log;
    }

    /// <summary>Card def id of each candidate: generated options, else the pile whose cards match.</summary>
    static List<int> CandidateIds(CombatState c, PendingCardSelection sel)
    {
        if (sel.GeneratedCandidates.Count > 0) return sel.Candidates.Select(i => sel.GeneratedCandidates[i]).ToList();
        var pile = sel.Kind.ChoosesFromDrawPile() ? c.DrawPile
            : sel.Kind.ToString().StartsWith("DiscardTo") ? c.DiscardPile
            : c.Hand;
        return sel.Candidates.Select(i => i < pile.Count ? pile[i].DefId : -1).ToList();
    }

    static Dictionary<string, int>? _bySnake;

    /// <summary>Game card id (UPPER_SNAKE) to the emulator's def id, through data/id_map.json.</summary>
    internal static int CardIdBySnake(string snake)
    {
        if (_bySnake == null)
        {
            var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!["cards"]!.AsObject();
            _bySnake = map.ToDictionary(kv => System.Text.RegularExpressions.Regex.Replace(kv.Key,
                "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant(), kv => (int)kv.Value!);
        }

        return _bySnake[snake];
    }

    /// <summary>Card pools for the batch maker: <c>EmuDump pools</c>.</summary>
    public static JsonObject Pools() => new()
    {
        ["ironclad"] = new JsonArray(Sts2Emulator.GeneratedData.CardPools.Ironclad.ToArray().Select(i => (JsonNode)i).ToArray()),
        ["colorless"] = new JsonArray(Sts2Emulator.GeneratedData.CardPools.Colorless.ToArray().Select(i => (JsonNode)i).ToArray()),
    };

    static JsonObject Dump(RunEngine e)
    {
        var c = e.State.ActiveCombat;
        var o = new JsonObject { ["in_combat"] = c != null && e.State.Phase == RunPhase.Combat, ["run_phase"] = e.State.Phase.ToString() };
        if (c == null) return o;
        var snap = JsonNode.Parse(Sts2Emulator.Interop.CombatSnapshot.Write(c))!;
        // Buff ids are the emulator's BuffId enum, not id_map's power ids: name them here.
        foreach (var holder in new[] { snap["player"]! }.Concat(snap["enemies"]!.AsArray().Select(x => x!)))
        {
            foreach (var b in holder["buffs"]!.AsArray())
            {
                b!["name"] = ((BuffId)(int)b["id"]!).ToString();
            }
        }

        o["snapshot"] = snap;
        o["draw_order"] = new JsonArray(c.DrawPile.Select(x => (JsonNode)new JsonArray(x.DefId, x.Upgraded)).ToArray());
        int N(CountingRandom? r) => r?.CallCount ?? -1;
        o["rng"] = new JsonObject
        {
            ["Shuffle"] = N(c.ShuffleRng), ["CombatTargets"] = N(c.TargetRng), ["MonsterAi"] = N(c.AiRng as CountingRandom),
            ["CombatCardSelection"] = N(c.CardSelectionRng), ["CombatCardGeneration"] = N(c.CardGenerationRng),
            ["CombatEnergyCosts"] = N(c.EnergyCostRng), ["CombatPotionGeneration"] = N(c.PotionGenerationRng),
            ["CombatOrbGeneration"] = N(c.OrbGenerationRng), ["Niche"] = N(c.NicheHpRng),
        };
        return o;
    }
}
