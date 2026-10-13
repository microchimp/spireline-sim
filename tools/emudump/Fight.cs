// Replays a fight the referee recorded (referee/driver/Combat.cs) in the emulator and dumps the
// state after each action in the same shape. Usage: EmuDump fight <referee out.json> > emu.json
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class Fight
{
    public static JsonObject Replay(JsonObject game)
    {
        var e = new RunEngine();
        int asc = (int)game["ascension"]!;
        e.Reset((string)game["seed"]!, asc);
        var s = e.State;
        var outp = new JsonObject { ["seed"] = s.StringSeed, ["neow_options"] = new JsonArray(s.NeowOptions.Select(i => (JsonNode)i).ToArray()) };
        var steps = new JsonArray();
        outp["steps"] = steps;
        try
        {
            e.Step((int)game["neow_pick"]!, -1, out _, out _, out _);
            int guard = 0;
            while (s.Phase != RunPhase.Map && s.Phase != RunPhase.Combat && guard++ < 10)
            {
                e.Step(0, -1, out _, out _, out _);  // first offered card, as the referee's selector
            }

            var room = game["room"]!.AsArray();
            (int Col, int Row) want = ((int)room[0]!, (int)room[1]!);
            int idx = Array.FindIndex(s.MapOptionCoords, c => c == want);
            if (idx < 0) throw new InvalidOperationException($"room {want} not offered; phase {s.Phase}");
            e.Step(idx, -1, out _, out _, out _);
            steps.Add(new JsonObject { ["action"] = null, ["state"] = Dump(e) });
            foreach (var step in game["steps"]!.AsArray().Skip(1))
            {
                var a = step!["action"]!.AsObject();
                if (s.ActiveCombat == null) break;
                if ((string)a["kind"]! == "end_turn")
                {
                    e.Step(s.ActiveCombat.Hand.Count, -1, out _, out _, out _);
                }
                else
                {
                    e.Step((int)a["hand_index"]!, (int)a["target_slot"]!, out _, out _, out _);
                }

                // A choice the card opened (Splash, Armaments, ...): the first option, as the
                // referee's FirstChoiceSelector answers it in the game.
                var chosen = new JsonArray();
                for (int g = 0; g < 5 && s.ActiveCombat?.PendingSelection is { } sel; g++)
                {
                    chosen.Add(new JsonObject
                    {
                        ["kind"] = sel.Kind.ToString(),
                        ["candidates"] = new JsonArray(sel.Candidates.Select(x => (JsonNode)x).ToArray()),
                        ["generated"] = new JsonArray(sel.GeneratedCandidates.Select(x => (JsonNode)x).ToArray()),
                    });
                    e.Step(0, -1, out _, out _, out _);
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
