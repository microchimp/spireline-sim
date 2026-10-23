// Re-anchoring (referee/SPEED_PLAN.md step 6): every recorded step of a referee fight becomes an
// independent test. For step j the engine is put into the GAME's state S(j-1), the game's
// action a(j) is applied, and the result is dumped to be compared with the game's S(j).
// Usage: EmuDump reanchor <referee out.json> [--seed SEED]... [--scramble] [--stale enemy,cards,relics,counters|all]
//        [--perturb-moves] > reanchor.json
//
// This is the adapter: it holds the loop over the record's steps and the output, and reaches the engine only through
// IEngine (Engine/IEngine.cs, ReanchorOpen and ReanchorStep; the types are in Engine/ReanchorTypes.cs). It names no engine
// type. The engine puts the game's snapshot (referee/driver/Combat.cs Snapshot()) over its own state after its previous
// step, which supplies what the snapshot cannot say: relic counters, enemy AI state, per-card modifiers, per-turn and
// per-combat counters, the potion belt. That state travels between steps as an opaque block, handed back unread. The
// tests of the anchoring itself (--scramble, --stale, --perturb-moves, --trace) are run by the engine, selected through
// the mode record. tools/fidelity/reanchor_compare.py judges the result and decides where a fight stops being
// re-anchorable. The old engine's side is Engines/Old/OldReanchor.cs, its snapshot loader Engines/Old/OldAnchor.cs.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines;

static class Reanchor
{
    /// <summary>Power keys the engine has a buff for, and buffs of its own that are no game power (<see cref="IEngine.PowerKeys"/>).</summary>
    public static JsonObject PowerKeys() => EngineFactory.Create().PowerKeys();

    /// <summary>
    /// Diagnostics that test the anchoring itself. <c>Scramble</c> wrecks every field the game's
    /// snapshot covers before it is written back, so a fight that matches anyway proves the write is
    /// complete. <c>Stale</c> starts every step from the fight's first state instead of the previous
    /// step's, with only the named groups of hidden state (enemy, cards, relics, counters) stale, to
    /// show what the snapshot alone cannot rebuild. <c>PerturbMoves</c> tests the move anchoring: before
    /// each step every cycling enemy is moved one position on (and re-announces), and the monster AI
    /// draws from another stream, so the moves must be put back on the game's for the fight to match.
    /// <c>Trace</c> records what happened inside each step (tools/emudump/Trace.cs) for steps up to
    /// <c>TraceTo</c> (all when null), and the run ends there. The first four are the engine's
    /// <see cref="ReanchorMode"/>; <c>TraceTo</c> is the loop's.
    /// </summary>
    public sealed record Mode(bool Scramble = false, HashSet<string>? Stale = null, bool PerturbMoves = false,
        bool Trace = false, int? TraceTo = null);

    public static JsonObject Run(JsonObject game, Mode? mode = null)
    {
        mode ??= new Mode();
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

            var opening = e.ReanchorOpen();
            var gsteps = game["steps"]!.AsArray();
            var gameChoices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
            var untagged = Fight.UntaggedChoices.Of(game);
            steps.Add(new JsonObject { ["action"] = null, ["state"] = opening.State });
            var carry = opening.Carry;
            var engineMode = new ReanchorMode(mode.Scramble, mode.Stale, mode.PerturbMoves, mode.Trace);
            Func<IScreenSource, JsonArray, List<JsonObject>>? answerOpen = untagged == null ? null : (src, log) => untagged.AnswerOpen(src, log);
            int last = mode.TraceTo is int upTo ? Math.Min(upTo + 1, gsteps.Count) : gsteps.Count;
            for (int j = 1; j < last; j++)
            {
                var rec = new JsonObject { ["action"] = gsteps[j]!["action"]!.AsObject().DeepClone() };
                steps.Add(rec);
                var outcome = e.ReanchorStep(carry, engineMode, new ReanchorInput(gsteps, j, gameChoices, answerOpen), rec);
                if (outcome.Stop) break;  // a step that cannot be tried ends it, as does a fight over in both
                carry = outcome.Carry;
            }
        }
        catch (Exception ex)
        {
            outp["error"] = ex.ToString();
        }

        return outp;
    }
}
