// The old engine's side of re-anchoring (referee/SPEED_PLAN.md step 6): every recorded step of a referee fight becomes
// an independent test. For step j the engine is put into the GAME's state S(j-1) (OldAnchor), the game's action a(j) is
// applied, and the result is dumped to be compared with the game's S(j) by tools/fidelity/reanchor_compare.py. The
// adapter (tools/emudump/Reanchor.cs) holds the loop and the output; this is what it asks of the engine through
// IEngine.ReanchorOpen and ReanchorStep, and the tests of the anchoring itself (Scramble, Stale, PerturbMoves, Trace).
//
// What the game's snapshot (referee/driver/Combat.cs Snapshot()) cannot say is taken from the engine's own state
// after its previous step, the block OldCarry: relic counters, enemy AI state, per-card modifiers, per-turn and
// per-combat counters, the potion belt. The snapshot's fields (player HP/block/energy/powers, the four piles in
// order, living enemies, RNG streams' seed and counter) are written over it, and the enemies' next moves where the
// record has them.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

/// <summary>What the old engine carries between two re-anchored steps: the combat and its stream as the step left them, and the fight's first state.</summary>
sealed class OldCarry(CombatState combat, CountingRandom rng, CombatState start) : EngineBlock
{
    internal CombatState Combat { get; } = combat;

    internal CountingRandom Rng { get; } = rng;

    /// <summary>The fight as it opened, which <c>Stale</c> falls back to for the groups it names.</summary>
    internal CombatState Start { get; } = start;
}

static class OldReanchor
{
    static OldAnchor.Tables? _tables;

    /// <summary>The id tables of OldAnchor, read once from data/id_map.json and tools/fidelity/power_aliases.json beside the program.</summary>
    internal static OldAnchor.Tables Tables => _tables ??= new OldAnchor.Tables(
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!.AsObject(),
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "power_aliases.json")))!.AsObject());

    internal static ReanchorOpening Open(OldEngine engine)
    {
        var s = engine.Run.State;
        var combat = s.ActiveCombat ?? throw new InvalidOperationException($"no combat after entering the room; phase {s.Phase}");
        var rng = s.ActiveCombatRng!;
        return new ReanchorOpening(DumpLite(combat, true), new OldCarry(combat, rng, combat.Clone()));
    }

    internal static ReanchorOutcome Step(OldEngine engine, OldCarry carry, ReanchorMode mode, ReanchorInput input, JsonObject rec)
    {
        var gsteps = input.GameSteps;
        int j = input.Step;
        var prior = gsteps[j - 1]!["state"]!.AsObject();
        var a = gsteps[j]!["action"]!.AsObject();
        var cloneSpan = Prof.Scope("clone");
        var q = carry.Combat.Clone();
        var qr = carry.Rng.Clone();
        cloneSpan.Dispose();
        if (mode.Stale != null) GoStale(q, carry.Start, mode.Stale);
        if (mode.Scramble) Scramble(q);
        var patched = new JsonArray();
        rec["patched"] = patched;
        string? problem;
        using (Prof.Scope("anchor")) problem = OldAnchor.Anchor(Tables, q, prior, patched);
        if (problem != null)
        {
            rec["anchor_error"] = problem;  // this step cannot be tried, so neither can the rest
            return new ReanchorOutcome(carry, Stop: true);
        }

        // The moves the enemies will make: the game's, where the engine's AI can be put there.
        if (mode.PerturbMoves) ShiftCycles(q);
        using (Prof.Scope("anchor_moves"))
        {
            var (status, unknown) = OldAnchor.AnchorMoves(q, prior, patched, i => OldAnchor.LaterMoves(gsteps, j, i));
            if (unknown != null) rec["moves_unknown"] = unknown;
            rec["moves"] = status;
        }

        if (mode.PerturbMoves && q.AiRng is CountingRandom ai0) q.AiRng = new CountingRandom(ai0.Seed ^ 0x2545F491, ai0.CallCount);
        var cloneSpan2 = Prof.Scope("clone");
        var pre = q.Clone();
        var preRng = qr.Clone();
        cloneSpan2.Dispose();
        var stepChoices = input.GameChoices.Where(c => (int?)c!["step"] == j && (string?)c["kind"] == "cards").ToList();
        try
        {
            if (mode.Trace)
            {
                rec["hidden_pre"] = OldTrace.Hidden(q);
                OldTrace.Begin(q);
            }

            bool terminal, won;
            JsonArray chosen;
            try
            {
                using (Prof.Scope("apply")) (terminal, won, chosen) = Apply(q, qr, a, stepChoices, input.AnswerOpen);
            }
            finally
            {
                if (mode.Trace) rec["trace"] = OldTrace.End();
            }

            if (mode.Trace) rec["hidden"] = OldTrace.Hidden(q);
            rec["choices"] = chosen;
            rec["fallback_rng_calls"] = qr.CallCount - preRng.CallCount;
            bool gameGoesOn = j + 1 < gsteps.Count && (gsteps[j]!["state"]!["in_combat"]?.GetValue<bool>() ?? false);
            // The fight ends here in the game too: the state to judge is the one the game's snapshot shows
            // after it, so the fight is finished the way the run finishes it (SyncAfterCombat, the
            // after-combat relics). Where the game goes on, the step differs in in_combat, as it always did.
            using (Prof.Scope("dumplite")) rec["state"] = terminal && !gameGoesOn ? EndState(engine.Run, q, qr, won) : DumpLite(q, !terminal);
            if (!terminal && q.PendingSelection != null) terminal = Finish(q, qr);
            if (terminal && !gameGoesOn) return new ReanchorOutcome(carry, Stop: true);
            // The engine ended a fight the game continues: carry on from the state before the action.
            var next = terminal ? new OldCarry(pre, preRng, carry.Start) : new OldCarry(q, qr, carry.Start);
            if (!terminal)
            {
                using (Prof.Scope("carry_moves")) CarryGameMoves(q, pre, preRng, a, stepChoices, gsteps[j]!["state"]!.AsObject(), rec);
            }

            return new ReanchorOutcome(next, Stop: false);
        }
        catch (Exception ex)
        {
            rec["error"] = ex.GetType().Name + ": " + ex.Message;
            rec["state"] = DumpLite(pre, true);
            return new ReanchorOutcome(new OldCarry(pre, preRng, carry.Start), Stop: false);
        }
    }

    /// <summary>
    /// The game's action, then the card choices the game made for it, as the plain replay applies them
    /// (a format-1 recording's untagged choices from <paramref name="answerOpen" />, which keeps track of
    /// those used; the ones this step used are added to <paramref name="choices" />, so a redo of the
    /// step in <see cref="CarryGameMoves" /> answers the same).
    /// </summary>
    static (bool Terminal, bool Won, JsonArray Chosen) Apply(CombatState q, CountingRandom qr, JsonObject a, List<JsonNode?> choices,
        Func<IScreenSource, JsonArray, List<JsonObject>>? answerOpen = null)
    {
        var first = StepOnce(q, qr, a);
        bool terminal = first.Terminal, won = first.Terminal && first.PlayerWon;
        var chosen = new JsonArray();
        void Answer(int i)
        {
            var r = CombatEngine.Step(q, i, qr);
            terminal |= r.Terminal;
            won |= r.Terminal && r.PlayerWon;
        }

        foreach (var choice in choices)
        {
            chosen.Add(OldChoices.Apply(() => q, Answer, choice!.AsObject()));
        }

        if (answerOpen != null)
        {
            choices.AddRange(answerOpen(OldChoices.Screens(() => q, Answer), chosen));
        }

        if (!terminal && q.PendingSelection is { } extra)  // a won fight leaves no screen to answer
        {
            chosen.Add(new JsonObject { ["unexpected_selection"] = extra.Kind.ToString() });
        }

        return (terminal, won, chosen);
    }

    /// <summary>
    /// Finishes a step left on a screen nobody answered (<see cref="Apply" /> logged it as unexpected),
    /// after the step has been judged as it stands. The step is stopped ON the screen, half done: a
    /// card's play in the middle of its OnPlay, a turn start in the middle of its hooks. Clearing the
    /// screen would leave that half for the next anchor to start from (the play never counted, an
    /// auto-play still queued: hidden state the anchor does not patch), so the engine's own answer
    /// -- decline when it may, else the first candidate -- finishes it. True when that ended the fight.
    /// </summary>
    internal static bool Finish(CombatState q, CountingRandom qr)
    {
        bool terminal = false;
        for (int guard = 0; guard < 64 && !terminal && q.PendingSelection is { } open; guard++)
        {
            terminal |= CombatEngine.Step(q, open.Skippable ? open.Candidates.Count : 0, qr).Terminal;
        }

        q.PendingSelection = null;
        return terminal;
    }

    /// <summary>Applies the game's action (end turn or a card play); the step's result, Terminal when the fight ended.</summary>
    static StepResult StepOnce(CombatState q, CountingRandom qr, JsonObject a)
    {
        if ((string)a["kind"]! == "end_turn") return CombatEngine.Step(q, q.Hand.Count, qr);
        if ((string)a["kind"]! == "potion")
        {
            if ((bool?)a["used"] == false) return new StepResult(Terminal: false, PlayerWon: false, Reward: 0f);  // the game cancelled the use
            return CombatEngine.Step(q, q.Hand.Count + 1 + (int)a["slot"]!, qr, (int)a["target_slot"]!);
        }
        // target_slot indexes the game's living enemies, and the anchor made the engine's roster the same list.
        return CombatEngine.Step(q, (int)a["hand_index"]!, qr, (int)a["target_slot"]!);
    }

    /// <summary>
    /// The state after the step that ended the fight in both the game and the engine. A won fight is put
    /// through the run's own end of combat on the engine <see cref="OldEngine.StartFight" /> made (DebugResolveCombat:
    /// the sync back to the run and the after-combat hooks, Burning Blood's heal among them), so the hit points,
    /// potions, relics and RNG positions written are the run's, as the game's snapshot shows them. A lost fight
    /// ends the run and is written as lost and nothing else (see OldDump.Final).
    /// The run's own streams stand wherever the engine's earlier steps left them, and the anchored combat's where
    /// the game's did (OldAnchor.Anchor, PatchRng); the sync moves the run's to the combat's and a stream cannot go back, so
    /// the run's start from the seed and end where the combat did.
    /// </summary>
    static JsonObject EndState(RunEngine e, CombatState q, CountingRandom qr, bool won)
    {
        if (!won) return DumpLite(q, false, OldDump.Final(e.State, false));
        e.State.ActiveCombat = q;
        e.BindActiveCombat();
        e.State.ActiveCombatRng = qr;
        e.State.Rng = new Sts2Emulator.Core.Rng.RunRngSet(e.State.StringSeed);
        if (e.DebugResolveCombat() == -1) throw new InvalidOperationException($"no fight to finish; phase {e.State.Phase}");
        return DumpLite(q, false, OldDump.Final(e.State, true));
    }

    /// <summary>
    /// After a step whose result announces another move than the game's for an enemy whose choice is a
    /// draw (a RandomBranchState), the step is tried again from the same state with the monster AI on
    /// other streams; an attempt that changes nothing but the moves, and in which that enemy picks the
    /// game's move, gives its AI memory to the state the next step starts from. The step's own verdict
    /// is unchanged: it is judged on what the engine did.
    /// </summary>
    static void CarryGameMoves(CombatState q, CombatState pre, CountingRandom preRng, JsonObject a, List<JsonNode?> choices,
        JsonObject g, JsonObject rec)
    {
        var gen = g["enemies"]!.AsArray();
        if (gen.Any(x => x!["intents"] == null) || q.AiRng is not CountingRandom) return;
        var living = q.Enemies.Where(x => x.Hp > 0).ToList();
        var glive = gen.Where(x => (int)x!["hp"]! > 0).ToList();
        if (living.Count != glive.Count
            || !living.Zip(glive).All(t => t.First.DefId == Tables.Enemies.GetValueOrDefault((string)t.Second!["name"]!, -1))) return;
        var wrong = Enumerable.Range(0, living.Count).Where(i => OldAnchor.EmuMove(living[i]) != OldAnchor.GameMove(glive[i]!)).ToList();
        if (wrong.Count == 0) return;
        string shown = Shown(q);
        var carried = new JsonArray();
        for (int seed = 1; seed <= 64 && wrong.Count > 0; seed++)
        {
            var t = pre.Clone();
            var tr = preRng.Clone();
            t.AiRng = new CountingRandom(unchecked(seed * (int)0x9E3779B1));
            try
            {
                if (Apply(t, tr, a, choices).Terminal || (t.PendingSelection != null && Finish(t, tr)) || Shown(t) != shown) continue;
            }
            catch (Exception)
            {
                continue;
            }

            var tl = t.Enemies.Where(x => x.Hp > 0).ToList();
            foreach (int i in wrong.ToList())
            {
                if (OldAnchor.EmuMove(tl[i]) != OldAnchor.GameMove(glive[i]!)) continue;
                CopyAi(tl[i], living[i]);
                wrong.Remove(i);
                carried.Add(i);
            }
        }

        if (carried.Count > 0) rec["moves_carried"] = carried;
    }

    /// <summary>The step's result without the enemies' moves and the monster AI stream: what a redo must leave alone.</summary>
    static string Shown(CombatState c)
    {
        var d = DumpLite(c, true);
        foreach (var en in d["snapshot"]!["enemies"]!.AsArray()) en!.AsObject().Remove("intent");
        d["rng"]!.AsObject().Remove("MonsterAi");
        return d.ToJsonString();
    }

    static void CopyAi(EnemyState from, EnemyState to)
    {
        to.CurrentIntent = from.CurrentIntent;
        to.SecondaryIntent = from.SecondaryIntent;
        to.MoveIndex = from.MoveIndex;
        to.LastMove = from.LastMove;
        to.LastMoveRepeats = from.LastMoveRepeats;
        to.OnceOnlyMoveUsed = from.OnceOnlyMoveUsed;
        to.MoveHistory = [.. from.MoveHistory];
        to.StartsOnBranch = from.StartsOnBranch;
        to.RollsNextMove = from.RollsNextMove;
        to.LastBranch = from.LastBranch;
        to.RepeatStreak = from.RepeatStreak;
        to.BranchCooldown = from.BranchCooldown;
    }

    // ── diagnostics ─────────────────────────────────────────────────────────────────

    static void ShiftCycles(CombatState q)
    {
        foreach (var en in q.Enemies.Where(x => x.Hp > 0))
        {
            if (!OldAnchor.Reselect(q, en, en.MoveIndex, out var same) || OldAnchor.EmuMove(same) != OldAnchor.EmuMove(en)
                || !OldAnchor.Reselect(q, en, en.MoveIndex + 1, out var next)) continue;
            en.MoveIndex = next.MoveIndex;
            en.CurrentIntent = next.CurrentIntent;
            en.SecondaryIntent = next.SecondaryIntent;
        }
    }

    static void Scramble(CombatState q)
    {
        // Piles are shuffled between and within, unless two cards that look alike differ underneath
        // (a Strike with Hidden Gem's replay and one without): the write-back cannot tell those
        // apart, so such a state keeps its piles as they are.
        var all = q.Hand.Concat(q.DrawPile).Concat(q.DiscardPile).Concat(q.ExhaustPile);
        bool alike = all.GroupBy(c => (c.DefId, c.Upgraded)).All(g => g.Distinct().Count() == 1);
        if (alike)
        {
            q.Hand = q.Hand.OrderByDescending(c => c.DefId).ThenBy(c => c.Upgraded).ToList();
            q.DrawPile = q.DrawPile.OrderByDescending(c => c.DefId).ThenBy(c => c.Upgraded).ToList();
            (q.DiscardPile, q.ExhaustPile) = (q.ExhaustPile, q.DiscardPile);
        }

        q.PlayerHp = 1; q.PlayerMaxHp += 7; q.PlayerBlock += 77; q.Energy = 0; q.Turn += 5;
        q.PlayerBuffs = q.PlayerBuffs.Where(b => OldAnchor.NotPowers.Contains(b.Id) || Tables.IsInternal(b.Id)).ToList();
        foreach (var en in q.Enemies)
        {
            en.Hp = 1; en.MaxHp += 3; en.Block += 5;
            en.Buffs = en.Buffs.Where(b => OldAnchor.NotPowers.Contains(b.Id) || Tables.IsInternal(b.Id)).ToList();
        }

        CountingRandom? Off(CountingRandom? r) => r == null ? null : new CountingRandom(r.Seed ^ 0x5a5a, 3);
        q.ShuffleRng = Off(q.ShuffleRng); q.TargetRng = Off(q.TargetRng); q.CardSelectionRng = Off(q.CardSelectionRng);
        q.CardGenerationRng = Off(q.CardGenerationRng); q.EnergyCostRng = Off(q.EnergyCostRng);
        q.PotionGenerationRng = Off(q.PotionGenerationRng); q.OrbGenerationRng = Off(q.OrbGenerationRng);
        q.NicheHpRng = Off(q.NicheHpRng);
        if (q.AiRng is CountingRandom ai) q.AiRng = Off(ai);
    }

    static readonly string[] Observed = ["PlayerHp", "PlayerMaxHp", "PlayerBlock", "Energy", "Turn"];

    internal static void GoStale(CombatState q, CombatState start, IReadOnlySet<string> groups)
    {
        bool all = groups.Contains("all");
        if (all || groups.Contains("enemy"))
        {
            var skip = new[] { "DefId", "Hp", "MaxHp", "Block", "Buffs" };
            foreach (var en in q.Enemies)
            {
                var from = start.Enemies.FirstOrDefault(x => x.DefId == en.DefId);
                if (from == null) continue;
                foreach (var f in typeof(EnemyState).GetFields().Where(f => !f.IsStatic && !skip.Contains(f.Name)))
                {
                    f.SetValue(en, f.GetValue(from));
                }
            }
        }

        if (all || groups.Contains("cards"))
        {
            // Every copy back to a plain card: cost changes, bonus damage, enchantments and the like are gone.
            // The uid is the card's identity, not a property the game's snapshot could have lost: kept.
            CardInstance Plain(CardInstance c) => new CardInstance(c.DefId, c.Upgraded).WithUid(c.UidOf());
            q.Hand = q.Hand.Select(Plain).ToList(); q.DrawPile = q.DrawPile.Select(Plain).ToList();
            q.DiscardPile = q.DiscardPile.Select(Plain).ToList(); q.ExhaustPile = q.ExhaustPile.Select(Plain).ToList();
        }

        if (all || groups.Contains("relics")) q.Relics = start.Relics.ToList();
        if (all || groups.Contains("counters"))
        {
            // [IdentityOnly] fields (the card uid counter) are identity, not state the game could have drifted from.
            foreach (var f in typeof(CombatState).GetFields().Where(f => !f.IsStatic && !Observed.Contains(f.Name)
                         && !f.IsDefined(typeof(IdentityOnlyAttribute), false)
                         && (f.FieldType == typeof(int) || f.FieldType == typeof(bool))))
            {
                f.SetValue(q, f.GetValue(start));
            }
        }
    }

    // ── what tools/fidelity/fight_compare.py reads, and no more ─────────────────────

    internal static JsonObject DumpLite(CombatState c, bool inCombat, JsonObject? final = null)
    {
        var o = new JsonObject { ["in_combat"] = inCombat, ["run_phase"] = inCombat ? "Combat" : "Over" };
        if (final != null) o["final"] = final;
        if (!inCombat) return o;
        JsonNode Card(CardInstance x) => new JsonObject { ["def_id"] = x.DefId, ["upgraded"] = x.Upgraded };
        JsonNode Buffs(List<BuffState> bs) => new JsonArray(bs.Select(b => (JsonNode)new JsonObject { ["id"] = (int)b.Id, ["name"] = b.Id.ToString(), ["amount"] = b.Magnitude, ["data"] = b.Data }).ToArray());
        o["snapshot"] = new JsonObject
        {
            ["player"] = new JsonObject
            {
                ["hp"] = c.PlayerHp, ["max_hp"] = c.PlayerMaxHp, ["block"] = c.PlayerBlock, ["energy"] = c.Energy,
                ["buffs"] = Buffs(c.PlayerBuffs),
            },
            ["piles"] = new JsonObject
            {
                ["hand"] = new JsonArray(c.Hand.Select(Card).ToArray()),
                ["discard"] = new JsonArray(c.DiscardPile.Select(Card).ToArray()),
                ["exhaust"] = new JsonArray(c.ExhaustPile.Select(Card).ToArray()),
            },
            ["enemies"] = new JsonArray(c.Enemies.Select(en => (JsonNode)new JsonObject
            {
                ["def_id"] = en.DefId, ["hp"] = en.Hp, ["max_hp"] = en.MaxHp, ["block"] = en.Block, ["buffs"] = Buffs(en.Buffs),
                ["intent"] = OldDump.IntentJson(en),
            }).ToArray()),
        };
        o["draw_order"] = new JsonArray(c.DrawPile.Select(x => (JsonNode)new JsonArray(x.DefId, x.Upgraded)).ToArray());
        if (OldDump.Extended) o["potions"] = OldSetup.Belt(c.PotionSlots, c.MaxPotionSlots);
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
