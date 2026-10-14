// Re-anchoring (referee/SPEED_PLAN.md step 6): every recorded step of a referee fight becomes an
// independent test. For step j the emulator is put into the GAME's state S(j-1), the game's
// action a(j) is applied, and the result is dumped to be compared with the game's S(j).
// Usage: EmuDump reanchor <referee out.json> [--seed SEED]... [--scramble] [--stale enemy,cards,relics,counters|all]
//        [--perturb-moves] > reanchor.json
//
// What the game's snapshot (referee/driver/Combat.cs Snapshot()) cannot say is taken from the
// emulator's own state after its previous step: relic counters, enemy AI state, per-card
// modifiers, per-turn and per-combat counters, the potion belt. The snapshot's fields (player
// HP/block/energy/powers, the four piles in order, living enemies, RNG streams' seed and counter)
// are written over it, and the enemies' next moves where the record has them (see "enemy moves"
// below). tools/fidelity/reanchor_compare.py judges the result and decides where a fight stops
// being re-anchorable.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class Reanchor
{
    static readonly Dictionary<string, int> EnemyBySnake = Load("enemies");
    static readonly Dictionary<string, BuffId> BuffByPower = BuffTable();

    // Representation, not rules (see tools/fidelity/fight_compare.py NOT_POWERS): the emulator
    // keeps a stun as a buff the game's power list never shows, so it is never overwritten.
    static readonly HashSet<BuffId> NotPowers = [BuffId.Stunned];

    /// <summary>The keys the game's own power list can hold (data/id_map.json "powers").</summary>
    static readonly HashSet<string> GamePowerKeys = Load("powers").Keys.Select(PowerKey).ToHashSet();

    /// <summary>Power keys the emulator has a buff for, and buffs of its own that are no game power.</summary>
    public static JsonObject PowerKeys() => new()
    {
        ["buff_keys"] = new JsonArray(BuffByPower.Keys.Select(k => (JsonNode)k).ToArray()),
        ["internal_keys"] = new JsonArray(BuffByPower.Keys.Where(k => !GamePowerKeys.Contains(k)).Select(k => (JsonNode)k).ToArray()),
    };

    /// <summary>The emulator buff for a game power id (VULNERABLE_POWER), or null.</summary>
    internal static BuffId? BuffFor(string gamePower) => BuffByPower.TryGetValue(PowerKey(gamePower), out var b) ? b : null;

    /// <summary>A buff only the emulator has (a game power it models in pieces, or a bookkeeping marker).</summary>
    static bool Internal(BuffId id) => !GamePowerKeys.Contains(PowerKey(id.ToString()));

    // tools/fidelity/power_aliases.json, the table fight_compare.py compares with: a game power the
    // emulator keeps under another name, and the TemporaryStrength/Dexterity/Focus subclasses it folds
    // into one buff. Writing back by name alone dropped them (Mangle's restore never came).
    static readonly JsonObject Aliases = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "power_aliases.json")))!.AsObject();
    static readonly Dictionary<string, string> GameRename = Aliases["emu_rename"]!.AsObject().ToDictionary(kv => (string)kv.Value!, kv => kv.Key);
    static readonly Dictionary<string, string> Folds = Aliases["folds"]!.AsObject()
        .SelectMany(kv => kv.Value!.AsArray().Select(sub => ((string)sub!, kv.Key))).ToDictionary(t => t.Item1, t => t.Item2);
    static readonly HashSet<BuffId> RenameTargets = GameRename.Values.Where(BuffByPower.ContainsKey).Select(k => BuffByPower[k]).ToHashSet();

    /// <summary>The game power an emulator-only buff is a piece of (InfernoSelfDamage of INFERNO), by name; null for pure bookkeeping.</summary>
    static string? OwnerPower(BuffId id)
    {
        if (OwnerCache.TryGetValue(id, out var owner)) return owner;
        string key = PowerKey(id.ToString());
        owner = GamePowerKeys.Where(p => key.StartsWith(p + "_", StringComparison.Ordinal)).OrderByDescending(p => p.Length).FirstOrDefault();
        OwnerCache[id] = owner;
        return owner;
    }

    static readonly Dictionary<BuffId, string?> OwnerCache = new();

    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    static string PowerKey(string name)
    {
        var n = Snake(name);
        return n.EndsWith("_POWER") ? n[..^"_POWER".Length] : n;
    }

    static Dictionary<string, int> Load(string kind)
    {
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))![kind]!.AsObject();
        return map.ToDictionary(kv => Snake(kv.Key), kv => (int)kv.Value!);
    }

    static Dictionary<string, BuffId> BuffTable()
    {
        var table = new Dictionary<string, BuffId>();
        foreach (var id in Enum.GetValues<BuffId>())
        {
            var key = PowerKey(id.ToString());
            // Two enum members with one key would make the game's power ambiguous: keep the
            // first and say so.
            if (!table.TryAdd(key, id)) Console.Error.WriteLine($"reanchor: power key {key} names both {table[key]} and {id}; using {table[key]}");
        }

        return table;
    }

    /// <summary>
    /// Diagnostics that test the anchoring itself. <c>Scramble</c> wrecks every field the game's
    /// snapshot covers before it is written back, so a fight that matches anyway proves the write is
    /// complete. <c>Stale</c> starts every step from the fight's first state instead of the previous
    /// step's, with only the named groups of hidden state (enemy, cards, relics, counters) stale, to
    /// show what the snapshot alone cannot rebuild. <c>PerturbMoves</c> tests the move anchoring: before
    /// each step every cycling enemy is moved one position on (and re-announces), and the monster AI
    /// draws from another stream, so the moves must be put back on the game's for the fight to match.
    /// </summary>
    /// <c>Trace</c> records what happened inside each step (tools/emudump/Trace.cs) for steps up to
    /// <c>TraceTo</c> (all when null), and the run ends there.
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
            Sts2Emulator.Core.Run.RunEngine e;
            using (Prof.Scope("enter_room")) e = Fight.EnterRoom(game, outp);
            var combat = e.State.ActiveCombat ?? throw new InvalidOperationException($"no combat after entering the room; phase {e.State.Phase}");
            var rng = e.State.ActiveCombatRng!;
            var gsteps = game["steps"]!.AsArray();
            var gameChoices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
            steps.Add(new JsonObject { ["action"] = null, ["state"] = DumpLite(combat, true) });
            var start = combat.Clone();
            int last = mode.TraceTo is int upTo ? Math.Min(upTo + 1, gsteps.Count) : gsteps.Count;
            for (int j = 1; j < last; j++)
            {
                var prior = gsteps[j - 1]!["state"]!.AsObject();
                var a = gsteps[j]!["action"]!.AsObject();
                var cloneSpan = Prof.Scope("clone");
                var q = combat.Clone();
                var qr = rng.Clone();
                cloneSpan.Dispose();
                if (mode.Stale != null) GoStale(q, start, mode.Stale);
                if (mode.Scramble) Scramble(q);
                var patched = new JsonArray();
                var rec = new JsonObject { ["action"] = a.DeepClone(), ["patched"] = patched };
                steps.Add(rec);
                string? problem;
                using (Prof.Scope("anchor")) problem = Anchor(q, prior, patched);
                if (problem != null)
                {
                    rec["anchor_error"] = problem;  // this step cannot be tried, so neither can the rest
                    break;
                }

                // The moves the enemies will make: the game's, where the emulator's AI can be put there.
                if (mode.PerturbMoves) ShiftCycles(q);
                using (Prof.Scope("anchor_moves")) rec["moves"] = AnchorMoves(q, prior, patched, rec, gsteps, j);
                if (mode.PerturbMoves && q.AiRng is CountingRandom ai0) q.AiRng = new CountingRandom(ai0.Seed ^ 0x2545F491, ai0.CallCount);
                var cloneSpan2 = Prof.Scope("clone");
                var pre = q.Clone();
                var preRng = qr.Clone();
                cloneSpan2.Dispose();
                var stepChoices = gameChoices.Where(c => (int?)c!["step"] == j && (string?)c["kind"] == "cards").ToList();
                try
                {
                    if (mode.Trace)
                    {
                        rec["hidden_pre"] = EmuTrace.Hidden(q);
                        EmuTrace.Begin(q);
                    }

                    bool terminal, won;
                    JsonArray chosen;
                    try
                    {
                        using (Prof.Scope("apply")) (terminal, won, chosen) = Apply(q, qr, a, stepChoices);
                    }
                    finally
                    {
                        if (mode.Trace) rec["trace"] = EmuTrace.End();
                    }

                    if (mode.Trace) rec["hidden"] = EmuTrace.Hidden(q);
                    rec["choices"] = chosen;
                    rec["fallback_rng_calls"] = qr.CallCount - preRng.CallCount;
                    bool gameGoesOn = j + 1 < gsteps.Count && (gsteps[j]!["state"]!["in_combat"]?.GetValue<bool>() ?? false);
                    // The fight ends here in the game too: the state to judge is the one the game's snapshot shows
                    // after it, so the fight is finished the way the run finishes it (SyncAfterCombat, the
                    // after-combat relics). Where the game goes on, the step differs in in_combat, as it always did.
                    using (Prof.Scope("dumplite")) rec["state"] = terminal && !gameGoesOn ? EndState(e, q, qr, won) : DumpLite(q, !terminal);
                    if (terminal && !gameGoesOn) break;
                    // The emulator ended a fight the game continues: carry on from the state before the action.
                    (combat, rng) = terminal ? (pre, preRng) : (q, qr);
                    if (!terminal)
                    {
                        using (Prof.Scope("carry_moves")) CarryGameMoves(q, pre, preRng, a, stepChoices, gsteps[j]!["state"]!.AsObject(), rec);
                    }
                }
                catch (Exception ex)
                {
                    rec["error"] = ex.GetType().Name + ": " + ex.Message;
                    rec["state"] = DumpLite(pre, true);
                    (combat, rng) = (pre, preRng);
                }
            }
        }
        catch (Exception ex)
        {
            outp["error"] = ex.ToString();
        }

        return outp;
    }

    /// <summary>The game's action, then the card choices the game made for it, as the plain replay applies them.</summary>
    static (bool Terminal, bool Won, JsonArray Chosen) Apply(CombatState q, CountingRandom qr, JsonObject a, List<JsonNode?> choices)
    {
        var first = StepOnce(q, qr, a);
        bool terminal = first.Terminal, won = first.Terminal && first.PlayerWon;
        var chosen = new JsonArray();
        foreach (var choice in choices)
        {
            chosen.Add(Fight.ApplyChoice(() => q, i =>
            {
                var r = CombatEngine.Step(q, i, qr);
                terminal |= r.Terminal;
                won |= r.Terminal && r.PlayerWon;
            }, choice!.AsObject()));
        }

        if (!terminal && q.PendingSelection is { } extra)  // a won fight leaves no screen to answer
        {
            chosen.Add(new JsonObject { ["unexpected_selection"] = extra.Kind.ToString() });
            q.PendingSelection = null;  // nobody is there to answer it; the next anchor starts clean
        }

        return (terminal, won, chosen);
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
        // target_slot indexes the game's living enemies, and Anchor made the emulator's roster the same list.
        return CombatEngine.Step(q, (int)a["hand_index"]!, qr, (int)a["target_slot"]!);
    }

    /// <summary>
    /// The state after the step that ended the fight in both the game and the emulator. A won fight is put
    /// through the run's own end of combat on the engine <see cref="Fight.EnterRoom" /> made (DebugResolveCombat:
    /// the sync back to the run and the after-combat hooks, Burning Blood's heal among them), so the hit points,
    /// potions, relics and RNG positions written are the run's, as the game's snapshot shows them. A lost fight
    /// ends the run and is written as lost and nothing else (see Fight.Final).
    /// The run's own streams stand wherever the emulator's earlier steps left them, and the anchored combat's where
    /// the game's did (Anchor, PatchRng); the sync moves the run's to the combat's and a stream cannot go back, so
    /// the run's start from the seed and end where the combat did.
    /// </summary>
    static JsonObject EndState(RunEngine e, CombatState q, CountingRandom qr, bool won)
    {
        if (!won) return DumpLite(q, false, Fight.Final(e.State, false));
        e.State.ActiveCombat = q;
        e.State.ActiveCombatRng = qr;
        e.State.Rng = new Sts2Emulator.Core.Rng.RunRngSet(e.State.StringSeed);
        if (e.DebugResolveCombat() == -1) throw new InvalidOperationException($"no fight to finish; phase {e.State.Phase}");
        return DumpLite(q, false, Fight.Final(e.State, true));
    }

    // ── enemy moves ─────────────────────────────────────────────────────────────────
    //
    // The game records each enemy's next move (referee/driver/Combat.cs Intents). Before a step the
    // emulator's announced move is compared with it in tools/fidelity/fight_compare.py's vocabulary
    // (game_intent / emu_intent): the kind of the first intent, and the damage-carrying intent's base
    // damage and hits. Where they differ, the emulator's AI is put on the game's move if that can be
    // done from the AI's own state; an end of turn whose moves stay unknown is left unjudged.

    static readonly Dictionary<string, string> GameKind = new()
    {
        ["Attack"] = "ATTACK", ["DeathBlow"] = "ATTACK", ["Defend"] = "DEFEND", ["Buff"] = "BUFF", ["Debuff"] = "DEBUFF",
        ["DebuffStrong"] = "DEBUFF", ["StatusCard"] = "DEBUFF", ["CardDebuff"] = "DEBUFF", ["Stun"] = "STUN",
    };

    /// <summary>The game's next move for one enemy, or null for a record made before intents were recorded.</summary>
    internal static string? GameMove(JsonNode enemy)
    {
        if (enemy["intents"] is not JsonArray its) return null;
        string kind = its.Count == 0 ? "NONE" : GameKind.GetValueOrDefault((string)its[0]![0]!, "OTHER");
        var atk = its.Select(x => x!.AsArray()).FirstOrDefault(x => x.Count >= 3);
        return atk == null ? kind : $"{kind} {(int)atk[1]!}x{(int)atk[2]!}";
    }

    internal static string EmuMove(EnemyState en)
    {
        if (BuffSystem.Get(en.Buffs, BuffId.Stunned) > 0) return "STUN";  // the game's StunIntent move
        var it = en.CurrentIntent;
        string kind = it.CarriesDamage ? "ATTACK" : it.Type switch
        {
            IntentType.Attack => "ATTACK", IntentType.Defend => "DEFEND", IntentType.Buff => "BUFF", IntentType.Debuff => "DEBUFF", _ => "OTHER",
        };
        static bool Hits(Intent x) => x.Type == IntentType.Attack || x.CarriesDamage;
        Intent? atk = Hits(it) ? it : en.SecondaryIntent is { } s && Hits(s) ? s : null;
        return atk is { } x ? $"{kind} {x.Magnitude}x{x.Hits}" : kind;
    }

    /// <summary>
    /// "same" when every living enemy announces the game's move, "set" when some had to be put there,
    /// "unknown" when one could not be (listed in moves_unknown), "unrecorded" for a record without intents.
    /// </summary>
    static string AnchorMoves(CombatState q, JsonObject g, JsonArray patched, JsonObject rec, JsonArray gsteps, int j)
    {
        var gen = g["enemies"]!.AsArray();
        if (gen.Any(x => x!["intents"] == null)) return "unrecorded";
        var unknown = new JsonArray();
        bool set = false;
        for (int i = 0; i < gen.Count && i < q.Enemies.Count; i++)
        {
            if (q.Enemies[i].Hp <= 0) continue;
            string want = GameMove(gen[i]!)!;
            if (EmuMove(q.Enemies[i]) == want) continue;
            if (SetMove(q, q.Enemies[i], want, LaterMoves(gsteps, j, i)))
            {
                patched.Add($"enemies[{i}].move");
                set = true;
            }
            else
            {
                unknown.Add(i);
            }
        }

        if (unknown.Count > 0)
        {
            rec["moves_unknown"] = unknown;
            return "unknown";
        }

        return set ? "set" : "same";
    }

    /// <summary>
    /// For an AI whose choice is a function of its MoveIndex alone (a cycle: re-selecting at its own
    /// position gives back what it announced, without a draw and without touching the AI's memory), a
    /// nearby position that announces the game's move. A cycle can announce the same move at two
    /// positions (Fuzzy Wurm Crawler's FIRST_ACID_GOOP and ACID_GOOP), so the positions are ranked by
    /// how many of the game's later moves the cycle then goes on to announce, then by distance.
    /// </summary>
    static bool SetMove(CombatState q, EnemyState en, string want, List<string> later)
    {
        if (!Reselect(q, en, en.MoveIndex, out var same) || EmuMove(same) != EmuMove(en)) return false;
        EnemyState? best = null;
        int bestScore = -1;
        foreach (int d in (int[])[1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6])
        {
            int m = en.MoveIndex + d;
            if (m < 0 || !Reselect(q, en, m, out var c) || EmuMove(c) != want) continue;
            int score = 0;
            while (score < later.Count && Reselect(q, en, m + score + 1, out var n) && EmuMove(n) == later[score]) score++;
            if (score > bestScore) (best, bestScore) = (c, score);
        }

        if (best == null) return false;
        en.MoveIndex = best.MoveIndex;
        en.CurrentIntent = best.CurrentIntent;
        en.SecondaryIntent = best.SecondaryIntent;
        return true;
    }

    /// <summary>
    /// The game's moves for enemy <paramref name="i"/> of step j-1's roster after each later end of turn,
    /// for as long as the roster keeps its shape (a death or a summon ends the list).
    /// </summary>
    static List<string> LaterMoves(JsonArray gsteps, int j, int i)
    {
        var roster = gsteps[j - 1]!["state"]!["enemies"]!.AsArray();
        string name = (string)roster[i]!["name"]!;
        var later = new List<string>();
        for (int k = j; k < gsteps.Count && later.Count < 6; k++)
        {
            if ((string?)gsteps[k]!["action"]?["kind"] != "end_turn") continue;
            var en = gsteps[k]!["state"]!["enemies"]?.AsArray();
            if (en == null || en.Count != roster.Count || (string?)en[i]!["name"] != name || (int)en[i]!["hp"]! <= 0) break;
            if (GameMove(en[i]!) is not { } m) break;
            later.Add(m);
        }

        return later;
    }

    /// <summary>A copy of the enemy re-selected at <paramref name="moveIndex"/>; false if that drew or changed the AI's memory.</summary>
    static bool Reselect(CombatState q, EnemyState en, int moveIndex, out EnemyState c)
    {
        c = en.Clone();
        c.MoveIndex = moveIndex;
        string before = AiMemory(c);
        var probe = new CountingRandom(0x5eed);
        EnemyAI.ChooseIntents([c], q.Turn, probe, probe, q.AscensionLevel);
        return probe.CallCount == 0 && AiMemory(c) == before;
    }

    static string AiMemory(EnemyState e) =>
        $"{e.LastMove},{e.LastMoveRepeats},{e.OnceOnlyMoveUsed},{string.Join('.', e.MoveHistory)},{e.StartsOnBranch},{e.RollsNextMove},{e.LastBranch},{e.RepeatStreak},{e.BranchCooldown}";

    /// <summary>
    /// After a step whose result announces another move than the game's for an enemy whose choice is a
    /// draw (a RandomBranchState), the step is tried again from the same state with the monster AI on
    /// other streams; an attempt that changes nothing but the moves, and in which that enemy picks the
    /// game's move, gives its AI memory to the state the next step starts from. The step's own verdict
    /// is unchanged: it is judged on what the emulator did.
    /// </summary>
    static void CarryGameMoves(CombatState q, CombatState pre, CountingRandom preRng, JsonObject a, List<JsonNode?> choices,
        JsonObject g, JsonObject rec)
    {
        var gen = g["enemies"]!.AsArray();
        if (gen.Any(x => x!["intents"] == null) || q.AiRng is not CountingRandom) return;
        var living = q.Enemies.Where(x => x.Hp > 0).ToList();
        var glive = gen.Where(x => (int)x!["hp"]! > 0).ToList();
        if (living.Count != glive.Count
            || !living.Zip(glive).All(t => t.First.DefId == EnemyBySnake.GetValueOrDefault((string)t.Second!["name"]!, -1))) return;
        var wrong = Enumerable.Range(0, living.Count).Where(i => EmuMove(living[i]) != GameMove(glive[i]!)).ToList();
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
                if (Apply(t, tr, a, choices).Terminal || Shown(t) != shown) continue;
            }
            catch (Exception)
            {
                continue;
            }

            var tl = t.Enemies.Where(x => x.Hp > 0).ToList();
            foreach (int i in wrong.ToList())
            {
                if (EmuMove(tl[i]) != GameMove(glive[i]!)) continue;
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
            if (!Reselect(q, en, en.MoveIndex, out var same) || EmuMove(same) != EmuMove(en)
                || !Reselect(q, en, en.MoveIndex + 1, out var next)) continue;
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
        q.PlayerBuffs = q.PlayerBuffs.Where(b => NotPowers.Contains(b.Id) || Internal(b.Id)).ToList();
        foreach (var en in q.Enemies)
        {
            en.Hp = 1; en.MaxHp += 3; en.Block += 5;
            en.Buffs = en.Buffs.Where(b => NotPowers.Contains(b.Id) || Internal(b.Id)).ToList();
        }

        CountingRandom? Off(CountingRandom? r) => r == null ? null : new CountingRandom(r.Seed ^ 0x5a5a, 3);
        q.ShuffleRng = Off(q.ShuffleRng); q.TargetRng = Off(q.TargetRng); q.CardSelectionRng = Off(q.CardSelectionRng);
        q.CardGenerationRng = Off(q.CardGenerationRng); q.EnergyCostRng = Off(q.EnergyCostRng);
        q.PotionGenerationRng = Off(q.PotionGenerationRng); q.OrbGenerationRng = Off(q.OrbGenerationRng);
        q.NicheHpRng = Off(q.NicheHpRng);
        if (q.AiRng is CountingRandom ai) q.AiRng = Off(ai);
    }

    static readonly string[] Observed = ["PlayerHp", "PlayerMaxHp", "PlayerBlock", "Energy", "Turn"];

    static void GoStale(CombatState q, CombatState start, HashSet<string> groups)
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
            CardInstance Plain(CardInstance c) => new(c.DefId, c.Upgraded);
            q.Hand = q.Hand.Select(Plain).ToList(); q.DrawPile = q.DrawPile.Select(Plain).ToList();
            q.DiscardPile = q.DiscardPile.Select(Plain).ToList(); q.ExhaustPile = q.ExhaustPile.Select(Plain).ToList();
        }

        if (all || groups.Contains("relics")) q.Relics = start.Relics.ToList();
        if (all || groups.Contains("counters"))
        {
            foreach (var f in typeof(CombatState).GetFields().Where(f => !f.IsStatic && !Observed.Contains(f.Name)
                         && (f.FieldType == typeof(int) || f.FieldType == typeof(bool))))
            {
                f.SetValue(q, f.GetValue(start));
            }
        }
    }

    // ── writing the game's state over the emulator's ────────────────────────────────

    /// <summary>
    /// Overwrites <paramref name="q"/>'s observable fields with the game's state; only fields that
    /// differ are touched, and each one touched is named in <paramref name="patched"/>. Returns a
    /// reason when the state cannot be built (an enemy or card the emulator has no id for, an
    /// enemy the game has and the emulator lacks).
    /// </summary>
    internal static string? Anchor(CombatState q, JsonObject g, JsonArray patched)
    {
        void Note(string what) => patched.Add(what);
        try
        {
            var p = g["player"]!.AsObject();
            int Int(JsonNode? n) => (int)n!;
            if (g["round"] is JsonNode round && q.Turn != (int)round - 1) { q.Turn = (int)round - 1; Note("turn"); }
            if (q.PlayerHp != Int(p["hp"])) { q.PlayerHp = Int(p["hp"]); Note("player.hp"); }
            if (q.PlayerMaxHp != Int(p["max_hp"])) { q.PlayerMaxHp = Int(p["max_hp"]); Note("player.max_hp"); }
            if (q.PlayerBlock != Int(p["block"])) { q.PlayerBlock = Int(p["block"]); Note("player.block"); }
            if (q.Energy != Int(p["energy"])) { q.Energy = Int(p["energy"]); Note("player.energy"); }
            if (PatchBuffs(q.PlayerBuffs, p["powers"]!.AsArray(), out var buffs, patched, "player.powers")) q.PlayerBuffs = buffs;

            PatchPiles(q, g, patched);

            string? roster = PatchEnemies(q, g["enemies"]!.AsArray(), patched);
            if (roster != null) return roster;

            PatchRng(q, g["rng"]!.AsObject(), patched);
            if (g["potions"] is JsonArray belt) PatchPotions(q, belt, patched);
            return null;
        }
        catch (KeyNotFoundException ex)
        {
            return "no emulator id for " + ex.Message;
        }
    }

    static bool PatchBuffs(List<BuffState> cur, JsonArray powers, out List<BuffState> result, JsonArray patched, string where)
    {
        var want = new List<BuffState>();
        var folded = new Dictionary<BuffId, int>();
        var present = new HashSet<string>();
        foreach (var pw in powers)
        {
            var name = (string)pw![0]!;
            int amount = (int)pw[1]!;
            string key = PowerKey(name);
            present.Add(key);
            if (amount == 0) continue;  // the emulator holds no buff at zero
            bool fold = Folds.TryGetValue(key, out var into);
            string emuKey = fold ? into! : GameRename.GetValueOrDefault(key, key);
            if (!BuffByPower.TryGetValue(emuKey, out var id)) continue;  // modelled in pieces, which Keep leaves alone
            if (fold) folded[id] = folded.GetValueOrDefault(id) + amount;
            else want.Add(new BuffState(id, amount));
        }

        want.AddRange(folded.Where(kv => kv.Value != 0).Select(kv => new BuffState(kv.Key, kv.Value)));

        // The emulator's own buffs are hidden state the snapshot cannot show, carried over: a piece of a
        // game power (Inferno's InfernoSelfDamage, Crimson Mantle's, Tank's, Beacon of Hope's ...) while
        // that power is there, and pure bookkeeping always. Dropping InfernoSelfDamage when every game
        // power had a buff stopped the next turn's Inferno.
        bool Keep(BuffState b) => NotPowers.Contains(b.Id)
            || (Internal(b.Id) && !RenameTargets.Contains(b.Id) && (OwnerPower(b.Id) is not { } owner || present.Contains(owner)));
        var have = cur.Where(b => !Keep(b)).ToList();
        // Compared on what the snapshot shows (id, amount). Data is hidden per-instance state (an
        // instanced power's second number, e.g. The Bomb's damage) that the game snapshot lacks.
        static (BuffId, int) Shown(BuffState b) => (b.Id, b.Magnitude);
        bool same = have.Count == want.Count && have.Select(Shown).OrderBy(x => x).SequenceEqual(want.Select(Shown).OrderBy(x => x));
        result = cur;
        if (same) return false;
        // Keep the emulator's order for powers it already has; the rest follow in the game's order.
        var next = new List<BuffState>();
        var left = want.ToList();
        foreach (var b in cur)
        {
            if (Keep(b)) { next.Add(b); continue; }
            int at = left.FindIndex(w => w.Id == b.Id);
            if (at < 0) continue;
            next.Add(left[at] with { Data = b.Data });  // the k-th instance keeps its own hidden Data
            left.RemoveAt(at);
        }

        next.AddRange(left);
        result = next;
        patched.Add(where);
        return true;
    }

    static void PatchPiles(CombatState q, JsonObject g, JsonArray patched)
    {
        var gh = Cards(g["hand"]!);
        var gd = Cards(g["discard"]!);
        var ge = Cards(g["exhaust"]!);
        var gw = Cards(g["draw"]!);
        bool same(List<CardInstance> have, List<(int Id, bool Up)> want) =>
            have.Count == want.Count && have.Select((c, i) => c.DefId == want[i].Id && c.Upgraded == want[i].Up).All(x => x);
        bool sh = same(q.Hand, gh), sd = same(q.DiscardPile, gd), se = same(q.ExhaustPile, ge), sw = same(q.DrawPile, gw);
        if (sh && sd && se && sw) return;

        // Cards that keep their identity (cost changes, bonus damage, enchantments, ...) are kept
        // where the pile still has them at the same place, then taken from whichever pile now holds
        // an identical card; a card the emulator has nowhere is built fresh.
        var piles = new[] { q.Hand, q.DiscardPile, q.ExhaustPile, q.DrawPile };
        var wants = new[] { gh, gd, ge, gw };
        var names = new[] { "hand", "discard", "exhaust", "draw" };
        var used = piles.Select(p => new bool[p.Count]).ToArray();
        var built = wants.Select(w => new CardInstance?[w.Count]).ToArray();
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < wants[k].Count && i < piles[k].Count; i++)
            {
                if (piles[k][i].DefId == wants[k][i].Id && piles[k][i].Upgraded == wants[k][i].Up)
                {
                    built[k][i] = piles[k][i];
                    used[k][i] = true;
                }
            }
        }

        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < wants[k].Count; i++)
            {
                if (built[k][i] != null) continue;
                var (id, up) = wants[k][i];
                CardInstance? found = null;
                for (int off = 0; off < 4 && found == null; off++)
                {
                    int m = (k + off) % 4;
                    for (int x = 0; x < piles[m].Count; x++)
                    {
                        if (used[m][x] || piles[m][x].DefId != id || piles[m][x].Upgraded != up) continue;
                        found = piles[m][x];
                        used[m][x] = true;
                        break;
                    }
                }

                built[k][i] = found ?? new CardInstance(id, up);
            }
        }

        var results = built.Select(b => b.Select(c => c!.Value).ToList()).ToArray();
        if (!sh) { q.Hand = results[0]; patched.Add("hand"); }
        if (!sd) { q.DiscardPile = results[1]; patched.Add("discard"); }
        if (!se) { q.ExhaustPile = results[2]; patched.Add("exhaust"); }
        if (!sw)
        {
            q.DrawPile = results[3];
            q.KnownTopCount = Math.Clamp(q.KnownTopCount, 0, q.DrawPile.Count);
            q.KnownBottomCount = Math.Clamp(q.KnownBottomCount, 0, q.DrawPile.Count - q.KnownTopCount);
            patched.Add("draw");
        }
    }

    static List<(int Id, bool Up)> Cards(JsonNode pile) =>
        pile.AsArray().Select(c => (Fight.CardIdBySnake((string)c![0]!), (bool)c[1]!)).ToList();

    static string? PatchEnemies(CombatState q, JsonArray gen, JsonArray patched)
    {
        var want = gen.Select(en => (Name: (string)en!["name"]!, Def: EnemyBySnake[(string)en["name"]!], Node: en)).ToList();
        // The game keeps only living creatures. Pair each with the emulator's creature of that
        // kind in order; emulator creatures nobody pairs with are gone in the game's state.
        var roster = new List<EnemyState>();
        int from = 0;
        foreach (var w in want)
        {
            int at = q.Enemies.FindIndex(from, en => en.DefId == w.Def);
            if (at < 0) return $"the game has {w.Name} where the emulator has none to pair with it";
            roster.Add(q.Enemies[at]);
            from = at + 1;
        }

        if (roster.Count != q.Enemies.Count || !roster.Zip(q.Enemies).All(t => ReferenceEquals(t.First, t.Second)))
        {
            q.Enemies = roster;
            patched.Add("enemies.roster");
        }

        for (int i = 0; i < want.Count; i++)
        {
            var en = roster[i];
            var n = want[i].Node;
            if (en.Hp != (int)n["hp"]!) { en.Hp = (int)n["hp"]!; patched.Add($"enemies[{i}].hp"); }
            if (en.MaxHp != (int)n["max_hp"]!) { en.MaxHp = (int)n["max_hp"]!; patched.Add($"enemies[{i}].max_hp"); }
            if (en.Block != (int)n["block"]!) { en.Block = (int)n["block"]!; patched.Add($"enemies[{i}].block"); }
            if (PatchBuffs(en.Buffs, n["powers"]!.AsArray(), out var buffs, patched, $"enemies[{i}].powers")) en.Buffs = buffs;
        }

        return null;
    }

    /// <summary>The potion belt, slot by slot, when the game's snapshot records it (setup fights).</summary>
    static void PatchPotions(CombatState q, JsonArray belt, JsonArray patched)
    {
        var want = belt.Select(p => p == null ? 0 : Setup.PotionId((string)p!) ?? throw new KeyNotFoundException("potion " + (string)p!)).ToArray();
        bool same = q.MaxPotionSlots == want.Length && want.Select((p, i) => q.PotionSlots[i] == p).All(x => x)
            && q.PotionSlots.Skip(want.Length).All(p => p == 0);
        if (same) return;
        Array.Clear(q.PotionSlots);
        for (int i = 0; i < Math.Min(want.Length, q.PotionSlots.Length); i++) q.PotionSlots[i] = want[i];
        q.MaxPotionSlots = want.Length;
        patched.Add("potions");
    }

    /// <summary>The streams' seed and counter come from the game; a stream already there stays as it is.</summary>
    static void PatchRng(CombatState q, JsonObject rng, JsonArray patched)
    {
        CountingRandom? Fit(string stream, CountingRandom? cur)
        {
            if (rng[stream] is not JsonObject g) return cur;
            int seed = unchecked((int)(uint)(long)g["seed"]!);
            int counter = (int)g["counter"]!;
            if (cur != null && cur.Seed == seed && cur.CallCount == counter) return cur;
            patched.Add($"rng.{stream}");
            return new CountingRandom(seed, counter);
        }

        q.ShuffleRng = Fit("Shuffle", q.ShuffleRng);
        q.TargetRng = Fit("CombatTargets", q.TargetRng);
        q.CardSelectionRng = Fit("CombatCardSelection", q.CardSelectionRng);
        q.CardGenerationRng = Fit("CombatCardGeneration", q.CardGenerationRng);
        q.EnergyCostRng = Fit("CombatEnergyCosts", q.EnergyCostRng);
        q.PotionGenerationRng = Fit("CombatPotionGeneration", q.PotionGenerationRng);
        q.OrbGenerationRng = Fit("CombatOrbGeneration", q.OrbGenerationRng);
        q.NicheHpRng = Fit("Niche", q.NicheHpRng);
        if (q.AiRng is CountingRandom ai) q.AiRng = Fit("MonsterAi", ai);
    }

    // ── what tools/fidelity/fight_compare.py reads, and no more ─────────────────────

    static JsonObject DumpLite(CombatState c, bool inCombat, JsonObject? final = null)
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
                ["intent"] = Fight.IntentJson(en),
            }).ToArray()),
        };
        o["draw_order"] = new JsonArray(c.DrawPile.Select(x => (JsonNode)new JsonArray(x.DefId, x.Upgraded)).ToArray());
        if (Setup.Extended) o["potions"] = Setup.Belt(c.PotionSlots, c.MaxPotionSlots);
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
