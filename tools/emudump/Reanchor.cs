// Re-anchoring (referee/SPEED_PLAN.md step 6): every recorded step of a referee fight becomes an
// independent test. For step j the emulator is put into the GAME's state S(j-1), the game's
// action a(j) is applied, and the result is dumped to be compared with the game's S(j).
// Usage: EmuDump reanchor <referee out.json> [--seed SEED]... [--scramble] [--stale enemy,cards,relics,counters|all] > reanchor.json
//
// What the game's snapshot (referee/driver/Combat.cs Snapshot()) cannot say is taken from the
// emulator's own state after its previous step: relic counters, enemy move/AI state and intent,
// per-card modifiers, per-turn and per-combat counters, the potion belt. The snapshot's
// fields (player HP/block/energy/powers, the four piles in order, living enemies, RNG
// streams' seed and counter) are written over it. tools/fidelity/reanchor_compare.py judges
// the result and decides where a fight stops being re-anchorable.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;

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

    /// <summary>A buff only the emulator has (a game power it models in pieces, or a bookkeeping marker).</summary>
    static bool Internal(BuffId id) => !GamePowerKeys.Contains(PowerKey(id.ToString()));

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
    /// show what the snapshot alone cannot rebuild.
    /// </summary>
    public sealed record Mode(bool Scramble = false, HashSet<string>? Stale = null);

    public static JsonObject Run(JsonObject game, Mode? mode = null)
    {
        mode ??= new Mode();
        var outp = new JsonObject();
        var steps = new JsonArray();
        outp["steps"] = steps;
        try
        {
            var e = Fight.EnterRoom(game, outp);
            var combat = e.State.ActiveCombat ?? throw new InvalidOperationException($"no combat after entering the room; phase {e.State.Phase}");
            var rng = e.State.ActiveCombatRng!;
            var gsteps = game["steps"]!.AsArray();
            var gameChoices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
            steps.Add(new JsonObject { ["action"] = null, ["state"] = DumpLite(combat, true) });
            var start = combat.Clone();
            for (int j = 1; j < gsteps.Count; j++)
            {
                var prior = gsteps[j - 1]!["state"]!.AsObject();
                var a = gsteps[j]!["action"]!.AsObject();
                var q = combat.Clone();
                var qr = rng.Clone();
                if (mode.Stale != null) GoStale(q, start, mode.Stale);
                if (mode.Scramble) Scramble(q);
                var patched = new JsonArray();
                var rec = new JsonObject { ["action"] = a.DeepClone(), ["patched"] = patched };
                steps.Add(rec);
                string? problem = Anchor(q, prior, patched);
                if (problem != null)
                {
                    rec["anchor_error"] = problem;  // this step cannot be tried, so neither can the rest
                    break;
                }

                var pre = q.Clone();
                var preRng = qr.Clone();
                try
                {
                    bool terminal = StepOnce(q, qr, a);
                    var chosen = new JsonArray();
                    foreach (var choice in gameChoices.Where(c => (int?)c!["step"] == j && (string?)c["kind"] == "cards"))
                    {
                        chosen.Add(Fight.ApplyChoice(() => q, i => terminal |= CombatEngine.Step(q, i, qr).Terminal, choice!.AsObject()));
                    }

                    if (!terminal && q.PendingSelection is { } extra)  // a won fight leaves no screen to answer
                    {
                        chosen.Add(new JsonObject { ["unexpected_selection"] = extra.Kind.ToString() });
                        q.PendingSelection = null;  // nobody is there to answer it; the next anchor starts clean
                    }

                    rec["choices"] = chosen;
                    rec["fallback_rng_calls"] = qr.CallCount - preRng.CallCount;
                    rec["state"] = DumpLite(q, !terminal);
                    bool gameGoesOn = j + 1 < gsteps.Count && (gsteps[j]!["state"]!["in_combat"]?.GetValue<bool>() ?? false);
                    if (terminal && !gameGoesOn) break;
                    // The emulator ended a fight the game continues: carry on from the state before the action.
                    (combat, rng) = terminal ? (pre, preRng) : (q, qr);
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

    /// <summary>Applies the game's action (end turn or a card play); true when the fight ended.</summary>
    static bool StepOnce(CombatState q, CountingRandom qr, JsonObject a)
    {
        if ((string)a["kind"]! == "end_turn") return CombatEngine.Step(q, q.Hand.Count, qr).Terminal;
        // target_slot indexes the game's living enemies, and Anchor made the emulator's roster the same list.
        return CombatEngine.Step(q, (int)a["hand_index"]!, qr, (int)a["target_slot"]!).Terminal;
    }

    // ── diagnostics ─────────────────────────────────────────────────────────────────

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
    static string? Anchor(CombatState q, JsonObject g, JsonArray patched)
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
        bool unmapped = false;
        foreach (var pw in powers)
        {
            var name = (string)pw![0]!;
            int amount = (int)pw[1]!;
            if (amount == 0) continue;  // the emulator holds no buff at zero
            if (!BuffByPower.TryGetValue(PowerKey(name), out var id)) { unmapped = true; continue; }
            want.Add(new BuffState(id, amount));
        }

        // A game power the emulator has no buff for lives in its own pieces (Crimson Mantle's two
        // buffs, Tank, Beacon of Hope, ...); those pieces are left alone, not overwritten.
        bool Keep(BuffState b) => NotPowers.Contains(b.Id) || (unmapped && Internal(b.Id));
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

    static JsonObject DumpLite(CombatState c, bool inCombat)
    {
        var o = new JsonObject { ["in_combat"] = inCombat, ["run_phase"] = inCombat ? "Combat" : "Over" };
        if (!inCombat) return o;
        JsonNode Card(CardInstance x) => new JsonObject { ["def_id"] = x.DefId, ["upgraded"] = x.Upgraded };
        JsonNode Buffs(List<BuffState> bs) => new JsonArray(bs.Select(b => (JsonNode)new JsonObject { ["id"] = (int)b.Id, ["name"] = b.Id.ToString(), ["amount"] = b.Magnitude }).ToArray());
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
