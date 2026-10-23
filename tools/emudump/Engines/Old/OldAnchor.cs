// The old engine's loader of a game snapshot into its combat: the game's recorded state (the referee's
// Snapshot(): player hit points, block, energy and powers, the four piles in order, the living enemies, the
// random streams' seed and counter, and the potion belt where the record has it) is written over a CombatState, and
// the enemies' next moves are put on the game's where the old engine's AI can be put there. What the snapshot cannot
// say (relic counters, enemy AI state, per-card modifiers, counters) stays as the combat has it.
//
// This is the old engine's loader, nothing else's. Round A1b may move this whole class into the engine DLL behind a
// native export for fight-start re-anchoring, so it depends on nothing of EmuDump: its inputs are the combat, the
// snapshot JSON, and the id tables handed in as <see cref="Tables"/> (built by the caller from data/id_map.json and
// tools/fidelity/power_aliases.json, which it reads); it reads no file, calls no other EmuDump class and keeps no state
// besides the engine-derived buff table. tools/emudump.Tests ReanchorBoundaryTests checks that.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;

namespace EmuDump.Engines.Old;

static class OldAnchor
{
    /// <summary>The game's id tables, as the loader reads them: ids in the game's UPPER_SNAKE spelling, mapped to the old engine's ids.</summary>
    internal sealed class Tables
    {
        /// <summary>Enemy ids to the old engine's enemy def ids.</summary>
        internal Dictionary<string, int> Enemies { get; }

        /// <summary>Card ids to card def ids.</summary>
        internal Dictionary<string, int> Cards { get; }

        /// <summary>Potion ids to potion ids.</summary>
        internal Dictionary<string, int> Potions { get; }

        /// <summary>The keys the game's own power list can hold (id_map.json "powers").</summary>
        internal HashSet<string> GamePowerKeys { get; }

        // power_aliases.json: a game power the engine keeps under another name, and the TemporaryStrength/Dexterity/Focus
        // subclasses it folds into one buff. Writing back by name alone dropped them (Mangle's restore never came).
        internal Dictionary<string, string> GameRename { get; }
        internal Dictionary<string, string> Folds { get; }
        internal HashSet<string> FoldTargets { get; }
        internal HashSet<BuffId> RenameTargets { get; }

        readonly Dictionary<BuffId, string?> _owners = new();

        /// <param name="idMap">data/id_map.json parsed: "cards", "enemies", "powers" and "potions" map the game's names to ids.</param>
        /// <param name="aliases">tools/fidelity/power_aliases.json parsed: "emu_rename" and "folds".</param>
        internal Tables(JsonObject idMap, JsonObject aliases)
        {
            Dictionary<string, int> Load(string kind) => idMap[kind]!.AsObject().ToDictionary(kv => Snake(kv.Key), kv => (int)kv.Value!);
            Enemies = Load("enemies");
            Cards = Load("cards");
            Potions = Load("potions");
            GamePowerKeys = Load("powers").Keys.Select(PowerKey).ToHashSet();
            GameRename = aliases["emu_rename"]!.AsObject().ToDictionary(kv => (string)kv.Value!, kv => kv.Key);
            Folds = aliases["folds"]!.AsObject()
                .SelectMany(kv => kv.Value!.AsArray().Select(sub => ((string)sub!, kv.Key))).ToDictionary(t => t.Item1, t => t.Item2);
            FoldTargets = Folds.Values.ToHashSet();
            RenameTargets = GameRename.Values.Where(BuffByPower.ContainsKey).Select(k => BuffByPower[k]).ToHashSet();
        }

        /// <summary>A buff only the engine has (a game power it models in pieces, or a bookkeeping marker).</summary>
        internal bool IsInternal(BuffId id) => !GamePowerKeys.Contains(PowerKey(id.ToString()));

        /// <summary>The game power an engine-only buff is a piece of (InfernoSelfDamage of INFERNO), by name; null for pure bookkeeping.</summary>
        internal string? OwnerPower(BuffId id)
        {
            if (_owners.TryGetValue(id, out var owner)) return owner;
            string key = PowerKey(id.ToString());
            owner = GamePowerKeys.Where(p => key.StartsWith(p + "_", StringComparison.Ordinal)).OrderByDescending(p => p.Length).FirstOrDefault();
            _owners[id] = owner;
            return owner;
        }
    }

    static readonly Dictionary<string, BuffId> BuffByPower = BuffTable();

    // Representation, not rules (see tools/fidelity/fight_compare.py NOT_POWERS): the engine keeps a stun as a buff the
    // game's power list never shows, so it is never overwritten.
    internal static readonly HashSet<BuffId> NotPowers = [BuffId.Stunned];

    /// <summary>Power keys the engine has a buff for (buff_keys), and the buffs of its own that are no game power (internal_keys).</summary>
    internal static JsonObject PowerKeys(Tables t) => new()
    {
        ["buff_keys"] = new JsonArray(BuffByPower.Keys.Select(k => (JsonNode)k).ToArray()),
        ["internal_keys"] = new JsonArray(BuffByPower.Keys.Where(k => !t.GamePowerKeys.Contains(k)).Select(k => (JsonNode)k).ToArray()),
    };

    /// <summary>The engine's buff for a game power id (VULNERABLE_POWER), or null.</summary>
    internal static BuffId? BuffFor(string gamePower) => BuffByPower.TryGetValue(PowerKey(gamePower), out var b) ? b : null;

    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    static string PowerKey(string name)
    {
        var n = Snake(name);
        return n.EndsWith("_POWER") ? n[..^"_POWER".Length] : n;
    }

    static Dictionary<string, BuffId> BuffTable()
    {
        var table = new Dictionary<string, BuffId>();
        foreach (var id in Enum.GetValues<BuffId>())
        {
            var key = PowerKey(id.ToString());
            // Two enum members with one key would make the game's power ambiguous: keep the first and say so.
            if (!table.TryAdd(key, id)) Console.Error.WriteLine($"reanchor: power key {key} names both {table[key]} and {id}; using {table[key]}");
        }

        return table;
    }

    // ── writing the game's state over the engine's ──────────────────────────────────

    /// <summary>
    /// Overwrites <paramref name="q"/>'s observable fields with the game's state; only fields that
    /// differ are touched, and each one touched is named in <paramref name="patched"/>. Returns a
    /// reason when the state cannot be built (an enemy or card the engine has no id for, an
    /// enemy the game has and the engine lacks).
    /// </summary>
    internal static string? Anchor(Tables t, CombatState q, JsonObject g, JsonArray patched)
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
            if (PatchBuffs(t, q.PlayerBuffs, p["powers"]!.AsArray(), out var buffs, patched, "player.powers")) q.PlayerBuffs = buffs;

            PatchPiles(t, q, g, patched);

            string? roster = PatchEnemies(t, q, g["enemies"]!.AsArray(), patched);
            if (roster != null) return roster;

            PatchRng(q, g["rng"]!.AsObject(), patched);
            if (g["potions"] is JsonArray belt) PatchPotions(t, q, belt, patched);
            return null;
        }
        catch (KeyNotFoundException ex)
        {
            return "no emulator id for " + ex.Message;
        }
    }

    internal static bool PatchBuffs(Tables t, List<BuffState> cur, JsonArray powers, out List<BuffState> result, JsonArray patched, string where)
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
            if (amount == 0) continue;  // the engine holds no buff at zero
            bool fold = t.Folds.TryGetValue(key, out var into);
            string emuKey = fold ? into! : t.GameRename.GetValueOrDefault(key, key);
            if (!BuffByPower.TryGetValue(emuKey, out var id)) continue;  // modelled in pieces, which Keep leaves alone
            // The game's own power of a fold target's name (BlockNextTurnPower beside SelfFormingClayPower) is part of the
            // same sum: the engine holds one buff, and two wanted entries of it could never match its one.
            if (fold || t.FoldTargets.Contains(emuKey)) folded[id] = folded.GetValueOrDefault(id) + amount;
            else want.Add(new BuffState(id, amount));
        }

        want.AddRange(folded.Where(kv => kv.Value != 0).Select(kv => new BuffState(kv.Key, kv.Value)));

        // The engine's own buffs are hidden state the snapshot cannot show, carried over: a piece of a
        // game power (Inferno's InfernoSelfDamage, Crimson Mantle's, Tank's, Beacon of Hope's ...) while
        // that power is there, and pure bookkeeping always. Dropping InfernoSelfDamage when every game
        // power had a buff stopped the next turn's Inferno.
        bool Keep(BuffState b) => NotPowers.Contains(b.Id)
            || (t.IsInternal(b.Id) && !t.RenameTargets.Contains(b.Id) && (t.OwnerPower(b.Id) is not { } owner || present.Contains(owner)));
        var have = cur.Where(b => !Keep(b)).ToList();
        // Compared on what the snapshot shows (id, amount). Data is hidden per-instance state (an
        // instanced power's second number, e.g. The Bomb's damage) that the game snapshot lacks.
        static (BuffId, int) Shown(BuffState b) => (b.Id, b.Magnitude);
        bool same = have.Count == want.Count && have.Select(Shown).OrderBy(x => x).SequenceEqual(want.Select(Shown).OrderBy(x => x));
        result = cur;
        if (same) return false;
        // Keep the engine's order for powers it already has; the rest follow in the game's order.
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

    static void PatchPiles(Tables t, CombatState q, JsonObject g, JsonArray patched)
    {
        var gh = Cards(t, g["hand"]!);
        var gd = Cards(t, g["discard"]!);
        var ge = Cards(t, g["exhaust"]!);
        var gw = Cards(t, g["draw"]!);
        bool same(List<CardInstance> have, List<(int Id, bool Up)> want) =>
            have.Count == want.Count && have.Select((c, i) => c.DefId == want[i].Id && c.Upgraded == want[i].Up).All(x => x);
        bool sh = same(q.Hand, gh), sd = same(q.DiscardPile, gd), se = same(q.ExhaustPile, ge), sw = same(q.DrawPile, gw);
        if (sh && sd && se && sw) return;

        // Cards that keep their identity (cost changes, bonus damage, enchantments, ...) are kept
        // where the pile still has them at the same place, then taken from whichever pile now holds
        // an identical card; a card the engine has nowhere is built fresh.
        var piles = new[] { q.Hand, q.DiscardPile, q.ExhaustPile, q.DrawPile };
        var wants = new[] { gh, gd, ge, gw };
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

        // A card matched in the old piles keeps its uid; one the game holds and the emulator did not gets a fresh one.
        Sts2Emulator.Core.CardUids.NumberNew(q);
    }

    static List<(int Id, bool Up)> Cards(Tables t, JsonNode pile) =>
        pile.AsArray().Select(c => (t.Cards[(string)c![0]!], (bool)c[1]!)).ToList();

    static string? PatchEnemies(Tables t, CombatState q, JsonArray gen, JsonArray patched)
    {
        var want = gen.Select(en => (Name: (string)en!["name"]!, Def: t.Enemies[(string)en["name"]!], Node: en)).ToList();
        // The game keeps only living creatures. Pair each with the engine's creature of that
        // kind in order; engine creatures nobody pairs with are gone in the game's state.
        var roster = new List<EnemyState>();
        int from = 0;
        foreach (var w in want)
        {
            int at = q.Enemies.FindIndex(from, en => en.DefId == w.Def);
            if (at < 0) return $"the game has {w.Name} where the emulator has none to pair with it";
            roster.Add(q.Enemies[at]);
            from = at + 1;
        }

        if (roster.Count != q.Enemies.Count || !roster.Zip(q.Enemies).All(x => ReferenceEquals(x.First, x.Second)))
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
            if (PatchBuffs(t, en.Buffs, n["powers"]!.AsArray(), out var buffs, patched, $"enemies[{i}].powers")) en.Buffs = buffs;
        }

        return null;
    }

    /// <summary>The potion belt, slot by slot, when the game's snapshot records it (setup fights).</summary>
    static void PatchPotions(Tables t, CombatState q, JsonArray belt, JsonArray patched)
    {
        var want = belt.Select(p => p == null ? 0
            : t.Potions.TryGetValue((string)p!, out int id) ? id : throw new KeyNotFoundException("potion " + (string)p!)).ToArray();
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

    // ── enemy moves ─────────────────────────────────────────────────────────────────
    //
    // The game records each enemy's next move (referee/driver/Combat.cs Intents). Before a step the
    // engine's announced move is compared with it in tools/fidelity/fight_compare.py's vocabulary
    // (game_intent / emu_intent): the kind of the first intent, and the damage-carrying intent's base
    // damage and hits. Where they differ, the engine's AI is put on the game's move if that can be
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
    /// "unknown" when one could not be (its index is in <c>Unknown</c>), "unrecorded" for a record without intents.
    /// <paramref name="later"/> gives, for the enemy at an index, the game's moves after the following ends of turn
    /// (<see cref="LaterMoves"/>), which rank the positions that announce the wanted move.
    /// </summary>
    internal static (string Status, JsonArray? Unknown) AnchorMoves(CombatState q, JsonObject g, JsonArray patched, Func<int, List<string>> later)
    {
        var gen = g["enemies"]!.AsArray();
        if (gen.Any(x => x!["intents"] == null)) return ("unrecorded", null);
        var unknown = new JsonArray();
        bool set = false;
        for (int i = 0; i < gen.Count && i < q.Enemies.Count; i++)
        {
            if (q.Enemies[i].Hp <= 0) continue;
            string want = GameMove(gen[i]!)!;
            if (EmuMove(q.Enemies[i]) == want) continue;
            if (SetMove(q, q.Enemies[i], want, later(i)))
            {
                patched.Add($"enemies[{i}].move");
                set = true;
            }
            else
            {
                unknown.Add(i);
            }
        }

        if (unknown.Count > 0) return ("unknown", unknown);
        return (set ? "set" : "same", null);
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
    internal static List<string> LaterMoves(JsonArray gsteps, int j, int i)
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
    internal static bool Reselect(CombatState q, EnemyState en, int moveIndex, out EnemyState c)
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
}
