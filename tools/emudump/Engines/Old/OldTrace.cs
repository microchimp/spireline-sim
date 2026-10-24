// Trace mode (EmuDump reanchor --trace), the old engine's side: what the emulator did inside each step, in the shape the
// referee's trace mode records for the game (referee/driver/Trace.cs), so tools/fidelity/trace_diff.py
// can align the two and name the first event that differs. OldReanchor.Step writes the result into the step's record
// (IEngine.ReanchorStep: trace, hidden_pre, hidden); nothing else reads it.
//
// Per step: "trace", an ordered list of
//   ["M", mark, subject, mini-state]   at the engine's CombatTrace marks (play/played, turn_end,
//                                      enemy_turn, move/moved, turn_start), with the state there;
//   ["R", stream, counter, caller]     every draw from a combat stream (counter after the draw;
//                                      caller: the first two engine frames above the stream);
// and "hidden": per-card cost and enchantment, relic counters, every buff with its hidden Data.
// Only for re-runs of fights that diverged: building a stack trace per draw is slow.
using System.Diagnostics;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;

namespace EmuDump.Engines.Old;

static class OldTrace
{
    static JsonArray? _events;
    static CombatState? _q;
    static Dictionary<int, string>? _cards, _enemies, _relics;

    static Dictionary<int, string> Names(string kind)
    {
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))![kind]!.AsObject();
        var o = new Dictionary<int, string>();
        foreach (var kv in map)
        {
            o.TryAdd((int)kv.Value!, Regex.Replace(kv.Key, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant());
        }

        return o;
    }

    static string Card(int id) => (_cards ??= Names("cards")).GetValueOrDefault(id, id.ToString());

    static string Enemy(int id) => (_enemies ??= Names("enemies")).GetValueOrDefault(id, id.ToString());

    static string Relic(int id) => (_relics ??= Names("relics")).GetValueOrDefault(id, id.ToString());

    /// <summary>Starts recording one step of <paramref name="q"/>.</summary>
    public static void Begin(CombatState q)
    {
        _events = new JsonArray();
        _q = q;
        _events.Add(new JsonArray("M", "start", null, Mini(q)));  // the re-anchored state the step starts from
        CombatTrace.Mark = OnMark;
        CombatTrace.RngCall = OnRng;
    }

    public static JsonArray End()
    {
        CombatTrace.Mark = null;
        CombatTrace.RngCall = null;
        var e = _events ?? new JsonArray();
        _events = null;
        _q = null;
        return e;
    }

    static void OnMark(string what, int subject, CombatState s)
    {
        if (what == "replay")
        {
            // A screen's answer runs the step again from its start (CombatEngine.Step,
            // HookChoiceReplay), and marks it again: what the stopped run marked goes, down to the
            // re-anchored start, so the trace holds the step once, as the game ran it.
            while (_events is { Count: > 1 })
            {
                _events.RemoveAt(_events.Count - 1);
            }

            return;
        }

        string? name = subject < 0 ? null : what is "move" or "moved" ? Enemy(subject) : Card(subject);
        _events?.Add(new JsonArray("M", what, name, Mini(s)));
    }

    static void OnRng(CountingRandom r) => _events?.Add(new JsonArray("R", Stream(r), r.CallCount, Caller()));

    static string Stream(CountingRandom r)
    {
        var q = _q;
        if (q == null) return "?";
        if (ReferenceEquals(r, q.ShuffleRng)) return "Shuffle";
        if (ReferenceEquals(r, q.TargetRng)) return "CombatTargets";
        if (ReferenceEquals(r, q.AiRng)) return "MonsterAi";
        if (ReferenceEquals(r, q.CardSelectionRng)) return "CombatCardSelection";
        if (ReferenceEquals(r, q.CardGenerationRng)) return "CombatCardGeneration";
        if (ReferenceEquals(r, q.EnergyCostRng)) return "CombatEnergyCosts";
        if (ReferenceEquals(r, q.PotionGenerationRng)) return "CombatPotionGeneration";
        if (ReferenceEquals(r, q.OrbGenerationRng)) return "CombatOrbGeneration";
        if (ReferenceEquals(r, q.NicheHpRng)) return "Niche";
        return "fallback";  // the combat's own stream, which no game stream matches
    }

    /// <summary>The first two engine frames above the stream: "CardEffects.ShufflePile &lt; CombatEngine.EndTurn".</summary>
    static string Caller()
    {
        var parts = new List<string>();
        foreach (var f in new StackTrace(2, false).GetFrames())
        {
            var m = f.GetMethod();
            var t = m?.DeclaringType;
            if (m == null || t == null || t == typeof(CountingRandom) || t == typeof(OldTrace) || t == typeof(CombatTrace)
                || (t.Namespace ?? "").StartsWith("System", StringComparison.Ordinal))
            {
                continue;
            }

            string mn = m.Name, tn = t.Name;
            if (mn.StartsWith('<')) mn = mn[1..mn.IndexOf('>')];  // a lambda or local function: its method
            if (ExtractedBody.Is(t, mn)) continue;  // an extracted <Model>_<Hook> body: the origin names the method it came from
            if (tn.StartsWith('<') && t.DeclaringType != null) tn = t.DeclaringType.Name;
            parts.Add(tn + "." + mn);
            if (parts.Count == 2) break;
        }

        return string.Join(" < ", parts);
    }

    static JsonArray Buffs(List<BuffState> bs) => new(bs.Select(b => (JsonNode)new JsonArray(b.Id.ToString(), b.Magnitude)).ToArray());

    /// <summary>The state at a mark: what tools/fidelity/trace_diff.py compares between the two sides.</summary>
    static JsonObject Mini(CombatState s) => new()
    {
        ["player"] = new JsonArray(s.PlayerHp, s.PlayerBlock, s.Energy),
        ["powers"] = Buffs(s.PlayerBuffs),
        ["enemies"] = new JsonArray(s.Enemies.Where(e => e.Hp > 0).Select(e => (JsonNode)new JsonArray(Enemy(e.DefId), e.Hp, e.Block, Buffs(e.Buffs))).ToArray()),
        ["hand"] = new JsonArray(s.Hand.Select(c => (JsonNode)Card(c.DefId)).ToArray()),
        ["n"] = new JsonArray(s.DrawPile.Count, s.DiscardPile.Count, s.ExhaustPile.Count),
    };

    /// <summary>What the game's snapshot does not show: per-card cost and enchantment, relic counters, buff Data.</summary>
    public static JsonObject Hidden(CombatState s)
    {
        var cards = new JsonArray();
        foreach (var (pile, list) in new[] { ("hand", s.Hand), ("draw", s.DrawPile), ("discard", s.DiscardPile), ("exhaust", s.ExhaustPile) })
        {
            foreach (var c in list)
            {
                int cost;
                try { cost = CombatEngine.EffectiveCost(c, s); }
                catch (Exception) { cost = -99; }
                cards.Add(new JsonArray(pile, Card(c.DefId), cost,
                    c.Enchantment == Enchantment.None ? null : c.Enchantment.ToString(), c.EnchantAmount));
            }
        }

        var powers = new JsonArray();
        foreach (var b in s.PlayerBuffs) powers.Add(new JsonArray("player", b.Id.ToString(), b.Magnitude, b.Data));
        for (int i = 0; i < s.Enemies.Count; i++)
        {
            foreach (var b in s.Enemies[i].Buffs) powers.Add(new JsonArray($"{Enemy(s.Enemies[i].DefId)}#{i}", b.Id.ToString(), b.Magnitude, b.Data));
        }

        return new JsonObject
        {
            ["cards"] = cards,
            ["relics"] = new JsonArray(s.Relics.Select(r => (JsonNode)new JsonArray(Relic(r.DefId), r.Counter)).ToArray()),
            ["powers"] = powers,
        };
    }
}
