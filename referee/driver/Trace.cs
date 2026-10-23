// Trace mode (spec "trace": true): what the game did inside each step, for re-runs of fights that
// diverged (a fight replays identically from its spec). Off in every normal run: nothing is patched
// or subscribed until a trace spec arrives, and the patches then only log while a traced fight runs.
//
// Per step: "trace", an ordered list of
//   ["M", mark, subject, mini-state]   marks the emulator's CombatTrace mirrors: play / played
//                                      (CombatHistory CardPlayStarted / CardPlayFinished), move
//                                      (MonsterModel.PerformMove entered) / moved (MonsterPerformedMove),
//                                      turn_end (Hook.BeforeTurnEnd, player side), enemy_turn /
//                                      turn_start (Hook.BeforeSideTurnStart for each side);
//   ["E", entry, ...]                  the game's own combat log (CombatManager.History entries);
//   ["A", action]                      each GameAction the executor runs (ActionExecutor.BeforeActionExecuted);
//   ["H", hook, side?]                 each Hook.Before* / Hook.After* event hook called;
//   ["R", stream, counter, caller]     each draw from a run or player Rng stream (Harmony postfix on Rng's
//                                      Next* methods; counter after the draw; the first two callers);
//   ["LB"|"L"|"LE", ...]              listener order, only when the spec also sets "trace_listeners": true
//                                      (ListenerTrace.cs); that also adds the Modify*, Should* and Try* hooks to the "H" events;
// and "hidden": per-card cost (CardEnergyCost with all modifiers), enchantment and affliction, relic
// counters and fields, every power's amount and fields. No game code here: only calls into its API.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

internal static class Trace
{
    private static bool _patched;
    private static bool _widened;  // the Modify*/Should*/Try* hook prefixes are installed (first fight with trace_listeners)
    private static Harmony? _harmony;
    private static JsonArray? _events;  // the step being recorded; null when no traced fight runs
    private static Player? _player;
    private static CombatHistory? _history;
    private static int _seen;
    private static readonly Dictionary<Rng, string> Streams = new(ReferenceEqualityComparer.Instance);
    private static readonly Dictionary<Rng, int> LastLogged = new(ReferenceEqualityComparer.Instance);

    internal static bool On => _events != null;

    /// <summary>The step being recorded (null when no traced fight runs), for ListenerTrace's records.</summary>
    internal static JsonArray? Events => _events;

    /// <summary>Starts tracing the fight that has just opened; events so far go to the first step.</summary>
    internal static void Begin(RunState run, Player player, bool listeners = false)
    {
        Patch();
        ListenerTrace.Enabled = listeners;
        if (listeners)
        {
            Widen();
            ListenerTrace.Install(_harmony!);
            ListenerTrace.Reset();
        }

        _player = player;
        Streams.Clear();
        foreach (var (set, prefix) in new (object, string)[] { (run.Rng, ""), (player.PlayerRng, "player.") })
        {
            foreach (PropertyInfo p in set.GetType().GetProperties().Where(p => p.PropertyType == typeof(Rng)))
            {
                if (p.GetValue(set) is Rng r) Streams.TryAdd(r, prefix + p.Name);
            }
        }

        _history = CombatManager.Instance.History;
        _seen = 0;
        _events = new JsonArray();
        OnHistoryChanged();  // the fight's opening entries
        _history.Changed += OnHistoryChanged;
        RunManager.Instance.ActionExecutor.BeforeActionExecuted += OnAction;
    }

    /// <summary>The events since the last call; the next step's start here.</summary>
    internal static JsonArray TakeStep()
    {
        var e = _events ?? new JsonArray();
        if (_events != null) _events = new JsonArray();
        return e;
    }

    internal static void End()
    {
        if (_history != null) _history.Changed -= OnHistoryChanged;
        RunManager.Instance.ActionExecutor.BeforeActionExecuted -= OnAction;
        _events = null;
        ListenerTrace.Enabled = false;
        _history = null;
        _player = null;
        Streams.Clear();
        LastLogged.Clear();
    }

    // ── patches (installed on the first traced fight, inert while _events is null) ──────────

    private static void Patch()
    {
        if (_patched) return;
        _patched = true;
        var h = _harmony = new Harmony("spireline.referee.trace");
        foreach (var m in typeof(Hook).GetMethods(BindingFlags.Public | BindingFlags.Static)
                     .Where(m => !m.IsGenericMethodDefinition && (m.Name.StartsWith("Before") || m.Name.StartsWith("After"))))
        {
            h.Patch(m, prefix: new HarmonyMethod(typeof(Trace), nameof(HookPrefix)));
        }

        h.Patch(AccessTools.Method(typeof(MonsterModel), nameof(MonsterModel.PerformMove)),
            prefix: new HarmonyMethod(typeof(Trace), nameof(MovePrefix)));
        // Which monster a MonsterAi draw belongs to: the draws come from the shared state machine.
        h.Patch(AccessTools.Method(typeof(MonsterModel), nameof(MonsterModel.RollMove)),
            prefix: new HarmonyMethod(typeof(Trace), nameof(RollPrefix)), postfix: new HarmonyMethod(typeof(Trace), nameof(RollPostfix)));
        foreach (var m in typeof(Rng).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                     .Where(m => m.DeclaringType == typeof(Rng) && !m.IsGenericMethodDefinition && m.Name.StartsWith("Next")))
        {
            h.Patch(m, postfix: new HarmonyMethod(typeof(Trace), nameof(RngPostfix)));
        }
    }

    /// <summary>With trace_listeners: the dispatchers (Modify*, Should*, Try*) log an "H" too, so each listener
    /// enumeration follows its own dispatcher's event. Installed on the first fight that asks; inert otherwise.</summary>
    private static void Widen()
    {
        if (_widened) return;
        _widened = true;
        foreach (var m in typeof(Hook).GetMethods(BindingFlags.Public | BindingFlags.Static)
                     .Where(m => !m.IsGenericMethodDefinition && (m.Name.StartsWith("Modify") || m.Name.StartsWith("Should") || m.Name.StartsWith("Try"))))
        {
            _harmony!.Patch(m, prefix: new HarmonyMethod(typeof(Trace), nameof(HookPrefix)));
        }
    }

    private static void HookPrefix(MethodBase __originalMethod, object[] __args)
    {
        if (_events == null) return;
        if (!ListenerTrace.Enabled && !__originalMethod.Name.StartsWith("Before") && !__originalMethod.Name.StartsWith("After")) return;
        var side = __args.OfType<CombatSide>().Cast<CombatSide?>().FirstOrDefault();
        string name = __originalMethod.Name;
        if (name == "BeforeTurnEnd" && side == CombatSide.Player) Mark("turn_end", null);
        if (name == "BeforeSideTurnStart") Mark(side == CombatSide.Player ? "turn_start" : "enemy_turn", null);
        _events.Add(side == null ? new JsonArray("H", name) : new JsonArray("H", name, side.ToString()));
    }

    private static void MovePrefix(MonsterModel __instance)
    {
        if (_events != null) Mark("move", __instance.Id.Entry);
    }

    private static void RollPrefix(MonsterModel __instance) => _events?.Add(new JsonArray("A", "RollMove " + Who(__instance.Creature)));

    private static void RollPostfix(MonsterModel __instance) => _events?.Add(new JsonArray("E", "MoveRolled", Who(__instance.Creature), __instance.NextMove?.Id));

    private static void RngPostfix(Rng __instance)
    {
        if (_events == null || ReferenceEquals(__instance, Rng.Chaotic)) return;  // Chaotic: presentation only
        // One draw, once: NextFloat(max) and the Gaussians call another Next* that logged it already.
        if (LastLogged.TryGetValue(__instance, out int last) && last == __instance.Counter) return;
        LastLogged[__instance] = __instance.Counter;
        string stream = Streams.TryGetValue(__instance, out var s) ? s : "?" + __instance.Seed;
        _events.Add(new JsonArray("R", stream, __instance.Counter, Caller()));
    }

    private static void OnAction(GameAction a) => _events?.Add(new JsonArray("A", a.GetType().Name));

    private static void OnHistoryChanged()
    {
        if (_events == null || _history == null) return;
        var all = _history.Entries.ToList();
        if (all.Count < _seen) _seen = 0;  // cleared
        for (; _seen < all.Count; _seen++)
        {
            Entry(all[_seen]);
        }
    }

    // ── what is written ────────────────────────────────────────────────────────────────

    private static void Mark(string what, string? subject) => _events?.Add(new JsonArray("M", what, subject, Mini()));

    private static void Entry(CombatHistoryEntry e)
    {
        switch (e)
        {
            case CardPlayStartedEntry s:
                Mark("play", s.CardPlay.Card.Id.Entry);
                return;
            case CardPlayFinishedEntry f:
                Mark("played", f.CardPlay.Card.Id.Entry);
                return;
            case MonsterPerformedMoveEntry m:
                _events!.Add(new JsonArray("E", "MonsterPerformedMove", m.Monster.Id.Entry, m.Move.Id));
                Mark("moved", m.Monster.Id.Entry);
                return;
            case DamageReceivedEntry d:
                _events!.Add(new JsonArray("E", "DamageReceived", Who(d.Receiver), d.Dealer == null ? null : Who(d.Dealer),
                    d.Result.UnblockedDamage, d.Result.BlockedDamage, d.CardSource?.Id.Entry));
                return;
            case BlockGainedEntry b:
                _events!.Add(new JsonArray("E", "BlockGained", Who(b.Actor), b.Amount));
                return;
            case PowerReceivedEntry p:
                _events!.Add(new JsonArray("E", "PowerReceived", Who(p.Actor), p.Power.Id.Entry, (double)p.Amount,
                    p.Applier == null ? null : Who(p.Applier)));
                return;
            case CardDrawnEntry c:
                _events!.Add(new JsonArray("E", "CardDrawn", c.Card.Id.Entry, c.FromHandDraw));
                return;
            case CardDiscardedEntry c:
                _events!.Add(new JsonArray("E", "CardDiscarded", c.Card.Id.Entry));
                return;
            case CardExhaustedEntry c:
                _events!.Add(new JsonArray("E", "CardExhausted", c.Card.Id.Entry));
                return;
            case CardGeneratedEntry c:
                _events!.Add(new JsonArray("E", "CardGenerated", c.Card.Id.Entry));
                return;
            case EnergySpentEntry c:
                _events!.Add(new JsonArray("E", "EnergySpent", c.Amount));
                return;
            default:
                _events!.Add(new JsonArray("E", e.GetType().Name.Replace("Entry", ""), Who(e.Actor)));
                return;
        }
    }

    private static string Who(Creature c)
    {
        if (c.IsPlayer) return "player";
        var combat = CombatManager.Instance.DebugOnlyGetState();
        int i = combat?.Enemies.ToList().IndexOf(c) ?? -1;
        return (c.Monster?.Id.Entry ?? "?") + "#" + i;
    }

    /// <summary>The first two frames above the stream: "CardPileCmd.Shuffle &lt; CardPileCmd.Draw" (async methods by name).</summary>
    private static string Caller()
    {
        var parts = new List<string>();
        foreach (var f in new StackTrace(2, false).GetFrames())
        {
            var m = f.GetMethod();
            var t = m?.DeclaringType;
            if (m == null || t == null || t == typeof(Rng) || t == typeof(Trace)) continue;
            string ns = t.Namespace ?? "";
            if (ns.StartsWith("System") || ns.StartsWith("HarmonyLib") || ns.StartsWith("MonoMod") || ns.StartsWith("Godot")) continue;
            string mn = m.Name, tn = t.Name;
            if (tn.StartsWith('<') && t.DeclaringType != null)
            {
                mn = tn[1..Math.Max(1, tn.IndexOf('>'))];  // an async state machine's MoveNext: its method
                tn = t.DeclaringType.Name;
            }

            if (mn.StartsWith('<')) mn = mn[1..Math.Max(1, mn.IndexOf('>'))];
            parts.Add(tn + "." + mn);
            if (parts.Count == 2) break;
        }

        return string.Join(" < ", parts);
    }

    private static JsonArray Powers(Creature c) => new(c.Powers.Select(p => (JsonNode)new JsonArray(p.Id.Entry, p.Amount)).ToArray());

    /// <summary>The state at a mark, in the shape tools/emudump/Trace.cs writes for the emulator.</summary>
    private static JsonObject Mini()
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var me = _player!.Creature;
        var pcs = _player.PlayerCombatState;
        return new JsonObject
        {
            ["player"] = new JsonArray(me.CurrentHp, me.Block, pcs?.Energy ?? 0),
            ["powers"] = Powers(me),
            ["enemies"] = new JsonArray((combat?.Enemies ?? Array.Empty<Creature>()).Where(e => e.CurrentHp > 0)
                .Select(e => (JsonNode)new JsonArray(e.Monster?.Id.Entry, e.CurrentHp, e.Block, Powers(e))).ToArray()),
            ["hand"] = new JsonArray((pcs?.Hand.Cards ?? Enumerable.Empty<CardModel>()).Select(c => (JsonNode)c.Id.Entry).ToArray()),
            ["n"] = new JsonArray(pcs?.DrawPile.Cards.Count ?? 0, pcs?.DiscardPile.Cards.Count ?? 0, pcs?.ExhaustPile.Cards.Count ?? 0),
        };
    }

    // ── hidden state ────────────────────────────────────────────────────────────────────

    private static readonly Dictionary<Type, FieldInfo[]> FieldCache = new();

    /// <summary>A model's own fields (its class and those between it and <paramref name="stop"/>) of simple types.</summary>
    private static JsonObject Fields(object o, Type stop)
    {
        var t0 = o.GetType();
        if (!FieldCache.TryGetValue(t0, out var fields))
        {
            var list = new List<FieldInfo>();
            for (var t = t0; t != null && t != stop && t != typeof(object); t = t.BaseType)
            {
                list.AddRange(t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                    .Where(f => f.FieldType.IsPrimitive || f.FieldType.IsEnum || f.FieldType == typeof(decimal) || f.FieldType == typeof(string)));
            }

            FieldCache[t0] = fields = list.ToArray();
        }

        var j = new JsonObject();
        foreach (var f in fields)
        {
            string name = f.Name.StartsWith('<') ? f.Name[1..f.Name.IndexOf('>')] : f.Name;
            object? v = f.GetValue(o);
            j[name] = v switch
            {
                null => null,
                bool b => b,
                string s => s,
                decimal d => (double)d,
                Enum en => en.ToString(),
                _ => JsonValue.Create(Convert.ToDouble(v)),
            };
        }

        return j;
    }

    /// <summary>What the step's snapshot does not show.</summary>
    internal static JsonObject Hidden()
    {
        var p = _player!;
        var pcs = p.PlayerCombatState;
        var cards = new JsonArray();
        if (pcs != null)
        {
            foreach (var (pile, list) in new[] { ("hand", pcs.Hand.Cards), ("draw", pcs.DrawPile.Cards), ("discard", pcs.DiscardPile.Cards), ("exhaust", pcs.ExhaustPile.Cards) })
            {
                foreach (var c in list)
                {
                    int cost;
                    try { cost = c.EnergyCost.CostsX ? -2 : c.EnergyCost.GetWithModifiers(CostModifiers.All); }
                    catch (Exception) { cost = -99; }
                    cards.Add(new JsonArray(pile, c.Id.Entry, cost, c.Enchantment?.Id.Entry, c.Enchantment?.Amount ?? 0, c.Affliction?.Id.Entry));
                }
            }
        }

        var relics = new JsonArray(p.Relics.Select(r => (JsonNode)new JsonArray(r.Id.Entry, r.ShowCounter ? r.DisplayAmount : null,
            Fields(r, typeof(RelicModel)))).ToArray());
        var powers = new JsonArray();
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var holders = new List<Creature> { p.Creature };
        holders.AddRange(combat?.Enemies ?? Array.Empty<Creature>());
        foreach (var c in holders)
        {
            foreach (var pw in c.Powers)
            {
                powers.Add(new JsonArray(Who(c), pw.Id.Entry, pw.Amount, Fields(pw, typeof(PowerModel))));
            }
        }

        return new JsonObject { ["cards"] = cards, ["relics"] = relics, ["powers"] = powers };
    }
}
