// Listener-order trace (spec "trace_listeners": true, with "trace": true; rebuild-plan D2, the "L patch").
// Off unless the spec asks. It only observes: Harmony postfixes on the game's two hook-listener
// iterators (CombatState.IterateHookListeners and RunState.IterateHookListeners) replace the returned
// enumerable with a wrapper that yields the same items in the same order, one at a time and as lazily as
// the game's own (the inner enumerable is enumerated once, disposed with the wrapper, and an exception
// from it passes through), and appends records to the step's trace (Trace.cs):
//
//   ["LB", enum_id, hook, "combat"|"run"]                    an enumeration is created (hook = the Hook method
//                                                            that asked for it, from the call stack; "?" if none)
//   ["L", enum_id, model, kind, owner, index]                one per yielded listener, as it is yielded
//   ["LE", enum_id]                                          the enumeration ends or is disposed
//
//   model  the model's id entry. kind: power, monster, relic, potion, orb, card, affliction, enchantment,
//          modifier, badge, achievement, singleton, other.
//   owner  [side, slot]: the owning creature, "Player" or "Enemy" and its index in the combat's allies or
//          enemies (a player outside combat: its index among the run's players); null for run and combat
//          level models (modifier, badge, singleton).
//   index  power: application order on its creature; relic: obtain index; potion: slot; orb: queue position;
//          monster: 0; card: [pile, position] with pile hand/draw/discard/exhaust/play (deck for a deck card in
//          a run enumeration); affliction and enchantment: the [pile, position] of the card they sit on.
//
// Enumerations nest (a listener's work dispatches another hook), so enum_id (counted per traced fight)
// tells them apart. A run enumeration that includes a combat's (RunState.IterateHookListeners(combat))
// logs the combat part itself; the nested combat iterator is left unwrapped so no listener is logged twice.
// The game builds an enumeration's listener list when it is first advanced, and a listener's work can then
// move cards or powers; so the metadata is the state at that moment (a snapshot taken just before the first
// advance; for the combat part of a run enumeration, just before that part is built), not at the yield. A
// listener missing from the snapshot (made during the dispatch) is described from the live state. No game
// code here: only calls into the game's API, and nothing is written except the trace output.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Spireline.Referee;

internal static class ListenerTrace
{
    /// <summary>True while a traced fight whose spec asked for it runs.</summary>
    internal static bool Enabled;

    private static bool _patched;
    private static int _nextId;
    private static RunHolder? _moving;  // the wrapped run enumeration that is advancing (its nested combat enumeration reports to it)
    private static RunState? _run;
    private static HashSet<string>? _publicHooks;
    private static readonly System.Text.RegularExpressions.Regex PatchedHook = new(@"\.Hooks\.Hook\.(\w+)_Patch\d+$");

    /// <summary>Starts a fight's enumeration ids at 0.</summary>
    internal static void Reset(RunState run)
    {
        _nextId = 0;
        _run = run;
        _moving = null;
    }

    /// <summary>Where each listener stood, owner and index, when an enumeration's list was built.</summary>
    private sealed class Snap : Dictionary<object, (JsonNode? Owner, JsonNode? Index)>
    {
        public Snap() : base(ReferenceEqualityComparer.Instance) { }
    }

    /// <summary>A run enumeration's own snapshot and the one its nested combat enumeration took when it was built.</summary>
    private sealed class RunHolder
    {
        public Snap? Own;
        public Snap? Child;
    }

    internal static void Install(Harmony h)
    {
        if (_patched) return;
        _patched = true;
        h.Patch(AccessTools.Method(typeof(CombatState), nameof(CombatState.IterateHookListeners)),
            postfix: new HarmonyMethod(typeof(ListenerTrace), nameof(CombatPostfix)));
        h.Patch(AccessTools.Method(typeof(RunState), nameof(RunState.IterateHookListeners), new[] { typeof(ICombatState) }),
            postfix: new HarmonyMethod(typeof(ListenerTrace), nameof(RunPostfix)));
    }

    private static void CombatPostfix(ref IEnumerable<AbstractModel> __result)
    {
        if (!Enabled || Trace.Events == null) return;
        // Inside a run enumeration's advance: the combat part the run yields (logged by the run's wrapper, once).
        if (_moving != null) __result = SnapChild(__result, _moving);
        else __result = Wrap(__result, _nextId++, null, HookName());
    }

    private static void RunPostfix(ref IEnumerable<AbstractModel> __result)
    {
        if (!Enabled || Trace.Events == null) return;
        __result = Wrap(__result, _nextId++, new RunHolder(), HookName());
    }

    private static IEnumerable<AbstractModel> Wrap(IEnumerable<AbstractModel> inner, int id, RunHolder? run, string hook)
    {
        Trace.Events?.Add(new JsonArray("LB", id, hook, run != null ? "run" : "combat"));
        try
        {
            using var e = inner.GetEnumerator();
            // The game builds its list in the first MoveNext: take the state it will read.
            Snap own = Take();
            if (run != null) run.Own = own;
            while (true)
            {
                bool more;
                var outer = _moving;
                if (run != null) _moving = run;
                try { more = e.MoveNext(); }
                finally { _moving = outer; }
                if (!more) yield break;
                AbstractModel m = e.Current;
                Trace.Events?.Add(Describe(id, m, run?.Child, own));
                yield return m;
            }
        }
        finally
        {
            Trace.Events?.Add(new JsonArray("LE", id));
        }
    }

    /// <summary>The combat enumeration a run enumeration nests: the same items, and a snapshot of the state its list is built from.</summary>
    private static IEnumerable<AbstractModel> SnapChild(IEnumerable<AbstractModel> inner, RunHolder run)
    {
        using var e = inner.GetEnumerator();
        run.Child = Take();
        while (e.MoveNext()) yield return e.Current;
    }

    // ── which Hook method asked ────────────────────────────────────────────────────────────

    /// <summary>The innermost public Hook method on the stack (async ones by their state machine's name), else the
    /// innermost Hook-hosted helper, else "?" and the first game frame (the dispatch is not a Hook method's).</summary>
    private static string HookName()
    {
        _publicHooks ??= new HashSet<string>(typeof(Hook).GetMethods(BindingFlags.Public | BindingFlags.Static).Select(m => m.Name));
        string? helper = null;
        string? caller = null;  // the first game frame, for an enumeration no Hook method asked for
        foreach (var f in new StackTrace(1, false).GetFrames())
        {
            var m = f.GetMethod();
            var t = m?.DeclaringType;
            if (m == null) continue;
            if (caller == null && t != null && !ReferenceEquals(t, typeof(ListenerTrace)) && t != typeof(CombatState) && t != typeof(RunState)
                && !(t.Namespace ?? "").StartsWith("System") && !(t.Namespace ?? "").StartsWith("HarmonyLib") && !(t.Namespace ?? "").StartsWith("MonoMod"))
            {
                string tn = t.Name, mn = m.Name;
                if (tn.StartsWith('<') && t.DeclaringType != null) { mn = tn[1..Math.Max(1, tn.IndexOf('>'))]; tn = t.DeclaringType.Name; }
                caller = tn + "." + mn;
            }

            string name;
            if (t == null)
            {
                // A Hook method that Harmony has patched (the trace's own H prefixes) runs as a dynamic method:
                // "<Type>.<Method>_Patch<n>", no declaring type.
                var dm = PatchedHook.Match(m.Name);
                if (!dm.Success) continue;
                name = dm.Groups[1].Value;
            }
            else if (t == typeof(Hook)) name = m.Name;
            else if (t.DeclaringType == typeof(Hook) && t.Name.StartsWith('<')) name = t.Name[1..Math.Max(1, t.Name.IndexOf('>'))];
            else continue;
            if (name.StartsWith("Iterate")) continue;
            if (_publicHooks.Contains(name)) return name;
            helper ??= name;
        }

        return helper ?? "?" + caller;
    }

    // ── what is written per listener ───────────────────────────────────────────────────────

    private static JsonArray Describe(int id, AbstractModel m, Snap? first, Snap second)
    {
        string kind = "other";
        JsonNode? owner = null;
        JsonNode? index = null;
        try
        {
            if (Kind(m) is string k && ((first != null && first.TryGetValue(m, out var hit)) || second.TryGetValue(m, out hit)))
            {
                return new JsonArray("L", id, m.Id.Entry, k, hit.Owner?.DeepClone(), hit.Index?.DeepClone());
            }

            switch (m)
            {
                case PowerModel p:
                    kind = "power";
                    owner = Owner(p.Owner, null);
                    index = Index(p.Owner.Powers, p);
                    break;
                case MonsterModel mo:
                    kind = "monster";
                    owner = Owner(mo.Creature, null);
                    index = 0;
                    break;
                case RelicModel r:
                    kind = "relic";
                    owner = Owner(r.Owner.Creature, r.Owner);
                    index = Index(r.Owner.Relics, r);
                    break;
                case PotionModel po:
                    kind = "potion";
                    owner = Owner(po.Owner.Creature, po.Owner);
                    index = Index(po.Owner.PotionSlots, po);
                    break;
                case OrbModel o:
                    kind = "orb";
                    owner = Owner(o.Owner.Creature, o.Owner);
                    index = Index(o.Owner.PlayerCombatState?.OrbQueue.Orbs, o);
                    break;
                case CardModel c:
                    kind = "card";
                    owner = Owner(c.Owner?.Creature, c.Owner);
                    index = Place(c);
                    break;
                case AfflictionModel a:
                    kind = "affliction";
                    owner = Owner(a.Card.Owner?.Creature, a.Card.Owner);
                    index = Place(a.Card);
                    break;
                case EnchantmentModel en:
                    kind = "enchantment";
                    owner = Owner(en.Card.Owner?.Creature, en.Card.Owner);
                    index = Place(en.Card);
                    break;
                case ModifierModel: kind = "modifier"; break;
                case BadgeModel: kind = "badge"; break;
                case AchievementModel: kind = "achievement"; break;
                default:
                    // The run's multiplayer scaling model and anything a mod adds: no owner.
                    kind = m.GetType().Name.Contains("Scaling") ? "singleton" : "other";
                    break;
            }
        }
        catch (Exception ex)
        {
            // Metadata only: a model that cannot be described is logged as such, the game is not disturbed.
            kind += "!" + ex.GetType().Name;
        }

        return new JsonArray("L", id, m.Id.Entry, kind, owner, index);
    }

    private static string? Kind(AbstractModel m) => m switch
    {
        PowerModel => "power", MonsterModel => "monster", RelicModel => "relic", PotionModel => "potion", OrbModel => "orb",
        CardModel => "card", AfflictionModel => "affliction", EnchantmentModel => "enchantment", _ => null,
    };

    /// <summary>The owner and index of every model the iterators list, as of now (the same lookups Describe makes live).</summary>
    private static Snap Take()
    {
        var s = new Snap();
        try
        {
            var combat = CombatManager.Instance.DebugOnlyGetState();
            var creatures = new List<Creature>();
            if (combat != null)
            {
                creatures.AddRange(combat.Allies);
                creatures.AddRange(combat.Enemies);
            }

            foreach (var c in creatures)
            {
                var owner = Owner(c, null);
                for (int i = 0; i < c.Powers.Count; i++) s[c.Powers[i]] = (owner, i);
                if (c.Monster != null) s[c.Monster] = (owner, 0);
                if (c.Player != null) TakePlayer(s, c.Player);
            }

            if (_run != null)
            {
                foreach (var p in _run.Players)
                {
                    if (combat == null || !creatures.Contains(p.Creature)) TakePlayer(s, p);
                    for (int i = 0; i < p.Deck.Cards.Count; i++) TakeCard(s, p, p.Deck.Cards[i], "deck", i);
                }
            }
        }
        catch (Exception)
        {
            // A partial snapshot: what is missing is described live (Describe), the game is not disturbed.
        }

        return s;
    }

    private static void TakePlayer(Snap s, Player p)
    {
        var owner = Owner(p.Creature, p);
        for (int i = 0; i < p.Relics.Count; i++) s[p.Relics[i]] = (owner, i);
        for (int i = 0; i < p.PotionSlots.Count; i++)
        {
            if (p.PotionSlots[i] != null) s[p.PotionSlots[i]!] = (owner, i);
        }

        var pcs = p.PlayerCombatState;
        if (pcs == null) return;
        for (int i = 0; i < pcs.OrbQueue.Orbs.Count; i++) s[pcs.OrbQueue.Orbs[i]] = (owner, i);
        foreach (var pile in pcs.AllPiles)
        {
            string name = pile.Type.ToString().ToLowerInvariant();
            for (int i = 0; i < pile.Cards.Count; i++) TakeCard(s, p, pile.Cards[i], name, i);
        }
    }

    private static void TakeCard(Snap s, Player p, CardModel c, string pile, int pos)
    {
        var owner = Owner(p.Creature, p);
        s[c] = (owner, new JsonArray(pile, pos));
        if (c.Affliction != null) s[c.Affliction] = (owner, new JsonArray(pile, pos));
        if (c.Enchantment != null) s[c.Enchantment] = (owner, new JsonArray(pile, pos));
    }

    private static JsonNode? Owner(Creature? c, Player? p)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        if (c != null && combat != null)
        {
            int i = Slot(combat.Allies, c);
            if (i >= 0) return new JsonArray("Player", i);
            i = Slot(combat.Enemies, c);
            if (i >= 0) return new JsonArray("Enemy", i);
        }

        if (p != null) return new JsonArray("Player", Slot(p.RunState.Players, p));
        return null;
    }

    private static int Slot<T>(IReadOnlyList<T>? list, object item) where T : class?
    {
        if (list == null) return -1;
        for (int i = 0; i < list.Count; i++)
        {
            if (ReferenceEquals(list[i], item)) return i;
        }

        return -1;
    }

    private static JsonNode? Index<T>(IReadOnlyList<T>? list, object item) where T : class? => Slot(list, item);

    /// <summary>[pile, position]: the deck, else the combat piles.</summary>
    private static JsonNode? Place(CardModel c)
    {
        var p = c.Owner;
        if (p == null) return null;
        int i = Slot(p.Deck.Cards, c);
        if (i >= 0) return new JsonArray("deck", i);
        var piles = p.PlayerCombatState?.AllPiles;
        if (piles != null)
        {
            foreach (var pile in piles)
            {
                i = Slot(pile.Cards, c);
                if (i >= 0) return new JsonArray(pile.Type.ToString().ToLowerInvariant(), i);
            }
        }

        return new JsonArray("none", -1);
    }
}
