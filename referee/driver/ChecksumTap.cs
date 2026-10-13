// The game's own desync checksums as a per-action oracle. In multiplayer the game hashes its full
// combat state (NetFullCombatState: creatures, piles, relics with their saved counters, RNG
// counters, ...) after every finished game action and at each turn-phase boundary, so peers can
// detect a divergence (ChecksumTracker.GenerateChecksum). Single player leaves the tracker off
// (RunManager.InitializeShared enables it for host/client/replay only); this tap switches it on for
// one fight and records, per recorded step, what the game generated:
//
//   step["checksums"] = [[context, action type or null, checksum], ...]   ("checksums": true)
//   step["net"]       = [{"ctx", "id", "state": <full state>}, ...]       ("trace": true, also)
//   result["checksum_cost"] = {"n", "gen_us", "tap_us"}                   (time inside the game's
//                                                                          GenerateChecksum, and here)
//
// Spec or job keys, spec wins. Off (the default) it does nothing at all: no patch, no subscription,
// no key in the result. Turning the tracker on adds no game rule; its only consumers in single
// player are this tap and the combat replay recorder.

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace Spireline.Referee;

internal static class ChecksumTap
{
    private static bool _on, _trace, _patched, _wasEnabled;
    private static ChecksumTracker? _tracker;
    private static GameAction? _action;
    private static long _t0, _genTicks, _tapTicks;
    private static int _count;
    private static string? _error;
    private static JsonArray _pending = new(), _pendingNet = new();
    private static (bool checksums, bool trace)? _jobDefaults;

    /// <summary>Job-level defaults, read once from the job file the driver was started with.</summary>
    private static (bool checksums, bool trace) JobDefaults()
    {
        if (_jobDefaults is { } d) return d;
        bool sums = false, trace = false;
        try
        {
            string? path = System.Environment.GetEnvironmentVariable("SPIRELINE_REFEREE_JOB");
            if (!string.IsNullOrEmpty(path) && JsonNode.Parse(File.ReadAllText(path)) is JsonObject job)
            {
                trace = (bool?)job["trace"] == true;
                sums = trace || (bool?)job["checksums"] == true;
            }
        }
        catch (Exception)
        {
            // The driver reports an unreadable job; here it only means "off".
        }

        _jobDefaults = (sums, trace);
        return (sums, trace);
    }

    /// <summary>Called once the run exists and before the fight's room is entered.</summary>
    internal static void Begin(JsonObject spec)
    {
        var (jobSums, jobTrace) = JobDefaults();
        _trace = (bool?)spec["trace"] ?? jobTrace;
        _on = _trace || ((bool?)spec["checksums"] ?? jobSums);
        if (!_on) return;
        Patch();
        _pending = new JsonArray();
        _pendingNet = new JsonArray();
        _genTicks = _tapTicks = 0;
        _count = 0;
        _error = null;
        _action = null;
        // A fresh tracker per run (RunManager.InitializeShared).
        _tracker = RunManager.Instance.ChecksumTracker;
        _wasEnabled = _tracker.IsEnabled;
        _tracker.IsEnabled = true;
        _tracker.ChecksumGenerated += OnChecksum;
    }

    /// <summary>Hands what was generated since the previous step to the step just recorded.</summary>
    internal static void Attach(JsonArray steps)
    {
        if (!_on || steps.Count == 0 || steps[^1] is not JsonObject step) return;
        step["checksums"] = _pending;
        _pending = new JsonArray();
        if (_trace)
        {
            step["net"] = _pendingNet;
            _pendingNet = new JsonArray();
        }
    }

    /// <summary>Switches the tracker back and writes the cost; the run is cleaned up after this.</summary>
    internal static void End(JsonObject result)
    {
        if (!_on) return;
        _on = false;
        if (_tracker != null)
        {
            _tracker.ChecksumGenerated -= OnChecksum;
            _tracker.IsEnabled = _wasEnabled;
            _tracker = null;
        }

        double us = 1e6 / Stopwatch.Frequency;
        result["checksum_cost"] = new JsonObject
        {
            ["n"] = _count, ["gen_us"] = Math.Round(_genTicks * us), ["tap_us"] = Math.Round(_tapTicks * us),
        };
        if (_pending.Count > 0) result["checksums_tail"] = _pending;  // generated after the last step
        if (_error != null) result["checksum_error"] = _error;
        _pending = new JsonArray();
        _pendingNet = new JsonArray();
    }

    /// <summary>
    /// Times the game's GenerateChecksum(context, action) and remembers which action it is for: the
    /// ChecksumGenerated event carries only the context text.
    /// </summary>
    private static void Patch()
    {
        if (_patched) return;
        _patched = true;
        var h = new Harmony("spireline.referee.checksum");
        h.Patch(AccessTools.Method(typeof(ChecksumTracker), nameof(ChecksumTracker.GenerateChecksum),
                new[] { typeof(string), typeof(GameAction) }),
            prefix: new HarmonyMethod(typeof(ChecksumTap), nameof(GeneratePrefix)),
            postfix: new HarmonyMethod(typeof(ChecksumTap), nameof(GeneratePostfix)));
    }

    private static void GeneratePrefix(GameAction? action)
    {
        _action = action;
        _t0 = Stopwatch.GetTimestamp();
    }

    private static void GeneratePostfix()
    {
        if (_on) _genTicks += Stopwatch.GetTimestamp() - _t0;
        _action = null;
    }

    private static void OnChecksum(NetChecksumData data, string context, NetFullCombatState state)
    {
        long t0 = Stopwatch.GetTimestamp();
        try
        {
            string? type = _action?.GetType().Name;
            _pending.Add(new JsonArray(type != null ? "action" : context, type, (long)data.checksum));
            if (_trace)
            {
                _pendingNet.Add(new JsonObject { ["ctx"] = context, ["id"] = (long)data.id, ["state"] = Full(state) });
            }

            _count++;
        }
        catch (Exception ex)
        {
            _error ??= ex.ToString();  // never let the tap break the game's action
        }

        _tapTicks += Stopwatch.GetTimestamp() - t0;
    }

    // ── NetFullCombatState as JSON: every field the game hashes, under its own name ──────────

    private static JsonObject Full(NetFullCombatState s) => new()
    {
        ["creatures"] = new JsonArray(s.Creatures.Select(c => (JsonNode)new JsonObject
        {
            ["monster"] = c.monsterId?.Entry,
            ["player"] = c.playerId,
            ["hp"] = c.currentHp, ["max_hp"] = c.maxHp, ["block"] = c.block,
            ["powers"] = new JsonArray(c.powers.Select(p => (JsonNode)new JsonArray(p.id.Entry, p.amount)).ToArray()),
        }).ToArray()),
        ["players"] = new JsonArray(s.Players.Select(p => (JsonNode)Player(p)).ToArray()),
        ["rng"] = new JsonObject
        {
            ["seed"] = s.Rng.Seed,
            ["counters"] = Counters(s.Rng.Counters.Select(kv => (kv.Key.ToString(), kv.Value))),
        },
        ["choice_ids"] = new JsonArray(s.nextChoiceIds.Select(i => (JsonNode)(long)i).ToArray()),
        ["reward_ids"] = new JsonArray(s.nextRewardIds.Select(i => (JsonNode)i).ToArray()),
        ["last_action"] = s.lastExecutedActionId,
        ["last_hook"] = s.lastExecutedHookId,
    };

    private static JsonObject Player(NetFullCombatState.PlayerState p)
    {
        var piles = new JsonObject();
        foreach (var pile in p.piles)
        {
            piles[pile.pileType.ToString()] = new JsonArray(pile.cards.Select(c => (JsonNode)Card(c)).ToArray());
        }

        var bag = new JsonObject();
        foreach (var kv in p.relicGrabBag.RelicIdLists.OrderBy(kv => kv.Key))
        {
            bag[kv.Key.ToString()] = new JsonArray(kv.Value.Select(m => (JsonNode)m.Entry).ToArray());
        }

        return new JsonObject
        {
            ["id"] = p.playerId, ["character"] = p.characterId.Entry,
            ["turn"] = p.turnNumber, ["phase"] = p.phase.ToString(),
            ["energy"] = p.energy, ["stars"] = p.stars, ["max_potions"] = p.maxPotionCount, ["gold"] = p.gold,
            ["piles"] = piles,
            ["potions"] = new JsonArray(p.potions.Select(x => (JsonNode)x.id.Entry).ToArray()),
            ["relics"] = new JsonArray(p.relics.Select(r => (JsonNode)Relic(r.relic)).ToArray()),
            ["orbs"] = new JsonArray(p.orbs.Select(o => (JsonNode)new JsonArray(o.id.Entry, o.passive, o.evoke)).ToArray()),
            ["rng"] = new JsonObject
            {
                ["seed"] = p.rngSet.Seed,
                ["counters"] = Counters(p.rngSet.Counters.Select(kv => (kv.Key.ToString(), kv.Value))),
            },
            ["grab_bag"] = bag,
        };
    }

    private static JsonObject Counters(System.Collections.Generic.IEnumerable<(string name, int n)> counters)
    {
        var o = new JsonObject();
        foreach (var (name, n) in counters.OrderBy(c => c.name, StringComparer.Ordinal)) o[name] = n;
        return o;
    }

    private static JsonObject Card(NetFullCombatState.CardState c)
    {
        var o = Serializable(c.card);
        if (c.affliction != null) o["affliction"] = new JsonArray(c.affliction.Entry, c.afflictionCount);
        if (c.energyCost is int cost) o["cost"] = cost;
        if (c.keywords != null) o["keywords"] = new JsonArray(c.keywords.Select(k => (JsonNode)k.ToString()).ToArray());
        return o;
    }

    private static JsonObject Serializable(SerializableCard c)
    {
        var o = new JsonObject { ["id"] = c.Id?.Entry, ["upgrades"] = c.CurrentUpgradeLevel };
        if (c.Enchantment is { } e)
        {
            var ench = new JsonObject { ["id"] = e.Id?.Entry, ["amount"] = e.Amount };
            if (Props(e.Props) is { } ep) ench["props"] = ep;
            o["enchantment"] = ench;
        }

        if (Props(c.Props) is { } props) o["props"] = props;
        if (c.FloorAddedToDeck is int floor) o["floor"] = floor;
        return o;
    }

    private static JsonObject Relic(SerializableRelic r)
    {
        var o = new JsonObject { ["id"] = r.Id?.Entry };
        if (Props(r.Props) is { } props) o["props"] = props;
        if (r.FloorAddedToDeck is int floor) o["floor"] = floor;
        return o;
    }

    /// <summary>A model's saved properties (its hidden counters and flags) as name: value.</summary>
    private static JsonObject? Props(SavedProperties? p)
    {
        if (p == null) return null;
        var o = new JsonObject();
        foreach (var x in p.ints ?? new()) o[x.name] = x.value;
        foreach (var x in p.bools ?? new()) o[x.name] = x.value;
        foreach (var x in p.strings ?? new()) o[x.name] = x.value;
        foreach (var x in p.intArrays ?? new()) o[x.name] = new JsonArray(x.value.Select(v => (JsonNode)v).ToArray());
        foreach (var x in p.modelIds ?? new()) o[x.name] = x.value?.Entry;
        foreach (var x in p.cards ?? new()) o[x.name] = Serializable(x.value);
        foreach (var x in p.cardArrays ?? new())
        {
            o[x.name] = new JsonArray(x.value.Select(v => (JsonNode)Serializable(v)).ToArray());
        }

        return o.Count > 0 ? o : null;
    }
}
