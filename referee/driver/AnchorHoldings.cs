using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;

namespace Spireline.Referee;

/// <summary>
/// The player's holdings as the lockstep's re-anchor reads them from "anchor_state" (Env.AnchorStateOf): per deck card its
/// enchantment, the enchantment's amount and, on Mad Science, the Tinker Time type and rider, and per relic the game's saved
/// counters by the game's own property names. Plain values in, JSON out, so the shape is testable without the game
/// (referee/driver.Tests); Env.cs reads the game's models and hands the values here. Every key is new to the message:
/// nothing that read "anchor_state" before reads these.
/// </summary>
internal static class AnchorHoldings
{
    /// <summary>
    /// One deck card: {"id", "up"} always; "ench" (the enchantment's entry) and "amount" when enchanted; "tinker_type" and
    /// "tinker_rider" (the game's CardType and rider enum values, -1 for a type not chosen) on a Mad Science.
    /// </summary>
    internal static JsonObject DeckCard(string id, bool upgraded, string? enchantment, int enchantAmount,
        int? tinkerType = null, int? tinkerRider = null)
    {
        var o = new JsonObject { ["id"] = id, ["up"] = upgraded ? 1 : 0 };
        if (enchantment != null)
        {
            o["ench"] = enchantment;
            o["amount"] = enchantAmount;
        }

        if (tinkerType is int type)
        {
            o["tinker_type"] = type;
            o["tinker_rider"] = tinkerRider ?? 0;
        }

        return o;
    }

    /// <summary>One relic: {"id"} and, when it saves any, "ints" and "bools" as {property: value} (the game's names).</summary>
    internal static JsonObject Relic(string id, IEnumerable<KeyValuePair<string, int>>? ints,
        IEnumerable<KeyValuePair<string, bool>>? bools)
    {
        var o = new JsonObject { ["id"] = id };
        var i = ints?.ToList();
        if (i is { Count: > 0 })
        {
            var m = new JsonObject();
            foreach (var kv in i) m[kv.Key] = kv.Value;
            o["ints"] = m;
        }

        var b = bools?.ToList();
        if (b is { Count: > 0 })
        {
            var m = new JsonObject();
            foreach (var kv in b) m[kv.Key] = kv.Value;
            o["bools"] = m;
        }

        return o;
    }
}
