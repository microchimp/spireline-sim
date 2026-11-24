using System.Collections.Generic;
using System.Text.Json.Nodes;

namespace Spireline.Referee;

/// <summary>
/// The fight actions the driver took without asking since the last asked decision (Env.Decide asks only with two or more legal
/// actions; with one it takes it). Each is [kind, label]: the decision kind ("combat", "select_cards_combat",
/// "card_reward_combat") and the one action's label as an asked decision would have listed it (["end"] for a lone End Turn).
/// The list goes out as "unasked" with the next asked decision and with the run's end message, then starts empty again, so the
/// lockstep's copy (referee/lockstep.py EmuLock._auto_combat) steps its End Turn exactly as often as the game took one, instead of
/// guessing the game's lone End Turns from the copy's own legal list (lane 21's group A2: the game's player had nothing left to
/// play and the game ended the turn; the copy still had a playable card and stood a turn behind). A decision outside a fight is
/// not listed. Plain values in and out, so the rule is testable without the game (referee/driver.Tests).
/// </summary>
internal sealed class UnaskedLog
{
    private readonly List<JsonNode> _since = new();

    /// <summary>The decision kinds of a fight, as referee/lockstep.py COMBAT_KINDS names them.</summary>
    internal static bool FightKind(string? kind) => kind is "combat" or "select_cards_combat" or "card_reward_combat";

    /// <summary>A lone action taken without asking: listed when its kind is a fight's (a copy of the label, which may be null).</summary>
    internal void Took(string kind, JsonNode? label)
    {
        if (FightKind(kind)) _since.Add(new JsonArray(JsonValue.Create(kind), label?.DeepClone()));
    }

    /// <summary>The lone actions listed since the last call, oldest first, and the list emptied.</summary>
    internal JsonArray Drain()
    {
        var a = new JsonArray(_since.ToArray());
        _since.Clear();
        return a;
    }

    /// <summary>A new run: nothing of the last one carries over.</summary>
    internal void Clear() => _since.Clear();

    internal int Count => _since.Count;
}
