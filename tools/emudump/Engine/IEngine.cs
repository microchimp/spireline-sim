// The one interface the oracle tools reach a simulation engine through, in the game's terms only:
// model ids are the game's UPPER_SNAKE strings as recordings write them (STRIKE_IRONCLAD), creatures
// are slots, cards on a screen are a candidate index plus a ModelId and the upgrade. No engine type
// (run engine, combat state, def-id ints, buff enums, RNG wrappers) appears in this folder, so a second
// engine can plug into the same adapters (referee/design/rebuild-plan.md section 1, "Oracle path").
// tools/emudump.Tests/EngineBoundaryTests.cs enforces the boundary.
// This round carries what Fight.cs needs; later rounds add members for Setup, NetState, Reanchor,
// HookProbe and Trace.
using System.Text.Json.Nodes;

namespace EmuDump.Engine;

/// <summary>A card offered on a screen: the game's model id and whether it is upgraded.</summary>
public sealed record Candidate(string ModelId, bool Upgraded);

/// <summary>
/// A screen the engine is waiting on. <see cref="Kind"/> is the engine's name for the screen (a string,
/// compared only for equality and shown in diagnostics). <see cref="MaxPicks"/> is null when the engine
/// asks for one pick at a time and has no count. <see cref="SourceModelId"/> is the model that opened it.
/// </summary>
public sealed record OpenScreen(string Kind, IReadOnlyList<Candidate> Candidates, int? MaxPicks, bool Skippable, string? SourceModelId);

/// <summary>Something that shows an open screen and answers it: an engine, or an adapter over a bare combat.</summary>
public interface IScreenSource
{
    /// <summary>The open screen, or null.</summary>
    OpenScreen? Screen { get; }

    /// <summary>Picks the candidate at this index of <see cref="Screen"/>.</summary>
    void Pick(int candidate);

    /// <summary>Declines the open screen (legal only when <see cref="OpenScreen.Skippable"/>).</summary>
    void Decline();
}

/// <summary>
/// What starts a fight: a referee record's spec. The typed members are read on demand (a record missing
/// one fails where the engine needs it, as the replay always did); <see cref="Record"/> is the whole
/// record, for the setup items and the choices the fight's start opens.
/// </summary>
public sealed class FightSpec(JsonObject record)
{
    public JsonObject Record { get; } = record;

    public string Seed => (string)Record["seed"]!;

    public int Ascension => (int)Record["ascension"]!;

    /// <summary>Cards the referee added to the deck before Neow, as game model ids.</summary>
    public IEnumerable<string> DeckAdd => (Record["deck_add"]?.AsArray() ?? new JsonArray()).Select(n => (string)n!);

    public int NeowPick => (int)Record["neow_pick"]!;

    /// <summary>The map room the fight is in, as the record's [column, row].</summary>
    public (int Col, int Row) Room => ((int)Record["room"]![0]!, (int)Record["room"]![1]!);
}

/// <summary>One simulation engine, driven in game terms.</summary>
public interface IEngine : IScreenSource
{
    /// <summary>The engine's name as <c>--engine</c> spells it: "old".</summary>
    string Name { get; }

    /// <summary>
    /// Runs the run up to the fight the spec describes and leaves the fight open: deck additions, the Neow
    /// pick, card choices, the walk into the room, the setup items and the choices the start opened.
    /// Writes what it produced into <paramref name="report"/> as it goes (seed, neow_options, setup_log,
    /// fight, start_choices), so a failure part-way leaves the keys written so far.
    /// </summary>
    void StartFight(FightSpec spec, JsonObject report);

    /// <summary>Whether a combat is held (a finished one stays until the run moves on).</summary>
    bool CombatOpen { get; }

    /// <summary>Plays the card at this hand index at the creature in this slot (-1: no target).</summary>
    void PlayCard(int handIndex, int targetSlot);

    /// <summary>Uses the potion in this belt slot at the creature in this slot (-1: no target).</summary>
    void UsePotion(int slot, int targetSlot);

    void EndTurn();

    /// <summary>The engine's native dump of the current step, in the shape tools/fidelity/fight_compare.py reads.</summary>
    JsonObject DumpStep();
}
