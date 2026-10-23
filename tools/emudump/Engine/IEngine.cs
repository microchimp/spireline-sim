// The one interface the oracle tools reach a simulation engine through, in the game's terms only:
// model ids are the game's UPPER_SNAKE strings as recordings write them (STRIKE_IRONCLAD), creatures
// are slots, cards on a screen are a candidate index plus a ModelId and the upgrade. No engine type
// (run engine, combat state, def-id ints, buff enums, RNG wrappers) appears in this folder, so a second
// engine can plug into the same adapters (referee/design/rebuild-plan.md section 1, "Oracle path").
// tools/emudump.Tests/EngineBoundaryTests.cs enforces the boundary.
// Fight.cs, Reanchor.cs and Program.cs's `fight --netstate` reach the engine through these members (the types Reanchor passes are in
// ReanchorTypes.cs); later rounds add members for Setup, HookProbe and Trace.
using System.Text.Json.Nodes;

namespace EmuDump.Engine;

/// <summary>A card offered on a screen: the game's model id and whether it is upgraded.</summary>
public sealed record Candidate(string ModelId, bool Upgraded);

/// <summary>
/// A screen the engine is waiting on. <see cref="Kind"/> is the engine's name for the screen (a string,
/// compared only for equality and shown in diagnostics). <see cref="MaxPicks"/> is null when the engine
/// asks for one pick at a time and has no count. <see cref="SourceModelId"/> is the model that opened it.
/// <see cref="Native"/> is the engine's own description of the candidates, for diagnostics only: adapters write
/// it out as it is and never read it (old: its card def ids).
/// </summary>
public sealed record OpenScreen(string Kind, IReadOnlyList<Candidate> Candidates, int? MaxPicks, bool Skippable, string? SourceModelId,
    JsonNode? Native = null);

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

/// <summary>
/// What a setup item came to: <see cref="Ok"/> when the engine built it, with a note or the reason it did not in
/// <see cref="Message"/> (null when there is nothing to say). A message that starts with "unsupported" marks a gap in the
/// engine's mirror, not a divergence; Setup.cs logs it as one.
/// </summary>
public readonly record struct SetupResult(bool Ok, string? Message);

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

    /// <summary>
    /// The game power ids (STRENGTH_POWER) the engine has a power for, as <c>buff_keys</c>, and the keys of its
    /// own powers that are no game power (a piece of one, or a bookkeeping marker), as <c>internal_keys</c>.
    /// tools/fidelity/reanchor_compare.py reads both.
    /// </summary>
    JsonObject PowerKeys();

    /// <summary>
    /// Opens re-anchoring on the fight <see cref="StartFight"/> left open (it throws when no combat is held): the dump
    /// of that state, and the block <see cref="ReanchorStep"/> starts from. The block is the engine's own previous
    /// state, which supplies what the game's snapshot cannot say (relic counters, enemy AI state, per-card
    /// modifiers, per-turn and per-combat counters, the potion belt).
    /// </summary>
    ReanchorOpening ReanchorOpen();

    /// <summary>
    /// Tries one recorded step on a fresh copy of the carried fight: loads the game's state after the step before,
    /// puts the enemies' moves on the game's where it can, applies the game's action and its choices, finishes a
    /// step left on a screen nobody answered, and writes what happened into <paramref name="record"/> (patched,
    /// anchor_error, moves, moves_unknown, hidden_pre, trace, hidden, choices, fallback_rng_calls, state,
    /// moves_carried, error), key by key as it goes, so a failure part-way leaves the keys written so far.
    /// </summary>
    ReanchorOutcome ReanchorStep(EngineBlock carry, ReanchorMode mode, ReanchorInput input, JsonObject record);

    /// <summary>
    /// The current combat in the shape of the game's NetFullCombatState (creatures, players with piles, potions,
    /// relics, orbs and RNG counters, ids as the game writes them), as tools/fidelity/netstate_diff.py reads it, or
    /// null when no combat is in progress. Fields the engine does not model are absent.
    /// </summary>
    JsonObject? GameState();

    /// <summary>
    /// The run's RNG streams' call counts under the game's names: <c>rng</c> (the run's streams) and <c>player_rng</c>
    /// (the player's), as the game's snapshots write them. Setup.cs logs them with each setup item.
    /// </summary>
    JsonObject RngCounters();

    // Setup items (Setup.cs): what the referee's setup did, redone in game terms. Each returns what it came to; an id
    // the engine has no model for throws KeyNotFoundException, which Setup.cs logs as an unsupported item.

    /// <summary>
    /// Obtains the relic (RelicCmd.Obtain on the map) and answers the selection its pickup opens with the picks the
    /// game made. <paramref name="cardChoices"/> are the game's own logged card choices for the item, as the record
    /// holds them (kind, step, options, picked): raw JSON for now.
    /// </summary>
    SetupResult ObtainRelic(string relicId, IReadOnlyList<JsonObject> cardChoices);

    /// <summary>PotionCmd.TryToProcure: the potion into the first free belt slot; not ok when there is none, or in a fight with Sozu.</summary>
    SetupResult ProcurePotion(string potionId);

    /// <summary>
    /// Adds a card made at the setup. <paramref name="pile"/> is "deck" (on the map), or "hand", "discard" or "draw" (in a
    /// fight); <paramref name="onTop"/> puts a card added to the draw pile on top instead of at the bottom.
    /// </summary>
    SetupResult AddCard(string cardId, bool upgraded, string pile, bool onTop);

    /// <summary>Applies a power with no applier, in a fight, to the player (<paramref name="targetSlot"/> -1) or the enemy in that slot.</summary>
    SetupResult ApplyPower(string powerId, int targetSlot, int amount);

    /// <summary>The player gains energy, in a fight.</summary>
    SetupResult GainEnergy(int amount);

    /// <summary>Sets the player's max HP, then current HP (capped at max), on the map; null keeps the value held.</summary>
    SetupResult SetHp(int? maxHp, int? currentHp);

    /// <summary>
    /// A reference-set item the members above do not cover (relics_set, deck_set, potions_set, player_load, act), as the
    /// record holds it: raw JSON for now. Null when the item is none of those.
    /// </summary>
    SetupResult? RefsetItem(JsonObject item, IReadOnlyList<JsonObject> cardChoices);

    /// <summary>
    /// The fight item: walks onto map option <paramref name="mapOption"/> but meets the encounter at this place in the
    /// game's act lists (act name, kind weak/normal/elite/boss, index). Returns the encounter's name for the setup log;
    /// throws InvalidOperationException when the place names no encounter. <paramref name="gameId"/> is the game's
    /// encounter id, for the messages.
    /// </summary>
    string EnterEncounter(int mapOption, string act, string kind, int index, string gameId);

    /// <summary>
    /// For the scenario generator (<c>EmuDump setupinfo</c>), independent of the run held: per game relic the follow-up
    /// its pickup opens on the map (<c>relics</c>), and <see cref="PowerKeys"/>' <c>buff_keys</c> and <c>internal_keys</c>.
    /// </summary>
    JsonObject SetupInfo();
}
