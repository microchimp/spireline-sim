// What re-anchoring passes between the Reanchor adapter and an engine (referee/SPEED_PLAN.md step 6; the
// members are on IEngine). Game terms and plain JSON only: the engine's own state between two steps travels in
// an EngineBlock that the adapter holds and hands back unread (referee/design/rebuild-plan.md section 1).
using System.Text.Json.Nodes;

namespace EmuDump.Engine;

/// <summary>
/// An engine-private block: whatever the engine carries from one call to the next that the game's record has no
/// field for (old: the combat it left, with its relic counters, enemy AI state and per-card modifiers, and the
/// random streams). Adapters hold it and pass it back to the engine that made it; they never read it.
/// </summary>
public abstract class EngineBlock;

/// <summary>
/// The tests of the anchoring itself, run by the engine on each step (<c>EmuDump reanchor</c> flags).
/// <c>Scramble</c> wrecks every field the game's snapshot covers before it is written back, so a fight that
/// matches anyway proves the write is complete. <c>Stale</c> names the groups of hidden state (enemy, cards,
/// relics, counters, or all) that start every step from the fight's first state instead of the previous step's.
/// <c>PerturbMoves</c> moves every cycling enemy one position on before each step, and the monster AI draws from
/// another stream, so the moves must be put back on the game's. <c>Trace</c> records what happens inside each step
/// (written as <c>hidden_pre</c>, <c>trace</c> and <c>hidden</c> in the step's record).
/// </summary>
public sealed record ReanchorMode(bool Scramble = false, IReadOnlySet<string>? Stale = null, bool PerturbMoves = false, bool Trace = false);

/// <summary>
/// One step to try, from the referee record: step <see cref="Step"/> is put into the game's state after step
/// <c>Step - 1</c>, the game's action of step <c>Step</c> is applied, and the choices the game made for it are answered.
/// </summary>
/// <param name="GameSteps">The record's <c>steps</c>: each has the game's <c>action</c> and its <c>state</c> after it.</param>
/// <param name="Step">The index into <paramref name="GameSteps"/> of the action to apply (at least 1).</param>
/// <param name="GameChoices">The record's <c>choices</c>: the card choices the game made, each tagged with its step.</param>
/// <param name="AnswerOpen">
/// A format-1 recording's untagged choices: answers every screen open on this source for which an unused choice fits,
/// and returns the choices used; null for a recording whose choices carry their step.
/// </param>
public sealed record ReanchorInput(JsonArray GameSteps, int Step, IReadOnlyList<JsonNode?> GameChoices,
    Func<IScreenSource, JsonArray, List<JsonObject>>? AnswerOpen);

/// <summary>The fight as re-anchoring starts it: the dump of the opening state (the record's step 0) and the carried block.</summary>
public sealed record ReanchorOpening(JsonObject State, EngineBlock Carry);

/// <summary>
/// What a step leaves: the block to carry into the next one, and whether re-anchoring stops here (the step could
/// not be tried, or the fight ended in the game and in the engine).
/// </summary>
public sealed record ReanchorOutcome(EngineBlock Carry, bool Stop);
