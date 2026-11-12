// Engine faults (docs/ENGINE_FAULTS.md): what the copy throws where the game's own code does not finish a step.
namespace Sts2Emulator.Core;

/// <summary>
/// A state where the game's own code never finishes the step: its loop has no exit there, and the game would draw or
/// play forever. The copy stops and says so instead of running until memory is gone. Pillage drawing for a Hellraiser
/// whose every auto-play is refused is the one known (engine fault class 2).
/// </summary>
public sealed class GameHangs(string message) : Exception(message);

/// <summary>
/// A step the copy will not follow further although the game might finish it: a chain deeper than the copy's stack
/// allows (Unceasing Top's draws, each a Strike Hellraiser plays into another empty hand).
/// </summary>
public sealed class EngineLimit(string message) : Exception(message);
