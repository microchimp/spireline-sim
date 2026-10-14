namespace Sts2Emulator.Core;

/// <summary>
/// A step that stopped on a screen raised in the middle of it, kept ON that screen
/// (<see cref="PendingCardSelection.Replay" />) so that the screen's answer can run the step again.
/// </summary>
/// <remarks>
/// <para>
/// The game awaits a screen where it is raised, and the code that raised it carries on from there
/// with the answer in hand: `StratagemPower.AfterShuffle` asks from inside `CardPileCmd.Draw`, so
/// the draw that shuffled goes on drawing after the pick; `TyrannyPower` and `EntropyPower` ask in
/// `AfterPlayerTurnStart`, so the turn start goes on after the pick; Knowledge Demon's curse is
/// chosen inside its move, before the player's turn starts. The emulator has no way to stop a
/// call stack halfway and resume it later.
/// </para>
/// <para>
/// So it does not resume: it replays. Every step starts from a copy of the combat
/// (<see cref="Before" />). A screen with no answer yet stops the step there, with the screen open
/// over the combat as it stands at that moment, and this record attached to the screen. The answer
/// runs the whole step again from <see cref="Before" /> with every answer given so far
/// (<see cref="Answers" />); each screen takes its answer where the game would await it, and the
/// step goes on to its end or to the next unanswered screen. The engine is deterministic given the
/// combat and its streams, so the replay reaches the same screen over the same piles.
/// </para>
/// <para>
/// It lives on the screen, not on the combat, so nothing can keep it once its screen is gone:
/// whatever closes, replaces or clears <c>PendingSelection</c> (a reset, a re-anchor, a snapshot
/// load) drops the stopped step with it. It is never changed once made, so a clone of the combat
/// shares it, and a copy that is resampled (<c>Clone(rngSeed)</c>) keeps the original
/// <see cref="Before" />: the rest of the step it is stopped in plays out on the streams the step
/// started with, and the resampled ones take over from the next step.
/// </para>
/// </remarks>
public sealed class HookChoiceReplay
{
    /// <summary>The combat as the step found it. Never changed: a replay copies it.</summary>
    public required CombatState Before { get; init; }

    /// <summary>The step's own stream as the step found it. Never changed either.</summary>
    public required CountingRandom RngBefore { get; init; }

    /// <summary>The step's action and target, as the caller gave them.</summary>
    public required int Action { get; init; }

    public required int TargetEnemyIndex { get; init; }

    /// <summary>The answers given so far, in the order the step asked for them.</summary>
    public required IReadOnlyList<int> Answers { get; init; }
}

/// <summary>
/// The step running now on this thread (<c>CombatEngine.Step</c> sets it and puts the previous one
/// back before it returns): which combat, what it can be replayed from, and where its answers come
/// from. Null outside a step, and for a step that cannot be replayed.
/// </summary>
internal sealed class StepInProgress
{
    public required CombatState State { get; init; }

    public required CombatState Before { get; init; }

    public required CountingRandom RngBefore { get; init; }

    public required int Action { get; init; }

    public required int TargetEnemyIndex { get; init; }

    public required List<int> Answers { get; init; }

    /// <summary>How many of <see cref="Answers" /> this run of the step has used.</summary>
    public int Used;

    /// <summary>The record a stopped step leaves on its screen.</summary>
    public HookChoiceReplay Stop() =>
        new()
        {
            Before = Before,
            RngBefore = RngBefore,
            Action = Action,
            TargetEnemyIndex = TargetEnemyIndex,
            Answers = [.. Answers],
        };
}

/// <summary>Unwinds a step to <c>CombatEngine.Step</c> at a screen with no answer yet.</summary>
internal sealed class HookChoicePending : Exception
{
    public static readonly HookChoicePending Instance = new();
}
