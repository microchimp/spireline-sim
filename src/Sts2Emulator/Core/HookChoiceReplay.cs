namespace Sts2Emulator.Core;

/// <summary>
/// A step that stopped on a screen a HOOK raised in the middle of it, kept so that the step can
/// be run again with the answer.
/// </summary>
/// <remarks>
/// <para>
/// The game awaits a hook's screen where the hook raised it, and its caller carries on from
/// there with the answer in hand: `StratagemPower.AfterShuffle` asks from inside
/// `CardPileCmd.Draw`, so the draw that shuffled goes on drawing after the pick; `TyrannyPower`
/// and `EntropyPower` ask in `AfterPlayerTurnStart`, so the turn start goes on to the intent
/// roll and the auto-pre-play phase after the pick. A card's own screen is the last thing its
/// effect does, so a step can simply end on it; a hook's screen is not, and the emulator has no
/// way to stop a call stack halfway and resume it later.
/// </para>
/// <para>
/// So it does not resume: it replays. Every step starts from a copy of the combat
/// (<see cref="Before" />). A hook screen that has no answer yet stops the step there, with the
/// screen open over the combat as it stands at that moment. The answer runs the whole step
/// again from <see cref="Before" /> with every answer given so far
/// (<see cref="Answers" />); each screen takes its answer where the game would await it, and the
/// step goes on to its end or to the next unanswered screen. The engine is deterministic given
/// the combat and its streams, so the replay reaches the same screen over the same piles.
/// </para>
/// <para>
/// A copy that is resampled (<c>Clone(rngSeed)</c>) keeps the original <see cref="Before" />, so
/// the rest of the step it is stopped in plays out on the streams the step started with; the
/// resampled ones take over from the next step.
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
    public required List<int> Answers { get; init; }

    /// <summary>How many of <see cref="Answers" /> the current run of the step has used.</summary>
    public int Used;

    /// <summary>
    /// <see cref="Before" /> and <see cref="RngBefore" /> are shared: nothing writes to them.
    /// </summary>
    public HookChoiceReplay Clone() =>
        new()
        {
            Before = Before,
            RngBefore = RngBefore,
            Action = Action,
            TargetEnemyIndex = TargetEnemyIndex,
            Answers = [.. Answers],
            Used = Used,
        };
}

/// <summary>Unwinds a step to <c>CombatEngine.Step</c> at a hook screen with no answer yet.</summary>
internal sealed class HookChoicePending : Exception
{
    public static readonly HookChoicePending Instance = new();
}
