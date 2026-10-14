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
/// shares it.
/// </para>
/// <para>
/// The replay runs on the streams and the draw order the step started with, because only those
/// reach the same screen over the same piles. A copy that stood on the screen may not be on them
/// any more: a search fork is determinized there (the native <c>Clone(resampleSeed)</c> with
/// <c>ReshuffleUnknownDrawOrder</c>, <c>RunEngine.Clone(resampleSeed)</c>,
/// <c>ForkActiveCombat</c>), and a loaded snapshot has streams of its own. So each answer takes
/// along the hidden state of the combat that gave it, as it stood on the screen
/// (<see cref="Hidden" />, <see cref="ScreenHiddenState" />), and the replay puts it back at that
/// screen, before the answer is resolved: the pick and everything after it play on the copy's
/// streams and order, wherever the determinization came from. For a copy nobody changed, it is
/// what the replay reached anyway. It used to keep the true streams after the pick, and later
/// put the determinization back only once the step was over: a search choosing what Burning Pact
/// exhausts drew the true next two cards in every fork.
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

    /// <summary>
    /// For each of <see cref="Answers" />, the hidden state of the combat that gave it, as it stood
    /// on the screen: what the replay puts back at that screen before it resolves the answer.
    /// </summary>
    public required IReadOnlyList<ScreenHiddenState> Hidden { get; init; }
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

    /// <summary>The step's stream, which a screen's answer resolves on.</summary>
    public required Random Rng { get; init; }

    public required int Action { get; init; }

    public required int TargetEnemyIndex { get; init; }

    public required List<int> Answers { get; init; }

    /// <inheritdoc cref="HookChoiceReplay.Hidden" />
    public required List<ScreenHiddenState> Hidden { get; init; }

    /// <summary>How many of <see cref="Answers" /> this run of the step has used.</summary>
    public int Used;

    /// <summary>
    /// The next answer, with the combat put back to the hidden state of the copy that gave it
    /// (<see cref="HookChoiceReplay.Hidden" />): the screen the replay has reached is the one that
    /// copy stood on. False when every answer is used: the step stops here.
    /// </summary>
    public bool TryNextAnswer(out int answer)
    {
        if (Used >= Answers.Count)
        {
            answer = -1;
            return false;
        }

        Hidden[Used].PutBack(State, Rng);
        answer = Answers[Used++];
        return true;
    }

    /// <summary>The record a stopped step leaves on its screen.</summary>
    public HookChoiceReplay Stop() =>
        new()
        {
            Before = Before,
            RngBefore = RngBefore,
            Action = Action,
            TargetEnemyIndex = TargetEnemyIndex,
            Answers = [.. Answers],
            Hidden = [.. Hidden],
        };
}

/// <summary>
/// What can differ between the combat that stopped on a screen and a copy answering it, as the copy
/// stood there: every stream the combat owns, the stream its steps are taken on, the draw pile's
/// order -- what a determinization changes -- and the screen's candidates, which a snapshot load
/// follows through its own shuffle. Taken from the combat an answer is given to, put back where the
/// replay reaches that screen (<see cref="HookChoiceReplay" />).
/// </summary>
/// <remarks>
/// The streams are put back in place (<see cref="CountingRandom.Become" />), not replaced, so code
/// in the middle of the step that holds one goes on with it. The draw pile is put back only as an
/// order: when its cards are not the same as the replay's, something other than a determinization
/// changed the combat on the screen, and the replay's own pile stands. The candidates are put back
/// only onto the same screen (kind, source and count), since the answer is an index into the screen
/// the copy showed.
/// </remarks>
public sealed class ScreenHiddenState
{
    internal ScreenHiddenState(CountingRandom?[] streams, CountingRandom? stepRng, CardInstance[] drawPile, int[]? candidates)
    {
        Streams = streams;
        StepRng = stepRng;
        DrawPile = drawPile;
        Candidates = candidates;
    }

    /// <summary>The combat's streams, in <c>StateCloning.StreamNames</c> order; null where it had none.</summary>
    internal CountingRandom?[] Streams { get; }

    /// <summary>The stream the answer was given on, when it can be put back.</summary>
    internal CountingRandom? StepRng { get; }

    internal CardInstance[] DrawPile { get; }

    /// <summary>The open screen's candidates, null when no screen was open.</summary>
    internal int[]? Candidates { get; }

    /// <summary><paramref name="state" />'s hidden state, with <paramref name="rng" /> as its step's stream.</summary>
    public static ScreenHiddenState Of(CombatState state, Random rng)
    {
        var streams = new CountingRandom?[StateCloning.StreamCount];
        for (int i = 0; i < streams.Length; i++)
        {
            streams[i] = StateCloning.Stream(state, i)?.Copy();
        }

        return new(streams, (rng as CountingRandom)?.Copy(), [.. state.DrawPile], state.PendingSelection?.Candidates.ToArray());
    }

    /// <summary>Puts this hidden state back on <paramref name="state" /> and <paramref name="rng" />.</summary>
    public void PutBack(CombatState state, Random rng)
    {
        for (int i = 0; i < Streams.Length; i++)
        {
            if (Streams[i] is not { } kept)
            {
                continue;
            }

            if (StateCloning.Stream(state, i) is { } live)
            {
                live.Become(kept);
            }
            else
            {
                StateCloning.SetStream(state, i, kept.Copy());
            }
        }

        if (StepRng is not null && rng is CountingRandom step)
        {
            step.Become(StepRng);
        }

        var pile = state.DrawPile;
        if (pile.Count == DrawPile.Length && !pile.SequenceEqual(DrawPile) && SameCards(pile, DrawPile))
        {
            for (int i = 0; i < DrawPile.Length; i++)
            {
                pile[i] = DrawPile[i];
            }
        }

        if (
            Candidates is { } shown
            && state.PendingSelection is { } open
            && open.Candidates.Count == shown.Length
            && !open.Candidates.SequenceEqual(shown)
        )
        {
            state.PendingSelection = open.Clone(open.Replay, candidates: [.. shown]);
        }
    }

    private static bool SameCards(List<CardInstance> pile, CardInstance[] kept)
    {
        var counts = new Dictionary<CardInstance, int>();
        foreach (var card in pile)
        {
            counts[card] = counts.GetValueOrDefault(card) + 1;
        }

        foreach (var card in kept)
        {
            if (counts.GetValueOrDefault(card) == 0)
            {
                return false;
            }

            counts[card]--;
        }

        return true;
    }
}

/// <summary>Unwinds a step to <c>CombatEngine.Step</c> at a screen with no answer yet.</summary>
internal sealed class HookChoicePending : Exception
{
    public static readonly HookChoicePending Instance = new();
}
