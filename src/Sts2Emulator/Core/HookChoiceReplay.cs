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
/// whatever closes, replaces or clears <c>PendingSelection</c> (a reset, a re-anchor) drops the
/// stopped step with it, and a document writes it down with the screen (<c>stopped_step</c>). It is
/// never changed once made, so a clone of the combat shares it.
/// </para>
/// <para>
/// The agent's document (<c>CombatSnapshot.Write</c>) writes the public part of it only: what kind of
/// step it is, the card or potion in play, the action, the answers given so far (a pick from the draw
/// pile by its card, not by its place in the true order) and the step's start as the player saw it --
/// the position the action was taken from, its unseen cards a multiset and no stream. Read back, that
/// start is the <see cref="Before" /> of a step from a document (<see cref="FromDocument" />): its
/// replay runs on the loaded copy's own streams and order and reaches the screen's place in the step,
/// not necessarily the screen's piles (what the step drew before the screen is the loaded world's), so
/// there the answering copy is put back WHOLE (<see cref="ScreenHiddenState" />), the document's
/// position exactly, and the answer and the rest of the step -- the played card to its pile and its
/// after-play hooks, the hand draw after a turn-start pick, the turn end after Well-Laid Plans' --
/// play on it. The save document (<c>CombatSnapshot.WriteSave</c>) adds the true world: the start's
/// draw order and streams and each answer's hidden state.
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

    /// <summary>
    /// The step's own stream as the step found it. Never changed either. Null for a step read from
    /// the agent's document, whose start has no true stream (<see cref="FromDocument" />).
    /// </summary>
    public required CountingRandom? RngBefore { get; init; }

    /// <summary>The step's action and target, as the caller gave them.</summary>
    public required int Action { get; init; }

    public required int TargetEnemyIndex { get; init; }

    /// <summary>The answers given so far, in the order the step asked for them.</summary>
    public required IReadOnlyList<StepAnswer> Answers { get; init; }

    /// <summary>
    /// For each of <see cref="Answers" />, the hidden state of the combat that gave it, as it stood
    /// on the screen: what the replay puts back at that screen before it resolves the answer.
    /// </summary>
    public required IReadOnlyList<ScreenHiddenState> Hidden { get; init; }

    /// <summary>
    /// Whether <see cref="Before" /> is the step's start as the agent's document gave it (public,
    /// in the loaded copy's world), not the true one: its replay reaches the screen's place in the
    /// step, and the answering copy is put back whole there (<see cref="ScreenHiddenState.Whole" />).
    /// </summary>
    public bool FromDocument => RngBefore is null;
}

/// <summary>
/// An answer a stopped step was given (<see cref="HookChoiceReplay.Answers" />): the index into its
/// screen's candidates, and the card it took when the screen chose from the draw pile
/// (<c>ChoosesFromDrawPile</c>), null otherwise. The agent's document names such a pick by its card
/// alone, since the index points into the true order; read back, its index is -1 and the replay takes
/// the candidate that offers that card in its own pile.
/// </summary>
public readonly record struct StepAnswer(int Index, CardInstance? Pick);

/// <summary>
/// The step running now on this thread (<c>CombatEngine.Step</c> sets it and puts the previous one
/// back before it returns): which combat, what it can be replayed from, and where its answers come
/// from. Null outside a step, and for a step that cannot be replayed.
/// </summary>
internal sealed class StepInProgress
{
    public required CombatState State { get; init; }

    public required CombatState Before { get; init; }

    /// <inheritdoc cref="HookChoiceReplay.RngBefore" />
    public required CountingRandom? RngBefore { get; init; }

    /// <summary>The step's stream, which a screen's answer resolves on.</summary>
    public required Random Rng { get; init; }

    public required int Action { get; init; }

    public required int TargetEnemyIndex { get; init; }

    public required List<StepAnswer> Answers { get; init; }

    /// <inheritdoc cref="HookChoiceReplay.Hidden" />
    public required List<ScreenHiddenState> Hidden { get; init; }

    /// <summary>How many of <see cref="Answers" /> this run of the step has used.</summary>
    public int Used;

    /// <summary>
    /// The next answer, with the combat put back to the hidden state of the copy that gave it
    /// (<see cref="HookChoiceReplay.Hidden" />): the screen the replay has reached is the one that
    /// copy stood on. False when every answer is used: the step stops here.
    /// </summary>
    /// <remarks>
    /// A pick from the draw pile that the agent's document named by card (answer -1) is the
    /// candidate that offers that card in the replay's own pile, the first when none does.
    /// </remarks>
    public bool TryNextAnswer(out int answer)
    {
        if (Used >= Answers.Count)
        {
            answer = -1;
            return false;
        }

        Hidden[Used].PutBack(State, Rng);
        answer = Answers[Used].Index;
        if (answer < 0 && Answers[Used].Pick is { } card && State.PendingSelection is { } open)
        {
            answer = Math.Max(0, open.Candidates.FindIndex(index => index >= 0 && index < State.DrawPile.Count && State.DrawPile[index] == card));
        }

        Used++;
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
/// <para>
/// A step read from the agent's document (<see cref="HookChoiceReplay.FromDocument" />) replays from
/// a start that is public, not true, so the replay reaches the screen's place in the step over piles
/// of its own world: there the copy answering is put back <see cref="Whole" />, in place (the lists,
/// the enemies and the streams are the replay's objects, refilled), so that code in the middle of the
/// step that holds one -- the turn end's baseline of the enemies, a play's target -- goes on with it.
/// An answer given before the document was written has nothing to put back (<see cref="None" />).
/// </para>
/// </remarks>
public sealed class ScreenHiddenState
{
    internal ScreenHiddenState(CountingRandom?[] streams, CountingRandom? stepRng, CardInstance[] drawPile, int[]? candidates, CombatState? whole = null)
    {
        Streams = streams;
        StepRng = stepRng;
        DrawPile = drawPile;
        Candidates = candidates;
        Whole = whole;
    }

    /// <summary>Nothing to put back: an answer a document gave, whose copy is gone.</summary>
    internal static readonly ScreenHiddenState None = new(new CountingRandom?[StateCloning.StreamCount], null, [], null);

    /// <summary>
    /// The whole copy that gave the answer, as it stood on the screen (its screen without the stopped
    /// step), for a step read from the agent's document; null otherwise.
    /// </summary>
    internal CombatState? Whole { get; }

    /// <summary>
    /// <paramref name="state" /> whole, with <paramref name="rng" /> as its step's stream: for a step
    /// read from the agent's document (<see cref="HookChoiceReplay.FromDocument" />).
    /// </summary>
    public static ScreenHiddenState WholeOf(CombatState state, Random rng)
    {
        var hidden = Of(state, rng);
        var whole = state.Clone();
        whole.PendingSelection = whole.PendingSelection?.Clone(replay: null);
        return new(hidden.Streams, hidden.StepRng, hidden.DrawPile, hidden.Candidates, whole);
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
        if (Whole is { } whole)
        {
            whole.CopyTo(state, inPlace: true);
            if (StepRng is not null && rng is CountingRandom wholeStep)
            {
                wholeStep.Become(StepRng);
            }

            return;
        }

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
