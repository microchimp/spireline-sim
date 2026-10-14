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
/// pile by its card, not by its place in the true order), the step's start as the player saw it -- the
/// position the action was taken from, its unseen cards a multiset and no stream -- and what the step
/// did before the screen that the player saw happen: the cards it took off the draw pile, in the order
/// it took them (<see cref="Taken" />: the hand it drew, the card Havoc or Mayhem auto-played off the
/// top, the picks), and each screen it reached (<see cref="Screens" />: kind, source card, the
/// candidates it showed, how many cards had been taken by then). Read back, that start is the
/// <see cref="Before" /> of a step from a document (<see cref="FromDocument" />), and its replay is held
/// to that record (<see cref="StepInProgress" />): every take off the draw pile takes the recorded card
/// (swapped into the place the replay takes from, which only reorders cards the player cannot see),
/// every screen before the last shows the recorded candidates, and every screen must be the recorded
/// one -- kind, source card, candidate count and the takes before it -- or the replay is REFUSED
/// (<see cref="StoppedStepMismatch" />), as are answers left over and a pick by a card no candidate
/// offers. At the last screen the answering copy is put back WHOLE (<see cref="ScreenHiddenState" />),
/// the document's position exactly, and the answer and the rest of the step -- the played card to its
/// pile and its after-play hooks, the hand draw after a turn-start pick, the turn end after Well-Laid
/// Plans' -- play on it. The save document (<c>CombatSnapshot.WriteSave</c>) adds the true world: the
/// start's draw order and streams and each answer's hidden state.
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

    /// <summary>
    /// The cards the step took off the draw pile before the screen it stopped on, in the order it took
    /// them (every take is <c>CombatState.RemoveFromDrawPileAt</c>): the hand it drew, a card it
    /// auto-played off the pile, a pick it was given. All of them are in sight on the screen -- in
    /// hand, in play, exhausted -- so the agent's document writes them; the cards still in the pile
    /// are not among them.
    /// </summary>
    public required IReadOnlyList<CardInstance> Taken { get; init; }

    /// <summary>
    /// Each screen the step reached, in the order it reached them: one per answer in
    /// <see cref="Answers" />, and last the screen it stopped on.
    /// </summary>
    public required IReadOnlyList<StepScreen> Screens { get; init; }
}

/// <summary>
/// A screen a stopped step reached (<see cref="HookChoiceReplay.Screens" />), as the player saw it:
/// its kind, the card that asked, how many candidates it offered and which (the cards for a screen
/// over a pile, the generated ids for a choose-a-card screen), and how many cards the step had taken
/// off the draw pile by then. A replay must reach the same screen there, or it is refused. A
/// draw-pile screen's cards are kept in pile order here and written as a sorted multiset
/// (<c>CombatSnapshot</c>): which cards it offered is public, where they lay is not.
/// </summary>
public sealed record StepScreen(CardSelectionKind Kind, int Source, int Count, int Taken, CardInstance[] Cards, int[] Generated)
{
    /// <summary><paramref name="open" /> over <paramref name="state" />, with <paramref name="taken" /> cards taken so far.</summary>
    internal static StepScreen Of(CombatState state, PendingCardSelection open, int taken)
    {
        var pile = open.CandidatePile(state);
        return new(
            open.Kind,
            open.SourceCardDefId,
            open.Candidates.Count,
            taken,
            pile is null ? [] : [.. open.Candidates.Where(index => index >= 0 && index < pile.Count).Select(index => pile[index])],
            pile is null ? [.. open.GeneratedCandidates] : []
        );
    }

    /// <summary>Why <paramref name="open" />, reached after <paramref name="taken" /> takes, is not this screen; null when it is.</summary>
    internal string? Differs(PendingCardSelection open, int taken) =>
        open.Kind != Kind ? $"a {open.Kind} screen where the step reached {Kind}"
        : open.SourceCardDefId != Source ? $"the {Kind} screen of card {open.SourceCardDefId} where the step's was card {Source}"
        : open.Candidates.Count != Count ? $"the {Kind} screen with {open.Candidates.Count} candidates where the step's had {Count}"
        : taken != Taken ? $"the {Kind} screen after {taken} cards taken off the draw pile where the step had taken {Taken}"
        : null;
}

/// <summary>
/// A replay of a stopped step that did not go the way the step went (<see cref="HookChoiceReplay" />):
/// it reached another screen, or took another card or another number of cards off the draw pile
/// before one, or had answers left over, or was answered by a card no candidate offers. The answer is
/// refused and the combat is put back on its screen as it was (<c>CombatEngine.Step</c>); a document
/// whose step cannot be replayed is refused at load and at export (<c>CombatSnapshot</c>). Never a
/// guess: the copy is never put back at a screen that is not its own.
/// </summary>
public sealed class StoppedStepMismatch(string message) : InvalidOperationException($"stopped step: {message}");

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
/// back before it returns): which combat, what it can be replayed from, where its answers come
/// from, and what it has done so far that a replay must do again. Null outside a step, and for a step
/// that cannot be replayed.
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

    /// <summary>
    /// The stopped step this run replays, whose record (<see cref="HookChoiceReplay.Taken" />,
    /// <see cref="HookChoiceReplay.Screens" />) it is held to on its way to the last screen the record
    /// reached; null for a step run afresh.
    /// </summary>
    public HookChoiceReplay? Record { get; init; }

    /// <summary>How many of <see cref="Answers" /> this run of the step has used.</summary>
    public int Used;

    /// <summary>The cards this run has taken off the draw pile, in order (<see cref="HookChoiceReplay.Taken" />).</summary>
    public List<CardInstance> Taken { get; } = [];

    /// <summary>The screens this run has reached, in order (<see cref="HookChoiceReplay.Screens" />).</summary>
    public List<StepScreen> Screens { get; } = [];

    /// <summary>
    /// Whether this run is still on its way to a screen <see cref="Record" /> reached, and so held to
    /// the record, takes and screens alike.
    /// </summary>
    private bool InRecord => Record is { } record && Used < record.Screens.Count;

    /// <summary>
    /// A take off the draw pile at <paramref name="index" />, before the pile gives the card up
    /// (<c>CombatState.RemoveFromDrawPileAt</c>, the one take). On the way to a recorded screen it must
    /// be the card the step took there. A step from the agent's document starts from a pile in the
    /// loader's order, so there the recorded card is swapped into <paramref name="index" /> first,
    /// from wherever it lies: a reorder of cards the player cannot see, which the last screen's whole
    /// copy replaces anyway. A card the pile does not hold, one take too many, or (in memory, where the
    /// replay is the step's own world) any other card is a replay gone another way: refused.
    /// </summary>
    public void OnTake(int index)
    {
        var pile = State.DrawPile;
        if (Record is { } record && InRecord)
        {
            if (Taken.Count >= record.Taken.Count)
            {
                throw new StoppedStepMismatch(
                    $"the replay takes card {Taken.Count + 1} off the draw pile on its way to screen {Used + 1}, where the step took {record.Taken.Count} before its last screen"
                );
            }

            var expected = record.Taken[Taken.Count];
            if (!pile[index].Equals(expected))
            {
                int at = record.FromDocument ? pile.IndexOf(expected) : -1;
                if (at < 0)
                {
                    throw new StoppedStepMismatch(
                        $"take {Taken.Count + 1} off the draw pile was card {expected.DefId}, which the replay cannot take there"
                    );
                }

                (pile[index], pile[at]) = (pile[at], pile[index]);
            }
        }

        Taken.Add(pile[index]);
    }

    /// <summary>
    /// A look at the card at <paramref name="index" /> that decides what the step does without
    /// taking it (<c>CombatEngine.PeekDrawPile</c>). A step from the agent's document shows there the
    /// card the step takes next, when the pile holds it: the step took it from there, or the step
    /// did not stop for it and takes it later (a draw from the top), and either way the replay then
    /// decides as the step did. The take itself is held to the record.
    /// </summary>
    public void OnPeek(int index)
    {
        var pile = State.DrawPile;
        if (Record is { FromDocument: true } record && InRecord && Taken.Count < record.Taken.Count)
        {
            var expected = record.Taken[Taken.Count];
            int at = pile[index].Equals(expected) ? -1 : pile.IndexOf(expected);
            if (at >= 0)
            {
                (pile[index], pile[at]) = (pile[at], pile[index]);
            }
        }
    }

    /// <summary>
    /// The next answer, at the screen the replay has reached (<see cref="State" />'s open screen),
    /// with the combat put back to the hidden state of the copy that gave it
    /// (<see cref="HookChoiceReplay.Hidden" />). False when every answer is used: the step stops here.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A screen the record reached must be the recorded one (<see cref="StepScreen.Differs" />: kind,
    /// source card, candidate count, the takes before it), or the replay is refused.
    /// </para>
    /// <para>
    /// On a step from the agent's document, an answer given before the document was written has no
    /// copy to put back: its screen shows the candidates the player saw (Seeker Strike's three by card,
    /// a choose-a-card screen's options), and the answer takes the card the player took there -- a
    /// pick from the draw pile is named by card (answer -1), a pick from the hand or the discard is the
    /// recorded screen's card at its index. When no candidate offers that card the replay is refused,
    /// never answered with another.
    /// </para>
    /// </remarks>
    public bool TryNextAnswer(out int answer)
    {
        var open = State.PendingSelection!;
        StepScreen? recorded = null;
        if (Record is { } record && InRecord)
        {
            if (record.Screens[Used].Differs(open, Taken.Count) is { } why)
            {
                throw new StoppedStepMismatch($"screen {Used + 1} of the replay is {why}");
            }

            if (record.FromDocument && Used < Answers.Count && Hidden[Used].Whole is null)
            {
                recorded = record.Screens[Used];
                open = ShowRecorded(recorded, open);
            }
        }

        if (Used >= Answers.Count)
        {
            Screens.Add(StepScreen.Of(State, open, Taken.Count));
            answer = -1;
            return false;
        }

        Hidden[Used].PutBack(State, Rng);
        open = State.PendingSelection!;
        Screens.Add(StepScreen.Of(State, open, Taken.Count));
        answer = Answers[Used].Index;
        if (Answers[Used].Pick is { } card && (answer < 0 || recorded is not null))
        {
            answer = Offering(open, State.DrawPile, card);
        }
        else if (answer < 0)
        {
            throw new StoppedStepMismatch($"answer {Used + 1} is neither an index nor a card");
        }
        else if (recorded is not null && answer < recorded.Cards.Length && open.CandidatePile(State) is { } pile)
        {
            answer = Offering(open, pile, recorded.Cards[answer], prefer: answer);
        }

        Used++;
        return true;
    }

    /// <summary>The candidate of <paramref name="open" /> that offers <paramref name="card" />, <paramref name="prefer" /> first; refused when none does.</summary>
    private int Offering(PendingCardSelection open, List<CardInstance> pile, CardInstance card, int prefer = -1)
    {
        bool Offers(int candidate) =>
            candidate >= 0 && candidate < open.Candidates.Count
            && open.Candidates[candidate] is int index && index >= 0 && index < pile.Count && pile[index].Equals(card);

        if (Offers(prefer))
        {
            return prefer;
        }

        int found = Enumerable.Range(0, open.Candidates.Count).FirstOrDefault(Offers, -1);
        return found >= 0
            ? found
            : throw new StoppedStepMismatch($"answer {Used + 1} takes card {card.DefId}, which no candidate of the replay's {open.Kind} screen offers");
    }

    /// <summary>
    /// <paramref name="open" /> showing what <paramref name="recorded" /> showed, where the replay's
    /// world may have rolled other candidates: a draw-pile screen's are the recorded cards' places in
    /// the replay's pile, a choose-a-card screen's options are the recorded ones. A screen over the
    /// hand or the discard shows those piles as they stand.
    /// </summary>
    private PendingCardSelection ShowRecorded(StepScreen recorded, PendingCardSelection open)
    {
        var pile = open.CandidatePile(State);
        if (pile is null)
        {
            if (!open.GeneratedCandidates.SequenceEqual(recorded.Generated))
            {
                open = open.Clone(open.Replay);
                open.GeneratedCandidates.Clear();
                open.GeneratedCandidates.AddRange(recorded.Generated);
            }
        }
        else if (open.Kind.ChoosesFromDrawPile())
        {
            var used = new bool[pile.Count];
            var places = new List<int>(recorded.Cards.Length);
            foreach (var card in recorded.Cards)
            {
                int at = Enumerable.Range(0, pile.Count).FirstOrDefault(i => !used[i] && pile[i].Equals(card), -1);
                if (at < 0)
                {
                    throw new StoppedStepMismatch($"screen {Used + 1} offered card {card.DefId}, which the replay's draw pile does not hold");
                }

                used[at] = true;
                places.Add(at);
            }

            places.Sort();
            if (!places.SequenceEqual(open.Candidates))
            {
                open = open.Clone(open.Replay, candidates: places);
            }
        }

        State.PendingSelection = open;
        return open;
    }

    /// <summary>
    /// The end of a run that did not stop: every answer must have been used, and every screen the
    /// record reached reached again.
    /// </summary>
    public void Finished()
    {
        if (Used < Answers.Count)
        {
            throw new StoppedStepMismatch($"the replay ended with {Answers.Count - Used} of its {Answers.Count} answers left over");
        }

        if (Record is { } record && Used < record.Screens.Count)
        {
            throw new StoppedStepMismatch($"the replay ended before screen {Used + 1} of the {record.Screens.Count} the step reached");
        }
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
            Taken = [.. Taken],
            Screens = [.. Screens],
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
