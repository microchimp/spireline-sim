using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Sts2Emulator.Tests.Invariants;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The emulator has no Play pile: <c>CombatState.HeldPlayedCard</c> holds the played card while the
/// screen its play opened stands (a step on a plain Random leaves it standing), and the step's tail
/// files it once the screen is closed. In the game the card moves from the Play pile to its result
/// pile once (`PlayCardAction.ExecuteAction`), so one Hidden Daggers played leaves one Hidden Daggers
/// in the discard pile, however the state was copied or saved in between. The rebuild lane saw a
/// second one after a search loaded a document taken mid-screen (docs/kits/HeldPlayedCard.md): the
/// slot had outlived its play, and the next step filed its card again.
/// </summary>
public class HeldPlayedCardTests
{
    private static CombatState HiddenDaggersFight(params CardInstance[] alsoInHand)
    {
        var fight = Fight
            .Hand([
                Card(SI.HiddenDaggers),
                Card(SI.StrikeSilent),
                Card(SI.DefendSilent),
                .. alsoInHand,
            ])
            .Energy(1)
            .Enemy(hp: 40);
        CardUids.NumberAll(fight.State);
        return fight.State;
    }

    /// <summary>Plays the Daggers on a plain Random: the screen is left standing and the card is held.</summary>
    private static CombatState DaggersHeld()
    {
        var state = HiddenDaggersFight(Card(SI.Neutralize));

        CombatEngine.Step(state, 0, new Random(0));

        Assert.NotNull(state.PendingSelection);
        Assert.Equal(SI.HiddenDaggers, state.HeldPlayedCard?.DefId);
        Assert.Equal(0, DaggersInPiles(state));
        AssertAtRest(state);
        return state;
    }

    private static int DaggersInPiles(CombatState s) =>
        s
            .Hand.Concat(s.DrawPile)
            .Concat(s.DiscardPile)
            .Concat(s.ExhaustPile)
            .Count(card => card.DefId == SI.HiddenDaggers);

    /// <summary>The rest-state invariants, among them: a held card has its screen open and is in no pile.</summary>
    private static void AssertAtRest(CombatState state) =>
        Assert.Empty(StateInvariants.Check(state));

    [Fact]
    public void AFiledCardIsNotFiledTwice()
    {
        var state = DaggersHeld();
        var held = state.HeldPlayedCard!.Value;
        // The broken shape: the screen closed and the card filed, the slot left set.
        state.DiscardPile.Add(held);
        state.PendingSelection = null;

        CombatEngine.Step(state, state.Hand.Count, new Random(0));

        Assert.Equal(1, DaggersInPiles(state));
        Assert.Null(state.HeldPlayedCard);
        AssertAtRest(state);
    }

    [Fact]
    public void AFiledCardUnderANewUidIsNotFiledTwice()
    {
        var state = DaggersHeld();
        var held = state.HeldPlayedCard!.Value;
        // What a document round trip gives: the same card, numbered afresh.
        state.DiscardPile.Add(held.WithUid(state.NewCardUid()));
        state.PendingSelection = null;

        CombatEngine.Step(state, state.Hand.Count, new Random(0));

        Assert.Equal(1, DaggersInPiles(state));
        Assert.Null(state.HeldPlayedCard);
        AssertAtRest(state);
    }

    [Fact]
    public void ATwinTheScreenDiscardedLastIsNotTheHeldCard()
    {
        // Two Hidden Daggers in hand; the played one's screen discards the other one last, so an
        // equal-valued card ends the discard pile while the held card is still its own, unfiled card.
        var state = HiddenDaggersFight(Card(SI.HiddenDaggers));
        var rng = new Random(0);

        CombatEngine.Step(state, 0, rng);
        Assert.NotNull(state.PendingSelection);
        CombatEngine.Step(state, 0, rng); // the Strike
        if (state.PendingSelection is not null)
        {
            CombatEngine.Step(state, 1, rng); // the twin, kept for last
        }

        Assert.Null(state.PendingSelection);
        Assert.Null(state.HeldPlayedCard);
        Assert.Equal(2, DaggersInPiles(state));
        Assert.Equal(2, state.Hand.Count(card => card.DefId == SI.Shiv));
        AssertAtRest(state);
    }

    [Fact]
    public void SnapshotLoadedSelectionFilesTheCardOnce()
    {
        var state = DaggersHeld();

        var read = CombatSnapshot.Read(CombatSnapshot.Write(state), seed: 1, out _);

        // A document has no play pile: the held card is written into its pile, and the slot is not restored.
        Assert.Null(read.HeldPlayedCard);
        Assert.Contains(read.DiscardPile, card => card.DefId == SI.HiddenDaggers);
        Assert.NotNull(read.PendingSelection);
        AnswerBothPicksThenEndTheTurn(read);
    }

    [Fact]
    public void SnapshotLoadedStoppedStepFilesTheCardOnce()
    {
        // A replayable step (CountingRandom) stops inside OnPlay instead: the screen carries the stopped
        // step and nothing is held; the document carries the stopped step.
        var fight = Fight
            .Hand(
                Card(SI.HiddenDaggers),
                Card(SI.StrikeSilent),
                Card(SI.DefendSilent),
                Card(SI.Neutralize)
            )
            .Energy(1)
            .Enemy(hp: 40);
        CardUids.NumberAll(fight.State);
        fight.Play();
        Assert.NotNull(fight.State.PendingSelection);
        Assert.Null(fight.State.HeldPlayedCard);

        var read = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 1, out _);

        Assert.Null(read.HeldPlayedCard);
        Assert.NotNull(read.PendingSelection);
        AnswerBothPicksThenEndTheTurn(read);
    }

    private static void AnswerBothPicksThenEndTheTurn(CombatState read)
    {
        var rng = new Random(0);
        CombatEngine.Step(read, 0, rng);
        AssertAtRest(read);
        if (read.PendingSelection is not null)
        {
            CombatEngine.Step(read, 0, rng);
            AssertAtRest(read);
        }

        Assert.Null(read.PendingSelection);
        Assert.Equal(2, read.Hand.Count(card => card.DefId == SI.Shiv));
        Assert.Equal(1, DaggersInPiles(read));
        Assert.Null(read.HeldPlayedCard);

        CombatEngine.Step(read, read.Hand.Count, rng);

        AssertAtRest(read);
        Assert.Equal(1, DaggersInPiles(read));
        Assert.Null(read.HeldPlayedCard);
    }
}
