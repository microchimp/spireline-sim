using System.Reflection;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// A step stopped on a screen raised inside it (HookChoiceReplay) through every lifecycle a combat
/// goes through: Clone and CopyTo, Reset, Snapshot, Reanchor. The stopped step lives on its screen
/// (`PendingCardSelection.Replay`), so whatever clears or replaces the screen drops it; nothing on
/// the combat itself may hold one. The round-2 branch kept it on the combat (`HookChoice`): a reset
/// or a re-anchor cleared the screen and left the stopped step behind, and the next step threw.
/// </summary>
public class StoppedStepLifecycleTests
{
    /// <summary>
    /// Every property of the screen and how each lifecycle treats it. A property added without an
    /// entry here fails <see cref="EveryScreenPropertyHasALifecycle" />: decide what Clone, Reset,
    /// a snapshot and a re-anchor do with it, then add it (and a behaviour test if it is not data).
    /// </summary>
    private static readonly Dictionary<string, string> ScreenLifecycle = new()
    {
        [nameof(PendingCardSelection.Kind)] = "data: cloned, dropped by Reset, written, cleared with the screen by Reanchor",
        [nameof(PendingCardSelection.Candidates)] = "data",
        [nameof(PendingCardSelection.SourceCardDefId)] = "data",
        [nameof(PendingCardSelection.Amount)] = "data",
        [nameof(PendingCardSelection.GeneratedCandidates)] = "data",
        [nameof(PendingCardSelection.AfterSelectionToHand)] = "data",
        [nameof(PendingCardSelection.Skippable)] = "data",
        [nameof(PendingCardSelection.GeneratedUpgraded)] = "data",
        [nameof(PendingCardSelection.GeneratedRetained)] = "data",
        [nameof(PendingCardSelection.Replay)] =
            "shared by Clone/CopyTo (never changed once made), dropped with the screen by Reset and Reanchor, "
            + "written by the save document only (stopped_step), never by the agent's (it reaches the true "
            + "world), and put back on the loaded screen: the tests below",
    };

    [Fact]
    public void EveryScreenPropertyHasALifecycle()
    {
        var properties = typeof(PendingCardSelection)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Select(p => p.Name)
            .Order()
            .ToList();

        Assert.Equal(ScreenLifecycle.Keys.Order().ToList(), properties);
    }

    [Fact]
    public void NoCombatFieldHoldsAStoppedStep()
    {
        // The stopped step must die with its screen, so it may only hang off the screen.
        foreach (var field in typeof(CombatState).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            Assert.False(
                field.FieldType == typeof(HookChoiceReplay)
                    || field.FieldType.Name == "StepInProgress",
                $"CombatState.{field.Name} holds a stopped step; keep it on PendingSelection.Replay"
            );
        }
    }

    private static Fight TyrannyFight() =>
        Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame, IC.TwinStrike)])
            .Enemy(hp: 500, intent: new Intent(IntentType.Attack, 7))
            .PlayerBuff(BuffId.Tyranny, 1);

    private static (Fight Fight, CountingRandom Rng) Stopped()
    {
        var fight = TyrannyFight();
        var rng = new CountingRandom(11);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        Assert.NotNull(fight.Pending!.Replay);
        return (fight, rng);
    }

    [Fact]
    public void CloneAndCopyToShareTheStoppedStep()
    {
        var (fight, rng) = Stopped();

        var clone = fight.State.Clone();
        var onto = new CombatState();
        fight.State.CopyTo(onto);

        Assert.Same(fight.Pending!.Replay, clone.PendingSelection!.Replay);
        Assert.Same(fight.Pending.Replay, onto.PendingSelection!.Replay);

        // Answering any of the three plays the same step out the same way.
        var r1 = rng.Clone();
        var r2 = rng.Clone();
        CombatEngine.Step(fight.State, 2, rng);
        CombatEngine.Step(clone, 2, r1);
        CombatEngine.Step(onto, 2, r2);
        var expected = CombatSnapshot.Write(fight.State);
        Assert.Equal(expected, CombatSnapshot.Write(clone));
        Assert.Equal(expected, CombatSnapshot.Write(onto));
        // The answered step left the original's record alone: the clone's was the same object.
        Assert.Null(fight.Pending);
    }

    [Fact]
    public void AResetDropsTheStoppedStepWithItsScreen()
    {
        // The probe that sank round 2: Tyranny's stop, a reset of the same pooled state, a step.
        var (fight, rng) = Stopped();

        CombatFactory.Reset(fight.State, rng);

        Assert.Null(fight.Pending);
        int turn = fight.State.Turn;
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        // The new combat's end of turn, not the old step brought back.
        Assert.Equal(turn + 1, fight.State.Turn);
        Assert.Empty(fight.State.ExhaustPile);
    }

    [Fact]
    public void AReanchorThatClearsTheScreenDropsTheStoppedStep()
    {
        // tools/emudump Reanchor: an unexpected screen is cleared (`q.PendingSelection = null`)
        // and the next anchored step starts clean.
        var (fight, rng) = Stopped();
        int turn = fight.State.Turn;

        fight.State.PendingSelection = null;

        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        Assert.Equal(turn + 1, fight.State.Turn);
    }

    /// <summary>
    /// The reviewer's probe: a save document written on Burning Pact's screen, read back, answered.
    /// The play is stopped inside OnPlay, so a document of the screen alone leaves the card in no pile
    /// and its after-play hooks unrun (7 cards become 6); the save document carries the stopped step,
    /// and the loaded copy's answer finishes the play: the card goes to the discard.
    /// </summary>
    [Fact]
    public void ASnapshotOfAStoppedPlayKeepsThePlayedCard()
    {
        var fight = Fight
            .Hand(Card(IC.BurningPact), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
            .Energy(3)
            .Draw([.. Pile(IC.Anger, IC.Inflame, IC.TwinStrike)])
            .Enemy(hp: 400);
        var rng = new CountingRandom(11);
        CombatEngine.Step(fight.State, 0, rng);
        Assert.NotNull(fight.Pending!.Replay);
        static int Cards(CombatState s) => s.Hand.Count + s.DrawPile.Count + s.DiscardPile.Count + s.ExhaustPile.Count;

        var loaded = CombatSnapshot.Read(CombatSnapshot.WriteSave(fight.State), seed: 0, out _);
        Assert.NotNull(loaded.PendingSelection!.Replay);
        // The load shuffled the three unseen cards with its own seed; hand it the original's order too.
        ShareStreams(from: fight.State, to: loaded);
        loaded.DrawPile = [.. fight.State.DrawPile];
        CombatEngine.Step(loaded, 0, rng.Clone());
        CombatEngine.Step(fight.State, 0, rng);

        Assert.Null(loaded.PendingSelection);
        Assert.Equal(7, Cards(loaded));
        Assert.Contains(loaded.DiscardPile, card => card.DefId == IC.BurningPact);
        Assert.Equal(CombatSnapshot.WriteSave(fight.State), CombatSnapshot.WriteSave(loaded));
    }

    /// <summary>
    /// A turn start stopped on Tyranny's screen: a save document written, read and answered, the rest
    /// of the turn start runs, as on the original, and a save document of the step's start and of
    /// every answer so far writes back the same.
    /// </summary>
    [Fact]
    public void ASnapshotOfAStoppedTurnStartGoesOnWithTheRestOfIt()
    {
        var (fight, rng) = Stopped();
        byte[] written = CombatSnapshot.WriteSave(fight.State);

        var loaded = CombatSnapshot.Read(written, seed: 0, out _);
        Assert.Equal(written, CombatSnapshot.WriteSave(loaded));
        ShareStreams(from: fight.State, to: loaded);
        CombatEngine.Step(loaded, 2, rng.Clone());
        CombatEngine.Step(fight.State, 2, rng);

        Assert.Null(loaded.PendingSelection);
        Assert.Equal(CombatSnapshot.WriteSave(fight.State), CombatSnapshot.WriteSave(loaded));
    }

    /// <summary>
    /// A draw-pile screen (Stratagem, inside the turn start's shuffle) loaded under another seed: the
    /// load shuffles the unordered middle and follows the candidates through it, and the answer picks
    /// the card the loaded screen offered at that index, not the card at that index of the true pile.
    /// </summary>
    [Fact]
    public void ALoadedDrawPileScreenIsAnsweredWithTheCardItOffers()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Enemy(hp: 400)
            .Draw()
            .Discard([.. Pile(IC.Anger, IC.DefendIronclad, IC.Inflame, IC.TwinStrike, IC.ShrugItOff, IC.PommelStrike)])
            .PlayerBuff(BuffId.StratagemPower, 1);
        var rng = new CountingRandom(11);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);

        foreach (int seed in new[] { 1, 2, 3 })
        {
            var loaded = CombatSnapshot.Read(CombatSnapshot.WriteSave(fight.State), seed, out _);
            var offered = loaded.DrawPile[loaded.PendingSelection!.Candidates[0]];

            CombatEngine.Step(loaded, 0, new CountingRandom(seed));

            Assert.Null(loaded.PendingSelection);
            Assert.Equal(offered, loaded.Hand[0]);
        }
    }

    private static readonly int[] Twelve =
    [
        IC.Anger, IC.DefendIronclad, IC.Inflame, IC.TwinStrike, IC.ShrugItOff, IC.PommelStrike,
        IC.IronWave, IC.Thunderclap, IC.Bash, IC.StrikeIronclad, IC.Armaments, IC.Headbutt,
    ];

    /// <summary>Burning Pact stopped on its screen, over twelve unseen cards in this order.</summary>
    private static Fight StoppedPact(int[] pile, int seed)
    {
        var fight = Fight
            .Hand(Card(IC.BurningPact), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
            .Seed(seed)
            .Energy(3)
            .Draw([.. Pile(pile)])
            .Enemy(hp: 400);
        CombatEngine.Step(fight.State, 0, new CountingRandom(seed));
        Assert.NotNull(fight.Pending!.Replay);
        return fight;
    }

    /// <summary>A search fork, as <c>Sts2_Clone(resample)</c> makes one: the streams reseeded, the unseen order reshuffled.</summary>
    private static CombatState Determinized(CombatState state, int seed)
    {
        var fork = state.Clone(seed);
        fork.ReshuffleUnknownDrawOrder(new CountingRandom(seed));
        return fork;
    }

    private static bool Holds(byte[] document, string text) =>
        document.AsSpan().IndexOf(System.Text.Encoding.UTF8.GetBytes(text)) >= 0;

    /// <summary>
    /// The reviewer's probe, round 3: a fork determinized at a stopped Burning Pact wrote the step's
    /// start in its TRUE world (stopped_step.before_draw, before_streams), which names the two cards
    /// the Pact really draws. The agent's document is the position as the player sees it: forks
    /// resampled differently write the original's document, byte for byte, with no stopped step in
    /// it, while the save document still holds the step.
    /// </summary>
    [Fact]
    public void ForksResampledDifferentlyAtAStoppedPlayWriteTheSameDocument()
    {
        var fight = StoppedPact(Twelve, seed: 77);
        byte[] original = CombatSnapshot.Write(fight.State);

        var forks = new[] { 4242, 7, 999, 31337 }.Select(seed => Determinized(fight.State, seed)).ToList();
        Assert.True(forks.Select(fork => string.Join(",", fork.DrawPile.Select(card => card.DefId))).Distinct().Count() > 1);

        Assert.False(Holds(original, "stopped_step"));
        Assert.All(forks, fork => Assert.Equal(original, CombatSnapshot.Write(fork)));
        Assert.True(Holds(CombatSnapshot.WriteSave(fight.State), "stopped_step"));
    }

    /// <summary>
    /// Two combats that look the same -- the same hand and the same unseen cards, in a different
    /// true order and on different streams -- write the same document at a stopped play, and at a
    /// stopped turn start, where the step's start holds the whole pile left in true order. Their save
    /// documents do not.
    /// </summary>
    [Fact]
    public void CombatsThatLookTheSameWriteTheSameDocumentWhereAStepStopped()
    {
        var pact = StoppedPact(Twelve, seed: 1);
        var twin = StoppedPact([.. Twelve.Reverse()], seed: 2);
        Assert.Equal(CombatSnapshot.Write(pact.State), CombatSnapshot.Write(twin.State));
        Assert.NotEqual(CombatSnapshot.WriteSave(pact.State), CombatSnapshot.WriteSave(twin.State));

        // Tyranny: the hand drawn before the screen is the same six Strikes (five, and Tyranny's
        // own draw); the seven cards under them are not in the same order.
        static Fight TurnStart(int[] under, int seed)
        {
            var fight = Fight
                .Hand(Card(IC.DefendIronclad))
                .Seed(seed)
                .Energy(3)
                .Draw([.. Pile([.. Enumerable.Repeat(IC.StrikeIronclad, 6), .. under])])
                .Enemy(hp: 500, intent: new Intent(IntentType.Attack, 7))
                .PlayerBuff(BuffId.Tyranny, 1);
            CombatEngine.Step(fight.State, fight.State.Hand.Count, new CountingRandom(seed));
            Assert.NotNull(fight.Pending!.Replay);
            return fight;
        }

        var start = TurnStart(Twelve[..7], seed: 3);
        var other = TurnStart([.. Twelve[..7].Reverse()], seed: 4);
        Assert.Equal(CombatSnapshot.Write(start.State), CombatSnapshot.Write(other.State));
        Assert.NotEqual(CombatSnapshot.WriteSave(start.State), CombatSnapshot.WriteSave(other.State));
    }

    /// <summary>
    /// The agent's document of a stopped play, loaded, is the screen alone, as a live document's is:
    /// there is no step to replay, and the answer exhausts and draws from the loaded copy's own pile.
    /// </summary>
    [Fact]
    public void TheAgentsDocumentOfAStoppedPlayLoadsAsTheScreenAlone()
    {
        var fight = StoppedPact(Twelve, seed: 77);

        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        Assert.Null(loaded.PendingSelection!.Replay);
        var top = loaded.DrawPile.Take(2).ToList();
        CombatEngine.Step(loaded, 0, new CountingRandom(5));

        Assert.Null(loaded.PendingSelection);
        Assert.Equal(top, loaded.Hand.TakeLast(2));
    }

    /// <summary>The document carries no RNG of its own: hand the loaded copy the original's streams.</summary>
    private static void ShareStreams(CombatState from, CombatState to)
    {
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            StateCloning.SetStream(to, i, StateCloning.Stream(from, i)?.Copy());
        }
    }

    [Fact]
    public void AStoppedStepLeavesNoStepRunningBehindIt()
    {
        // The running step is the thread's for the length of one Step call. A second combat
        // stepped after the first stopped stops on its own screen, with its own record.
        var (first, _) = Stopped();
        var second = TyrannyFight();
        var rng = new CountingRandom(5);

        CombatEngine.Step(second.State, second.State.Hand.Count, rng);

        Assert.NotNull(second.Pending!.Replay);
        Assert.NotSame(first.Pending!.Replay, second.Pending.Replay);
    }
}
