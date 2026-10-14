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
            + "not written by a snapshot (the loaded screen is answered alone): the tests below",
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

    [Fact]
    public void ASnapshotCarriesTheScreenAndItsAnswerResolvesItAlone()
    {
        var (fight, _) = Stopped();

        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 0, out _);

        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, loaded.PendingSelection!.Kind);
        Assert.Null(loaded.PendingSelection.Replay);
        var picked = loaded.Hand[2];
        CombatEngine.Step(loaded, 2, new CountingRandom(3));
        Assert.Null(loaded.PendingSelection);
        Assert.Equal([picked], loaded.ExhaustPile);
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
