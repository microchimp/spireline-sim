using System.Reflection;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Sts2Emulator.Tests.Invariants;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Every field the selection mechanism adds, through every lifecycle a state goes through:
/// Clone and CopyTo, Reset, a snapshot, a re-anchor. Round 2 of the turn-start branch added a
/// field (the stopped step on the combat) that Reset and Reanchor did not know about, and the
/// next step threw inside a C export. A field added here without an entry fails
/// <see cref="EveryFieldTheMechanismAddsHasAnEntry" />: decide what each lifecycle does with it,
/// add the entry, and point it at the test that shows it.
/// </summary>
public unsafe class SelectionLifecycleTests
{
    /// <summary>(type, member) -> Clone/CopyTo; Reset; Snapshot; Reanchor, each with its test.</summary>
    public static readonly Dictionary<(Type Type, string Member), string[]> Fields = new()
    {
        [(typeof(PendingCardSelection), nameof(PendingCardSelection.Replay))] =
        [
            "Clone/CopyTo: shared, never changed once made; a copy determinized on the screen puts its own streams and draw order back at the screen, before the answer -- StoppedStepLifecycleTests.CloneAndCopyToShareTheStoppedStep, DeterminizedStopTests",
            "Reset: dropped with the screen, the native handle wiped too -- AResetDropsTheStoppedStepWithItsScreen, ANativeResetAfterAStoppedStepIsANewCombat",
            "Snapshot: the agent's document writes its public part (stopped_step: the kind of step, the card or potion in play, the answers, a draw-pile pick by card, the start as the player saw it), the same for every copy that looks the same, and its load finishes the step as answering in memory does; the save document adds the true world (the start's order and streams, each answer's index and hidden state) -- StoppedStepDocumentTests, StoppedStepLifecycleTests.ASnapshotOfAStoppedPlayKeepsThePlayedCard, ASnapshotOfAStoppedTurnStartGoesOnWithTheRestOfIt, ALoadedDrawPileScreenIsAnsweredWithTheCardItOffers, ForksResampledDifferentlyAtAStoppedPlayWriteTheSameDocument, CombatsThatLookTheSameWriteTheSameDocumentWhereAStepStopped, TheAgentsDocumentOfAStoppedPlayFinishesThePlay",
            "Reanchor: an unanswered stop is finished by the emulator's own answer, then the screen goes -- EmuDump.Tests ReanchorFinishTests",
        ],
        [(typeof(RunState), nameof(RunState.PendingSelectionOptional))] =
        [
            "Clone: copied -- StateCloningTests.EveryRunStateScalarFieldSurvivesTheClone, ARunCloneKeepsAnOptionalDeckSelection",
            "Reset: cleared by RunEngine.Reset and by the selection's end -- TheOptionalFlagEndsWithTheSelectionAndTheRun",
            "Snapshot: the run has no snapshot document; the DONE action is in the action mask -- OptionalDeckSelectionTests",
            "Reanchor: combat only; the run replays (codex Walker, RunReplay) close it with DONE",
        ],
        [(typeof(CombatState), nameof(CombatState.TurnStartHeldPhase))] =
        [
            "Clone/CopyTo: copied -- StateCloningTests.EveryScalarFieldSurvivesTheClone",
            "Reset: StartFirstTurn writes it, and the native handle is wiped first -- ANativeResetAfterAStoppedStepIsANewCombat",
            "Snapshot: carried (turn_start_held_phase) -- CombatSnapshotTests.EveryOptionalFieldSetToANonDefaultValueRoundTripsExactly",
            "Reanchor: kept from the emulator's own last step; the anchor writes only what the game shows",
        ],
        [(typeof(CombatState), nameof(CombatState.TurnStartHeldRelic))] =
        [
            "Clone/CopyTo: copied -- StateCloningTests.EveryScalarFieldSurvivesTheClone",
            "Reset: as TurnStartHeldPhase",
            "Snapshot: carried (turn_start_held_relic)",
            "Reanchor: as TurnStartHeldPhase",
        ],
    };

    [Fact]
    public void EveryFieldTheMechanismAddsHasAnEntry()
    {
        foreach (var ((type, member), lifecycles) in Fields)
        {
            Assert.True(
                type.GetMember(member, BindingFlags.Public | BindingFlags.Instance).Length == 1,
                $"{type.Name}.{member} is listed but does not exist"
            );
            Assert.Equal(4, lifecycles.Length);
        }

        // The screen's own properties all have an entry in StoppedStepLifecycleTests; the
        // stopped step may hang nowhere else, on the run no more than on the combat.
        foreach (var field in typeof(RunState).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            Assert.False(
                field.FieldType == typeof(HookChoiceReplay) || field.FieldType.Name == "StepInProgress",
                $"RunState.{field.Name} holds a stopped step; keep it on PendingSelection.Replay"
            );
        }
    }

    private static int Create(int seed)
    {
        delegate* unmanaged<int, int> f = &NativeExports.Sts2_Create;
        return f(seed);
    }

    private static void Destroy(int handle)
    {
        delegate* unmanaged<int, void> f = &NativeExports.Sts2_Destroy;
        f(handle);
    }

    private static void Reset(int handle)
    {
        delegate* unmanaged<int, int*, void> f = &NativeExports.Sts2_Reset;
        fixed (int* obs = new int[NativeExports.OBS_SIZE])
        {
            f(handle, obs);
        }
    }

    private static void ResetEncounter(int handle, int encounterId)
    {
        delegate* unmanaged<int, int, int*, void> f = &NativeExports.Sts2_ResetEncounter;
        fixed (int* obs = new int[NativeExports.OBS_SIZE])
        {
            f(handle, encounterId, obs);
        }
    }

    private static void Step(int handle, int action)
    {
        delegate* unmanaged<int, int, int*, float*, int> f = &NativeExports.Sts2_Step;
        float reward;
        fixed (int* obs = new int[NativeExports.OBS_SIZE])
        {
            f(handle, action, obs, &reward);
        }
    }

    /// <summary>
    /// Every scalar field of the handle's combat set off its value, the way an episode leaves
    /// counters and latches behind, and a screen left stopped in the middle of a step.
    /// </summary>
    private static void Dirty(int handle)
    {
        var state = NativeExports.StateOf(handle)!;
        // A real stopped step: Tyranny's exhaust at the turn start the end of turn runs into.
        BuffSystem.Apply(state.PlayerBuffs, BuffId.Tyranny, 1);
        Step(handle, state.Hand.Count);
        Assert.NotNull(state.PendingSelection?.Replay);
        int stamp = 1000;
        foreach (var field in typeof(CombatState).GetFields(BindingFlags.Public | BindingFlags.Instance))
        {
            if (field.FieldType == typeof(int))
            {
                field.SetValue(state, stamp++);
            }
            else if (field.FieldType == typeof(bool))
            {
                field.SetValue(state, !(bool)field.GetValue(state)!);
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ANativeResetAfterAStoppedStepIsANewCombat(bool namedEncounter)
    {
        // The pooled combat is reused across episodes (Sts2_Reset and the others), so a field
        // the reset does not write is the last episode's: 69 of them were, and the stopped step
        // was until it moved onto the screen. After a reset the reused handle must be, field
        // for field, a handle that never played.
        int reused = Create(seed: 21);
        int fresh = Create(seed: 21);
        try
        {
            Dirty(reused);
            if (namedEncounter)
            {
                ResetEncounter(reused, RunConstants.SlitheringStranglerEncounterId);
                ResetEncounter(fresh, RunConstants.SlitheringStranglerEncounterId);
            }
            else
            {
                Reset(reused);
                Reset(fresh);
            }

            Assert.Empty(DeepState.Diff(NativeExports.StateOf(fresh), NativeExports.StateOf(reused), limit: 12));

            // And it plays on: the old stopped step is not brought back by the next step.
            var state = NativeExports.StateOf(reused)!;
            int turn = state.Turn;
            Step(reused, state.Hand.Count);
            Assert.True(state.Turn == turn + 1 || state.PendingSelection is { Replay: not null });
        }
        finally
        {
            Destroy(reused);
            Destroy(fresh);
        }
    }

    [Fact]
    public void TheOptionalFlagEndsWithTheSelectionAndTheRun()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Relics.Add(new RelicInstance(RelicEffects.Kifuda, 0));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.Kifuda);
        engine.State.Phase = RunPhase.TransformSelect;
        Assert.True(engine.State.PendingSelectionOptional);

        // DONE ends the selection, and the flag with it.
        engine.Step(engine.State.Deck.Count, -1, out _, out _, out _);
        Assert.False(engine.State.PendingSelectionOptional);

        // A run reset on an open one starts without it.
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.Kifuda);
        Assert.True(engine.State.PendingSelectionOptional);
        engine.Reset("ABCDEF");
        Assert.False(engine.State.PendingSelectionOptional);
    }

    [Fact]
    public void ARunCloneKeepsAnOptionalDeckSelection()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.GnarledHammer);

        var copy = engine.State.Clone();

        Assert.True(copy.PendingSelectionOptional);
        Assert.Equal(engine.State.PendingSelectionKind, copy.PendingSelectionKind);
    }
}

