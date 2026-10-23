using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>Leaks round follow-ups (Spireline DESIGN.md 7.6 L3 and L4).</summary>
public class ResampleLeakFollowupTests
{
    private const string Seed = "84G7XUWQD6";

    private static RunEngine AtMap()
    {
        var e = new RunEngine();
        e.Reset(Seed, 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }

    /// <summary>
    /// The cursor has walked past three events and the run met one of them; the other two were
    /// skipped (disallowed when it came by) and never shown. A clone keeps what the run met, in
    /// place, and puts the cursor after it. Slot 0 (the Ancient room's) and the skipped events are
    /// dealt again with the rest.
    /// </summary>
    [Fact]
    public void EventsTheCursorSkippedAreNotKeptInTheCurrentActsPrefix()
    {
        var e = AtMap();
        var events = e.State.Acts[0].Events;
        e.State.EventSequenceIndex = 4;
        e.State.VisitedEventIds.Clear();
        e.State.VisitedEventIds.Add(events[2]);

        var skippedMoved = false;
        var openingMoved = false;
        for (int seed = 0; seed < 200; seed++)
        {
            var clone = e.Clone(seed);
            var cloned = clone.State.Acts[0].Events;
            Assert.Equal(events.Length, cloned.Length);
            Assert.Equal(events.Order(), cloned.Order());
            Assert.Equal(events[2], cloned[1]);
            Assert.Equal(2, clone.State.EventSequenceIndex);
            skippedMoved |= cloned[3] != events[3];
            openingMoved |= cloned[0] != events[0];
        }

        Assert.True(skippedMoved, "an event the cursor skipped is still where it was dealt in every clone");
        Assert.True(openingMoved, "the event under the Ancient room's slot is still where it was dealt");
    }

    /// <summary>
    /// L4: the observation, as the engine writes it, holds no encounter id. The ids planted in
    /// MapChoices are ones no other observation field can take.
    /// </summary>
    [Fact]
    public void TheRunObservationHoldsNoEncounterIdFromMapChoices()
    {
        var e = AtMap();
        for (int i = 0; i < RunConstants.MapChoices; i++)
        {
            e.State.MapChoices[i] = 7_123_450 + i;
        }

        var obs = new int[1 << 14];
        var info = new int[1 << 8];
        e.WriteObservation(obs);
        e.WriteInfo(info);
        Assert.DoesNotContain(obs, v => v is >= 7_123_450 and < 7_123_450 + 64);
        Assert.DoesNotContain(info, v => v is >= 7_123_450 and < 7_123_450 + 64);
    }
}
