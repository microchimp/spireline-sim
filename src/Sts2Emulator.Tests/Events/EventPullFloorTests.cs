using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The act's event ring is pulled while the room is being CREATED (RunManager.CreateRoom), and
/// EnterMapPointInternal only appends the floor to the map point history after that. A floor-gated
/// event's IsAllowed therefore reads TotalFloor without the room it is about to fill: walking onto
/// the 7th floor, Slippery Bridge (TotalFloor &gt; 6) is still refused. Found by the run referee
/// (seed 84G7XUWQD6: the game dealt Doors of Light and Dark where the emulator dealt the bridge).
/// </summary>
public class EventPullFloorTests
{
    [Theory]
    [InlineData(RunConstants.EventSlipperyBridge, 7, false)]
    [InlineData(RunConstants.EventSlipperyBridge, 8, true)]
    [InlineData(RunConstants.EventPunchOff, 6, false)]
    [InlineData(RunConstants.EventPunchOff, 7, true)]
    public void TheRingReadsTheFloorsBeforeTheRoom(int eventId, int floorEntered, bool dealt)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        s.EventSequence = [eventId, RunConstants.EventSunkenTreasury];
        s.EventSequenceIndex = 0;
        s.VisitedEventIds.Clear();
        s.Floor = floorEntered;  // the map move has already counted the room being entered

        RunNonCombatEffects.EnterEvent(s);

        Assert.Equal(dealt ? eventId : RunConstants.EventSunkenTreasury, s.EventId);
    }
}
