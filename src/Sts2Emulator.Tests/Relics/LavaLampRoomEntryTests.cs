using System.Linq;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `LavaLamp.AfterRoomEntered` clears TookDamageThisCombat on entering ANY room. The
/// emulator cleared it only as a fight began, so a rest site, shop or event after a hit
/// still read the last fight's damage (run referee, seed RXVFG7FZ86).
/// </summary>
public class LavaLampRoomEntryTests
{
    [Fact]
    public void EnteringARestSiteClearsTheFlag()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Phase = RunPhase.Map;
        int i = Enumerable.Range(0, engine.State.MapNodeTypes.Length)
            .First(k => engine.State.MapNodeTypes[k] != RunConstants.NodeNone);
        engine.State.MapNodeTypes[i] = RunConstants.NodeRest;
        engine.State.TookUnblockedDamageThisCombat = true;

        engine.Step(i, -1, out _, out _, out _);

        Assert.Equal(RunPhase.Rest, engine.State.Phase);
        Assert.False(engine.State.TookUnblockedDamageThisCombat);
    }
}
