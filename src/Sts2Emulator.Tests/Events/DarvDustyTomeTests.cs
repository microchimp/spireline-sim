using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Darv's GenerateInitialOptions calls DustyTome.SetupForPlayer as it builds the tome's option,
/// so the tome's card is a PlayerRng.Rewards draw on entering Darv, taken or not (run referee,
/// seed 6HMCK2BXTJ: the game's rewards counter one ahead on entering Darv).
/// </summary>
public class DarvDustyTomeTests
{
    [Fact]
    public void TheTomesCardIsDrawnWhenItIsOffered()
    {
        int found = 0;
        for (int i = 0; i < 60 && found < 3; i++)
        {
            var engine = new RunEngine();
            engine.Reset($"DARV{i:00}TOME");
            var s = engine.State;
            s.CurrentActIndex = 1;
            int before = s.PlayerRng.Rewards.CallCount;
            int[] options = RunNonCombatEffects.GenerateAncientOptions(s, RunConstants.AncientDarv);
            if (!options.Contains(RunConstants.DarvDustyTome))
            {
                Assert.Equal(before, s.PlayerRng.Rewards.CallCount);
                continue;
            }

            found++;
            Assert.Equal(before + 1, s.PlayerRng.Rewards.CallCount);
            int card = s.EventValue0!.Value;
            RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicDustyTome);
            Assert.Equal(before + 1, s.PlayerRng.Rewards.CallCount);
            Assert.Equal(card, s.Deck[^1].DefId);
            Assert.True(s.Deck[^1].Upgraded);
        }

        Assert.True(found > 0);
    }
}
