// The codex walker names the game's encounter models with the emulator's encounter names
// (Names.EncounterModelToEmulator) and compares them with the fight the run engine dealt. A weak
// encounter and its normal twin are two models and two emulator ids, so they must be two names:
// the walker's alias table once folded Corpse Slugs' and Seapunk's normal models into the weak
// names, which made every normal fight look like a mismatch once the engine dealt the normal id.
using EmuDump.Codex;
using Sts2Emulator.Core.Run;
using Xunit;

public class EncounterNameTests
{
    [Theory]
    [InlineData("ENCOUNTER.CORPSE_SLUGS_WEAK", RunConstants.CorpseSlugsEncounterId)]
    [InlineData("ENCOUNTER.CORPSE_SLUGS_NORMAL", RunConstants.CorpseSlugsNormalEncounterId)]
    [InlineData("ENCOUNTER.SEAPUNK_WEAK", RunConstants.SeapunkEncounterId)]
    [InlineData("ENCOUNTER.SEAPUNK_NORMAL", RunConstants.SeapunkNormalEncounterId)]
    public void EachModelOfAPairNamesItsOwnId(string model, int encounterId)
    {
        Assert.Equal(Names.EncounterName(encounterId), Names.EncounterModelToEmulator(model));
    }

    /// <summary>Every weak/normal pair the acts deal (the game's Weak and Normal models).</summary>
    [Theory]
    [InlineData("NIBBITS_WEAK", "NIBBITS_NORMAL")]
    [InlineData("SLIMES_WEAK", "SLIMES_NORMAL")]
    [InlineData("CORPSE_SLUGS_WEAK", "CORPSE_SLUGS_NORMAL")]
    [InlineData("SEAPUNK_WEAK", "SEAPUNK_NORMAL")]
    [InlineData("BOWLBUGS_WEAK", "BOWLBUGS_NORMAL")]
    [InlineData("EXOSKELETONS_WEAK", "EXOSKELETONS_NORMAL")]
    [InlineData("SCROLLS_OF_BITING_WEAK", "SCROLLS_OF_BITING_NORMAL")]
    [InlineData("TUNNELER_WEAK", "TUNNELER_NORMAL")]
    public void AWeakModelAndItsNormalTwinAreTwoEncounters(string weak, string normal)
    {
        string w = Names.EncounterModelToEmulator("ENCOUNTER." + weak);
        string n = Names.EncounterModelToEmulator("ENCOUNTER." + normal);
        Assert.NotEqual(w, n);
        var known = Enumerable.Range(0, 256).Select(Names.EncounterName).ToHashSet();
        Assert.Contains(w, known);
        Assert.Contains(n, known);
    }
}
