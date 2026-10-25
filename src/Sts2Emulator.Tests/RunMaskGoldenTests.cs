using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Pins the run's action mask, all 256 entries, on every screen with a belt held and with none, to the values it had
/// before the mask versions existed (D32). Version 1 is the default of every run and must stay byte-for-byte this.
/// The values were taken on main cf74c742 (the commit before the versioned masks) and are not to be regenerated from
/// the code they pin.
/// </summary>
public class RunMaskGoldenTests
{
    private static readonly RunPhase[] Screens =
    [
        RunPhase.CardReward,
        RunPhase.TransformSelect,
        RunPhase.BundleSelect,
        RunPhase.CrystalSphere,
        RunPhase.Map,
        RunPhase.Rest,
        RunPhase.Shop,
        RunPhase.Treasure,
        RunPhase.RelicReward,
        RunPhase.Event,
    ];

    internal static readonly string[][] Belts = [[], ["BlockPotion", "BloodPotion", "FruitJuice"]];

    internal static RunEngine At(RunPhase phase, params string[] belt)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Phase = phase;
        engine.State.MaxPotionSlots = 3;
        Array.Clear(engine.State.PotionSlots);
        for (int i = 0; i < belt.Length; i++)
        {
            engine.State.PotionSlots[i] = RunNonCombatEffects.NamedPotion(belt[i]);
        }

        return engine;
    }

    internal static int[] Mask(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return mask;
    }

    internal static string Describe(int[] mask)
    {
        ulong hash = 14695981039346656037UL;
        foreach (int value in mask)
        {
            hash = unchecked((hash ^ (uint)value) * 1099511628211UL);
        }

        return $"{hash:x16}:" + string.Join(',', Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0));
    }

    private const string Golden =
        "CardReward/0=8665dce34ae37f89:0,1,2,3\n" +
        "CardReward/3=8665dce34ae37f89:0,1,2,3\n" +
        "TransformSelect/0=4ffd1ef3aa2da1ee:0,1,2,3,4,5,6,7,8,9,10\n" +
        "TransformSelect/3=4ffd1ef3aa2da1ee:0,1,2,3,4,5,6,7,8,9,10\n" +
        "BundleSelect/0=64f9281574e2349f:0,1\n" +
        "BundleSelect/3=64f9281574e2349f:0,1\n" +
        "CrystalSphere/0=d80ac658736bb725:\n" +
        "CrystalSphere/3=d80ac658736bb725:\n" +
        "Map/0=4c4e92fd21806786:0,1,2\n" +
        "Map/3=1d3b1988ba29c117:0,1,2,243,244,248,249,250\n" +
        "Rest/0=2ae1de2f4b7f1c9c:0,1,3\n" +
        "Rest/3=fbce64bae428762d:0,1,3,243,244,248,249,250\n" +
        "Shop/0=4a79dba93aa58423:13,14\n" +
        "Shop/3=798d551da1fc2a92:13,14,243,244,248,249,250\n" +
        "Treasure/0=9df37c724a089f22:3\n" +
        "Treasure/3=6ee002fde2b1f8b3:3,243,244,248,249,250\n" +
        "RelicReward/0=9df37c724a089f22:3\n" +
        "RelicReward/3=6ee002fde2b1f8b3:3,243,244,248,249,250\n" +
        "Event/0=8665dce34ae37f89:0,1,2,3\n" +
        "Event/3=b5795657b23a25f8:0,1,2,3,243,244,248,249,250";

    [Fact]
    public void VersionOneMasksAreWhatTheyWereBeforeTheVersionsExisted()
    {
        var lines = new List<string>();
        foreach (var phase in Screens)
        {
            foreach (var belt in Belts)
            {
                lines.Add($"{phase}/{belt.Length}={Describe(Mask(At(phase, belt)))}");
            }
        }

        string actual = string.Join('\n', lines);
        Assert.True(actual == Golden, "actual masks:\n" + actual);
    }
}
