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

    // Version 2 (D32), taken on the commit before version 3 (90dbcd0d, fork main d857c286 plus A11 and A12): the four overlay
    // screens carry the discard slots 248.. of the held potions and nothing else is added.
    private const string GoldenVersionTwo =
        "CardReward/0=8665dce34ae37f89:0,1,2,3\n" +
        "CardReward/3=50272d31a3d6420a:0,1,2,3,248,249,250\n" +
        "TransformSelect/0=4ffd1ef3aa2da1ee:0,1,2,3,4,5,6,7,8,9,10\n" +
        "TransformSelect/3=863bcea5513adf6d:0,1,2,3,4,5,6,7,8,9,10,248,249,250\n" +
        "BundleSelect/0=64f9281574e2349f:0,1\n" +
        "BundleSelect/3=2eba7863cdd4f720:0,1,248,249,250\n" +
        "CrystalSphere/0=d80ac658736bb725:\n" +
        "CrystalSphere/3=a1cc16a6cc5e79a6:248,249,250\n" +
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

    // Version 3 (D34): the same table plus, on the four overlay screens, the use slots of the AnyTime potions held (Blood
    // Potion, Fruit Juice: 243, 244). The seven screens whose belt was open before are the same lines as in version 1.
    private const string GoldenVersionThree =
        "CardReward/0=8665dce34ae37f89:0,1,2,3\n" +
        "CardReward/3=b5795657b23a25f8:0,1,2,3,243,244,248,249,250\n" +
        "TransformSelect/0=4ffd1ef3aa2da1ee:0,1,2,3,4,5,6,7,8,9,10\n" +
        "TransformSelect/3=20e9a57f42d6fb7f:0,1,2,3,4,5,6,7,8,9,10,243,244,248,249,250\n" +
        "BundleSelect/0=64f9281574e2349f:0,1\n" +
        "BundleSelect/3=940ca189dc38db0e:0,1,243,244,248,249,250\n" +
        "CrystalSphere/0=d80ac658736bb725:\n" +
        "CrystalSphere/3=071e3fccdac25d94:243,244,248,249,250\n" +
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

    // Welcome to Wongos (D34): the page's whole mask at four purses (50 buys nothing, 100 the bargain bin, 250 also the featured item,
    // 300 also the mystery box), with no belt and with the three-potion belt. The Leave (action 3) is on the native mask at every
    // version: versions 1, 2, 3 and 4 are the same four lines (version 4 only tells the policy-side reader to show Leave).
    private const string GoldenWongos =
        "Wongos/50/0=9df37c724a089f22:3\n" +
        "Wongos/50/3=6ee002fde2b1f8b3:3,243,244,248,249,250\n" +
        "Wongos/100/0=536800a5ef15cb27:0,3\n" +
        "Wongos/100/3=827b7a1a566c7196:0,3,243,244,248,249,250\n" +
        "Wongos/250/0=2ae1de2f4b7f1c9c:0,1,3\n" +
        "Wongos/250/3=fbce64bae428762d:0,1,3,243,244,248,249,250\n" +
        "Wongos/300/0=8665dce34ae37f89:0,1,2,3\n" +
        "Wongos/300/3=b5795657b23a25f8:0,1,2,3,243,244,248,249,250";

    private static string WongosTable(int version)
    {
        var lines = new List<string>();
        foreach (int purse in new[] { 50, 100, 250, 300 })
        {
            foreach (var belt in Belts)
            {
                var engine = At(RunPhase.Event, belt);
                engine.State.Gold = purse;
                RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventWelcomeToWongos);
                engine.MaskVersion = version;
                lines.Add($"Wongos/{purse}/{belt.Length}={Describe(Mask(engine))}");
            }
        }

        return string.Join('\n', lines);
    }

    private static string Table(int version)
    {
        var lines = new List<string>();
        foreach (var phase in Screens)
        {
            foreach (var belt in Belts)
            {
                var engine = At(phase, belt);
                engine.MaskVersion = version;
                lines.Add($"{phase}/{belt.Length}={Describe(Mask(engine))}");
            }
        }

        return string.Join('\n', lines);
    }

    [Fact]
    public void VersionOneMasksAreWhatTheyWereBeforeTheVersionsExisted()
    {
        string actual = Table(1);
        Assert.True(actual == Golden, "actual masks:\n" + actual);
    }

    [Fact]
    public void VersionTwoMasksAreWhatTheyWereBeforeVersionThree()
    {
        string actual = Table(2);
        Assert.True(actual == GoldenVersionTwo, "actual masks:\n" + actual);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void TheWongosPageCarriesItsLeaveOnTheMaskInEveryVersion(int version)
    {
        string actual = WongosTable(version);
        Assert.True(actual == GoldenWongos, "actual masks:\n" + actual);
    }

    [Fact]
    public void VersionFourIsByteIdenticalToVersionThreeOnEveryScreenAndOnTheWongosPage()
    {
        Assert.Equal(Table(3), Table(4));
        Assert.Equal(WongosTable(3), WongosTable(4));
        Assert.True(Table(4) == GoldenVersionThree, "actual masks:\n" + Table(4));
    }

    [Fact]
    public void VersionThreeMasksAreVersionTwosPlusTheUseSlotsOfTheUsablePotions()
    {
        string actual = Table(3);
        Assert.True(actual == GoldenVersionThree, "actual masks:\n" + actual);
    }
}
